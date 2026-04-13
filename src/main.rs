#[path = "../benchmarks/bench.rs"]
mod bench;
mod pre_processing;
mod sequential;
mod rayon;
mod std_thread;

use bench::Results;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::Instant;

const TRAIN_BIN: &str = "data/train.bin";
const TEST_BIN:  &str = "data/test.bin";
const K: usize = 3;
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];
const ISO_TARGETS: &[f64] = &[0.50, 0.75, 0.90];

fn load_or_preprocess() -> (pre_processing::MnistDataset, pre_processing::MnistDataset) {
    let train_bin = Path::new(TRAIN_BIN);
    let test_bin  = Path::new(TEST_BIN);

    if train_bin.exists() && test_bin.exists() {
        println!("Loading preprocessed data...");
        let train = pre_processing::load_preprocessed(train_bin).expect("failed to load train.bin");
        let test  = pre_processing::load_preprocessed(test_bin).expect("failed to load test.bin");
        (train, test)
    } else {
        println!("Preprocessed files not found — loading raw data and binarizing...");
        let mut train = pre_processing::load(
            Path::new("data/train-images.idx3-ubyte"),
            Path::new("data/train-labels.idx1-ubyte"),
        )
        .expect("failed to load training data");

        let mut test = pre_processing::load(
            Path::new("data/t10k-images.idx3-ubyte"),
            Path::new("data/t10k-labels.idx1-ubyte"),
        )
        .expect("failed to load test data");

        train.binarize(128);
        test.binarize(128);

        train.save_preprocessed(train_bin).expect("failed to save train.bin");
        test.save_preprocessed(test_bin).expect("failed to save test.bin");
        println!("Saved preprocessed files to {TRAIN_BIN} and {TEST_BIN}");

        (train, test)
    }
}

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn result_header(title: &str) {
    let iso_label = ISO_TARGETS
        .iter()
        .map(|&e| format!("E={:.0}%", e * 100.0))
        .collect::<Vec<_>>()
        .join(", ");
    println!("\n{}", title);
    println!(
        "  {:<14} | {:>10} | {:>10} | {:>8} | {:>8} | {:>11} | {:>23} | {:>10}",
        "method", "time", "samples/s", "speedup", "effic.", "overhead",
        format!("isoeff. ({})", iso_label),
        "mem delta"
    );
    println!("  {}", "-".repeat(112));
}

fn result_row(label: &str, threads: usize, r: &Results, baseline: Option<&Results>) {
    let iso = ISO_TARGETS
        .iter()
        .map(|&e| match baseline {
            Some(b) => format!("{:.2}x", r.isoefficiency(b, threads, e)),
            None    => "—".into(),
        })
        .collect::<Vec<_>>();

    let speedup_str  = baseline.map(|b| format!("{:.3}x", r.speedup(b))).unwrap_or_else(|| "—".into());
    let effic_str    = baseline.map(|b| format!("{:.3}",  r.efficiency(b, threads))).unwrap_or_else(|| "—".into());
    let overhead_str = baseline
        .map(|b| format!("{:.1}ms", r.parallel_overhead(b, threads).as_secs_f64() * 1000.0))
        .unwrap_or_else(|| "—".into());

    println!(
        "  {:<14} | {:>10.3}s | {:>10.1} | {:>8} | {:>8} | {:>11} | {:>7} {:>7} {:>7} | {:>10}",
        label,
        r.elapsed.as_secs_f64(),
        r.throughput(),
        speedup_str,
        effic_str,
        overhead_str,
        iso[0], iso[1], iso[2],
        format!("{:+} KB", r.memory_delta_kb),
    );
}

fn scaling_header() {
    let iso_label = ISO_TARGETS
        .iter()
        .map(|&e| format!("E={:.0}%", e * 100.0))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "  {:>7} | {:>10} | {:>10} | {:>8} | {:>8} | {:>11} | {:>23} | {:>10}",
        "threads", "time", "samples/s", "speedup", "effic.", "overhead",
        format!("isoeff. ({})", iso_label),
        "mem delta"
    );
    println!("  {}", "-".repeat(108));
}

fn scaling_row(threads: usize, r: &Results, baseline: &Results) {
    let iso = ISO_TARGETS
        .iter()
        .map(|&e| format!("{:.2}x", r.isoefficiency(baseline, threads, e)))
        .collect::<Vec<_>>();

    println!(
        "  {:>7} | {:>10.3}s | {:>10.1} | {:>8} | {:>8} | {:>11} | {:>7} {:>7} {:>7} | {:>10}",
        threads,
        r.elapsed.as_secs_f64(),
        r.throughput(),
        format!("{:.3}x", r.speedup(baseline)),
        format!("{:.3}",  r.efficiency(baseline, threads)),
        format!("{:.1}ms", r.parallel_overhead(baseline, threads).as_secs_f64() * 1000.0),
        iso[0], iso[1], iso[2],
        format!("{:+} KB", r.memory_delta_kb),
    );
}


fn run_scalability(
    train: &pre_processing::MnistDataset,
    test: &pre_processing::MnistDataset,
) {
    println!("\nRunning sequential baseline...");
    let baseline = sequential::bench(train, test, K);

    let mut rayon_results: Vec<Results> = Vec::new();
    let mut thread_results: Vec<Results> = Vec::new();

    println!("\n--- Rayon scaling (k={}, train={}, test={}) ---\n", K, train.len(), test.len());
    scaling_header();
    for &n in THREAD_COUNTS {
        let r = rayon::bench_with_threads(train, test, K, n);
        scaling_row(n, &r, &baseline);
        rayon_results.push(r);
    }

    println!("\n--- std::thread scaling (k={}, train={}, test={}) ---\n", K, train.len(), test.len());
    scaling_header();
    for &n in THREAD_COUNTS {
        let r = std_thread::bench(train, test, K, n);
        scaling_row(n, &r, &baseline);
        thread_results.push(r);
    }

    println!("\n  Efficiency = speedup / threads  (1.0 = perfect linear scaling)");
    println!("  Isoeff.   = problem size growth needed to maintain target efficiency");

    // Save all scaling rows to scalability.csv
    let mut file = csv_open_append_path(SCALABILITY_CSV_PATH);
    for (&n, r) in THREAD_COUNTS.iter().zip(rayon_results.iter()) {
        csv_write_row(&mut file, "rayon", K, test.len(), n, r, Some(&baseline));
    }
    for (&n, r) in THREAD_COUNTS.iter().zip(thread_results.iter()) {
        csv_write_row(&mut file, "std::thread", K, test.len(), n, r, Some(&baseline));
    }
    println!("\n  Scalability results appended to '{SCALABILITY_CSV_PATH}'");
}

/// Time to spawn and join `n` empty std::thread threads — pure spawn cost.
fn measure_spawn_time(n: usize) -> std::time::Duration {
    let start = Instant::now();
    thread::scope(|s| {
        for _ in 0..n { s.spawn(|| {}); }
    });
    start.elapsed()
}

/// Time to initialise a Rayon thread pool with `n` threads.
fn measure_rayon_pool_init(n: usize) -> std::time::Duration {
    let start = Instant::now();
    let _pool = ::rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("failed to build pool");
    start.elapsed()
}

/// Time for Rayon to dispatch and collect `task_count` empty tasks on `n` threads.
fn measure_rayon_dispatch(n: usize, task_count: usize) -> std::time::Duration {
    use ::rayon::prelude::*;
    let pool = ::rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("failed to build pool");
    let start = Instant::now();
    pool.install(|| (0..task_count).into_par_iter().for_each(|_| {}));
    start.elapsed()
}

fn run_overhead_analysis(test: &pre_processing::MnistDataset) {
    let task_count = test.len();

    println!("\n=== Overhead Analysis (task_count={}) ===\n", task_count);

    // Header
    println!(
        "  {:>7} | {:>14} | {:>14} | {:>16} | {:>16}",
        "threads",
        "spawn+join",
        "rayon pool init",
        "rayon dispatch",
        "spawn cost/thread",
    );
    println!("  {}", "-".repeat(80));

    for &n in THREAD_COUNTS {
        // Run each measurement multiple times and take the minimum to reduce noise
        let spawn     = (0..5).map(|_| measure_spawn_time(n))          .min().unwrap();
        let pool_init = (0..5).map(|_| measure_rayon_pool_init(n))     .min().unwrap();
        let dispatch  = (0..5).map(|_| measure_rayon_dispatch(n, task_count)).min().unwrap();

        let spawn_per_thread = spawn.as_secs_f64() * 1000.0 / n as f64;

        println!(
            "  {:>7} | {:>12.3}ms | {:>13.3}ms | {:>14.3}ms | {:>14.3}ms",
            n,
            spawn.as_secs_f64()     * 1000.0,
            pool_init.as_secs_f64() * 1000.0,
            dispatch.as_secs_f64()  * 1000.0,
            spawn_per_thread,
        );
    }

    println!("\n  spawn+join     — cost of creating and joining N empty std::thread threads");
    println!("  rayon pool init — cost of building a fresh ThreadPool with N threads");
    println!("  rayon dispatch  — cost of distributing {} empty tasks across N threads", task_count);
    println!("  spawn cost/thread — average per-thread spawn overhead");
}

const CSV_PATH: &str = "results/results.csv";
const SCALABILITY_CSV_PATH: &str = "results/scalability.csv";

fn csv_open_append_path(path: &str) -> std::fs::File {
    use std::fs::OpenOptions;
    use std::io::Write as IoWrite;

    std::fs::create_dir_all("results").expect("failed to create results/ directory");
    let exists = std::path::Path::new(path).exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|_| panic!("failed to open {path}"));

    if !exists {
        writeln!(
            file,
            "method,k,test_size,threads,time_s,samples_per_s,speedup,efficiency,\
             overhead_ms,isoeff_50,isoeff_75,isoeff_90,mem_delta_kb,accuracy_pct"
        )
        .expect("failed to write CSV header");
    }
    file
}

fn csv_open_append() -> std::fs::File {
    csv_open_append_path(CSV_PATH)
}

fn csv_write_row(
    file: &mut std::fs::File,
    method: &str,
    k: usize,
    test_size: usize,
    threads: usize,
    r: &Results,
    baseline: Option<&Results>,
) {
    use std::io::Write as IoWrite;

    let speedup  = baseline.map(|b| r.speedup(b));
    let effic    = baseline.map(|b| r.efficiency(b, threads));
    let overhead = baseline.map(|b| r.parallel_overhead(b, threads).as_secs_f64() * 1000.0);
    let iso50    = baseline.map(|b| r.isoefficiency(b, threads, 0.50));
    let iso75    = baseline.map(|b| r.isoefficiency(b, threads, 0.75));
    let iso90    = baseline.map(|b| r.isoefficiency(b, threads, 0.90));

    writeln!(
        file,
        "{},{},{},{},{:.6},{:.3},{},{},{},{},{},{},{},{:.2}",
        method, k, test_size, threads,
        r.elapsed.as_secs_f64(),
        r.throughput(),
        speedup .map(|v| format!("{:.6}", v)).unwrap_or_default(),
        effic   .map(|v| format!("{:.6}", v)).unwrap_or_default(),
        overhead.map(|v| format!("{:.3}", v)).unwrap_or_default(),
        iso50   .map(|v| format!("{:.6}", v)).unwrap_or_default(),
        iso75   .map(|v| format!("{:.6}", v)).unwrap_or_default(),
        iso90   .map(|v| format!("{:.6}", v)).unwrap_or_default(),
        r.memory_delta_kb,
        r.accuracy(),
    )
    .expect("failed to write CSV row");
}

fn save_csv(
    k: usize,
    test_size: usize,
    num_threads: usize,
    seq: &Results,
    ray: &Results,
    thr: &Results,
) {
    let mut file = csv_open_append();
    csv_write_row(&mut file, "sequential",  k, test_size, 1,           seq, None);
    csv_write_row(&mut file, "rayon",       k, test_size, num_threads, ray, Some(seq));
    csv_write_row(&mut file, "std::thread", k, test_size, num_threads, thr, Some(seq));
    println!("\n  Results appended to '{CSV_PATH}'");
}

fn print_menu(test_size: usize, max: usize) {
    println!("\n=== MNIST k-nn Classifier ===");
    println!("  1) Change test size  (current: {}/{})", test_size, max);
    println!("  2) Sequential");
    println!("  3) Rayon parallel");
    println!("  4) std::thread parallel");
    println!("  5) Run all");
    println!("  6) Scalability analysis");
    println!("  7) Overhead analysis");
    println!("  q) Quit");
}

fn change_test_size(current: usize, max: usize) -> usize {
    println!("\nCurrent test size : {}", current);
    println!("Available samples : {}", max);
    loop {
        let input = prompt(&format!("New size (1–{}): ", max));
        match input.parse::<usize>() {
            Ok(n) if n >= 1 && n <= max => {
                println!("Test size set to {}.", n);
                return n;
            }
            _ => println!("Please enter a number between 1 and {}.", max),
        }
    }
}

fn main() {
    let num_threads = std::env::args()
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let (train, test_full) = load_or_preprocess();
    let max_test = test_full.len();
    let mut test_size = max_test;

    println!("\nTraining samples : {}", train.len());
    println!("Test samples     : {}", max_test);
    println!("Image size       : {} bytes ({} pixels)", train.image_size, train.image_size * 8);
    println!("Threads          : {}", num_threads);

    loop {
        let test = test_full.subset(test_size);
        print_menu(test_size, max_test);
        match prompt("\nChoice: ").as_str() {
            "1" => test_size = change_test_size(test_size, max_test),
            "2" => {
                let title = format!("Sequential k-nn  (k={}, test={})", K, test.len());
                let r = sequential::bench(&train, &test, K);
                result_header(&title);
                result_row("sequential", 1, &r, None);
            }
            "3" => {
                let title = format!("Rayon k-nn  (k={}, test={}, threads={})", K, test.len(), num_threads);
                let r = rayon::bench(&train, &test, K);
                result_header(&title);
                result_row("rayon", num_threads, &r, None);
            }
            "4" => {
                let title = format!("std::thread k-nn  (k={}, test={}, threads={})", K, test.len(), num_threads);
                let r = std_thread::bench(&train, &test, K, num_threads);
                result_header(&title);
                result_row("std::thread", num_threads, &r, None);
            }
            "5" => {
                let title = format!("All methods  (k={}, test={}, threads={})", K, test.len(), num_threads);
                let seq = sequential::bench(&train, &test, K);
                let ray = rayon::bench(&train, &test, K);
                let thr = std_thread::bench(&train, &test, K, num_threads);
                result_header(&title);
                result_row("sequential",  1,           &seq, None);
                result_row("rayon",       num_threads, &ray, Some(&seq));
                result_row("std::thread", num_threads, &thr, Some(&seq));
                save_csv(K, test.len(), num_threads, &seq, &ray, &thr);
            }
            "6" => run_scalability(&train, &test),
            "7" => run_overhead_analysis(&test),
            "q" | "Q" => {
                println!("Goodbye.");
                break;
            }
            other => println!("Unknown option '{other}', please try again."),
        }
    }
}
