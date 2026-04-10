use rust_final::mnist;
use rust_final::{sequential, rayon, std_thread};
use rust_final::bench::Results;
use std::path::Path;

// Number of test samples to classify — kept small so the test finishes quickly.
// The full 60k training set is always used to keep distances realistic.
const TEST_SAMPLES: usize = 100;
const K: usize = 3;
const NUM_THREADS: usize = 4;
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

fn load_data() -> (mnist::MnistDataset, mnist::MnistDataset) {
    let train = mnist::load_preprocessed(Path::new("data/train.bin"))
        .expect("data/train.bin not found — run `cargo run` once first to generate it");
    let test = mnist::load_preprocessed(Path::new("data/test.bin"))
        .expect("data/test.bin not found — run `cargo run` once first to generate it");
    (train, test)
}

fn print_row(label: &str, r: &Results, baseline: Option<&Results>, num_threads: usize) {
    let speedup    = baseline.map(|b| r.speedup(b));
    let efficiency = baseline.map(|b| r.efficiency(b, num_threads));

    println!(
        "  {:<14} | {:>10.3}s | {:>8.2}% | {:>10.1} | {:>8} | {:>8} | {:>10}",
        label,
        r.elapsed.as_secs_f64(),
        r.accuracy(),
        r.throughput(),
        match speedup    { Some(s) => format!("{:.3}x", s),  None => "baseline".into() },
        match efficiency { Some(e) => format!("{:.3}",  e),  None => "—".into()        },
        format!("{:+} KB", r.memory_delta_kb),
    );
}

#[test]
fn benchmark_knn() {
    let (train, test_full) = load_data();
    let test = test_full.subset(TEST_SAMPLES);

    println!("\n=== KNN Benchmark (k={}, train={}, test={}) ===\n", K, train.len(), test.len());
    println!(
        "  {:<14} | {:>11} | {:>9} | {:>10} | {:>8} | {:>8} | {:>10}",
        "method", "time", "accuracy", "samples/s", "speedup", "effic.", "mem delta"
    );
    println!("  {}", "-".repeat(84));

    let seq    = sequential::bench(&train, &test, K);
    let ray    = rayon::bench(&train, &test, K);
    let thread = std_thread::bench(&train, &test, K, NUM_THREADS);

    print_row("sequential",  &seq,    None,          1);
    print_row("rayon",       &ray,    Some(&seq),    NUM_THREADS);
    print_row("std::thread", &thread, Some(&seq),    NUM_THREADS);

    println!("\n  Threads used for parallel runs: {}", NUM_THREADS);
    println!("  Efficiency = speedup / num_threads  (1.0 = perfect linear scaling)\n");

    // Sanity check: all methods should agree on accuracy within 1%
    let acc_diff_rayon  = (seq.accuracy() - ray.accuracy()).abs();
    let acc_diff_thread = (seq.accuracy() - thread.accuracy()).abs();
    assert!(acc_diff_rayon  < 1.0, "rayon accuracy diverged from sequential by {:.2}%",  acc_diff_rayon);
    assert!(acc_diff_thread < 1.0, "thread accuracy diverged from sequential by {:.2}%", acc_diff_thread);
}

fn print_scaling_row(threads: usize, r: &Results, baseline: &Results) {
    println!(
        "  {:>7} | {:>10.3}s | {:>8.2}% | {:>10.1} | {:>8} | {:>8} | {:>10}",
        threads,
        r.elapsed.as_secs_f64(),
        r.accuracy(),
        r.throughput(),
        format!("{:.3}x", r.speedup(baseline)),
        format!("{:.3}",  r.efficiency(baseline, threads)),
        format!("{:+} KB", r.memory_delta_kb),
    );
}

fn print_scaling_header() {
    println!(
        "  {:>7} | {:>11} | {:>9} | {:>10} | {:>8} | {:>8} | {:>10}",
        "threads", "time", "accuracy", "samples/s", "speedup", "effic.", "mem delta"
    );
    println!("  {}", "-".repeat(80));
}

#[test]
fn scaling_rayon() {
    let (train, test_full) = load_data();
    let test = test_full.subset(TEST_SAMPLES);
    let baseline = sequential::bench(&train, &test, K);

    println!(
        "\n=== Rayon scaling (k={}, train={}, test={}) ===\n",
        K, train.len(), test.len()
    );
    print_scaling_header();

    for &n in THREAD_COUNTS {
        let r = rayon::bench_with_threads(&train, &test, K, n);
        print_scaling_row(n, &r, &baseline);
        assert!(
            (r.accuracy() - baseline.accuracy()).abs() < 1.0,
            "rayon ({} threads) accuracy diverged from sequential", n
        );
    }
    println!();
}

#[test]
fn scaling_std_thread() {
    let (train, test_full) = load_data();
    let test = test_full.subset(TEST_SAMPLES);
    let baseline = sequential::bench(&train, &test, K);

    println!(
        "\n=== std::thread scaling (k={}, train={}, test={}) ===\n",
        K, train.len(), test.len()
    );
    print_scaling_header();

    for &n in THREAD_COUNTS {
        let r = std_thread::bench(&train, &test, K, n);
        print_scaling_row(n, &r, &baseline);
        assert!(
            (r.accuracy() - baseline.accuracy()).abs() < 1.0,
            "std::thread ({} threads) accuracy diverged from sequential", n
        );
    }
    println!();
}
