use rust_final::pre_processing;
use rust_final::{sequential, rayon, std_thread};
use rust_final::bench::Results;
use std::path::Path;

// Number of test samples to classify — kept small so the test finishes quickly.
// The full 60k training set is always used to keep distances realistic.
const TEST_SAMPLES: usize = 100;
const K: usize = 3;
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

fn load_data() -> (pre_processing::MnistDataset, pre_processing::MnistDataset) {
    let train = pre_processing::load_preprocessed(Path::new("data/train.bin"))
        .expect("data/train.bin not found — run `cargo run` once first to generate it");
    let test = pre_processing::load_preprocessed(Path::new("data/test.bin"))
        .expect("data/test.bin not found — run `cargo run` once first to generate it");
    (train, test)
}

#[test]
fn benchmark_knn() {
    let (train, test_full) = load_data();
    let test = test_full.subset(TEST_SAMPLES);

    let seq    = sequential::bench(&train, &test, K);
    let ray    = rayon::bench(&train, &test, K);
    let n = num_threads();
    let thread = std_thread::bench(&train, &test, K, n);

    println!("\n=== K-nn Benchmark (k={}, train={}, test={}) ===\n", K, train.len(), test.len());
    print_benchmark_header();
    print_scaling_row_labeled("sequential",                    1, &seq,    &seq);
    print_scaling_row_labeled(&format!("rayon ({} threads)", n), n, &ray,    &seq);
    print_scaling_row_labeled(&format!("std::thread ({n}t)"),  n, &thread, &seq);
    println!();

    // Sanity check: all methods should agree on accuracy within 1%
    let acc_diff_rayon  = (seq.accuracy() - ray.accuracy()).abs();
    let acc_diff_thread = (seq.accuracy() - thread.accuracy()).abs();
    assert!(acc_diff_rayon  < 1.0, "rayon accuracy diverged from sequential by {:.2}%",  acc_diff_rayon);
    assert!(acc_diff_thread < 1.0, "thread accuracy diverged from sequential by {:.2}%", acc_diff_thread);
}

const ISOEFFICIENCY_TARGETS: &[f64] = &[0.50, 0.75, 0.90];

fn print_scaling_row(threads: usize, r: &Results, baseline: &Results) {
    let iso: Vec<String> = ISOEFFICIENCY_TARGETS
        .iter()
        .map(|&e| format!("{:.2}x", r.isoefficiency(baseline, threads, e)))
        .collect();

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

fn print_benchmark_header() {
    let targets: Vec<String> = ISOEFFICIENCY_TARGETS
        .iter()
        .map(|&e| format!("E={:.0}%", e * 100.0))
        .collect();

    println!(
        "  {:<14} | {:>11} | {:>10} | {:>8} | {:>8} | {:>11} | {:>23} | {:>10}",
        "method", "time", "samples/s", "speedup", "effic.", "overhead",
        format!("isoeff. ({}, {}, {})", targets[0], targets[1], targets[2]),
        "mem delta"
    );
    println!("  {}", "-".repeat(108));
}

fn print_scaling_row_labeled(label: &str, threads: usize, r: &Results, baseline: &Results) {
    let iso: Vec<String> = ISOEFFICIENCY_TARGETS
        .iter()
        .map(|&e| format!("{:.2}x", r.isoefficiency(baseline, threads, e)))
        .collect();

    let (speedup_str, effic_str, overhead_str) = if std::ptr::eq(r, baseline) {
        ("baseline".to_string(), "—".to_string(), "—".to_string())
    } else {
        (
            format!("{:.3}x", r.speedup(baseline)),
            format!("{:.3}",  r.efficiency(baseline, threads)),
            format!("{:.1}ms", r.parallel_overhead(baseline, threads).as_secs_f64() * 1000.0),
        )
    };

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

fn print_scaling_header() {
    let targets: Vec<String> = ISOEFFICIENCY_TARGETS
        .iter()
        .map(|&e| format!("E={:.0}%", e * 100.0))
        .collect();

    println!(
        "  {:>7} | {:>11} | {:>10} | {:>8} | {:>8} | {:>11} | {:>23} | {:>10}",
        "threads", "time", "samples/s", "speedup", "effic.", "overhead",
        format!("isoeff. ({}, {}, {})", targets[0], targets[1], targets[2]),
        "mem delta"
    );
    println!("  {}", "-".repeat(108));
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
