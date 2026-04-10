use crate::bench::{process_memory_kb, Results};
use crate::mnist::MnistDataset;
use std::thread;
use std::time::Instant;

fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

fn classify_one(train: &MnistDataset, image: &[u8], k: usize) -> u8 {
    let mut distances: Vec<(u32, u8)> = (0..train.len())
        .map(|i| (hamming_distance(image, train.image(i)), train.labels[i]))
        .collect();

    distances.select_nth_unstable_by_key(k - 1, |&(dist, _)| dist);

    let mut counts = [0u32; 10];
    for &(_, label) in &distances[..k] {
        counts[label as usize] += 1;
    }

    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label as u8)
        .unwrap()
}

/// Spawns scoped threads that borrow `train` and `test` directly — no cloning.
/// `thread::scope` guarantees all threads finish before the scope exits,
/// which satisfies the borrow checker without needing `Arc` or `'static`.
fn run_scoped(
    train: &MnistDataset,
    test: &MnistDataset,
    k: usize,
    num_threads: usize,
) -> Vec<u8> {
    let test_len = test.len();
    let chunk_size = test_len.div_ceil(num_threads);

    thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                s.spawn(move || {
                    let start = t * chunk_size;
                    let end = (start + chunk_size).min(test_len);
                    (start..end)
                        .map(|i| classify_one(train, test.image(i), k))
                        .collect::<Vec<u8>>()
                })
            })
            .collect();

        handles
            .into_iter()
            .flat_map(|h| h.join().expect("thread panicked"))
            .collect()
    })
}

/// Runs the std::thread parallel KNN and returns metrics without printing progress.
pub fn bench(train: &MnistDataset, test: &MnistDataset, k: usize, num_threads: usize) -> Results {
    let mem_before = process_memory_kb();
    let start = Instant::now();

    let predictions = run_scoped(train, test, k, num_threads);

    let elapsed = start.elapsed();
    let mem_after = process_memory_kb();

    let correct = predictions
        .iter()
        .zip(test.labels.iter())
        .filter(|&(&pred, &truth)| pred == truth)
        .count();

    Results {
        elapsed,
        correct,
        total: test.len(),
        memory_delta_kb: mem_after - mem_before,
    }
}

pub fn run(train: &MnistDataset, test: &MnistDataset, k: usize, num_threads: usize) {
    println!(
        "Running std::thread parallel KNN (k={}, threads={})...",
        k, num_threads
    );
    let start = Instant::now();

    let predictions = run_scoped(train, test, k, num_threads);

    let elapsed = start.elapsed();

    let correct = predictions
        .iter()
        .zip(test.labels.iter())
        .filter(|&(&pred, &truth)| pred == truth)
        .count();

    println!("\nstd::thread parallel KNN results (k={}):", k);
    println!("  Threads : {}", num_threads);
    println!("  Correct : {}/{}", correct, test.len());
    println!(
        "  Accuracy: {:.2}%",
        100.0 * correct as f64 / test.len() as f64
    );
    println!("  Time    : {:.2?}", elapsed);
}
