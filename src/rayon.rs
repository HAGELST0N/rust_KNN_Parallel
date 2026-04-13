use crate::bench::{process_memory_kb, Results};
use crate::pre_processing::MnistDataset;
use ::rayon::prelude::*;
use std::time::Instant;

// Hamming distance is more efficient than euclidean for binary data
// performs bitwise XOR to find where images are different and counts the differences
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

// Difference from sequential classify_one: .into_par_iter() allows parallel iteration
fn classify_one(train: &MnistDataset, image: &[u8], k: usize) -> u8 {
    // Compute distance from this image to every training sample — parallelised
    let mut distances: Vec<(u32, u8)> = (0..train.len())
        .into_par_iter()
        .map(|i| (hamming_distance(image, train.image(i)), train.labels[i]))
        .collect();

    // Partial sort brings the k smallest distances to the front O(n) on average
    // Highly useful for implementations where I only need the lowest few distances
    distances.select_nth_unstable_by_key(k - 1, |&(dist, _)| dist);

    // Majority vote over the k nearest neighbours
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

/// Runs the Rayon parallel k-nn and returns metrics without printing progress.
/// Uses the default global thread pool.
pub fn bench(train: &MnistDataset, test: &MnistDataset, k: usize) -> Results {
    bench_with_threads(train, test, k, ::rayon::current_num_threads())
}

/// Runs the Rayon parallel k-nn with a specific thread count.
/// Builds an isolated thread pool so the global pool is not affected.
pub fn bench_with_threads(
    train: &MnistDataset,
    test: &MnistDataset,
    k: usize,
    num_threads: usize,
) -> Results {
    let pool = ::rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool");

    let mem_before = process_memory_kb();
    let start = Instant::now();

    let predictions: Vec<u8> = pool.install(|| {
        (0..test.len())
            .into_par_iter()
            .map(|i| classify_one(train, test.image(i), k))
            .collect()
    });

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

