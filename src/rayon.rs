use crate::bench::{process_memory_kb, Results, DistanceMetric};
use crate::pre_processing::MnistDataset;
use ::rayon::prelude::*;
use std::time::Instant;

fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

fn euclidean_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut sum = 0u32;
    for (&ax, &bx) in a.iter().zip(b.iter()) {
        for bit in 0..8u8 {
            let a_bit = (ax >> bit) & 1;
            let b_bit = (bx >> bit) & 1;
            let diff = a_bit as i32 - b_bit as i32;
            sum += (diff * diff) as u32;
        }
    }
    sum
}

// Difference from sequential classify_one: .into_par_iter() parallelises the
// distance calculations across training samples.
fn classify_one(train: &MnistDataset, image: &[u8], k: usize, metric: DistanceMetric) -> u8 {
    let mut distances: Vec<(u32, u8)> = (0..train.len())
        .into_par_iter()
        .map(|i| {
            let dist = match metric {
                DistanceMetric::Hamming   => hamming_distance(image, train.image(i)),
                DistanceMetric::Euclidean => euclidean_distance(image, train.image(i)),
            };
            (dist, train.labels[i])
        })
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

/// Runs the Rayon parallel k-nn and returns metrics. Uses the default global thread pool.
pub fn bench(train: &MnistDataset, test: &MnistDataset, k: usize, metric: DistanceMetric) -> Results {
    bench_with_threads(train, test, k, metric, ::rayon::current_num_threads())
}

/// Runs the Rayon parallel k-nn with a specific thread count.
/// Builds an isolated thread pool so the global pool is not affected.
pub fn bench_with_threads(
    train: &MnistDataset,
    test: &MnistDataset,
    k: usize,
    metric: DistanceMetric,
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
            .map(|i| classify_one(train, test.image(i), k, metric))
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
        distance_metric: metric,
    }
}
