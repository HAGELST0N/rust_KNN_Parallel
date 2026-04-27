use crate::bench::{process_memory_kb, Results, DistanceMetric};
use crate::pre_processing::MnistDataset;
use std::time::Instant;

// Hamming distance: XOR each byte pair and count differing bits via POPCNT.
// Works directly on bit-packed data (98 bytes per image).
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

// Euclidean distance: unpack each bit-packed byte into 8 individual binary
// pixels (0 or 1) and compute the squared L2 norm across all 784 pixels.
// For binary data this equals Hamming distance numerically, but the computation
// path is genuinely different — 784 subtractions and multiplications instead of
// 98 XOR + POPCNT operations — making it useful for benchmarking.
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

fn classify_one(train: &MnistDataset, image: &[u8], k: usize, metric: DistanceMetric) -> u8 {
    let mut distances: Vec<(u32, u8)> = (0..train.len())
        .map(|i| {
            let dist = match metric {
                DistanceMetric::Hamming   => hamming_distance(image, train.image(i)),
                DistanceMetric::Euclidean => euclidean_distance(image, train.image(i)),
            };
            (dist, train.labels[i])
        })
        .collect();

    // Partial sort: brings the k smallest distances to the front in O(n) average
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

pub fn bench(train: &MnistDataset, test: &MnistDataset, k: usize, metric: DistanceMetric) -> Results {
    let mem_before = process_memory_kb();
    let start = Instant::now();

    let correct = test
        .labels
        .iter()
        .enumerate()
        .filter(|&(i, &true_label)| classify_one(train, test.image(i), k, metric) == true_label)
        .count();

    let elapsed = start.elapsed();
    let mem_after = process_memory_kb();

    Results {
        elapsed,
        correct,
        total: test.len(),
        memory_delta_kb: mem_after - mem_before,
        distance_metric: metric,
    }
}
