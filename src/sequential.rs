use crate::bench::{process_memory_kb, Results};
use crate::mnist::MnistDataset;
use std::time::Instant;

/// Hamming distance on bit-packed images: XOR each byte then count differing bits.
/// Equivalent to squared Euclidean distance for binary pixel data, with no floats needed.
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

fn classify_one(train: &MnistDataset, image: &[u8], k: usize) -> u8 {
    // Compute distance from this image to every training sample
    let mut distances: Vec<(u32, u8)> = (0..train.len())
        .map(|i| (hamming_distance(image, train.image(i)), train.labels[i]))
        .collect();

    // Partial sort: bring the k smallest distances to the front
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

/// Runs the sequential KNN and returns metrics without printing progress.
pub fn bench(train: &MnistDataset, test: &MnistDataset, k: usize) -> Results {
    let mem_before = process_memory_kb();
    let start = Instant::now();

    let correct = test
        .labels
        .iter()
        .enumerate()
        .filter(|&(i, &true_label)| classify_one(train, test.image(i), k) == true_label)
        .count();

    let elapsed = start.elapsed();
    let mem_after = process_memory_kb();

    Results {
        elapsed,
        correct,
        total: test.len(),
        memory_delta_kb: mem_after - mem_before,
    }
}

pub fn run(train: &MnistDataset, test: &MnistDataset, k: usize) {
    println!("Running sequential KNN (k={})...", k);
    let start = Instant::now();

    let mut correct = 0usize;
    for (i, &true_label) in test.labels.iter().enumerate() {
        let image = test.image(i);
        let predicted = classify_one(train, &image, k);
        if predicted == true_label {
            correct += 1;
        }
        if (i + 1) % 500 == 0 {
            println!(
                "  [{}/{}] running accuracy: {:.2}%",
                i + 1,
                test.len(),
                100.0 * correct as f64 / (i + 1) as f64
            );
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nSequential KNN results (k={}):",
        k
    );
    println!("  Correct : {}/{}", correct, test.len());
    println!(
        "  Accuracy: {:.2}%",
        100.0 * correct as f64 / test.len() as f64
    );
    println!("  Time    : {:.2?}", elapsed);
}
