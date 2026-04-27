use rust_final::pre_processing::MnistDataset;
use rust_final::{rayon, sequential};

fn make_dataset() -> (MnistDataset, MnistDataset) {
    let train = MnistDataset {
        images: vec![
            0x00, // label 0
            0x00, // label 0
            0x00, // label 0
            0xFF, // label 1
            0xFF, // label 1
            0xFF, // label 1
        ],
        labels: vec![0, 0, 0, 1, 1, 1],
        image_size: 1,
    };
    let test = MnistDataset {
        images: vec![0x01, 0xFE],
        labels: vec![0, 1],
        image_size: 1,
    };
    (train, test)
}

#[test]
fn correct_classification_k3() {
    let (train, test) = make_dataset();
    let results = rayon::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.correct, 2);
    assert_eq!(results.accuracy(), 100.0);
}

#[test]
fn correct_classification_k1() {
    let (train, test) = make_dataset();
    let results = rayon::bench(&train, &test, 1, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.correct, 2);
    assert_eq!(results.accuracy(), 100.0);
}

#[test]
fn total_matches_test_set_size() {
    let (train, test) = make_dataset();
    let results = rayon::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.total, test.len());
}

#[test]
fn throughput_is_positive() {
    let (train, test) = make_dataset();
    let results = rayon::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert!(results.throughput() > 0.0);
}

#[test]
fn agrees_with_sequential() {
    let (train, test) = make_dataset();
    let seq = sequential::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    let ray = rayon::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(seq.correct, ray.correct);
}

#[test]
fn thread_counts_agree() {
    let (train, test) = make_dataset();
    let r1 = rayon::bench_with_threads(&train, &test, 3, 1, rust_final::bench::DistanceMetric::Hamming);
    let r2 = rayon::bench_with_threads(&train, &test, 3, 2, rust_final::bench::DistanceMetric::Hamming);
    let r4 = rayon::bench_with_threads(&train, &test, 3, 4, rust_final::bench::DistanceMetric::Hamming);
    let r8 = rayon::bench_with_threads(&train, &test, 3, 8, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(r1.correct, r2.correct);
    assert_eq!(r1.correct, r4.correct);
    assert_eq!(r1.correct, r8.correct);
}
