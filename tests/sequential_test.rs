use rust_final::pre_processing::MnistDataset;
use rust_final::sequential;

/// Builds a tiny synthetic dataset.
/// Class 0: all-zero images (0x00). Class 1: all-one images (0xFF).
/// Image size is 1 byte (8 bit-packed pixels) to keep arithmetic trivial.
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
    // 0x01 (1 bit set)  → nearest to class 0
    // 0xFE (7 bits set) → nearest to class 1
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
    let results = sequential::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.correct, 2);
    assert_eq!(results.accuracy(), 100.0);
}

#[test]
fn correct_classification_k1() {
    let (train, test) = make_dataset();
    let results = sequential::bench(&train, &test, 1, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.correct, 2);
    assert_eq!(results.accuracy(), 100.0);
}

#[test]
fn total_matches_test_set_size() {
    let (train, test) = make_dataset();
    let results = sequential::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert_eq!(results.total, test.len());
}

#[test]
fn throughput_is_positive() {
    let (train, test) = make_dataset();
    let results = sequential::bench(&train, &test, 3, rust_final::bench::DistanceMetric::Hamming);
    assert!(results.throughput() > 0.0);
}
