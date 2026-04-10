#[path = "../benchmarks/bench.rs"]
mod bench;
mod mnist;
mod sequential;
mod rayon;
mod std_thread;

use std::path::Path;


const TRAIN_BIN: &str = "data/train.bin";
const TEST_BIN:  &str = "data/test.bin";

fn load_or_preprocess() -> (mnist::MnistDataset, mnist::MnistDataset) {
    let train_bin = Path::new(TRAIN_BIN);
    let test_bin  = Path::new(TEST_BIN);

    if train_bin.exists() && test_bin.exists() {
        println!("Loading preprocessed data...");
        let train = mnist::load_preprocessed(train_bin).expect("failed to load train.bin");
        let test  = mnist::load_preprocessed(test_bin).expect("failed to load test.bin");
        (train, test)
    } else {
        println!("Preprocessed files not found — loading raw data and binarizing...");
        let mut train = mnist::load(
            Path::new("data/train-images.idx3-ubyte"),
            Path::new("data/train-labels.idx1-ubyte"),
        )
        .expect("failed to load training data");

        let mut test = mnist::load(
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

fn main() {
    let num_threads = std::env::args()
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let (train, test) = load_or_preprocess();

    println!("Training samples : {}", train.len());
    println!("Test samples     : {}", test.len());
    println!("Image size       : {} bytes ({} pixels)", train.image_size, train.image_size * 8);
    println!("Threads          : {}", num_threads);

    sequential::run(&train, &test, 3);
    rayon::run(&train, &test, 3);
    std_thread::run(&train, &test, 3, num_threads);
}
