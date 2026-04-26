Rust Parallel Image Classification by Patrick Hagelston

---DESCRIPTION---
This project tests the efficacy of algorithms that classify images from the MINST handwritten digits dataset using the k-nearest neighbors algorithm (KNN). Three implementations of KNN were tested: sequential processing, parallel processing using rayon and parallel processing using std::thread. Multiple evaluation metrics were measured with derivative metrics being calculated from them.

---MEASURED METRICS---

| Metric | Description |
|---|---|
| `elapsed` | Wall-clock time via `Instant::now()` / `start.elapsed()` |
| `correct` | Count of correctly classified samples, incremented per prediction |
| `total` | Number of test samples processed |
| `memory_delta_kb` | RSS before and after via `sysinfo::process().memory()` |

---CALCULATED METRICS---

| Metric | Formula |
|---|---|
| `accuracy` | `correct / total` |
| `throughput` | `total / elapsed` |
| `speedup` | `baseline.elapsed / elapsed` |
| `efficiency` | `speedup / threads` |
| `overhead` | `elapsed × threads − sequential.elapsed` |
| `isoefficiency` | `(E / (1 - E)) × overhead / sequential.elapsed` |

---REQUIREMENTS---
x86-64 CPU
512 MB of available RAM
8 available CPU threads

Rust 1.85 or newer
Cargo (comes with rust)

Tested on Windows 10 and 11, but should function on MacOS and Linux

---USER GUIDE---
1. Download data folder at https://drive.google.com/drive/folders/1LrArqP033e3OqMRsjIJzF5wdm9p2UuB6?usp=sharing (must be logged into champlain email to access)

2. Place data folder inside the root directory (rust_KNN_parallel)

3. From the terminal navigate to the root directory and execute:

cargo run --release

4. Select your option from the menu:

| Option | Description |
|---|---|
| `1` Change test size | Set how many test samples to classify (1 – 10,000) |
| `2` Sequential | Runs KNN on a single thread and displays results |
| `3` Rayon parallel | Runs KNN using Rayon, automatically using all available threads |
| `4` std::thread parallel | Runs KNN using std::thread with all available threads |
| `5` Run all | Runs all three implementations back-to-back, using sequential as the baseline for relative metrics, and saves results to `results/results.csv` |
| `6` Scalability analysis | Sweeps Rayon and std::thread across 1, 2, 4, and 8 threads and saves results to `results/scalability.csv` |
| `7` Overhead analysis | Measures raw thread spawn time, Rayon pool initialisation time, and Rayon dispatch time |
| `q` Quit | Exits the program |