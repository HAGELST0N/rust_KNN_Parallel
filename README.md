Rust Parallel Image Classification by Patrick Hagelston

---DESCRIPTION---
This project tests the efficacy of algorithms that classify images from the MINST handwritten digits dataset using the k-nearest neighbors algorithm (KNN). Three implementations of KNN were tested: sequential processing, parallel processing using rayon and parallel processing using std::thread. Multiple evaluation metrics were measured with derivative metrics being calculated from them.

---MEASURED METRICS---
elapsed             wall-clock time via Instant::now() / start.elapsed()
correct             count of correctly classified samples, incremented per prediction
total               number of test samples processed
memory_delta_kb     RSS before and after via sysinfo::process().memory()

---CALCULATED METRICS---
accuracy:	    correct / total
throughput:	    total / elapsed
speedup:	    baseline.elapsed / elapsed
efficiency:	    speedup / threads
overhead:	    elapsed × threads − sequential.elapsed
isoefficiency:	(target_efficiency / (1-target_efficiency)) × overhead / sequential.elapsed

---REQUIREMENTS---
x86-64 CPU
~200 MB RAM
8 CPU threads

Rust 1.85 or newer
Cargo (comes with rust)

Tested on Windows 10 and 11, but should function on MacOS and Linux
---USER GUIDE---
1. Download data folder at https://drive.google.com/drive/folders/1LrArqP033e3OqMRsjIJzF5wdm9p2UuB6?usp=sharing (must be logged into champlain email to access)

2. Place data folder inside the project folder (rust_final) 

