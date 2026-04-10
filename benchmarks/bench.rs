use std::time::Duration;
use sysinfo::{Pid, System};

/// Metrics returned by every benchmark function.
pub struct Results {
    pub elapsed: Duration,
    pub correct: usize,
    pub total: usize,
    pub memory_delta_kb: i64, // RSS change during the run (KB)
}

impl Results {
    pub fn accuracy(&self) -> f64 {
        100.0 * self.correct as f64 / self.total as f64
    }

    /// How many times faster this run was compared to `baseline`.
    pub fn speedup(&self, baseline: &Results) -> f64 {
        baseline.elapsed.as_secs_f64() / self.elapsed.as_secs_f64()
    }

    /// Fraction of ideal linear speedup actually achieved (1.0 = perfect).
    pub fn efficiency(&self, baseline: &Results, num_threads: usize) -> f64 {
        self.speedup(baseline) / num_threads as f64
    }

    /// Samples classified per second.
    pub fn throughput(&self) -> f64 {
        self.total as f64 / self.elapsed.as_secs_f64()
    }
}

/// Snapshot of the current process RSS in KB.
pub fn process_memory_kb() -> i64 {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new();
    sys.refresh_process(pid);
    sys.process(pid)
        .map(|p| (p.memory() / 1024) as i64)
        .unwrap_or(0)
}
