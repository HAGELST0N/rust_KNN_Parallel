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

    /// Parallel overhead: total CPU time spent minus the sequential baseline.
    /// Represents time wasted on coordination (spawning, stealing, synchronisation)
    /// rather than useful work. Formula: T(n) * n - T(1).
    pub fn parallel_overhead(&self, baseline: &Results, num_threads: usize) -> Duration {
        let total_cpu = self.elapsed.mul_f64(num_threads as f64);
        total_cpu.saturating_sub(baseline.elapsed)
    }

    /// Isoefficiency scale factor: how many times larger the problem must grow
    /// to maintain `target_efficiency` with `num_threads` threads.
    ///
    /// Formula: K * T_overhead(n) / T(1)  where K = E / (1 - E)
    ///
    /// - < 1.0 — current problem is already large enough to hit the target
    /// - = 1.0 — current problem is exactly sufficient
    /// - > 1.0 — problem must grow by this factor to reach the target efficiency
    pub fn isoefficiency(&self, baseline: &Results, num_threads: usize, target_efficiency: f64) -> f64 {
        let k = target_efficiency / (1.0 - target_efficiency);
        let t_overhead = self.parallel_overhead(baseline, num_threads).as_secs_f64();
        let t1 = baseline.elapsed.as_secs_f64();
        k * t_overhead / t1
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
