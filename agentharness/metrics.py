"""Metrics for a batch of agent runs.

Takes the RunResults your agent actually produced and turns them into
the numbers that matter for a before/after comparison: how many runs
were served from cache (near-zero cost), how many tokens were spent
overall, and roughly how many tokens a naive version (no cache, no
budget trimming) would have spent on the same tasks.

The "naive baseline" estimate is necessarily approximate — we don't
actually re-run every task without optimization just to measure it.
It's a labeled estimate, not a measured fact, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentharness.types import RunResult


@dataclass
class MetricsReport:
    total_runs: int
    cache_hits: int
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_naive_tokens: int
    estimated_tokens_saved: int
    estimated_savings_pct: float

    def summary(self) -> str:
        return (
            f"Runs: {self.total_runs}  |  Cache hits: {self.cache_hits} "
            f"({self.cache_hit_rate:.0%})\n"
            f"Actual tokens used: {self.total_tokens:,} "
            f"(input: {self.total_input_tokens:,}, output: {self.total_output_tokens:,})\n"
            f"Estimated naive-baseline tokens: {self.estimated_naive_tokens:,}\n"
            f"Estimated tokens saved: {self.estimated_tokens_saved:,} "
            f"({self.estimated_savings_pct:.1f}%)"
        )


def compute_metrics(results: list[RunResult], avg_tokens_per_cache_hit_task: int = 1500) -> MetricsReport:
    """Compute a metrics report from a batch of completed runs.

    `avg_tokens_per_cache_hit_task` is the estimate of what a cache-hit
    task *would have* cost if it had run normally — used only to compute
    the naive-baseline comparison, since a real run never happened for
    those tasks. Tune this from your own observed average of non-cached
    runs for a more accurate estimate.
    """
    total_runs = len(results)
    cache_hits = sum(1 for r in results if r.cache_hit)
    cache_hit_rate = cache_hits / total_runs if total_runs else 0.0

    total_input_tokens = sum(r.input_tokens for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)
    total_tokens = total_input_tokens + total_output_tokens

    # Naive baseline: every run, including cache hits, costs the average
    # of what your *actual* non-cached runs cost.
    non_cached = [r for r in results if not r.cache_hit]
    if non_cached:
        avg_real_run_tokens = sum(r.input_tokens + r.output_tokens for r in non_cached) // len(non_cached)
    else:
        avg_real_run_tokens = avg_tokens_per_cache_hit_task

    estimated_naive_tokens = avg_real_run_tokens * total_runs
    estimated_tokens_saved = max(estimated_naive_tokens - total_tokens, 0)
    estimated_savings_pct = (
        (estimated_tokens_saved / estimated_naive_tokens * 100) if estimated_naive_tokens else 0.0
    )

    return MetricsReport(
        total_runs=total_runs,
        cache_hits=cache_hits,
        cache_hit_rate=cache_hit_rate,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        estimated_naive_tokens=estimated_naive_tokens,
        estimated_tokens_saved=estimated_tokens_saved,
        estimated_savings_pct=estimated_savings_pct,
    )