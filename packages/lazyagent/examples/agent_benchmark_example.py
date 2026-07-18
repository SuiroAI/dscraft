"""Runnable demo: run the small LazyAgent benchmark suite end-to-end.

Imports and calls the real `benchcraft_lazyagent` package API -- it does
not reimplement any task/adapter/scoring logic inline (per CLAUDE.md's
"no net-new scripts" rule). Run with:

    python packages/lazyagent/examples/agent_benchmark_example.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchcraft_lazyagent import default_task_suite, rule_based_agent, run_benchmark


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lazyagent-example-") as tmp_dir:
        tasks = default_task_suite(Path(tmp_dir))

        print(f"Running {len(tasks)} LazyAgent benchmark tasks: {[t.name for t in tasks]}")
        report = run_benchmark(tasks, rule_based_agent)

        print()
        print("Per-task results:")
        for result in report.results:
            status = "PASS" if result.success else "FAIL"
            print(
                f"  [{status}] {result.task_name} "
                f"(latency={result.latency_seconds:.4f}s) -- {result.detail}"
            )

        print()
        print(f"Aggregate pass rate:   {report.pass_rate:.2%}")
        print(f"Mean latency:          {report.mean_latency_seconds:.4f}s")


if __name__ == "__main__":
    main()
