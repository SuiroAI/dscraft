"""Runnable demo: the prompt-injection probe run N times through the real
shared sandbox executor, printing the resulting leaderboard summary.

This script only imports and calls the real `benchcraft_lazyred` package
API -- per CLAUDE.md's "no net-new scripts" rule, it does not reimplement
any probe/detector/sandbox logic inline.

Run with (after installing lazycore and lazyred -- see README):

    python packages/lazyred/examples/prompt_injection_probe_example.py
"""

from __future__ import annotations

from lazycore.sandbox import get_default_executor

from benchcraft_lazyred import (
    PromptInjectionAdapter,
    build_probe_sandbox_policy,
    default_payload_variations,
    run_leaderboard,
)


def main() -> None:
    """Run the prompt-injection probe through the real sandbox and print a report.

    Builds the shared sandbox executor and LazyRed's mode-specific policy,
    runs :class:`PromptInjectionAdapter` against 8 payload variations (a mix
    of known injection triggers and benign controls), and prints the
    resulting :class:`~benchcraft_lazyred.leaderboard.LeaderboardReport`
    summary to stdout.
    """
    # The shared lazycore sandbox executor (§2.3) -- picks the real
    # SeatbeltSandboxExecutor on macOS. LazyRed supplies its own
    # mode-specific SandboxPolicy on top (build_probe_sandbox_policy),
    # never a second executor implementation.
    policy = build_probe_sandbox_policy()
    executor = get_default_executor(policy)

    adapter = PromptInjectionAdapter()

    # 8 payload variations: cycles through the probe's known injection
    # triggers plus benign control payloads, so the leaderboard shows a
    # realistic (non-trivial) failure rate rather than 100% or 0%.
    payloads = default_payload_variations(8)

    print(f"Running {adapter.probe_id!r} probe against the naive local "
          f"target for {len(payloads)} payload variations, via "
          f"{type(executor).__name__}...\n")

    report = run_leaderboard(adapter, executor, payloads)

    print(report.format_summary())


if __name__ == "__main__":
    main()
