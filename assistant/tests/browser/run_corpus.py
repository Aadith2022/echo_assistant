"""Run the archetype corpus and report a pass RATE.

Most of what decides whether a browser task succeeds is model judgement, which
is not deterministic - the same test passes, then fails, then passes, with no
code change. A suite reporting "3 failed" from one pass of each case cannot
tell a real regression from noise, so it produces confident conclusions from
insufficient evidence.

Every archetype therefore runs N times and the report is "4/5 passed". A change
is worth keeping when it moves the rate.

Usage:
    python -m tests.browser.run_corpus                 # all, 3 runs each
    python -m tests.browser.run_corpus --runs 5
    python -m tests.browser.run_corpus --only autocomplete,calendar
    python -m tests.browser.run_corpus --baseline out.json   # save for comparison
    python -m tests.browser.run_corpus --compare out.json    # diff against it
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from tests.browser import corpus, fixtures, vendors  # noqa: E402
from tests.browser.corpus import ERROR, HONEST_FAIL, PASS, WRONG  # noqa: E402

_MARK = {PASS: "ok", HONEST_FAIL: "honest-fail", WRONG: "WRONG", ERROR: "error"}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_once(archetype, verbose: bool) -> dict:
    from browser import action_cache, plan_cache
    from browser.task_runner import runner
    from guardrails import confirmation
    from llm.metrics import metrics

    # Each run starts cold: a cache hit would measure the cache, not the agent.
    action_cache.clear()
    plan_cache.clear()
    fixtures.reset()
    if archetype.setup:
        archetype.setup()

    # Auto-approve, so confirmation prompts don't stall an unattended run. The
    # gating itself is covered by the dedicated purchase-gate test.
    confirmation.set_handler(lambda r: True)

    started = time.monotonic()
    try:
        result = runner.run(
            task=archetype.task,
            start_url=archetype.start_url,
            allowed_domains=archetype.domains,
            on_progress=(lambda m: print(f"      {m}")) if verbose else None,
        )
    except Exception as e:  # noqa: BLE001 - a crash is a result too
        result = f"EXCEPTION: {type(e).__name__}: {e}"
    elapsed = time.monotonic() - started

    verdict, reason = archetype.check(result, list(fixtures.EVENTS))
    snap = metrics.snapshot()
    return {
        "verdict": verdict,
        "reason": reason,
        "seconds": round(elapsed, 1),
        "calls": sum(s.calls for s in snap.values()),
        "result": (result or "")[:400],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser archetype corpus.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default="", help="comma-separated archetype names")
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "comma-separated archetype names to skip. Cost is very unevenly "
            "distributed - one measured pass spent 69%% of its wall time and "
            "half its model calls on multi-vendor alone - so skipping the "
            "expensive ones makes a routine check affordable. Use the full "
            "corpus for baselines."
        ),
    )
    parser.add_argument("--baseline", default="", help="write results to this file")
    parser.add_argument("--compare", default="", help="compare against a saved file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    config.BROWSER_HEADLESS = False

    fixtures.serve(8900)
    vendors.serve_all()

    selected = corpus.CORPUS
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        selected = [a for a in corpus.CORPUS if a.name in wanted]
        if not selected:
            print(f"No archetypes matched {sorted(wanted)}")
            return 2
    if args.exclude:
        skipped = {n.strip() for n in args.exclude.split(",")}
        selected = [a for a in selected if a.name not in skipped]
        # Named explicitly in the output: a pass rate means something different
        # when part of the suite did not run, and a report that quietly hides
        # which part is how a partial run gets read as a full one.
        print(f"Skipping: {', '.join(sorted(skipped))}\n")

    print(f"Running {len(selected)} archetypes x {args.runs} runs\n")
    report: dict[str, dict] = {}

    for archetype in selected:
        print(f"--- {archetype.name}  ({archetype.pattern})")
        runs = []
        for i in range(args.runs):
            outcome = _run_once(archetype, args.verbose)
            runs.append(outcome)
            mark = _MARK[outcome["verdict"]]
            detail = f" - {outcome['reason']}" if outcome["reason"] else ""
            print(f"    run {i + 1}: {mark:12} {outcome['seconds']:6.1f}s "
                  f"{outcome['calls']:3d} calls{detail}")
            # Show what it actually said when it did not pass. Without this the
            # report says a run failed but not why, and diagnosing means
            # re-running by hand - which is how a plain crash spent two rounds
            # being mistaken for the agent giving up early.
            if outcome["verdict"] != PASS:
                said = outcome["result"].replace("\n", " ")[:220]
                print(f"        said: {said!r}")

        passes = sum(1 for r in runs if r["verdict"] == PASS)
        wrongs = sum(1 for r in runs if r["verdict"] == WRONG)
        errors = sum(1 for r in runs if r["verdict"] == ERROR)
        # Errored runs are not evidence either way, so they leave the pass rate
        # rather than dragging it down - a rate computed over runs that never
        # happened is not a measurement of the agent.
        counted = len(runs) - errors
        report[archetype.name] = {
            "pass_rate": (passes / counted) if counted else 0.0,
            "passes": passes,
            "runs": counted,
            "errors": errors,
            "wrong": wrongs,
            "median_seconds": statistics.median(r["seconds"] for r in runs),
            "median_calls": statistics.median(r["calls"] for r in runs),
            "expected_to_fail": archetype.expected_to_fail,
            "detail": runs,
        }
        print()

    _summarise(report)

    if args.baseline:
        with open(args.baseline, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nBaseline written to {args.baseline}")

    if args.compare:
        _compare(report, args.compare)

    # A known gap failing is not a build failure; a confidently wrong answer is.
    unexpected = [
        n for n, r in report.items()
        if not r["expected_to_fail"] and r["runs"] and r["pass_rate"] < 1.0
    ]
    any_wrong = [n for n, r in report.items() if r["wrong"]]
    return 1 if (unexpected or any_wrong) else 0


def _summarise(report: dict) -> None:
    print("=" * 78)
    print(f"{'archetype':22} {'pass rate':>10} {'wrong':>6} {'err':>4} {'med s':>7} {'med calls':>10}")
    print("-" * 78)
    for name, r in report.items():
        rate = f"{r['passes']}/{r['runs']}" if r["runs"] else "-"
        flag = "  (known gap)" if r["expected_to_fail"] else ""
        print(f"{name:22} {rate:>10} {r['wrong']:>6} {r['errors']:>4} "
              f"{r['median_seconds']:>7.1f} {r['median_calls']:>10.0f}{flag}")
    print("-" * 78)

    total_runs = sum(r["runs"] for r in report.values())
    total_pass = sum(r["passes"] for r in report.values())
    total_wrong = sum(r["wrong"] for r in report.values())
    total_err = sum(r["errors"] for r in report.values())
    expected = [r for r in report.values() if not r["expected_to_fail"]]
    exp_runs = sum(r["runs"] for r in expected)
    exp_pass = sum(r["passes"] for r in expected)

    if not total_runs:
        print("No runs produced a usable result - every run errored.")
        if total_err:
            print(f"({total_err} infrastructure failures: check API quota and network.)")
        print("=" * 78)
        return

    print(f"overall            {total_pass}/{total_runs} "
          f"({total_pass / total_runs:.0%})   "
          f"excluding known gaps: {exp_pass}/{exp_runs} "
          f"({exp_pass / exp_runs:.0%})" if exp_runs else "")
    if total_err:
        print(f"infrastructure errors (excluded from rates): {total_err}"
              "   <-- check API quota/network before reading anything else")
    # Called out separately because it is the failure mode with no user-visible
    # signal - the agent sounds just as confident when it is wrong.
    print(f"confidently WRONG answers: {total_wrong}"
          + ("   <-- fix these first" if total_wrong else "   (none)"))
    print("=" * 78)


def _compare(report: dict, path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except OSError as e:
        print(f"\nCould not read baseline {path}: {e}")
        return

    print("\nCompared with baseline:")
    for name, r in report.items():
        if name not in old:
            print(f"  {name:22} new")
            continue
        delta = r["pass_rate"] - old[name]["pass_rate"]
        if abs(delta) < 1e-9:
            continue
        arrow = "improved" if delta > 0 else "REGRESSED"
        print(f"  {name:22} {arrow}: {old[name]['passes']}/{old[name]['runs']}"
              f" -> {r['passes']}/{r['runs']}")


if __name__ == "__main__":
    sys.exit(main())
