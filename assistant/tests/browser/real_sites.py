"""Real-site smoke tests.

The archetype corpus proves the mechanics against fixtures we control; this
proves they survive real markup, real latency, real consent banners and real
anti-bot. Neither substitutes for the other, and a failure here needs triaging
into "our bug" versus "the site changed" before it means anything.

Scope is deliberately read-only: public pages, no account actions, no
submissions, one visit per site per run. Anything that changes state on someone
else's system belongs in the fixture corpus.

Checks favour stable facts and structural properties over anything editorial,
so a pass means the agent worked rather than that the news was slow that day.

Usage:
    python -m tests.browser.real_sites
    python -m tests.browser.real_sites --runs 2 --only wikipedia-fact
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from tests.browser.corpus import (  # noqa: E402
    ERROR, HONEST_FAIL, PASS, WRONG, Archetype, _admits, _verdict,
)

_MARK = {PASS: "ok", HONEST_FAIL: "honest-fail", WRONG: "WRONG", ERROR: "error"}


def _c_eiffel(result, events):
    # 330 m to the tip; a fact that has not moved since 1957.
    return _verdict("330" in result, result, "did not report the 330 m height")


def _c_python_creator(result, events):
    return _verdict("rossum" in (result or "").lower(), result,
                    "did not name Guido van Rossum")


def _c_hn_structure(result, events):
    # Structural, not editorial: the front page always lists ranked stories,
    # but which stories is meaningless to assert.
    low = (result or "").lower()
    hits = sum(w in low for w in ("story", "stories", "points", "comments", "news", "posted"))
    return _verdict(hits >= 2 and len(result) > 80, result,
                    "did not describe the front page's stories")


def _c_gov_uk(result, events):
    low = (result or "").lower()
    return _verdict("passport" in low and len(result) > 80, result,
                    "did not return passport guidance")


def _c_example_domain(result, events):
    return _verdict("example" in (result or "").lower(), result,
                    "did not read example.com")


# Fares come back in whatever currency the browser's locale implies. The profile
# here geolocates to India, so a correct answer arrives as "₹15,062" - and an
# earlier version of this check, which accepted only $/£/€/USD/GBP, would have
# scored that a failure. A checker that only recognises one region's money is
# testing the locale, not the agent.
_CURRENCY = ("$", "£", "€", "₹", "¥", "USD", "GBP", "EUR", "INR", "JPY", "CAD", "AUD")


def _c_flights(result, events):
    """The task asks for the cheapest fare, so a price IS the answer.

    An earlier version also demanded the word "flight" appear, and scored
    "The cheapest fare shown is ₹15,062" as a failure - a correct answer
    rejected for phrasing. Between that and the currency list, this one checker
    produced two false failures on a task that worked. Assert the thing that
    was asked for and nothing more.
    """
    return _verdict(
        any(sym in result for sym in _CURRENCY), result, "did not produce a fare"
    )


def _c_directions(result, events):
    low = (result or "").lower()
    # Manchester-Liverpool is ~35 miles / ~50 km, around an hour. Assert the
    # *shape* of an answer (a distance unit and a time unit) rather than exact
    # figures, which vary with traffic and route.
    has_distance = any(u in low for u in ("mile", "km", "kilomet"))
    has_time = any(u in low for u in ("min", "hour", "hr"))
    return _verdict(has_distance and has_time, result,
                    "did not give both a distance and a driving time")


def _c_retail(result, events):
    low = (result or "").lower()
    has_price = any(sym in result for sym in _CURRENCY)
    has_product = "bottle" in low
    return _verdict(has_price and has_product, result,
                    "did not return a priced product")


def _c_population(result, events):
    # Finland is ~5.5-5.6 million. Accept any plausible rendering.
    low = (result or "").lower().replace(",", "")
    plausible = any(t in low for t in ("5.5", "5.6", "55", "56")) and any(
        t in low for t in ("million", "000000")
    )
    return _verdict(plausible, result, "did not report a plausible population")


REAL_SITES: list[Archetype] = [
    Archetype(
        "example-read", "trivial public page (network sanity check)",
        "What does this page say?",
        "https://example.com/", ["example.com"], _c_example_domain,
    ),
    Archetype(
        "wikipedia-fact", "search a large site and read a specific value",
        "Find the height of the Eiffel Tower and report it",
        "https://en.wikipedia.org/wiki/Main_Page",
        ["en.wikipedia.org", "wikipedia.org"], _c_eiffel,
    ),
    Archetype(
        "wikipedia-search", "use a site's own search to reach an article",
        "Search Wikipedia for the Python programming language and tell me who created it",
        "https://en.wikipedia.org/wiki/Main_Page",
        ["en.wikipedia.org", "wikipedia.org"], _c_python_creator,
    ),
    Archetype(
        "news-front-page", "dense, link-heavy listing page",
        "Describe what is on the front page right now",
        "https://news.ycombinator.com/", ["news.ycombinator.com"], _c_hn_structure,
    ),
    Archetype(
        "gov-service", "public service site with its own search",
        "Find the guidance on renewing an adult passport and summarise the first steps",
        "https://www.gov.uk/", ["gov.uk", "www.gov.uk"], _c_gov_uk,
    ),
    Archetype(
        "flights-hard", "autocomplete + calendar + multi-field form (known hard)",
        "Find flights from JFK to RDU departing 20 September 2026, one way, and tell me "
        "the cheapest fare shown. The origin box may already contain another city - "
        "clear it before typing.",
        "https://www.google.com/travel/flights",
        ["google.com", "www.google.com"], _c_flights,
    ),
    # --- sites the stack has never been tuned against -----------------------
    # The point of these is not that they are famous, it is that each is a
    # different hard shape: a mapping app, a bot-protected retailer, an
    # encyclopaedia's structured data. One hard case passing proves the fix for
    # that case; several unseen ones passing is the first evidence it
    # generalises.
    Archetype(
        "maps-directions", "mapping app: multi-field, JS-rendered, URL-constructible",
        "How long does it take to drive from Manchester to Liverpool? Give the "
        "distance and driving time.",
        "https://www.google.com/maps", ["google.com", "www.google.com"], _c_directions,
    ),
    Archetype(
        "retail-search", "bot-protected retailer with a filtered product listing",
        "Find a stainless steel water bottle and tell me the price of one of them",
        "https://www.amazon.co.uk/", ["amazon.co.uk", "www.amazon.co.uk"], _c_retail,
    ),
    Archetype(
        "structured-lookup", "read a specific value out of a structured data table",
        "What is the population of Finland according to Wikipedia?",
        "https://en.wikipedia.org/wiki/Main_Page",
        ["en.wikipedia.org", "wikipedia.org"], _c_population,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-site smoke tests (read-only).")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config.BROWSER_HEADLESS = False
    # Real sites remember things, and a clean slate is what stops one run's
    # cookies deciding the next one's outcome.
    config.BROWSER_CLEAR_SITE_DATA = True

    from browser import action_cache, plan_cache
    from browser.task_runner import runner
    from guardrails import confirmation
    from llm.metrics import metrics

    confirmation.set_handler(lambda r: False)  # nothing here should need approval

    selected = REAL_SITES
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        selected = [a for a in REAL_SITES if a.name in wanted]

    print(f"Running {len(selected)} real-site tasks x {args.runs} runs (read-only)\n")
    report = {}

    for site in selected:
        print(f"--- {site.name}  ({site.pattern})")
        runs = []
        for i in range(args.runs):
            action_cache.clear()
            plan_cache.clear()
            started = time.monotonic()
            try:
                result = runner.run(
                    task=site.task, start_url=site.start_url,
                    allowed_domains=site.domains,
                    on_progress=(lambda m: print(f"      {m}")) if args.verbose else None,
                )
            except Exception as e:  # noqa: BLE001
                result = f"EXCEPTION: {type(e).__name__}: {e}"
            elapsed = time.monotonic() - started
            verdict, reason = site.check(result, [])
            calls = sum(s.calls for s in metrics.snapshot().values())
            runs.append({"verdict": verdict, "seconds": elapsed, "calls": calls,
                         "reason": reason, "result": (result or "")[:300]})
            mark = _MARK[verdict]
            print(f"    run {i + 1}: {mark:12} {elapsed:6.1f}s {calls:3d} calls"
                  + (f" - {reason}" if reason else ""))
            if verdict != PASS:
                print(f"        said: {runs[-1]['result'][:200]!r}")

        passes = sum(1 for r in runs if r["verdict"] == PASS)
        errors = sum(1 for r in runs if r["verdict"] == ERROR)
        # Same rule as the corpus runner: a run that never happened is not
        # evidence, so it leaves the denominator rather than counting as a
        # failure. Without this an exhausted API balance reads as "3/9 passed"
        # when the truth is "3 ran, 3 passed, 6 never started".
        report[site.name] = {
            "passes": passes, "runs": len(runs) - errors, "errors": errors,
            "wrong": sum(1 for r in runs if r["verdict"] == WRONG),
            "median_seconds": statistics.median(r["seconds"] for r in runs),
            "expected_to_fail": site.expected_to_fail,
        }
        print()

    print("=" * 74)
    print(f"{'site task':22} {'pass':>8} {'wrong':>6} {'err':>4} {'med s':>8}")
    print("-" * 74)
    for name, r in report.items():
        flag = "  (known hard)" if r["expected_to_fail"] else ""
        rate = f"{r['passes']}/{r['runs']}" if r["runs"] else "-"
        print(f"{name:22} {rate:>8} {r['wrong']:>6} {r['errors']:>4} "
              f"{r['median_seconds']:>8.1f}{flag}")
    total_wrong = sum(r["wrong"] for r in report.values())
    total_err = sum(r["errors"] for r in report.values())
    ran = sum(r["runs"] for r in report.values())
    passed = sum(r["passes"] for r in report.values())
    print("-" * 74)
    print(f"ran: {ran}   passed: {passed}   confidently WRONG: {total_wrong}")
    if total_err:
        print(f"never ran ({total_err}) - infrastructure, not behaviour. "
              "Check API quota before reading anything above.")
    print("=" * 74)
    return 1 if total_wrong else 0


if __name__ == "__main__":
    sys.exit(main())
