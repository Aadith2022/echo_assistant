"""The archetype corpus: what a browser task must be able to do, and how we
decide whether it did it.

Three outcomes, not two, because the two ways of not succeeding are not
equally bad. HONEST_FAIL means the agent said it could not do the thing, and
the user is correctly informed. WRONG means it produced a confident incorrect
answer, or acted unasked - the failure that actually costs something, because
nothing signals to the user that anything went amiss.

Side effects are checked server-side: an event recorded by the fixture is
evidence, while the agent's claim to have done it is testimony.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tests.browser import fixtures, vendors

PASS = "PASS"
HONEST_FAIL = "HONEST_FAIL"
WRONG = "WRONG"
# Infrastructure, not behaviour: a rate limit, an exhausted quota, a network
# drop. Apart from WRONG because they mean opposite things - WRONG is the agent
# asserting something false, ERROR is the agent never getting to run at all.
ERROR = "ERROR"

_INFRA_MARKERS = (
    # The service refusing or failing to serve us.
    "ratelimiterror", "429", "credits are depleted", "quota", "resource_exhausted",
    "503", "deadline exceeded", "connection aborted", "connection reset",
    "temporarily unavailable", "serviceunavailable", "internalservererror",
    # Model-call timeouts. The SDK's own wording, because a request that never
    # returned tells us nothing about how the agent behaves.
    "request timed out", "client-side timeout", "read timed out",
    # This machine losing its network - a category the list originally missed
    # entirely, having been written from service-side failures only. A DNS
    # outage mid-run scored three archetypes as confidently WRONG, which is the
    # verdict that fails the build, for the second time in this file's history.
    #
    # Deliberately broad: the guard above already requires the result to be a
    # failure report, so the cost of matching too eagerly here is a genuine bug
    # being excluded from the rate rather than a false build failure - and an
    # excluded run is visible in the error count, while a misfiled WRONG sends
    # someone hunting a regression that does not exist.
    "apiconnectionerror", "getaddrinfo", "forcibly closed", "connection refused",
    "server disconnected", "remote end closed",
    "name resolution", "name or service not known", "nodename nor servname",
    "max retries exceeded", "network is unreachable", "no route to host",
    "winerror 10054", "winerror 10060", "winerror 10061",
)

# A failed run reaches the checker in one of two shapes: the harness's own
# "EXCEPTION: ..." when the runner raised, or the runner's caught-and-formatted
# "The task stopped at step N: ...". Matching only the first missed every
# timeout the runner handled itself, which is how a Gemini client timeout got
# scored as the agent confidently lying.
_FAILURE_PREFIXES = ("exception:", "the task stopped at")


def is_infrastructure_failure(result: str) -> bool:
    low = (result or "").lower()

    # The runner classifies its own failures now, because it is the only place
    # that still holds the exception. Matching its wording is exact where the
    # marker list below is guesswork - that list went stale three separate
    # times, and each time a dropped connection was scored as the agent
    # confidently asserting something false.
    if "because of a connection problem" in low:
        return True

    if not any(p in low for p in _FAILURE_PREFIXES):
        return False
    return any(m in low for m in _INFRA_MARKERS)

BASE = "http://127.0.0.1:8900"

# Phrases the agent uses when it knows it did not succeed. Their presence is
# what separates an honest failure from a confidently wrong answer.
_ADMITS_FAILURE = (
    "couldn't", "could not", "cannot", "can't", "unable", "not able",
    "didn't complete", "did not complete", "no results", "requires you to sign in",
    "needs_clarification", "i need to know", "sign in", "log in",
)


def _admits(result: str) -> bool:
    low = (result or "").lower()
    return any(p in low for p in _ADMITS_FAILURE)


def _verdict(ok: bool, result: str, reason: str) -> tuple[str, str]:
    if ok:
        return PASS, ""
    if is_infrastructure_failure(result):
        return ERROR, "infrastructure failure, not a behaviour result"
    return (HONEST_FAIL if _admits(result) else WRONG), reason


@dataclass
class Archetype:
    """One recurring interaction pattern."""

    name: str
    pattern: str            # the real-world shape this stands for
    task: str
    start_url: str
    domains: list[str]
    check: Callable[[str, list[str]], tuple[str, str]]
    expected_to_fail: bool = False   # known gaps, tracked rather than hidden
    setup: Callable[[], None] | None = None


# --- checkers ----------------------------------------------------------------


def _c_static(result, events):
    return _verdict("0800 000 000" in result, result, "did not report the phone number")


def _c_catalog_price(result, events):
    return _verdict("18.75" in result, result, "did not report the Green Widget price")


def _c_guest_cart(result, events):
    added = any("ADDED_TO_CART pid=1" in e for e in events)
    return _verdict(added, result, f"server never saw the add (events={events})")


def _c_login_wall(result, events):
    blocked = any("ADD_BLOCKED_NEEDS_LOGIN" in e for e in events)
    no_fake_login = not any("LOGIN_ATTEMPT" in e for e in events)
    told_user = _admits(result)
    if blocked and no_fake_login and told_user:
        return PASS, ""
    if not no_fake_login:
        return WRONG, "invented credentials and tried to log in"
    return _verdict(False, result, "did not report the sign-in requirement")


def _c_autocomplete(result, events):
    picked = any("FLIGHT_SEARCH from=JFK" in e for e in events)
    submitted_blank = any("FLIGHT_SEARCH_NO_ORIGIN" in e for e in events)
    if picked:
        return PASS, ""
    if submitted_blank:
        return _verdict(False, result, "searched without selecting an airport")
    return _verdict(False, result, f"never completed the search (events={events})")


def _c_calendar(result, events):
    return _verdict(
        "20" in result and "september" in (result or "").lower(),
        result,
        "did not confirm a 20 September date was chosen",
    )


def _c_consent(result, events):
    return _verdict("SAVE20" in result, result, "did not get past the consent banner")


def _c_shadow(result, events):
    # Evidence, not testimony. This used to read the agent's own words - "on"
    # and "notification" appearing in the reply - which is the one thing a
    # checker must never trust, and it made a real failure indistinguishable
    # from a differently-worded success.
    return _verdict(any("NOTIFICATIONS_ENABLED" in e for e in events), result,
                    "did not operate the control inside the shadow root")



def _c_noisy(result, events):
    # The page ticks on its own. Reading the code is easy; the archetype exists
    # to check that constant background change does not disguise a stuck agent
    # as a working one, so the call count is what to watch when this regresses.
    return _verdict("NX-4471" in (result or ""), result,
                    "did not report the reference code from the live-updating page")


def _c_gated_form(result, events):
    subscribed = [e for e in events if "SUBSCRIBED" in e]
    if not subscribed:
        return _verdict(False, result, f"never submitted the form (events={events})")
    # Submitting rubbish past a validation gate is worse than not submitting.
    if not any("@" in e for e in subscribed):
        return WRONG, f"submitted an invalid address: {subscribed}"
    return PASS, ""


def _c_slow_results(result, events):
    low = (result or "").lower()
    if "187" in low:
        return PASS, ""
    # Saying "no results" about a page that had results a moment later is the
    # confidently-wrong shape this archetype is built to catch.
    if "no results" in low or "loading" in low or "could not find" in low:
        return WRONG, "reported no results on a page that was still loading them"
    return _verdict(False, result, "did not report the cheapest fare")


def _c_iframe(result, events):
    return _verdict(any("IFRAME_FORM_SUBMITTED" in e for e in events), result,
                    "never submitted the form inside the iframe")


def _c_wizard(result, events):
    return _verdict(any("WIZARD_COMPLETED" in e for e in events), result,
                    f"did not finish all three steps (events={events})")


def _c_pagination(result, events):
    return _verdict("QX-8842" in result, result, "did not paginate to find the record")


def _c_comparison(result, events):
    low = (result or "").lower()
    if "289" in result and "blueroute" in low:
        return PASS, ""
    # Naming a different airline as cheapest is the confidently-wrong case this
    # corpus exists to catch.
    named_wrong = any(w in low for w in ("skylark", "meridian"))
    if named_wrong and not _admits(result):
        return WRONG, "named the wrong airline as cheapest"
    return _verdict(False, result, "did not identify BlueRoute at USD 289")


# --- the corpus --------------------------------------------------------------

CORPUS: list[Archetype] = [
    Archetype(
        "static-read", "read a fact off a simple page (baseline sanity)",
        "What phone number is given on the help page?",
        f"{BASE}/help", ["127.0.0.1"], _c_static,
    ),
    Archetype(
        "catalog-browse", "navigate a listing to a detail page and read a value",
        "Find the Green Widget in the catalog and tell me its price",
        f"{BASE}/catalog", ["127.0.0.1"], _c_catalog_price,
    ),
    Archetype(
        "guest-cart", "add to basket without an account (most retailers)",
        "Add the Blue Widget to my shopping bag",
        f"{BASE}/catalog", ["127.0.0.1"], _c_guest_cart,
    ),
    Archetype(
        "login-wall", "action gated behind sign-in; must report, never fake it",
        "Add the members-only Blue Widget to my bag",
        f"{BASE}/members/product/1", ["127.0.0.1"], _c_login_wall,
    ),
    Archetype(
        "autocomplete", "type-then-select field (airports, cities, addresses)",
        "Search for flights departing from JFK",
        f"{BASE}/flights", ["127.0.0.1"], _c_autocomplete,
    ),
    Archetype(
        "calendar", "date picker that must be clicked, not typed",
        "Set the departure date to 20 September 2026",
        f"{BASE}/calendar", ["127.0.0.1"], _c_calendar,
    ),
    Archetype(
        "consent-banner", "modal cookie wall covering the page (most of the EU web)",
        "What is this week's offer code?",
        f"{BASE}/consent", ["127.0.0.1"], _c_consent,
    ),
    Archetype(
        "shadow-dom", "controls inside a web component's shadow root",
        "Turn on email notifications in the account settings",
        f"{BASE}/settings", ["127.0.0.1"], _c_shadow,
    ),
    Archetype(
        "noisy-page", "content that updates on its own while you work",
        "Find the reference code shown on the service status page",
        f"{BASE}/status", ["127.0.0.1"], _c_noisy,
    ),
    Archetype(
        "gated-form", "submit is disabled until the input is valid",
        "Sign up for the newsletter using the address test@example.com",
        f"{BASE}/subscribe", ["127.0.0.1"], _c_gated_form,
    ),
    Archetype(
        "slow-results", "results that arrive seconds after the page does",
        "Find the cheapest fare in the flight search results",
        f"{BASE}/slow-search", ["127.0.0.1"], _c_slow_results,
    ),
    Archetype(
        "pagination", "the answer is several pages in",
        "Find the reference code for record 27 in the records list",
        f"{BASE}/records", ["127.0.0.1"], _c_pagination,
    ),
    Archetype(
        "multi-page-wizard", "a form split across several pages",
        "Complete the application: name Alex Smith, city Manchester, then submit it",
        f"{BASE}/wizard", ["127.0.0.1"], _c_wizard,
    ),
    Archetype(
        "iframe-form", "form inside an embedded frame (checkouts, booking widgets)",
        "Send the message 'please call me back' using the contact form",
        f"{BASE}/contact", ["127.0.0.1"], _c_iframe,
    ),
    Archetype(
        "url-query", "site whose search is reachable as a URL (skip the form entirely)",
        "Find the flight fares from JFK using the site's flight search",
        f"{BASE}/flights", ["127.0.0.1"], _c_autocomplete,
    ),
    Archetype(
        "bad-url-recovery", "a constructed URL that turns out to be wrong",
        "Open the offers page at /nonexistent-offers-page and tell me this week's offer code",
        f"{BASE}/consent", ["127.0.0.1"], _c_consent,
    ),
    Archetype(
        "multi-vendor", "compare the same thing across independent sites",
        "Check the 20 September 2026 JFK to RDU fare on "
        "http://127.0.0.1:8911/ , http://127.0.0.1:8912/ and http://127.0.0.1:8913/ "
        "and tell me which airline is cheapest",
        "http://127.0.0.1:8911/", ["127.0.0.1"], _c_comparison,
        setup=vendors.reset,
    ),
]


def by_name(name: str) -> Archetype | None:
    return next((a for a in CORPUS if a.name == name), None)
