"""Properties of the step loop, over randomly generated page behaviour.

The rest of the suite is regression tests, which prove only that we fixed what
we saw - a suite built that way goes blind exactly where nobody has looked. Each
guard in `_run_step` was added in response to one site, and the risk is that it
encodes that site rather than the rule.

So this asserts the rules instead, against pages that behave arbitrarily:
actions that work, do nothing, or raise; vetoes; a page that twitches without
progressing; a page that changes wholesale. Whatever the sequence, these hold.

Seeded, so a failure is reproducible - the scenario is printed with its seed.

Run:  python -m tests.browser.test_loop_properties
"""

from __future__ import annotations

import random
import types as pytypes

import config
import browser.page_actions as pa
import browser.task_runner as tr

_failures: list[str] = []

# The loop may end a step early - that is what the stall, veto and repetition
# guards are for - but it may never run past its guard or fail to return.
SCENARIOS = 300


class _Obs:
    raw_chars = 4000

    class state:
        elements: list = []
        url = "http://x/"
        digest = pytypes.SimpleNamespace(page_type="form")

        @staticmethod
        def render_for_actor() -> str:
            return ""


class _Validator:
    """Stands in for the Validator model, which the loop consults when the
    deterministic check is ambiguous. Always pessimistic, so scenarios cannot
    accidentally pass by being rescued."""

    @staticmethod
    def ask(prompt, schema):
        return tr.StepVerdict(succeeded=False, reason="stub")


class Scenario:
    """One arbitrarily-behaving page.

    Each round independently decides whether the action executes, whether the
    page moves, and whether the Critic objects - which is a fair model of the
    web, where none of those are correlated in any way the agent can rely on.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.p_veto = rng.choice([0.0, 0.1, 0.5])
        self.p_raise = rng.choice([0.0, 0.1, 0.3])
        self.p_moves = rng.choice([0.0, 0.3, 0.9, 1.0])
        self.p_substantial = rng.choice([0.0, 0.5, 1.0])
        self.n_actions = rng.choice([1, 2, 5])       # how varied the Actor is
        self.p_complete = rng.choice([0.0, 0.05])
        self.rounds = 0
        self.executed = 0
        self._tick = 0
        self._hash = 0

    def describe(self) -> str:
        return (
            f"veto={self.p_veto} raise={self.p_raise} moves={self.p_moves} "
            f"substantial={self.p_substantial} actions={self.n_actions} "
            f"complete={self.p_complete}"
        )

    # --- the pieces the loop talks to ---------------------------------------

    def decide(self, step, task, url, attempt, progress, last_error="",
               already_done=None, answers="", baseline=None):
        self.rounds += 1
        label = f"Button {self.rng.randrange(self.n_actions)}"
        self._label = label
        complete = self.rng.random() < self.p_complete
        return [tr.ActorDecision(action="click", ref=1)], _Obs(), None, complete

    def gate(self, step, task, decision, label, control=""):
        if self.rng.random() < self.p_veto:
            return False, "the safety reviewer blocked that", False
        return True, "", False

    def execute(self, *a, **k):
        if self.rng.random() < self.p_raise:
            raise RuntimeError("element not found")
        self.executed += 1
        if self.rng.random() < self.p_moves:
            self._hash += 1
            if self.rng.random() < self.p_substantial:
                self._tick += 1

    def signature(self):
        return {
            "url": "http://x/",
            "title": "t",
            "elements": self._tick,
            "text_len": self._tick * 100,
            "text_hash": str(self._hash),
        }


# One runner, reused. Constructing `BrowserTaskRunner` builds a real Critic,
# which builds an API client and an SSL context - about a second each, which is
# invisible once and fatal three hundred times. The Critic is switched off here
# regardless: `_gate` is stubbed, so a live one would only be a slow no-op.
_RUNNER: "tr.BrowserTaskRunner | None" = None


def _runner_singleton() -> "tr.BrowserTaskRunner":
    global _RUNNER
    if _RUNNER is None:
        config.CRITIC_ENABLED = False
        _RUNNER = tr.BrowserTaskRunner()
        _RUNNER._validator = _Validator()
    return _RUNNER


def run_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    sc = Scenario(rng)

    tr.page_actions = pytypes.SimpleNamespace(
        current_url=lambda: "http://x/",
        DigestBaseline=pa.DigestBaseline,
        observe_page=lambda goal="", baseline=None: _Obs(),
        observe_structure=lambda: _Obs(),
        clear_digest_cache=lambda: None,
    )
    tr.action_cache = pytypes.SimpleNamespace(
        lookup=lambda *a: None, store=lambda *a: None,
        invalidate=lambda *a: None, match_element=lambda *a: None,
    )
    tr._page_signature = sc.signature
    tr.log_event = lambda *a, **k: None

    runner = _runner_singleton()
    runner._decide = sc.decide
    runner._gate = sc.gate
    runner._execute = sc.execute
    runner._label_for = lambda obs, d: sc._label
    runner._kind_for = lambda *a, **k: "button"
    runner._control_for = lambda *a, **k: "button"
    runner._run_step("step", "task", None, lambda m: None)
    return sc


def main() -> int:
    config.BROWSER_MAX_ITERATIONS_PER_STEP = 20
    config.BROWSER_MAX_STALLS_PER_STEP = 4
    config.BROWSER_MAX_VETOES_PER_STEP = 3
    config.BROWSER_MAX_RETRIES_PER_STEP = 2

    guard = config.BROWSER_MAX_ITERATIONS_PER_STEP
    # A round can be refunded - a veto never touched the page - so the ceiling
    # on *rounds* is the round guard plus the refunds the loop is allowed to
    # grant. Anything beyond that is a budget leak.
    ceiling = guard + config.BROWSER_MAX_VETOES_PER_STEP + 1

    worst = 0
    stuck_runs: list[int] = []
    for seed in range(SCENARIOS):
        try:
            sc = run_scenario(seed)
        except Exception as e:  # noqa: BLE001
            _failures.append(f"seed {seed} raised {type(e).__name__}: {e}")
            continue

        worst = max(worst, sc.rounds)

        # P1: it returned at all (reaching here proves termination).
        # P2: it never ran past what the budgets allow.
        if sc.rounds > ceiling:
            _failures.append(
                f"seed {seed} ran {sc.rounds} rounds (ceiling {ceiling}) - {sc.describe()}"
            )

        # P3: a page that never moves must be abandoned quickly, whatever the
        # Actor does. This is the property that stops a 47-round 'Close menu'
        # loop, and it must not depend on the Actor repeating itself exactly.
        if sc.p_moves == 0.0 and sc.p_veto == 0.0 and sc.p_raise == 0.0:
            if sc.rounds > config.BROWSER_MAX_STALLS_PER_STEP + 2:
                _failures.append(
                    f"seed {seed} spent {sc.rounds} rounds on a page that never moved "
                    f"- {sc.describe()}"
                )
            stuck_runs.append(sc.rounds)

        # P4: a page that always moves substantially and is never complete
        # should get the full budget - the guards must not cut short a step
        # that is working.
        if (sc.p_moves == 1.0 and sc.p_substantial == 1.0 and sc.p_veto == 0.0
                and sc.p_raise == 0.0 and sc.p_complete == 0.0):
            if sc.rounds != guard:
                _failures.append(
                    f"seed {seed} cut a working step short at {sc.rounds}/{guard} "
                    f"- {sc.describe()}"
                )

    print(f"  scenarios run          : {SCENARIOS}")
    print(f"  longest step           : {worst} rounds (ceiling {ceiling})")
    if stuck_runs:
        print(f"  dead pages abandoned in: {min(stuck_runs)}-{max(stuck_runs)} rounds")
    print()
    if _failures:
        for f in _failures[:10]:
            print(f"  FAIL  {f}")
        print(f"\nFAILED: {len(_failures)} scenarios")
        return 1
    print("OK - properties hold across every generated scenario")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
