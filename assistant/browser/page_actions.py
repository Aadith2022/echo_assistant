"""Observing and acting on a page.

Observation is the security-critical half:

    DOM enumeration (ours)  ─┐
                             ├─> PageState ─> Actor
    page prose ─> sanitize ─> Quarantined LLM ─┘

The element list is built by us, and every element gets a `data-echo-ref`
number we assign. The Actor's entire vocabulary for "that element" is one of
those numbers - it cannot emit a CSS selector, so a page cannot induce it to
act on something that was never enumerated. Page prose reaches a model only
through `guardrails.prompt_injection_detector`, schema-forced.

When the DOM is unreadable - a canvas app, closed shadow roots, a legacy
frameset - observation falls back to vision via Playwright's
`page.screenshot()` rather than `mss`, which captures whatever window happens
to be in the foreground.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import config
from browser.content_extractor import sanitize
from browser.engine import engine, human_mouse_path, random_delay
from browser.page_state import InteractiveElement, PageDigest, PageState
from guardrails.audit_log import log_event
from guardrails.origin_sets import OriginSet
from guardrails.risk_classifier import is_credential_field
from guardrails.prompt_injection_detector import quarantine_extract, quarantine_extract_image

logger = logging.getLogger(__name__)

_MAX_ELEMENTS = 120
_MAX_PROSE_CHARS = 12000

# Reuse of Quarantined digests for page content we have already read this
# session. Small and process-local - see the note in observe_page.
_digest_cache: dict[str, PageDigest] = {}
_DIGEST_CACHE_MAX = 64
# Below this much readable content with no interactive elements, treat the DOM
# as unusable and escalate to vision rather than handing the Actor an empty page.
_DOM_UNRELIABLE_CHARS = 120

# THE definition of "an element the agent can act on". Everything that needs to
# agree about that imports this rather than restating it - a second copy in
# task_runner's page signature drifted twice, each time leaving stall detection
# blind to elements the agent could see.
INTERACTIVE_SELECTOR = ",".join((
    'a[href]', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
    'summary', '[contenteditable="true"]', '[onclick]',
    '[role="button"]', '[role="link"]', '[role="textbox"]', '[role="combobox"]',
    '[role="checkbox"]', '[role="radio"]', '[role="switch"]',
    '[role="tab"]', '[role="menuitem"]', '[role="menuitemradio"]',
    '[role="menuitemcheckbox"]', '[role="treeitem"]',
    '[role="option"]', '[role="gridcell"]', '[role="spinbutton"]',
))

# Enumerate visible, actionable elements and stamp each with our own ref.
# Runs in the page, so it must be plain ES5-compatible JS with no build step.
_ENUMERATE_JS = r"""
(args) => {
  const OFFSET = args.offset || 0;
  const LIMIT = args.max || 120;
  // Modern apps express most controls through ARIA roles rather than native
  // tags, and anything missing here is invisible to the agent. Autocomplete
  // suggestions ([role="option"]) and calendar days ([role="gridcell"]) were
  // the expensive omissions: without them a form could be typed into but not
  // completed, which reads as the model failing rather than never being shown
  // the controls.
  const SELECTOR = '__INTERACTIVE_SELECTOR__';

  // Clear previous stamps, shadow roots included. Refs restart at 0 each time,
  // so a stale stamp left inside a component makes [data-echo-ref="3"] match
  // two elements and Playwright's strict mode refuses to act on either.
  (function clearStamps(node, depth) {
    if (depth > 8) return;
    try {
      node.querySelectorAll('[data-echo-ref]').forEach(function (el) {
        el.removeAttribute('data-echo-ref');
      });
      node.querySelectorAll('*').forEach(function (el) {
        if (el.shadowRoot) clearStamps(el.shadowRoot, depth + 1);
      });
    } catch (e) { /* detached or cross-origin subtree */ }
  })(document, 0);

  // If a modal is open, only its contents are actionable - everything behind
  // it is covered, and clicking it fails with "subtree intercepts pointer
  // events" after a full timeout. A dialog often contains a duplicate of the
  // control that opened it, so the unscoped list offers an unreachable twin.
  let root = document;
  const modals = Array.prototype.slice.call(
    document.querySelectorAll('[role="dialog"][aria-modal="true"], dialog[open]')
  ).filter(function (d) {
    const r = d.getBoundingClientRect();
    const s = window.getComputedStyle(d);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
  });
  if (modals.length) root = modals[modals.length - 1];

  const seen = new Set();
  const out = [];
  // Refs continue across frames so they stay unique for the whole page.
  let ref = OFFSET;

  // querySelectorAll does not cross shadow boundaries, and whole design
  // systems render their entire UI into shadow roots - without this walk the
  // agent reports "no interactive elements" on a page full of them. Only OPEN
  // roots are reachable; closed ones fall through to the vision path.
  const nodes = [];
  function collect(node, depth) {
    if (depth > 8) return;
    let found;
    try { found = node.querySelectorAll(SELECTOR); } catch (e) { return; }
    for (let i = 0; i < found.length; i++) nodes.push(found[i]);

    let all;
    try { all = node.querySelectorAll('*'); } catch (e) { return; }
    for (let i = 0; i < all.length; i++) {
      if (all[i].shadowRoot) collect(all[i].shadowRoot, depth + 1);
    }
  }
  collect(root, 0);
  for (let i = 0; i < nodes.length && out.length < LIMIT; i++) {
    const el = nodes[i];
    if (seen.has(el)) continue;
    seen.add(el);

    if (el.disabled === true) continue;
    if (el.getAttribute('aria-hidden') === 'true') continue;

    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) continue;

    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    if (parseFloat(style.opacity || '1') < 0.05) continue;

    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();

    // `kind` is what the Actor sees: a coarse vocabulary of what you can do to
    // the thing. `control` is the widget role underneath, kept for risk
    // classification - a dropdown suggestion and a Place Order button are both
    // 'button' to the Actor and must not be to the risk tier.
    let kind = 'other';
    let control = role || tag;
    if (tag === 'a' || role === 'link') kind = 'link';
    else if (tag === 'button' || role === 'button' || type === 'submit' || type === 'button') kind = 'button';
    else if (tag === 'select') kind = 'select';
    else if (type === 'checkbox' || type === 'radio' || role === 'checkbox'
             || role === 'radio' || role === 'switch') kind = 'checkbox';
    // Suggestions and calendar days are things you click, so they are buttons
    // to the Actor - which also makes check_action_allowed refuse an attempt
    // to type into one.
    else if (role === 'option' || role === 'gridcell' || role === 'menuitem'
             || role === 'menuitemradio' || role === 'menuitemcheckbox'
             || role === 'tab' || role === 'treeitem') kind = 'button';
    else if (tag === 'input' || tag === 'textarea' || role === 'textbox'
             || role === 'combobox' || el.isContentEditable) kind = 'input';

    let label = el.getAttribute('aria-label')
      || el.getAttribute('placeholder')
      || el.getAttribute('title')
      || (el.innerText || '').trim()
      || el.getAttribute('alt')
      || el.getAttribute('name')
      || el.getAttribute('value')
      || '';
    label = String(label).replace(/\s+/g, ' ').trim().slice(0, 140);
    // An unlabelled element is unusable by the Actor and only adds noise to a
    // list that is already capped.
    if (!label) continue;

    // Never surface secrets: a model does not need a password field's contents
    // in order to decide whether to type into it.
    let value = '';
    if (kind === 'input' || kind === 'select') {
      value = type === 'password' ? '(hidden)' : String(el.value || '').slice(0, 80);
    } else if (kind === 'checkbox') {
      value = el.checked ? 'checked' : 'unchecked';
    }

    el.setAttribute('data-echo-ref', String(ref));
    out.push({
      ref: ref,
      kind: kind,
      control: control,
      label: label,
      value: value,
      href: kind === 'link' ? (el.href || '') : '',
      inViewport: rect.top < window.innerHeight && rect.bottom > 0
    });
    ref++;
  }
  return out;
}
"""
_ENUMERATE_JS = _ENUMERATE_JS.replace("__INTERACTIVE_SELECTOR__", INTERACTIVE_SELECTOR)


@dataclass
class Observation:
    """A PageState plus the bookkeeping that must never reach a model."""

    state: PageState
    # The URL redaction registry, kept out of PageState: the Actor works in
    # [url:N] handles and only this side can resolve one back to an address,
    # which is then still checked against the Origin Set.
    urls: list[str] = field(default_factory=list)
    raw_chars: int = 0
    used_vision: bool = False

    def resolve_handle(self, handle: str) -> str | None:
        import re

        match = re.fullmatch(r"\[?url:(\d+)\]?|(\d+)", str(handle).strip(), re.IGNORECASE)
        if not match:
            return None
        index = int(match.group(1) or match.group(2))
        return self.urls[index] if 0 <= index < len(self.urls) else None


# --- observation -------------------------------------------------------------


def _has_modal(page) -> bool:
    try:
        return bool(
            page.evaluate(
                """() => !!document.querySelector(
                     '[role="dialog"][aria-modal="true"], dialog[open]')"""
            )
        )
    except Exception:
        return False


def _raw_snapshot(page) -> dict:
    """Collect everything we need from the page in one browser-thread hop.

    Enumeration walks every frame, because `querySelectorAll` stops at the
    frame boundary and embedding is how the web ships payment forms, booking
    widgets and consent managers. Refs continue counting across frames so they
    stay unique, and `_locator` finds the owning frame by searching for the
    stamp rather than keeping a ref-to-frame map that re-enumeration would
    invalidate. When a modal is open, everything behind it - iframes included -
    is unreachable, so only the top frame is walked.
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=config.BROWSER_TIMEOUT_SECONDS * 1000)
    except Exception:
        logger.debug("wait_for_load_state timed out; observing anyway", exc_info=True)

    # `domcontentloaded` fires when the HTML is parsed, which on a JS-driven
    # site is long before it has content - observing then captures "Loading
    # results..." and the agent reasons about a page that does not exist yet.
    # Best-effort and short: a site with persistent polling never goes idle.
    try:
        page.wait_for_load_state("networkidle", timeout=config.BROWSER_SETTLE_MS)
    except Exception:
        logger.debug("Page did not go idle; observing as-is", exc_info=True)

    frames = [page.main_frame] if _has_modal(page) else list(page.frames)

    elements: list[dict] = []
    prose_parts: list[str] = []
    for frame in frames:
        if len(elements) >= _MAX_ELEMENTS:
            break
        try:
            found = frame.evaluate(
                _ENUMERATE_JS,
                {"offset": len(elements), "max": _MAX_ELEMENTS - len(elements)},
            )
            elements.extend(found or [])
        except Exception:
            # Detached, still loading, or an opaque cross-origin frame we
            # cannot script. Skipping one frame must not lose the others.
            logger.debug("Enumeration failed in frame %s", getattr(frame, "url", "?"), exc_info=True)

        try:
            text = frame.evaluate(
                "() => (document.body && document.body.innerText) ? document.body.innerText : ''"
            )
            if text and text.strip():
                prose_parts.append(text)
        except Exception:
            pass

    return {
        "url": page.url,
        "title": (page.title() or "")[:200],
        "elements": elements,
        "prose": "\n\n".join(prose_parts)[:_MAX_PROSE_CHARS],
        "frames": len(frames),
    }


def _screenshot(page) -> bytes:
    return page.screenshot(full_page=False, type="png")


def _unanalyzed_state(snapshot: dict, elements: list[InteractiveElement]) -> PageState:
    """A PageState whose prose was deliberately not read.

    The placeholder text is explicit rather than empty so that nothing
    downstream - a model, a log, a checker - can mistake "not extracted" for
    "the page said nothing".
    """
    return PageState(
        url=snapshot["url"],
        title=snapshot["title"],
        digest=PageDigest(
            page_type="(not analyzed)",
            summary="Structural observation only - page content was not read this round.",
        ),
        elements=elements,
    )


@dataclass(frozen=True)
class DigestBaseline:
    """What the page looked like the last time its prose was actually read.

    Passed to `observe_page` so it can decide whether reading the prose again
    is worth a Quarantined call. See `_prose_changed_materially`.
    """

    url: str
    chars: int


# How much the readable text must change before the page is worth re-reading.
# Interaction changes prose slightly and constantly - a typed value, a list
# opening - while a page that has become something else moves the character
# count by far more. A heuristic, and treated as one: the exact guarantees are
# the URL check below and the final `read_outcome` pass, which always reads the
# page in full.
_MATERIAL_PROSE_DELTA_RATIO = 0.25
_MATERIAL_PROSE_DELTA_MIN = 300


def _prose_changed_materially(url: str, text: str, baseline: DigestBaseline) -> bool:
    if url != baseline.url:
        return True
    delta = abs(len(text) - baseline.chars)
    return delta > max(_MATERIAL_PROSE_DELTA_MIN, baseline.chars * _MATERIAL_PROSE_DELTA_RATIO)


def observe_page(goal: str = "", baseline: DigestBaseline | None = None) -> Observation:
    """Read the current page into a PageState the Actor can act on.

    With a `baseline`, the prose only goes to the Quarantined LLM when the page
    has materially changed. Otherwise the element list is returned with the
    digest explicitly marked as not analyzed - deliberately not "reuse the
    previous digest", which would assert a stale reading as current and keep
    saying `injection_detected: false` about a page that had grown hostile text.
    """
    snapshot = engine.submit(_raw_snapshot)

    sanitized = sanitize(snapshot["prose"])
    urls = list(sanitized.urls)

    elements: list[InteractiveElement] = []
    for item in snapshot["elements"]:
        href_handle = ""
        href = item.get("href") or ""
        if href:
            if href not in urls:
                urls.append(href)
            href_handle = f"[url:{urls.index(href)}]"

        # Labels are attacker-controlled too, so they get the same stripping.
        label = sanitize(item.get("label", ""), max_chars=140).text
        elements.append(
            InteractiveElement(
                ref=int(item["ref"]),
                kind=item.get("kind", "other"),
                control=str(item.get("control", "")),
                label=label,
                value=str(item.get("value", ""))[:80],
                href_handle=href_handle,
            )
        )

    dom_unusable = not elements and len(sanitized.text.strip()) < _DOM_UNRELIABLE_CHARS
    used_vision = False

    # Not applied when the DOM is unusable: there the prose is the only signal,
    # and skipping it would hand the Actor nothing at all.
    if baseline is not None and not dom_unusable:
        if not _prose_changed_materially(snapshot["url"], sanitized.text, baseline):
            logger.info("Page not materially changed - structural observation, no model call")
            return Observation(
                state=_unanalyzed_state(snapshot, elements),
                urls=urls,
                raw_chars=len(snapshot["prose"]),
                used_vision=False,
            )

    # Consecutive steps often observe the same page, so digests are keyed on a
    # hash of the sanitized content plus the goal - a different goal wants
    # different key_facts - and any real change to the page misses. In-memory
    # and per-process: page content goes stale in seconds.
    digest_key = ""
    if not dom_unusable:
        digest_key = hashlib.sha256(
            f"{snapshot['url']}||{goal}||{sanitized.text}".encode("utf-8")
        ).hexdigest()
        cached_digest = _digest_cache.get(digest_key)
        if cached_digest is not None:
            logger.info("Reusing page digest (unchanged content) - no model call")
            return Observation(
                state=PageState(
                    url=snapshot["url"],
                    title=snapshot["title"],
                    digest=cached_digest,
                    elements=elements,
                ),
                urls=urls,
                raw_chars=len(snapshot["prose"]),
                used_vision=False,
            )

    if dom_unusable:
        # Canvas app, closed shadow roots, or a page that hadn't rendered.
        logger.info("DOM looks unusable (%d chars, 0 elements); falling back to vision", len(sanitized.text))
        log_event("browser_vision_fallback", url=snapshot["url"])
        try:
            image = engine.submit(_screenshot)
            digest = quarantine_extract_image(image, PageDigest, goal)
            used_vision = True
        except Exception:
            logger.exception("Vision fallback failed; returning an empty digest")
            digest = PageDigest(
                page_type="unreadable",
                summary="The page could not be read from either the DOM or a screenshot.",
            )
    else:
        digest = quarantine_extract(
            sanitized.text or "(the page contained no readable text)", PageDigest, goal
        )

    if digest_key:
        if len(_digest_cache) >= _DIGEST_CACHE_MAX:
            _digest_cache.clear()
        _digest_cache[digest_key] = digest

    state = PageState(
        url=snapshot["url"],
        title=snapshot["title"],
        digest=digest,
        elements=elements,
    )
    return Observation(
        state=state,
        urls=urls,
        raw_chars=len(snapshot["prose"]),
        used_vision=used_vision,
    )


def read_outcome(task: str) -> "TaskOutcome":
    """Read the final page and judge it against the user's goal.

    Goes through the Quarantined LLM like every other page read - the final
    page is no more trustworthy than any other, and a page claiming "Order
    complete!" is not evidence that anything completed.
    """
    from browser.page_state import TaskOutcome

    snapshot = engine.submit(_raw_snapshot)
    sanitized = sanitize(snapshot["prose"])
    return quarantine_extract(
        f"Page title: {snapshot['title']}\nURL: {snapshot['url']}\n\n"
        f"{sanitized.text or '(the page contained no readable text)'}",
        TaskOutcome,
        goal=task,
    )


def clear_digest_cache() -> None:
    """Called between tasks - a new task should read the world fresh."""
    _digest_cache.clear()


def observe_structure() -> Observation:
    """Enumerate the page's elements only - no digest, no model call.

    This is what makes an action-cache hit free: replaying a known step needs
    only the element list, which our own JavaScript builds, and it is the
    digest that costs a Quarantined call.
    """
    snapshot = engine.submit(_raw_snapshot)

    urls: list[str] = []
    elements: list[InteractiveElement] = []
    for item in snapshot["elements"]:
        href_handle = ""
        href = item.get("href") or ""
        if href:
            if href not in urls:
                urls.append(href)
            href_handle = f"[url:{urls.index(href)}]"
        elements.append(
            InteractiveElement(
                ref=int(item["ref"]),
                kind=item.get("kind", "other"),
                control=str(item.get("control", "")),
                label=sanitize(item.get("label", ""), max_chars=140).text,
                value=str(item.get("value", ""))[:80],
                href_handle=href_handle,
            )
        )

    return Observation(
        state=_unanalyzed_state(snapshot, elements),
        urls=urls,
        # Reported even though nothing was extracted - a later DigestBaseline
        # comparison measures against it.
        raw_chars=len(snapshot["prose"]),
        used_vision=False,
    )


# --- actions -----------------------------------------------------------------


def _locator(page, ref: int):
    """Find the element with this ref, in whichever frame holds it.

    Searching beats a ref-to-frame map, which would have to survive every
    re-enumeration and re-resolve; a `count()` per frame is cheap.
    """
    selector = f'[data-echo-ref="{ref}"]'
    for frame in page.frames:
        try:
            locator = frame.locator(selector)
            if locator.count() > 0:
                return locator
        except Exception:
            continue
    # Nothing matched; return the main-frame locator so the caller fails with
    # Playwright's own message rather than a None.
    return page.locator(selector)


def resolve_by_label(label: str, kind: str = "") -> int:
    """Find an element by its label on the page as it looks right now.

    Used when replaying a cached batch: a later action's target may exist only
    because an earlier one created it, so each resolves as it runs rather than
    from one snapshot taken before the batch started.
    """
    if not label:
        raise ValueError("Cannot resolve an element without a label.")

    from browser.action_cache import CachedAction, match_element

    fresh = observe_structure()
    resolved = match_element(
        CachedAction(action="", element_kind=kind, element_label=label), fresh.state.elements
    )
    if resolved is None:
        raise ValueError(f"Element {label!r} is not on the page.")
    return resolved


def ensure_ref(ref: int, label: str = "", kind: str = "") -> int:
    """Re-resolve an element ref that may have gone stale, returning a live one.

    The Quarantined extraction leaves a several-second gap between observing a
    page and acting on it, and a JS-heavy site can re-render in that window,
    destroying the stamps. Re-enumeration is free, so recovery is: enumerate
    again and match by label, as the action cache does.
    """
    if engine.submit(lambda page: _locator(page, ref).count()) > 0:
        return ref

    if not label:
        raise ValueError(f"Element {ref} is no longer on the page.")

    logger.info("Ref %s went stale; re-resolving by label %r", ref, label)
    from browser.action_cache import CachedAction, match_element

    fresh = observe_structure()
    resolved = match_element(
        CachedAction(action="", element_kind=kind, element_label=label),
        fresh.state.elements,
    )
    if resolved is None:
        raise ValueError(f"Element {label!r} is no longer on the page.")
    log_event("browser_ref_recovered", old_ref=ref, new_ref=resolved, label=label)
    return resolved


def navigate(url: str, origin_set: OriginSet) -> str:
    """Go to a URL, after checking it against the task's Origin Set."""
    origin_set.check_url(url, action="navigate")

    def _go(page):
        random_delay()
        response = page.goto(url, wait_until="domcontentloaded")
        # An allowed host can redirect to a disallowed one - an open redirect
        # would otherwise be a free way out of the Origin Set - so the
        # destination is checked too, not just the address we requested.
        final_url = page.url
        if final_url != url:
            origin_set.check_url(final_url, action="redirect")
        status = response.status if response else "unknown"
        return f"Navigated to {final_url} (status {status})"

    result = engine.submit(_go)
    log_event("browser_navigate", url=url, task=origin_set.task)
    return result


def click(ref: int, origin_set: OriginSet) -> str:
    """Click an enumerated element, with human-ish pointer movement."""

    def _click(page):
        locator = _locator(page, ref)
        locator.wait_for(state="visible", timeout=config.BROWSER_ACTION_TIMEOUT_SECONDS * 1000)
        box = locator.bounding_box()
        random_delay()
        if box:
            human_mouse_path(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        # Explicit: click() runs its own actionability wait on the page default
        # (30s), so a visible-but-covered element would cost half a minute.
        locator.click(timeout=config.BROWSER_ACTION_TIMEOUT_SECONDS * 1000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            # Plenty of clicks don't navigate; not waiting successfully is fine.
            pass
        if not origin_set.allows_url(page.url):
            # A click can navigate anywhere; land back inside the scope.
            page.go_back()
            origin_set.check_url(page.url, action="click-navigation")
        return f"Clicked element {ref}. Now at {page.url}"

    result = engine.submit(_click)
    log_event("browser_click", ref=ref, task=origin_set.task)
    return result


FILLABLE_KINDS = {"input", "select"}


def check_action_allowed(action: str, kind: str, label: str = "") -> None:
    """Reject an action that cannot possibly work on this element kind.

    Without this, filling a link goes to Playwright, which retries for the full
    timeout before failing. Failing in milliseconds with a message that says
    why lets the Actor correct itself on the next round.
    """
    if action == "fill" and is_credential_field(label):
        # Refused outright, before the Critic and before any confirmation:
        # there is no value the agent could put here that is not invented, and
        # a wrong password submitted repeatedly locks the user out.
        log_event("credential_fill_refused", label=label)
        raise ValueError(
            f"I will not type anything into {label!r}. It asks for a credential, and "
            "any value I supplied would be invented. Report that this site requires "
            "the user to sign in themselves, and stop."
        )

    if action == "fill" and kind not in FILLABLE_KINDS:
        raise ValueError(
            f"Cannot type into {label!r} - it is a {kind}, not a text field. "
            "If it opens or reveals a field, click it first."
        )


def fill(ref: int, text: str, origin_set: OriginSet) -> str:
    """Type into an enumerated field, keystroke by keystroke."""

    def _fill(page):
        locator = _locator(page, ref)
        locator.wait_for(state="visible", timeout=config.BROWSER_ACTION_TIMEOUT_SECONDS * 1000)
        random_delay()

        # Resolve exactly once, to focus. Everything after goes through the
        # keyboard, which targets whatever has focus rather than re-resolving.
        # Framework-driven inputs re-render on focus, destroying the stamp, so
        # a second resolve waits out its whole timeout on a dead node.
        locator.click(timeout=config.BROWSER_ACTION_TIMEOUT_SECONDS * 1000)

        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")

        # Real keystrokes: setting the value in one shot skips the input and
        # keydown events form validators listen for, and is a bot signal.
        page.keyboard.type(text, delay=_typing_delay())
        return f"Filled element {ref}."

    result = engine.submit(_fill)
    # The value is deliberately not logged - fields hold passwords and card
    # numbers.
    log_event("browser_fill", ref=ref, chars=len(text), task=origin_set.task)
    return result


def press(key: str, origin_set: OriginSet) -> str:
    def _press(page):
        random_delay()
        page.keyboard.press(key)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        return f"Pressed {key}. Now at {page.url}"

    return engine.submit(_press)


def back(origin_set: OriginSet) -> str:
    """Go back one page.

    The recovery move for a dead end; everything else in the action set moves
    forward. Still Origin-Set checked, because history can lead somewhere the
    current task is no longer allowed to be.
    """

    def _back(page):
        random_delay()
        page.go_back(wait_until="domcontentloaded")
        return page.url

    url = engine.submit(_back)
    origin_set.check_url(url, action="back")
    log_event("browser_back", url=url, task=origin_set.task)
    return f"Went back. Now at {url}"


def scroll(amount: int = 600) -> str:
    def _scroll(page):
        random_delay(0.1, 0.3)
        page.mouse.wheel(0, amount)
        return f"Scrolled {amount}px."

    return engine.submit(_scroll)


def page_looks_usable() -> bool:
    """Cheap, model-free check that a page actually loaded something.

    Used after a constructed URL, which is a guess about a site's query format
    - and a wrong guess yields a shell that would read as a genuine "no
    results". Deliberately permissive: anything ambiguous passes to the Actor.
    """

    def _check(page):
        try:
            text = page.evaluate(
                "() => (document.body && document.body.innerText) "
                "? document.body.innerText.trim() : ''"
            )
            controls = page.evaluate(
                "() => document.querySelectorAll('a,button,input,select,textarea').length"
            )
            title = (page.title() or "").lower()
        except Exception:
            return True  # can't tell; let the Actor look

        if len(text) < 60 and controls < 3:
            return False  # blank shell

        opening = text[:200].lower()
        error_markers = (
            "404", "page not found", "not found", "no longer available",
            "something went wrong", "an error occurred", "bad request",
            "forbidden", "access denied", "service unavailable",
        )
        if any(m in title for m in ("404", "not found", "error")):
            return False
        # Only the opening of the body, so an article about error pages is not
        # mistaken for one.
        return not any(m in opening for m in error_markers)

    try:
        return bool(engine.submit(_check))
    except Exception:
        return True


def current_url() -> str:
    return engine.submit(lambda page: page.url)


def _typing_delay() -> int:
    import random

    return random.randint(35, 110)
