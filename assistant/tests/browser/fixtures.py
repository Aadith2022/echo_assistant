"""Archetype fixtures - interaction *patterns*, not copies of particular sites.

Each one isolates a shape that recurs across thousands of real pages, so a fix
driven by it generalises instead of special-casing whichever site we happened
to test against. A hard flight-search page is not a site to support; it is
autocomplete plus a calendar plus a multi-field form.

A fixture must match reality in both directions. Weaker than reality invents
bugs - an empty-cart page with no navigation is a dead end no real store ships,
and it sends the agent in circles. More convenient than reality hides them:
real pages have consent banners, ARIA-role widgets, iframes and shadow roots,
and leaving those out makes the suite pass while the product fails.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
import urllib.parse

# Recorded server-side so a checker can assert what actually happened, rather
# than trusting the agent's own account of it.
EVENTS: list[str] = []
CARTS: dict[str, list[str]] = {}

_NAV = (
    '<nav><a href="/">Home</a> | <a href="/catalog">Catalog</a> | '
    '<a href="/cart">Bag</a> | <a href="/help">Help</a></nav>'
)

PRODUCTS = {
    "1": ("Blue Widget", "24.99"),
    "2": ("Red Widget", "31.50"),
    "3": ("Green Widget", "18.75"),
}

AIRPORTS = [
    ("JFK", "John F. Kennedy International Airport, New York"),
    ("LGA", "LaGuardia Airport, New York"),
    ("EWR", "Newark Liberty International, New York"),
    ("RDU", "Raleigh-Durham International Airport, North Carolina"),
]


def _page(title: str, body: str, banner: bool = False) -> bytes:
    # A consent banner that genuinely covers the page, as real ones do: the
    # agent must dismiss it before anything underneath is clickable.
    consent = (
        """
        <div id="consent" role="dialog" aria-modal="true"
             style="position:fixed;inset:0;background:#fff;z-index:9999;padding:2em">
          <h2>We value your privacy</h2>
          <p>We use cookies to improve your experience.</p>
          <button id="accept" onclick="document.getElementById('consent').remove()">
            Accept all</button>
          <button id="reject" onclick="document.getElementById('consent').remove()">
            Reject non-essential</button>
        </div>"""
        if banner
        else ""
    )
    return f"""<!doctype html><html><head><title>{title}</title></head>
<body>{consent}<h1>Archetype Co.</h1>{_NAV}{body}</body></html>""".encode()


# --- individual archetype pages ---------------------------------------------


def _catalog() -> str:
    items = "".join(
        f'<div><h3>{n}</h3><p>Price: GBP {p}</p>'
        f'<a href="/product/{i}">View {n}</a></div>'
        for i, (n, p) in PRODUCTS.items()
    )
    return f"<h2>Catalog</h2>{items}"


def _autocomplete_page() -> str:
    """Type-then-select. The field is only valid once a suggestion is chosen -
    exactly how airport, city and address inputs behave everywhere."""
    options = "".join(
        f'<li role="option" data-code="{c}" onclick="pick(this)">{c} - {n}</li>'
        for c, n in AIRPORTS
    )
    return f"""
    <h2>Book a flight</h2>
    <label for="from">Origin</label>
    <input id="from" role="combobox" autocomplete="off" placeholder="Origin airport"
           oninput="show()">
    <ul id="suggestions" role="listbox" style="display:none">{options}</ul>
    <input type="hidden" id="fromcode">
    <p id="chosen">No origin selected.</p>
    <form action="/flights/results" method="GET">
      <input type="hidden" name="from" id="fromfield">
      <button type="submit" id="searchbtn">Search flights</button>
    </form>
    <script>
      function show() {{ document.getElementById('suggestions').style.display = 'block'; }}
      function pick(el) {{
        var c = el.getAttribute('data-code');
        document.getElementById('from').value = el.innerText;
        document.getElementById('fromfield').value = c;
        document.getElementById('chosen').innerText = 'Origin selected: ' + c;
        document.getElementById('suggestions').style.display = 'none';
      }}
    </script>"""


def _calendar_page() -> str:
    """A date picker where typing does nothing - the day must be clicked, and
    the right month reached first."""
    days = "".join(
        f'<td role="gridcell" onclick="pickDay({d})">{d}</td>' for d in range(1, 29)
    )
    return f"""
    <h2>Choose a departure date</h2>
    <p>Showing: <span id="month">August 2026</span></p>
    <button id="next" onclick="nextMonth()">Next month</button>
    <table role="grid"><tr>{days}</tr></table>
    <p id="picked">No date chosen.</p>
    <script>
      var m = 8;
      function nextMonth() {{
        m += 1;
        document.getElementById('month').innerText =
          (m === 9 ? 'September' : m === 10 ? 'October' : 'Month ' + m) + ' 2026';
      }}
      function pickDay(d) {{
        document.getElementById('picked').innerText =
          'Departure set to ' + d + ' ' + document.getElementById('month').innerText;
      }}
    </script>"""


def _shadow_page() -> str:
    """The entire control lives in an open shadow root, as component libraries
    ship it. Invisible to querySelectorAll from the light DOM."""
    return """
    <h2>Account settings</h2>
    <my-settings></my-settings>
    <p id="status">Notifications are OFF.</p>
    <script>
      class MySettings extends HTMLElement {
        constructor() {
          super();
          const r = this.attachShadow({mode: 'open'});
          r.innerHTML =
            '<label for="n">Email notifications</label>' +
            '<button id="toggle">Turn notifications on</button>';
          r.getElementById('toggle').addEventListener('click', function () {
            document.getElementById('status').innerText = 'Notifications are ON.';
            // Reported to the server so the checker can assert on evidence
            // rather than on the agent's account of itself. A real settings
            // toggle persists too - a fixture that only changed local text was
            // both less realistic and untestable.
            try { fetch('/notify-on', {method: 'POST'}); } catch (e) {}
          });
        }
      }
      customElements.define('my-settings', MySettings);
    </script>"""



def _noisy_page() -> str:
    """The answer sits on a page that will not hold still.

    Real pages tick: live prices, "3 people are viewing this", relative
    timestamps, carousels. The stall detector asks "did anything change", and a
    page changing on its own answers yes forever - so a stuck agent can click
    the same dead button indefinitely without ever tripping a guard.

    This archetype is not about reading the value. It is about whether a
    working agent can still be told apart from a stuck one on a noisy page.
    """
    return """
    <h2>Service status</h2>
    <p>Reference code: <strong>NX-4471</strong></p>
    <p id="ticker">Live viewers: 0</p>
    <script>
      let n = 0;
      setInterval(function () {
        n += 1;
        document.getElementById('ticker').innerText =
          'Live viewers: ' + n + ' - updated ' + new Date().toISOString();
      }, 400);
    </script>"""


def _gated_form() -> str:
    """Submit stays disabled until the input is actually valid.

    The failure this guards against is an agent that clicks a dead control
    repeatedly instead of working out why nothing happens - and reports success
    because "the page changed" when it never submitted.
    """
    return """
    <h2>Newsletter signup</h2>
    <form id="f" method="POST" action="/subscribe">
      <label for="email">Email address</label>
      <input id="email" name="email" placeholder="Email address">
      <button id="go" type="submit" disabled>Subscribe</button>
    </form>
    <script>
      const email = document.getElementById('email');
      email.addEventListener('input', function () {
        document.getElementById('go').disabled = email.value.indexOf('@') === -1;
      });
    </script>"""


def _slow_results(delay_ms: int = 2500) -> str:
    """Results that arrive well after the page does.

    Everything the agent can see says "no results" for the first few seconds.
    Reporting that is a confidently wrong answer, and it is the single easiest
    one to produce by accident - the DOM is settled, the request is finished,
    and the content simply is not there yet.
    """
    return f"""
    <h2>Search flights</h2>
    <div id="results"><p>Loading results...</p></div>
    <script>
      setTimeout(function () {{
        document.getElementById('results').innerHTML =
          '<ul><li>Meridian Air - GBP 214</li>' +
          '<li>Northwind - GBP 187</li>' +
          '<li>SkyLark - GBP 259</li></ul>';
      }}, {delay_ms});
    </script>"""


def _iframe_page() -> str:
    """A form in a same-origin iframe - the shape of every embedded checkout,
    booking widget and support form."""
    return """
    <h2>Contact us</h2>
    <p>Fill in the form below and we will get back to you.</p>
    <iframe id="formframe" src="/embedded-form" width="600" height="300"></iframe>"""


def _wizard(step: int) -> str:
    if step == 1:
        return """<h2>Step 1 of 3: your details</h2>
        <form action="/wizard/2" method="POST">
          <label for="name">Full name</label><input id="name" name="name">
          <button type="submit">Continue to step 2</button></form>"""
    if step == 2:
        return """<h2>Step 2 of 3: delivery</h2>
        <form action="/wizard/3" method="POST">
          <label for="city">City</label><input id="city" name="city">
          <button type="submit">Continue to step 3</button></form>"""
    return """<h2>Step 3 of 3: confirm</h2>
        <form action="/wizard/done" method="POST">
          <button type="submit">Submit application</button></form>"""


def _paginated(page: int) -> str:
    """The target is deliberately on page 3, so it cannot be found without
    paginating."""
    rows = "".join(
        f"<li>Record {i}: status normal</li>" for i in range((page - 1) * 10, page * 10)
    )
    special = "<li><strong>Record 27: reference code QX-8842</strong></li>" if page == 3 else ""
    nxt = f'<a href="/records?page={page + 1}">Next page</a>' if page < 4 else ""
    return f"<h2>Records - page {page}</h2><ul>{rows}{special}</ul>{nxt}"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, status: int = 200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in headers or []:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, to: str):
        self.send_response(302)
        self.send_header("Location", to)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/home"):
            return self._send(_page("Archetype Co.", "<p>Welcome.</p>" + _catalog()))
        if path == "/catalog":
            return self._send(_page("Catalog", _catalog()))
        if path == "/consent":
            return self._send(
                _page("Special offers", "<h2>Special offers</h2>"
                      "<p>This week's offer code is SAVE20.</p>", banner=True)
            )
        if path.startswith("/product/"):
            pid = path.rsplit("/", 1)[-1]
            if pid not in PRODUCTS:
                return self._send(_page("Not found", "<p>No such product.</p>"), 404)
            name, price = PRODUCTS[pid]
            return self._send(_page(name, f"""
                <h2>{name}</h2><p>Price: GBP {price}</p><p>In stock.</p>
                <form method="POST" action="/add"><input type="hidden" name="pid" value="{pid}">
                <button type="submit">Add to bag</button></form>"""))
        if path == "/cart":
            cart = CARTS.setdefault("s1", [])
            if not cart:
                body = "<p>Your bag is empty.</p>"
            else:
                lines = "".join(f"<li>{PRODUCTS[i][0]} - GBP {PRODUCTS[i][1]}</li>" for i in cart)
                body = f"<p>Your bag has {len(cart)} item(s).</p><ul>{lines}</ul>"
            return self._send(_page("Your bag", body))
        if path == "/members/product/1":
            return self._send(_page("Members Blue Widget", """
                <h2>Blue Widget (members only)</h2><p>Price: GBP 19.99</p>
                <form method="POST" action="/members/add">
                <button type="submit">Add to bag</button></form>"""))
        if path == "/login":
            return self._send(_page("Sign in", """
                <h2>Sign in</h2><p>You must sign in to buy members-only items.</p>
                <form method="POST" action="/login">
                <label for="u">Email</label><input id="u" name="u">
                <label for="p">Password</label><input id="p" name="p" type="password">
                <button type="submit">Sign in</button></form>"""))
        if path == "/flights":
            return self._send(_page("Book a flight", _autocomplete_page()))
        if path == "/flights/results":
            code = (query.get("from") or [""])[0]
            if not code:
                EVENTS.append("FLIGHT_SEARCH_NO_ORIGIN")
                return self._send(_page("Flights", "<p>Please choose an origin airport first.</p>"))
            EVENTS.append(f"FLIGHT_SEARCH from={code}")
            return self._send(_page("Flights", f"""
                <h2>Flights from {code}</h2>
                <p>AC101 departs 07:15 - fare GBP 212</p>
                <p>AC204 departs 14:40 - fare GBP 268</p>"""))
        if path == "/calendar":
            return self._send(_page("Choose a date", _calendar_page()))
        if path == "/settings":
            return self._send(_page("Settings", _shadow_page()))
        if path == "/contact":
            return self._send(_page("Contact us", _iframe_page()))
        if path == "/embedded-form":
            # Served without the outer chrome, as a real embedded widget is.
            return self._send(b"""<!doctype html><html><body>
                <form method="POST" action="/embedded-form">
                <label for="msg">Your message</label>
                <input id="msg" name="msg" placeholder="Your message">
                <button type="submit">Send message</button></form></body></html>""")
        if path == "/wizard" or path == "/wizard/1":
            return self._send(_page("Application", _wizard(1)))
        if path == "/records":
            page = int((query.get("page") or ["1"])[0])
            return self._send(_page(f"Records page {page}", _paginated(page)))
        if path == "/status":
            return self._send(_page("Status", _noisy_page()))
        if path == "/subscribe":
            return self._send(_page("Newsletter", _gated_form()))
        if path == "/slow-search":
            return self._send(_page("Search", _slow_results()))
        if path == "/help":
            return self._send(_page("Help", "<h2>Help</h2><p>Call us on 0800 000 000.</p>"))
        return self._send(_page("Not found", "<p>Nothing here.</p>"), 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        data = urllib.parse.parse_qs(self.rfile.read(length).decode())

        if path == "/subscribe":
            email = (data.get("email") or [""])[0]
            EVENTS.append(f"SUBSCRIBED email={email!r}")
            return self._send(_page("Newsletter", "<p>Thanks for subscribing.</p>"))
        if path == "/notify-on":
            EVENTS.append("NOTIFICATIONS_ENABLED")
            return self._send("<p>ok</p>")
        if path == "/add":
            pid = (data.get("pid") or ["1"])[0]
            CARTS.setdefault("s1", []).append(pid)
            EVENTS.append(f"ADDED_TO_CART pid={pid}")
            return self._redirect("/cart")
        if path == "/members/add":
            EVENTS.append("ADD_BLOCKED_NEEDS_LOGIN")
            return self._redirect("/login")
        if path == "/login":
            EVENTS.append(f"LOGIN_ATTEMPT user={(data.get('u') or [''])[0]!r}")
            return self._send(_page("Sign in", "<p style='color:red'>Incorrect email or password.</p>"
                                    "<form method='POST' action='/login'>"
                                    "<label for='u'>Email</label><input id='u' name='u'>"
                                    "<label for='p'>Password</label><input id='p' name='p' type='password'>"
                                    "<button type='submit'>Sign in</button></form>"))
        if path == "/embedded-form":
            EVENTS.append(f"IFRAME_FORM_SUBMITTED msg={(data.get('msg') or [''])[0]!r}")
            return self._send(b"<!doctype html><html><body><p>Thanks, we got your message.</p></body></html>")
        if path.startswith("/wizard/"):
            stage = path.rsplit("/", 1)[-1]
            if stage == "done":
                EVENTS.append("WIZARD_COMPLETED")
                return self._send(_page("Done", "<h2>Application submitted</h2>"
                                        "<p>Your reference is AP-5591.</p>"))
            EVENTS.append(f"WIZARD_STEP {stage}")
            return self._send(_page(f"Step {stage}", _wizard(int(stage))))
        return self._send(_page("Not found", "<p>Nothing here.</p>"), 404)


class _Server(socketserver.ThreadingTCPServer):
    # Chrome opens several connections at once; a single-threaded server
    # serialises them until a request trips Playwright's navigation timeout,
    # which then looks like a site problem rather than a harness one.
    allow_reuse_address = True
    daemon_threads = True


def serve(port: int = 8900):
    httpd = _Server(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def reset():
    EVENTS.clear()
    CARTS.clear()
