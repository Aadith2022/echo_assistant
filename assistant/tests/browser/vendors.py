"""Three independent vendor sites, for the comparison archetype.

Separate origins on purpose: nothing links to anything else, which is what
makes comparison structurally different from following links around one site,
and what forced the split between plan-supplied and page-supplied navigation.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
import time

FARES = {
    8911: ("SkyLark Air", 412),
    8912: ("BlueRoute", 289),        # the correct answer
    8913: ("Meridian Airways", 355),
}

HITS: list[tuple[float, int]] = []


def _make(port: int):
    name, fare = FARES[port]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            HITS.append((time.monotonic(), port))
            # Real sites take a moment; without it the cost of visiting
            # several is invisible and fan-out cannot be measured.
            time.sleep(1.2)
            if self.path.startswith("/search"):
                body = f"""<!doctype html><html><head><title>{name} results</title></head>
                <body><h1>{name}</h1><nav><a href="/">Home</a></nav>
                <h2>JFK to RDU, 20 September 2026</h2>
                <table><tr><th>Flight</th><th>Depart</th><th>Fare</th></tr>
                <tr><td>{name[:2].upper()}101</td><td>07:15</td><td>USD {fare}</td></tr>
                <tr><td>{name[:2].upper()}204</td><td>14:40</td><td>USD {fare + 60}</td></tr>
                </table></body></html>""".encode()
            else:
                body = f"""<!doctype html><html><head><title>{name}</title></head>
                <body><h1>{name}</h1><nav><a href="/">Home</a></nav>
                <p>Low fares across the US.</p>
                <p><a href="/search?from=JFK&to=RDU&date=2026-09-20">
                Search JFK to RDU on 20 September 2026</a></p></body></html>""".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = S(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def serve_all():
    return [_make(p) for p in FARES]


def reset():
    HITS.clear()
