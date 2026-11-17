"""Expose two Linux RAM metrics on a Pi's lab address."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Metrics(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        with open("/proc/meminfo", encoding="utf-8") as source:
            memory = dict(line.split(":", 1) for line in source)
        body = "".join(
            f"node_memory_{key}_bytes {int(memory[key].split()[0]) * 1024}\n"
            for key in ("MemTotal", "MemAvailable")
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="Pi's PulseNet Ethernet address")
    args = parser.parse_args()
    ThreadingHTTPServer((args.address, 9100), Metrics).serve_forever()
