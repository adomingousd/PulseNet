"""Read live Pi memory from configured Node Exporter endpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen


def read_memory(target):
    names = {"node_memory_MemTotal_bytes", "node_memory_MemAvailable_bytes"}
    values = {}
    with urlopen(f"http://{target}/metrics", timeout=2) as response:
        for line in response:
            fields = line.split()
            if len(fields) >= 2 and fields[0].decode() in names:
                values[fields[0].decode()] = float(fields[1])
                if len(values) == len(names):
                    break
    total = values["node_memory_MemTotal_bytes"]
    available = values["node_memory_MemAvailable_bytes"]
    if not 0 <= available <= total or total <= 0:
        raise ValueError("Invalid memory totals")
    return 100 * (1 - available / total)


def read_node(node):
    name, target = node
    try:
        memory = f"{read_memory(target):.1f}%"
        status = "Telemetry reachable"
    except (OSError, ValueError, KeyError) as error:
        memory = "Unavailable"
        status = f"Telemetry unavailable ({error})"
    return "".join(f"<td>{escape(value)}</td>" for value in (name, target, memory, status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--node", action="append", required=True, metavar="NAME=HOST:PORT")
    args = parser.parse_args()
    nodes = [option.partition("=")[::2] for option in args.node]
    if any(not name or not target for name, target in nodes):
        parser.error("--node requires NAME=HOST:PORT")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = "".join(f"<tr>{row}</tr>" for row in pool.map(read_node, nodes))
            updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
            page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>PulseNet | Live Pi memory</title>
<style>
body {{font: 1rem system-ui; margin: 3rem auto; max-width: 60rem; padding: 0 1rem}}
table {{border-collapse: collapse; width: 100%}}
td, th {{padding: .7rem; border-bottom: 1px solid #ccc; text-align: left}}
</style></head><body>
<h1>PulseNet live Pi memory</h1><p>Last poll (Jetson clock) {updated}. Refreshes every 5 seconds.</p>
<table><thead><tr><th>Node</th><th>Exporter</th><th>Memory used</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>""".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

    with HTTPServer((args.bind, args.port), Handler) as server:
        print(f"PulseNet live memory at http://{args.bind}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
