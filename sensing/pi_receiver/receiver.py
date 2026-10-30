#!/usr/bin/env python3
"""
PulseNet BLE receiver.

Each Pi runs one copy of this script. It connects to the one sensing board
assigned to this Pi (a Seeed or an ESP32-C6), subscribes to the board's data
characteristic, and saves every packet it receives to a local file.

If the board gets unplugged, restarts, or goes out of range, the script
notices and keeps trying to reconnect on its own. The raw data never leaves
this Pi; that's the whole point of federated learning.

Normally this runs as a systemd service (see pulsenet-receiver.service).
To run it by hand for debugging, stop the service first, then:
    sudo -u pulsenet-rx /opt/pulsenet/venv/bin/python /opt/pulsenet/receiver.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import signal
import sys
import time
import tomllib  # built into Python 3.11+, so no extra dependency for reading the config
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

# bleak is the Bluetooth library. On Linux it talks to BlueZ (the OS Bluetooth stack).
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.exc import BleakError
from bleak.uuids import normalize_uuid_str

LOG = logging.getLogger("pulsenet.receiver")

# Where the config lives by default. Can be overridden with the PULSENET_CONFIG
# environment variable or the --config flag.
DEFAULT_CONFIG = os.environ.get("PULSENET_CONFIG", "/etc/pulsenet/node.toml")

# Patterns used to sanity-check the config so a typo fails loudly at startup
# instead of causing weird behavior later.
MAC_RE = re.compile(r"^[0-9A-F]{2}(?::[0-9A-F]{2}){5}$")      # e.g. F0:1E:BA:62:59:5D
LABEL_RE = re.compile(r"^PN-[SE]\d{2}$")                       # e.g. PN-S01 or PN-E01
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """All the settings for this node, loaded once at startup. frozen=True means read-only."""
    label: str              # board name, e.g. PN-S01
    address: str            # board's BLE address
    char_uuid: str          # which characteristic to subscribe to for data
    data_dir: Path          # where packet files get written
    scan_timeout_s: float   # how long to look for the board each attempt
    connect_timeout_s: float
    backoff_initial_s: float  # first wait between failed attempts
    backoff_max_s: float      # longest we'll ever wait between attempts
    max_packet_bytes: int     # anything bigger than this gets dropped


def _bounded(section: dict, key: str, default: float, lo: float, hi: float) -> float:
    """Read a number from the config and make sure it's in a sensible range."""
    value = section.get(key, default)
    # bool is technically an int in Python, so reject it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"receiver.{key} must be a number")
    if not lo <= value <= hi:
        raise ValueError(f"receiver.{key}={value} is outside the allowed range [{lo}, {hi}]")
    return float(value)


def _default_data_dir() -> str:
    """Figure out where to save data if the config doesn't say.

    When running under systemd, StateDirectory= creates /var/lib/pulsenet for us
    and passes its path in $STATE_DIRECTORY. Otherwise fall back to the standard
    Linux location for app data.
    """
    state = os.environ.get("STATE_DIRECTORY", "")
    # systemd can pass several paths separated by ":"; we only need the first one
    return state.split(":")[0] if state else "/var/lib/pulsenet"


def load_config(path: Path) -> Config:
    """Read the TOML config file, validate everything, and return a Config."""
    with path.open("rb") as fh:  # tomllib requires the file opened in binary mode
        raw = tomllib.load(fh)

    board = raw.get("board")
    receiver = raw.get("receiver", {})  # [receiver] is optional; defaults are used if missing
    if not isinstance(board, dict) or not isinstance(receiver, dict):
        raise ValueError("config must contain a [board] table (and optionally [receiver])")

    # These three are required. A KeyError means one is missing.
    try:
        label = str(board["label"]).strip()
        address = str(board["address"]).strip().upper()
        # normalize_uuid_str lowercases the UUID so comparisons are consistent
        char_uuid = normalize_uuid_str(str(board["char_uuid"]).strip())
    except KeyError as exc:
        raise ValueError(f"missing required key in [board]: {exc}") from None

    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"board.label {label!r} must look like PN-S01 or PN-E01")
    if not MAC_RE.fullmatch(address):
        raise ValueError(f"board.address {address!r} is not a valid Bluetooth address")
    if not UUID_RE.fullmatch(char_uuid):
        raise ValueError(f"board.char_uuid {char_uuid!r} is not a valid 128-bit UUID")

    # resolve() turns it into a clean absolute path
    data_dir = Path(str(receiver.get("data_dir", _default_data_dir()))).expanduser().resolve()
    # backoff_max has to be at least backoff_initial, so read initial first
    backoff_initial = _bounded(receiver, "backoff_initial_s", 2.0, 0.5, 60.0)

    return Config(
        label=label,
        address=address,
        char_uuid=char_uuid,
        data_dir=data_dir,
        scan_timeout_s=_bounded(receiver, "scan_timeout_s", 10.0, 1.0, 120.0),
        connect_timeout_s=_bounded(receiver, "connect_timeout_s", 20.0, 5.0, 120.0),
        backoff_initial_s=backoff_initial,
        backoff_max_s=_bounded(receiver, "backoff_max_s", 60.0, backoff_initial, 600.0),
        max_packet_bytes=int(_bounded(receiver, "max_packet_bytes", 512, 1, 4096)),
    )


# --------------------------------------------------------------------------- #
# Saving packets to disk (raw data never leaves this Pi)
# --------------------------------------------------------------------------- #
class DataWriter:
    """Writes one line of JSON per packet to <data_dir>/<label>_<date>.jsonl.

    A new file is started each day (UTC) so files don't grow forever and are
    easy to find by date.
    """

    def __init__(self, data_dir: Path, label: str) -> None:
        self.data_dir = data_dir
        self.label = label
        self.count = 0                  # total packets written since startup
        self._fh: TextIO | None = None  # currently open file
        self._day: str | None = None    # which day's file is open
        # 0o750 = owner can read/write, group can read, everyone else gets nothing
        self.data_dir.mkdir(mode=0o750, parents=True, exist_ok=True)

    def _open_for(self, day: str) -> None:
        """Close the current file (if any) and open the file for the given day."""
        self.close()
        path = self.data_dir / f"{self.label}_{day}.jsonl"
        # O_APPEND: always add to the end, never overwrite existing data.
        # O_NOFOLLOW: refuse to open the file if someone replaced it with a symlink
        #             (stops us from being tricked into writing somewhere else).
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        fd = os.open(path, flags, 0o640)  # 0o640 = owner read/write, group read only
        # buffering=1 means each line is flushed to disk as soon as it's written
        self._fh = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        self._day = day
        LOG.info("writing packets to %s", path)

    def write(self, data: bytes) -> None:
        """Save one packet."""
        now_ns = time.time_ns()
        day = time.strftime("%Y-%m-%d", time.gmtime(now_ns / 1e9))
        # Switch files at midnight UTC, or open the first file on the first packet
        if day != self._day or self._fh is None:
            self._open_for(day)
        self.count += 1
        record = {
            "t_ns": now_ns,          # when the Pi received it (Unix time in nanoseconds, UTC)
            "board": self.label,
            "rx_seq": self.count,    # our own counter; the firmware's sequence number comes in Phase 1
            "len": len(data),
            "hex": data.hex(),       # store raw bytes as hex so any binary data is safe in JSON
        }
        assert self._fh is not None
        # separators=(",", ":") removes extra spaces to keep lines compact
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        """Close the current file. Safe to call even if nothing is open."""
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------- #
# Bluetooth connection handling
# --------------------------------------------------------------------------- #
class Receiver:
    """Keeps this Pi connected to its board and passes every packet to DataWriter."""

    def __init__(self, cfg: Config, stop: asyncio.Event) -> None:
        self.cfg = cfg
        self.stop = stop  # gets set when we're told to shut down (Ctrl+C or systemctl stop)
        self.writer = DataWriter(cfg.data_dir, cfg.label)

    async def run(self) -> None:
        """Main loop: connect, receive data, and reconnect forever until told to stop."""
        backoff = self.cfg.backoff_initial_s
        while not self.stop.is_set():
            had_session = False
            try:
                had_session = await self._session()
            except (BleakError, TimeoutError, OSError) as exc:
                # Expected Bluetooth problems (board vanished mid-connect, timeout, etc.)
                LOG.warning("BLE error: %s", exc or type(exc).__name__)
            except Exception:
                # Anything unexpected: log the full error but keep running.
                # A receiver that crashes would stop collecting data.
                LOG.exception("unexpected error in BLE session")

            if self.stop.is_set():
                break

            # If we were connected and then lost it, retry quickly.
            # If we couldn't even find the board, wait longer each time (exponential backoff)
            # so we're not constantly scanning when the board is off.
            if had_session:
                backoff = self.cfg.backoff_initial_s
            # Add +/-20% randomness so multiple Pis don't all retry at the exact same moment
            delay = backoff * random.uniform(0.8, 1.2)
            LOG.info("retrying in %.1f s", delay)
            await self._sleep_unless_stopped(delay)
            if not had_session:
                backoff = min(backoff * 2, self.cfg.backoff_max_s)

    async def _session(self) -> bool:
        """One full attempt: find the board, connect, subscribe, and stream until disconnect.

        Returns True if we actually connected (even if it dropped later),
        False if the board couldn't be found at all.
        """
        cfg = self.cfg

        # Step 1: look for our specific board. On Linux, bleak has to see the
        # board advertising before it can connect to it.
        LOG.info("scanning for %s (%s), up to %.0f s", cfg.label, cfg.address, cfg.scan_timeout_s)
        device = await BleakScanner.find_device_by_address(cfg.address, timeout=cfg.scan_timeout_s)
        if device is None:
            LOG.info("%s not found", cfg.label)
            return False

        # This event gets set by bleak when the connection drops
        disconnected = asyncio.Event()

        def on_disconnect(_client: BleakClient) -> None:
            disconnected.set()

        start_count = self.writer.count

        # Step 2: connect. Using "async with" guarantees we disconnect cleanly
        # when this block ends, no matter how it ends.
        async with BleakClient(
            device,
            disconnected_callback=on_disconnect,
            timeout=cfg.connect_timeout_s,
        ) as client:
            LOG.info("connected to %s", cfg.label)

            # Step 3: subscribe. From here on, bleak calls _on_notify for every packet.
            await client.start_notify(cfg.char_uuid, self._on_notify)
            LOG.info("subscribed to %s", cfg.char_uuid)

            # Step 4: sit here until either the board disconnects or we're told to stop.
            waiters = [
                asyncio.create_task(disconnected.wait()),
                asyncio.create_task(self.stop.wait()),
            ]
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                # Cancel whichever wait didn't finish so it doesn't hang around
                for task in waiters:
                    task.cancel()

            # If we're shutting down on purpose (board still connected), unsubscribe politely.
            # If the board already dropped, there's nothing to unsubscribe from.
            if client.is_connected:
                with contextlib.suppress(BleakError, TimeoutError):
                    await client.stop_notify(cfg.char_uuid)

        LOG.info(
            "disconnected from %s (%d packets this session)",
            cfg.label,
            self.writer.count - start_count,
        )
        return True

    def _on_notify(self, _char: BleakGATTCharacteristic, data: bytearray) -> None:
        """Called by bleak every time the board sends a packet."""
        # Drop anything unreasonably large instead of writing it
        if len(data) > self.cfg.max_packet_bytes:
            LOG.warning("dropping oversized packet (%d bytes)", len(data))
            return
        try:
            self.writer.write(bytes(data))
        except OSError:
            # e.g. disk full. Log it, but don't kill the connection over one packet.
            LOG.exception("failed to write packet to disk")
            return
        # Only print each packet in --verbose mode (it would flood the logs otherwise)
        if LOG.isEnabledFor(logging.DEBUG):
            # repr() escapes any control characters, so a bad packet can't mess up
            # the terminal by injecting escape codes.
            text = bytes(data).decode("utf-8", errors="replace")
            LOG.debug("rx %3d B  %s  %r", len(data), data.hex(" "), text)

    async def _sleep_unless_stopped(self, seconds: float) -> None:
        """Wait for the given time, but wake up right away if we're told to stop."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
async def _amain(cfg: Config) -> None:
    """Set up shutdown handling, then run the receiver until stopped."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # SIGINT = Ctrl+C, SIGTERM = what systemd sends on "systemctl stop".
    # Either one sets the stop event so we can shut down cleanly.
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    receiver = Receiver(cfg, stop)
    LOG.info("PulseNet receiver starting for %s (%s)", cfg.label, cfg.address)
    try:
        await receiver.run()
    finally:
        # Always close the data file, even if something went wrong
        receiver.writer.close()
        LOG.info("PulseNet receiver stopped (%d packets total)", receiver.writer.count)


def main() -> int:
    parser = argparse.ArgumentParser(description="PulseNet BLE receiver")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"path to TOML config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--verbose", action="store_true", help="log every received packet")
    args = parser.parse_args()

    # Log to stderr; when running as a service, systemd's journal picks this up
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("bleak").setLevel(logging.WARNING)  # bleak's own logs are very chatty
    # Any file we create gets no permissions for "other" users
    os.umask(0o027)

    try:
        cfg = load_config(Path(args.config).expanduser())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        LOG.error("config error: %s", exc)
        # Exit code 2 = bad config. The systemd unit is set to NOT restart on this,
        # since restarting won't fix a broken config.
        return 2

    asyncio.run(_amain(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())