#!/usr/bin/env bash
# PulseNet BLE receiver installer (Raspberry Pi OS / Debian-based Linux with systemd + BlueZ).
#
# Sets up everything a node needs to run the receiver as a service:
# a service account, the code, a Python environment, the node's config, and the systemd unit.
#
# Safe to run again: re-running upgrades the code and dependencies.
# An existing /etc/pulsenet/node.toml is never overwritten.
#
# Usage (run from the directory containing this script):
#   sudo bash install.sh --label PN-S01 --address F0:1E:BA:62:59:5D \
#        --char-uuid 6e400003-b5a3-f393-e0a9-e50e24dcca9e
#
# Options:
#   --label LABEL      Board label, e.g. PN-S01        (required on first install)
#   --address ADDR     Board BLE address               (required on first install)
#   --char-uuid UUID   128-bit characteristic UUID     (required on first install)
#   --no-start         Install but do not enable/start the service

# Stop on any error (-e), on undefined variables (-u), and on failures inside pipes (pipefail)
set -euo pipefail

# Folder this script is in, so it works no matter where you run it from
SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SRC_DIR

# Where everything gets installed (standard Linux locations)
readonly PREFIX="/opt/pulsenet"                 # code + Python environment
readonly CONF_DIR="/etc/pulsenet"               # config
readonly CONF_FILE="${CONF_DIR}/node.toml"
readonly SVC_USER="pulsenet-rx"                 # account the service runs as
readonly UNIT="pulsenet-receiver.service"
readonly WHEELS_DIR="${SRC_DIR}/wheels"         # offline Python packages (if present)

# Values filled in from the command-line arguments
label="" address="" char_uuid="" start=1

# Small helpers for printing messages
die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }
# Make sure an option like --label actually has a value after it
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

# ---- Read command-line arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)     need_value "$@"; label="$2";     shift 2 ;;
        --address)   need_value "$@"; address="$2";   shift 2 ;;
        --char-uuid) need_value "$@"; char_uuid="$2"; shift 2 ;;
        --no-start)  start=0; shift ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;   # print the usage comment at the top
        *)           die "unknown option: $1 (see --help)" ;;
    esac
done

# ---- Preflight checks: make sure this machine can run the receiver ----
[[ ${EUID} -eq 0 ]] || die "run with sudo"
command -v python3   >/dev/null || die "python3 not found"
command -v systemctl >/dev/null || die "systemd is required"
# BlueZ creates the bluetooth group; if it's missing, Bluetooth isn't set up
getent group bluetooth >/dev/null || die "group 'bluetooth' not found (is BlueZ installed?)"

# Make sure all the files we need are here, and none have Windows line endings
# (Windows CRLF endings break bash scripts and systemd units)
for f in receiver.py requirements.txt "${UNIT}"; do
    [[ -f "${SRC_DIR}/${f}" ]] || die "missing ${f} next to install.sh"
    if grep -q $'\r' "${SRC_DIR}/${f}"; then
        die "${f} has Windows (CRLF) line endings; convert it to LF"
    fi
done

# ---- 1. Create the service account ----
# A system account with no home folder and no login shell. Nobody can log in as it;
# it exists only to run the receiver with minimal permissions.
if ! id -u "${SVC_USER}" >/dev/null 2>&1; then
    log "creating system user ${SVC_USER}"
    useradd --system --user-group --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin "${SVC_USER}"
fi

# ---- 2. Install the code ----
# Owned by root and not writable by the service, so the service can't modify its own code
log "installing code to ${PREFIX}"
install -d -m 0755 -o root -g root "${PREFIX}"
install -m 0644 -o root -g root "${SRC_DIR}/receiver.py"      "${PREFIX}/receiver.py"
install -m 0644 -o root -g root "${SRC_DIR}/requirements.txt" "${PREFIX}/requirements.txt"

# ---- 3. Python environment ----
# Create the virtual environment the first time only
if [[ ! -x "${PREFIX}/venv/bin/python" ]]; then
    log "creating virtual environment"
    python3 -m venv "${PREFIX}/venv" || die "venv creation failed (is python3-venv installed?)"
fi

pip_args=(--disable-pip-version-check --quiet)
if [[ -d "${WHEELS_DIR}" ]]; then
    # Offline install: only use the package files in wheels/, never the internet.
    # --no-deps because requirements.txt already lists every package we need.
    log "installing dependencies offline from ${WHEELS_DIR}"
    "${PREFIX}/venv/bin/pip" install "${pip_args[@]}" --no-index --no-deps \
        --find-links "${WHEELS_DIR}" -r "${PREFIX}/requirements.txt"
else
    # No wheels/ folder: download from PyPI (needs internet)
    log "no wheels/ directory found; installing dependencies from PyPI"
    "${PREFIX}/venv/bin/pip" install "${pip_args[@]}" -r "${PREFIX}/requirements.txt"
fi

# ---- 4. Node config ----
# Root owns the config folder; the service account can read it but not change it
install -d -m 0750 -o root -g "${SVC_USER}" "${CONF_DIR}"

if [[ -f "${CONF_FILE}" ]]; then
    # Never overwrite an existing config, so re-running the installer is safe
    log "keeping existing ${CONF_FILE}"
else
    [[ -n "${label}" && -n "${address}" && -n "${char_uuid}" ]] \
        || die "first install needs --label, --address and --char-uuid"

    # Normalize case: addresses uppercase, UUIDs lowercase
    address="${address^^}"
    char_uuid="${char_uuid,,}"

    # Validate the arguments strictly. Besides catching typos, this stops anyone
    # from sneaking extra lines into the config file through an argument.
    [[ "${label}"     =~ ^PN-[SE][0-9]{2}$ ]]              || die "label must look like PN-S01 or PN-E01"
    [[ "${address}"   =~ ^([0-9A-F]{2}:){5}[0-9A-F]{2}$ ]] || die "invalid BLE address: ${address}"
    [[ "${char_uuid}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
        || die "char-uuid must be a full 128-bit UUID"

    log "writing ${CONF_FILE}"
    # Write to a temp file first, then move it into place with the right
    # owner and permissions, so the real file is never half-written
    tmp="$(mktemp)"
    trap 'rm -f "${tmp}"' EXIT    # delete the temp file when the script exits
    # NOTE: everything between "<<EOF" and "EOF" is written to the file exactly as-is,
    # so it must stay unindented, and the closing EOF must start at column 0.
    cat >"${tmp}" <<EOF
# PulseNet receiver config for $(hostname)
[board]
label = "${label}"
address = "${address}"
char_uuid = "${char_uuid}"

[receiver]
scan_timeout_s = 10
connect_timeout_s = 20
backoff_initial_s = 2
backoff_max_s = 15
max_packet_bytes = 512
EOF
    # 0640 = root can read/write, service group can read, nobody else
    install -m 0640 -o root -g "${SVC_USER}" "${tmp}" "${CONF_FILE}"
fi

# ---- 5. Install and start the systemd service ----
log "installing ${UNIT}"
install -m 0644 -o root -g root "${SRC_DIR}/${UNIT}" "/etc/systemd/system/${UNIT}"
systemctl daemon-reload   # tell systemd to pick up the new or changed unit file

if [[ ${start} -eq 1 ]]; then
    systemctl enable --quiet "${UNIT}"   # start automatically at boot
    systemctl restart "${UNIT}"          # restart (not just start) so upgrades take effect now
    log "service running. Check with: systemctl status ${UNIT%.service}"
else
    log "installed but not started (--no-start)"
fi