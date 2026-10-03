#!/usr/bin/env bash
# Keep the Kokoro speech server (:8880) running across reboots and logouts.
#
# Why this exists: RUNNING-ON-MAC.md starts Kokoro as a foreground
# `./start-gpu_mac.sh` in a terminal. Nothing restarts it, so the first reboot —
# or closing that window — takes speech down silently, and the only visible
# symptom is the agent's hourly narration job logging a connection error while
# the digests themselves look perfectly fine. That outage ran for days before
# anyone noticed. A LaunchAgent makes the server's uptime the machine's problem
# rather than something a person has to remember.
#
# Usage:
#   ./scripts/install-kokoro-service.sh            install (or update) and start
#   ./scripts/install-kokoro-service.sh --uninstall stop and remove
#   ./scripts/install-kokoro-service.sh --status    is it loaded, is it answering
#
# Environment (all have safe defaults; real values belong in .deploy.env):
#   KOKORO_DIR   the Kokoro-FastAPI clone      default ./Kokoro-FastAPI
#   KOKORO_PORT  port it listens on            default 8880
#   KOKORO_LOG   where its output goes         default $KOKORO_DIR/kokoro.log
#   BREW_PREFIX  Homebrew prefix, for espeak-ng  default /opt/homebrew
#
# A LaunchAgent, not a LaunchDaemon, deliberately: Metal needs a logged-in GUI
# session, and a daemon running before login gets no GPU at all. The tradeoff is
# that speech is down until someone logs in, which is the same condition the
# hourly retry already exists to cover.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .deploy.env ] && . ./.deploy.env

LABEL="com.homelab.kokoro-speech"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

KOKORO_DIR="${KOKORO_DIR:-$PWD/Kokoro-FastAPI}"
KOKORO_PORT="${KOKORO_PORT:-8880}"
KOKORO_LOG="${KOKORO_LOG:-${KOKORO_DIR}/kokoro.log}"
BREW_PREFIX="${BREW_PREFIX:-/opt/homebrew}"

probe() {
  curl -fsS --max-time 5 "http://127.0.0.1:${KOKORO_PORT}/v1/audio/voices" >/dev/null 2>&1
}

case "${1:-install}" in
--status)
  if launchctl list "$LABEL" >/dev/null 2>&1; then
    echo "✓ ${LABEL}: loaded"
    launchctl list "$LABEL" | grep -E '"(PID|LastExitStatus)"' || true
  else
    echo "✗ ${LABEL}: not loaded"
  fi
  # Loaded is not answering: the model download on a cold start takes minutes,
  # and a crash loop also reports as loaded.
  if probe; then
    echo "✓ speech: answering on :${KOKORO_PORT}"
  else
    echo "✗ speech: no answer on :${KOKORO_PORT} (starting, or failed — see ${KOKORO_LOG})"
  fi
  exit 0
  ;;
--uninstall)
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed ${LABEL}"
  exit 0
  ;;
install) ;;
*)
  echo "usage: $0 [install|--uninstall|--status]" >&2
  exit 2
  ;;
esac

# Fail loudly rather than installing an agent that can only crash-loop. launchd
# reports a missing program as a numeric exit status in `launchctl list` and
# nowhere else, which is a miserable thing to debug later.
if [ ! -x "${KOKORO_DIR}/start-gpu_mac.sh" ]; then
  echo "FATAL: no start-gpu_mac.sh in ${KOKORO_DIR}" >&2
  echo "       clone it first — see RUNNING-ON-MAC.md, or set KOKORO_DIR." >&2
  exit 1
fi
if [ ! -x "${BREW_PREFIX}/bin/espeak-ng" ]; then
  echo "FATAL: espeak-ng not found at ${BREW_PREFIX}/bin — run: brew install espeak-ng" >&2
  exit 1
fi

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "FATAL: uv is not on PATH; start-gpu_mac.sh needs it" >&2
  exit 1
fi

mkdir -p "$(dirname "$PLIST")" "$(dirname "$KOKORO_LOG")"

# `start-gpu_mac.sh` rather than uvicorn directly, even though it re-runs
# `uv pip install -e .` and the model download on every launch. Both are
# no-ops once satisfied, and duplicating its env setup here would drift
# silently the next time upstream changes it. It resolves PROJECT_ROOT from
# $(pwd), hence WorkingDirectory below.
cat >"$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${KOKORO_DIR}/start-gpu_mac.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${KOKORO_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$(dirname "$UV_BIN"):${BREW_PREFIX}/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTORCH_ENABLE_MPS_FALLBACK</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>${KOKORO_LOG}</string>
    <key>StandardErrorPath</key>
    <string>${KOKORO_LOG}</string>
</dict>
</plist>
PLIST_EOF

# bootout first so this is idempotent: bootstrapping an already-loaded label is
# an error, and re-running after editing anything above is the normal case.
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "installed ${PLIST}"
echo "waiting for :${KOKORO_PORT} to answer (a cold start downloads the model)…"

# Verified over the port rather than trusting "loaded": a crash-looping agent
# and a healthy one look identical to launchctl.
for _ in $(seq 1 60); do
  if probe; then
    echo "✓ speech: answering on :${KOKORO_PORT}"
    exit 0
  fi
  sleep 5
done

echo "✗ speech: still no answer after 5 minutes — check ${KOKORO_LOG}" >&2
exit 1
