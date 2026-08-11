#!/bin/sh
# Nightly CouchDB backup, for the QNAP host's cron.
#
# This exists because scripts/backup.sh cannot run here: it verifies the dump
# with `python3 -c`, and THIS NAS HAS NO python3. A backup script that dies
# halfway through leaves you with no backup and a cron mail nobody reads, so
# rather than depend on an interpreter that is not installed, this one verifies
# with the tools the host actually has (curl, gzip, sed).
#
# Two more things about this host shape the script:
#   * /bin/bash is a symlink to sh, so this is POSIX — no `set -o pipefail`,
#     no [[ ]], no arrays.
#   * curl lives in /sbin, which IS on the non-interactive PATH here.
#
# It reaches CouchDB on 127.0.0.1:5984, which docker-compose.nas.yml publishes
# on loopback only. The LAN cannot reach that port, and must not.
#
# Usage:
#   COUCHDB_PASSWORD=... ./scripts/backup-nas.sh [output-dir]
#
# Cron (30 3 * * *), keeping the password out of the crontab itself:
#   30 3 * * * cd /share/Container/podcast-digest && . ./.backup-env && ./scripts/backup-nas.sh ./backups >> ./backups/backup.log 2>&1
#
# scripts/backup.sh remains the portable version for hosts that do have python3
# — including the Mac, which is where the migration restore is driven from.

set -eu

OUT_DIR="${1:-./backups}"
COUCHDB_URL="${COUCHDB_URL:-http://127.0.0.1:5984}"
COUCHDB_USER="${COUCHDB_USER:-podagent}"
COUCHDB_DB="${COUCHDB_DB:-podcast_agent}"
KEEP="${KEEP:-14}"
#: Floor below which the dump is assumed broken rather than merely small. The
#: real corpus is tens of MB; anything under this is an error page or a stub.
MIN_BYTES="${MIN_BYTES:-10240}"

if [ -z "${COUCHDB_PASSWORD:-}" ]; then
  echo "FATAL: COUCHDB_PASSWORD is not set" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TMP="$OUT_DIR/.${COUCHDB_DB}-${STAMP}.json.gz.tmp"
FINAL="$OUT_DIR/${COUCHDB_DB}-${STAMP}.json.gz"

# Any exit before the final mv leaves no half-written file claiming to be a backup.
trap 'rm -f "$TMP"' EXIT

echo "Backing up ${COUCHDB_URL}/${COUCHDB_DB} -> ${FINAL}"

# attachments=true inlines the gzipped transcripts as base64 so they travel with
# the dump. Without them a later re-score means re-running ASR from audio.
curl -fsS --max-time 900 \
  -u "${COUCHDB_USER}:${COUCHDB_PASSWORD}" \
  -H 'Accept: application/json' \
  "${COUCHDB_URL}/${COUCHDB_DB}/_all_docs?include_docs=true&attachments=true" \
  | gzip -9 > "$TMP" || true

# --- verification -----------------------------------------------------------
# Without pipefail, curl's exit status is lost above, so the checks below are
# not belt-and-braces — they ARE the error detection. Each catches a different
# way this goes wrong, and a truncated backup that looks fine is worse than an
# obvious failure because it is only discovered when it is needed.

# 1. Is the gzip stream complete? Catches a connection dropped mid-transfer.
if ! gzip -t "$TMP" 2>/dev/null; then
  echo "FATAL: gzip stream is corrupt or truncated" >&2
  exit 1
fi

# `wc -c` rather than `stat`: stat's size flag is -c on GNU and -f on BSD, and
# picking one makes the script silently wrong on the other. wc is POSIX.
SIZE=$(wc -c < "$TMP" | tr -d ' ')

# 2. Is it actually an _all_docs body? Catches auth failures and error JSON.
HEAD=$(gzip -dc "$TMP" | head -c 200)
case "$HEAD" in
  *'"total_rows"'*) ;;
  *)
    echo "FATAL: response is not an _all_docs body (auth failure? wrong db?)" >&2
    echo "  got: $HEAD" >&2
    exit 1
    ;;
esac

# 3. Does the JSON actually finish? Catches a body cut short by a timeout,
#    which is the failure the gzip check alone can miss when the stream itself
#    closed cleanly.
TAIL=$(gzip -dc "$TMP" | tail -c 4)
case "$TAIL" in
  *']}') ;;
  *)
    echo "FATAL: JSON body is incomplete — backup truncated" >&2
    exit 1
    ;;
esac

# 4. Floor check, for the case where all of the above pass on a near-empty db.
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
  echo "FATAL: backup is ${SIZE} bytes, below the ${MIN_BYTES} floor" >&2
  exit 1
fi

# Whitespace-tolerant on purpose: CouchDB emits compact JSON, but a proxy or a
# future version that pretty-prints would otherwise turn this into a silent "?".
ROWS=$(echo "$HEAD" | sed -n 's/.*"total_rows":[[:space:]]*\([0-9][0-9]*\).*/\1/p')

mv "$TMP" "$FINAL"
trap - EXIT
echo "OK: ${ROWS:-?} documents, $(du -h "$FINAL" | cut -f1)"

# --- rotation ---------------------------------------------------------------
if [ "$KEEP" -gt 0 ]; then
  ls -1t "$OUT_DIR/${COUCHDB_DB}-"*.json.gz 2>/dev/null \
    | tail -n +$((KEEP + 1)) \
    | while read -r old; do
        echo "Rotating out $(basename "$old")"
        rm -f "$old"
      done
fi
