#!/usr/bin/env bash
# Back up the podcast-agent CouchDB state database (roadmap F5).
#
# The digests themselves are Markdown in a synced folder and are already covered
# by that sync. What is NOT covered is this database: losing it means re-fetching,
# re-transcribing and re-summarising everything, which costs hours of NAS CPU.
#
# Attachments (the gzipped transcripts) are included, because they are what makes
# a later re-score cheap — without them a profile change means re-running ASR.
#
# Usage:
#   COUCHDB_PASSWORD=... ./scripts/backup.sh [output-dir]
#
# Environment:
#   COUCHDB_URL       default http://127.0.0.1:5984
#   COUCHDB_USER      default podagent
#   COUCHDB_PASSWORD  required
#   COUCHDB_DB        default podcast_agent
#   KEEP              number of backups to retain, default 14

set -euo pipefail

OUT_DIR="${1:-./backups}"
COUCHDB_URL="${COUCHDB_URL:-http://127.0.0.1:5984}"
COUCHDB_USER="${COUCHDB_USER:-podagent}"
COUCHDB_DB="${COUCHDB_DB:-podcast_agent}"
KEEP="${KEEP:-14}"

if [[ -z "${COUCHDB_PASSWORD:-}" ]]; then
  echo "FATAL: COUCHDB_PASSWORD is not set" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TMP="$OUT_DIR/.${COUCHDB_DB}-${STAMP}.json.gz.tmp"
FINAL="$OUT_DIR/${COUCHDB_DB}-${STAMP}.json.gz"

echo "Backing up ${COUCHDB_URL}/${COUCHDB_DB} -> ${FINAL}"

# include_docs gives full bodies; attachments=true inlines them base64 so the
# transcripts travel with the dump rather than being silently dropped.
curl -fsS --max-time 600 \
  -u "${COUCHDB_USER}:${COUCHDB_PASSWORD}" \
  -H 'Accept: application/json' \
  "${COUCHDB_URL}/${COUCHDB_DB}/_all_docs?include_docs=true&attachments=true" \
  | gzip -9 > "$TMP"

# Verify before publishing the file: a truncated backup that looks fine is worse
# than an obvious failure, because it is only discovered when it is needed.
DOC_COUNT="$(gzip -dc "$TMP" | python3 -c '
import json, sys
payload = json.load(sys.stdin)
rows = payload.get("rows")
if rows is None:
    print("no rows key in response", file=sys.stderr)
    raise SystemExit(1)
print(len(rows))
')"

mv "$TMP" "$FINAL"
echo "OK: ${DOC_COUNT} documents, $(du -h "$FINAL" | cut -f1)"

# Rotate, newest first.
if [[ "$KEEP" -gt 0 ]]; then
  # shellcheck disable=SC2012 — filenames are timestamped and shell-safe.
  ls -1t "$OUT_DIR/${COUCHDB_DB}-"*.json.gz 2>/dev/null \
    | tail -n +$((KEEP + 1)) \
    | while read -r old; do
        echo "Rotating out $(basename "$old")"
        rm -f "$old"
      done
fi
