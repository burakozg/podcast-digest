#!/usr/bin/env bash
# Restore a podcast-agent CouchDB backup produced by backup.sh (roadmap F5).
#
# Restores into a database that must NOT already exist, so a restore can never
# silently merge into live data — create a scratch database, verify it, then
# swap. That is deliberate: the common restore mistake is discovering afterwards
# that you merged two half-states.
#
# Usage:
#   COUCHDB_PASSWORD=... ./scripts/restore.sh backups/podcast_agent-....json.gz [target-db]

set -euo pipefail

BACKUP="${1:?usage: restore.sh <backup.json.gz> [target-db]}"
COUCHDB_URL="${COUCHDB_URL:-http://127.0.0.1:5984}"
COUCHDB_USER="${COUCHDB_USER:-podagent}"
TARGET_DB="${2:-${COUCHDB_DB:-podcast_agent}_restored}"

if [[ -z "${COUCHDB_PASSWORD:-}" ]]; then
  echo "FATAL: COUCHDB_PASSWORD is not set" >&2
  exit 2
fi
[[ -f "$BACKUP" ]] || { echo "FATAL: no such backup: $BACKUP" >&2; exit 2; }

AUTH="${COUCHDB_USER}:${COUCHDB_PASSWORD}"

# Branch on the actual status code, not on curl's exit status.
#
# This check is the only thing standing between a restore and a live database,
# and `curl -fsS` alone cannot carry it: -f fails on ANY non-2xx, so a 401 was
# indistinguishable from a 404 and the script read "auth rejected me" as
# "database does not exist" and carried on to create it. Reaching the wrong
# server — an SSH tunnel fighting a local container for port 5984, say — is
# exactly when that matters, because it is exactly when the credentials are
# wrong for whatever answered.
# curl already writes "000" to stdout when it cannot connect, so `|| echo 000`
# would concatenate a second one and produce a status nothing matches.
STATUS=$(curl -sS -o /dev/null -w '%{http_code}' -u "$AUTH" "${COUCHDB_URL}/${TARGET_DB}" 2>/dev/null) || true
STATUS=${STATUS:-000}
case "$STATUS" in
  200)
    echo "FATAL: database '${TARGET_DB}' already exists at ${COUCHDB_URL}." >&2
    echo "Restore into a new name, verify it, then swap. Refusing to merge." >&2
    exit 3
    ;;
  404) : ;;  # the only state we may proceed from
  401 | 403)
    echo "FATAL: ${COUCHDB_URL} rejected the credentials (HTTP ${STATUS})." >&2
    echo "Check COUCHDB_USER/COUCHDB_PASSWORD are the ones for THAT server —" >&2
    echo "when restoring over an SSH tunnel they are the remote host's, not this one's." >&2
    exit 4
    ;;
  000)
    echo "FATAL: could not reach ${COUCHDB_URL}." >&2
    echo "Is the server up, and is the tunnel (if any) actually bound?" >&2
    exit 5
    ;;
  *)
    echo "FATAL: unexpected HTTP ${STATUS} from ${COUCHDB_URL}/${TARGET_DB}." >&2
    exit 6
    ;;
esac

echo "Creating ${COUCHDB_URL}/${TARGET_DB}"
curl -fsS -X PUT -u "$AUTH" "${COUCHDB_URL}/${TARGET_DB}" > /dev/null

# Strip _rev so documents are inserted fresh, and keep inlined attachments.
# _design/* documents (the Mango indexes) are skipped on purpose: the agent
# recreates them at startup via ensure_setup(). Expect the restored database to
# report fewer documents than the backup by exactly that number.
DOCS="$(mktemp)"
trap 'rm -f "$DOCS"' EXIT
gzip -dc "$BACKUP" | python3 -c '
import json, sys
rows = json.load(sys.stdin).get("rows", [])
docs = []
for row in rows:
    doc = row.get("doc")
    if not doc or doc.get("_id", "").startswith("_design/"):
        continue
    doc.pop("_rev", None)
    # Inlined attachments carry "data"; drop the stub marker so CouchDB stores them.
    for attachment in (doc.get("_attachments") or {}).values():
        attachment.pop("stub", None)
        attachment.pop("revpos", None)
        attachment.pop("digest", None)
        attachment.pop("length", None)
        attachment.pop("encoded_length", None)
        attachment.pop("encoding", None)
    docs.append(doc)
json.dump({"docs": docs, "new_edits": True}, sys.stdout)
print(f"prepared {len(docs)} documents", file=sys.stderr)
' > "$DOCS"

echo "Loading into ${TARGET_DB}"
RESULT="$(curl -fsS --max-time 600 -X POST \
  -u "$AUTH" \
  -H 'Content-Type: application/json' \
  --data-binary "@${DOCS}" \
  "${COUCHDB_URL}/${TARGET_DB}/_bulk_docs")"

# Note: this whole program is inside shell single quotes, so it must not contain
# a single quote of its own — hence %-formatting rather than f-strings here.
echo "$RESULT" | python3 -c '
import json, sys
results = json.load(sys.stdin)
errors = [r for r in results if r.get("error")]
print("restored %d documents, %d errors" % (len(results) - len(errors), len(errors)))
for row in errors[:10]:
    print("  %s: %s %s" % (row.get("id"), row.get("error"), row.get("reason")),
          file=sys.stderr)
raise SystemExit(1 if errors else 0)
'

echo
echo "Restored into '${TARGET_DB}'. Verify, then point couchdb.db at it"
echo "(or delete the old database and rename)."
