#!/usr/bin/env bash
# Restore a backup into the NAS's CouchDB, from this Mac.
#
# `scripts/restore.sh` cannot do this job against this NAS, for two independent
# reasons, and neither has a workaround on the NAS side:
#
#   1. The NAS has no python3, so the script cannot run there.
#   2. This QNAP's sshd refuses port forwarding —
#        channel 2: open failed: administratively prohibited
#      — so it cannot be run from here against a tunnelled 127.0.0.1 either.
#
# The database is also deliberately published on the NAS's loopback only, so
# there is no LAN address to aim at, and opening one for a restore would undo
# the single most important property of the network layout.
#
# So the work is split across the two machines along the line of what each one
# actually has: this Mac has python3 and does the JSON transform; the NAS has
# curl and does the HTTP. The prepared body is streamed between them over plain
# SSH stdin, which is the one channel this NAS does allow.
#
#   [Mac] gunzip → python3 transform → gzip ──ssh──> [NAS] gunzip → curl → CouchDB
#
# The password is never sent: the remote side reads it from the .env already
# sitting in the app directory.
#
# Usage:
#   ./qnap/restore-nas.sh backups/podcast_agent-<stamp>.json.gz [target-db]

set -euo pipefail

BACKUP="${1:?usage: restore-nas.sh <backup.json.gz> [target-db]}"
TARGET_DB="${2:-podcast_agent}"

# Real per-deployment values live in .deploy.env, git-ignored (see
# deploy.env.example) — auto-sourced here so nothing needs exporting by hand.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "${REPO_ROOT}/.deploy.env" ] && . "${REPO_ROOT}/.deploy.env"

NAS_HOST="${NAS_HOST:-deploy@nas.local}"
NAS_SSH_PORT="${NAS_SSH_PORT:-22}"
NAS_APP_DIR="${NAS_APP_DIR:-/share/Container/podcast-digest}"
NAS_COUCHDB="${NAS_COUCHDB:-http://127.0.0.1:5984}"

[[ -f "$BACKUP" ]] || { echo "FATAL: no such backup: $BACKUP" >&2; exit 2; }

ssh_nas() { ssh -p "$NAS_SSH_PORT" "$NAS_HOST" "$@"; }

# Everything the NAS side needs to find its own credentials, in one place.
REMOTE_PREAMBLE="PW=\$(grep -E '^COUCHDB_PASSWORD=' '$NAS_APP_DIR/.env' | cut -d= -f2-);
  [ -n \"\$PW\" ] || { echo 'FATAL: COUCHDB_PASSWORD not found in $NAS_APP_DIR/.env' >&2; exit 2; }"

# --- pre-flight -------------------------------------------------------------
# Branch on the status code, never on curl's exit status: -f fails on any
# non-2xx, which makes a 401 look exactly like a 404 and turns "auth rejected
# me" into "the database does not exist, go ahead and create it".
echo "== checking ${NAS_COUCHDB}/${TARGET_DB} on the NAS =="
STATUS=$(ssh_nas "$REMOTE_PREAMBLE
  curl -sS -o /dev/null -w '%{http_code}' -u \"podagent:\$PW\" '$NAS_COUCHDB/$TARGET_DB' 2>/dev/null || true")
STATUS=${STATUS:-000}

case "$STATUS" in
  404) echo "   does not exist yet — good" ;;
  200)
    echo "FATAL: '${TARGET_DB}' already exists on the NAS." >&2
    echo "Restore into a new name, verify, then swap. Refusing to merge." >&2
    exit 3
    ;;
  401 | 403)
    echo "FATAL: the NAS's CouchDB rejected the credentials in $NAS_APP_DIR/.env (HTTP $STATUS)." >&2
    exit 4
    ;;
  000)
    echo "FATAL: could not reach $NAS_COUCHDB from the NAS." >&2
    echo "Is couchdb-podcast up?  docker compose ... up -d couchdb-podcast" >&2
    exit 5
    ;;
  *)
    echo "FATAL: unexpected HTTP $STATUS from the NAS's CouchDB." >&2
    exit 6
    ;;
esac

echo "== creating ${TARGET_DB} =="
ssh_nas "$REMOTE_PREAMBLE
  curl -fsS -X PUT -u \"podagent:\$PW\" '$NAS_COUCHDB/$TARGET_DB' > /dev/null"

# --- transform here, POST there ---------------------------------------------
# Same document handling as scripts/restore.sh: _rev stripped so documents are
# inserted fresh, attachment stubs cleaned so the inlined bodies are stored
# rather than treated as references, and _design/* skipped because the agent
# recreates its Mango indexes at startup.
echo "== streaming $(du -h "$BACKUP" | cut -f1) into _bulk_docs =="
RESULT=$(
  gzip -dc "$BACKUP" \
    | python3 -c '
import json, sys
rows = json.load(sys.stdin).get("rows", [])
docs = []
for row in rows:
    doc = row.get("doc")
    if not doc or doc.get("_id", "").startswith("_design/"):
        continue
    doc.pop("_rev", None)
    for attachment in (doc.get("_attachments") or {}).values():
        for key in ("stub", "revpos", "digest", "length", "encoded_length", "encoding"):
            attachment.pop(key, None)
    docs.append(doc)
print(f"prepared {len(docs)} documents", file=sys.stderr)
json.dump({"docs": docs, "new_edits": True}, sys.stdout)
' \
    | gzip -1 \
    | ssh_nas "$REMOTE_PREAMBLE
        gunzip -c | curl -fsS --max-time 900 -X POST \
          -u \"podagent:\$PW\" \
          -H 'Content-Type: application/json' \
          --data-binary @- \
          '$NAS_COUCHDB/$TARGET_DB/_bulk_docs'"
)

# --- report -----------------------------------------------------------------
printf '%s' "$RESULT" | python3 -c '
import json, sys
results = json.load(sys.stdin)
errors = [r for r in results if r.get("error")]
print("restored %d documents, %d errors" % (len(results) - len(errors), len(errors)))
for row in errors[:10]:
    print("  %s: %s %s" % (row.get("id"), row.get("error"), row.get("reason")), file=sys.stderr)
raise SystemExit(1 if errors else 0)
'

echo
echo "Restored into '${TARGET_DB}' on the NAS."
echo "Next: check the stored console override before starting the agent (DEPLOY-NAS.md step 5)."
