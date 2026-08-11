# Deploying to the NAS

The permanent home. Both services run as containers — CouchDB and the agent — with all
state under one directory the NAS's own backup jobs can see.

This replaces the native `uv run` setup in [RUNNING-ON-MAC.md](RUNNING-ON-MAC.md), which
existed because Ollama needed Metal and a Linux container could not reach it. Model work
is now cloud (OpenRouter primary, Anthropic fallback), so nothing about the workload
requires the Mac.

```
                     ┌─────────────────────────────────────────┐
 LAN ── 10.0.0.2 ──│ podcast-agent   (console :8080)         │
                     │      │ internal bridge                  │
                     │ couchdb-podcast (no LAN address at all) │
                     └─────────────────────────────────────────┘
        /share/Container/podcast-digest/{couchdb,work,digests,backups}
```

---

## What is specific to this NAS

Each of these cost something to find out. They are why `docker-compose.nas.yml` and
`qnap/*.sh` exist rather than just running the base compose file.

| | |
|---|---|
| **Architecture** | NAS is `x86_64`, the Mac is arm64. The image **must** be cross-built for `linux/amd64`. Wrong arch = `exec format error`, a silent non-start rather than a partial failure. |
| **No `scp`/`sftp`** | This QNAP's sshd exposes no SFTP subsystem. `scp` fails with `subsystem request failed on channel 0`, exit 255 — and **silently** when a script wraps it in `2>/dev/null \|\| true`. Everything moves over plain SSH pipes. |
| **No `python3`** | `scripts/backup.sh` and `scripts/restore.sh` both need it. The restore is therefore driven **from the Mac**; the nightly backup uses `qnap/backup-nas.sh`, which verifies with curl/gzip/sed instead. |
| **`/bin/bash` is a symlink to `sh`** | Anything run on the NAS host must be POSIX. No `pipefail`, no `[[ ]]`. |
| **`docker` is not on `PATH`** | Only wired in for interactive logins. Scripts use the full path `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker`. |
| **qnet has no embedded DNS** | Service names do not resolve on it, and container-to-container traffic over it has been observed to fail outright. The agent keeps the `internal` bridge for reaching CouchDB, and uses qnet only for its LAN address. |
| **qnet does NOT isolate the host** | Unlike standard macvlan, the NAS host reaches `10.0.0.2:8080` fine (verified: HTTP 200). Both that and the published `127.0.0.1:8080` work from the NAS, so either is usable for host-side checks. |
| **Memory** | ~7.8 GB total, ~3 GB genuinely free, shared with Home Assistant, traefik, two other digest services and the wyoming voice stack. Hence `small.en` ASR and a 2 GB agent limit. |

---

## One-time setup

### 1. Directories

```bash
export NAS=deploy@nas.local P=44
export D=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
export APP=/share/Container/podcast-digest

ssh -p $P $NAS "mkdir -p $APP/couchdb/data $APP/couchdb/config \
                         $APP/work/audio $APP/digests $APP/backups"
```

They must be created by the SSH user, not by Docker. Docker creates missing bind-mount
paths as `root:root`, and the agent — uid 10001 in the image, read-only root filesystem,
every capability dropped — cannot then write to them. `docker-compose.nas.yml` runs the
agent as `1002:100` to match. CouchDB needs no such help: its entrypoint starts as root
and chowns its own data directory.

### 2. Files

`scp` does not work here. Use SSH pipes:

```bash
for f in docker-compose.yml docker-compose.nas.yml config.yaml; do
  ssh -p $P $NAS "cat > $APP/$f" < $f
done
ssh -p $P $NAS "mkdir -p $APP/scripts && cat > $APP/scripts/backup-nas.sh" < qnap/backup-nas.sh
ssh -p $P $NAS "chmod +x $APP/scripts/backup-nas.sh"
```

**Verify they landed.** A silent transfer failure is exactly the shape of bug this NAS
produces:

```bash
ssh -p $P $NAS "cd $APP && wc -c docker-compose.yml docker-compose.nas.yml config.yaml"
```

Compare against `wc -c` locally. `config.yaml` should be byte-identical to the repo's —
it is the declared baseline, and a NAS-only edit is how it silently goes stale.

### 3. `.env`

Write it directly on the NAS (never commit it):

```bash
ssh -p $P $NAS "cat > $APP/.env" <<'EOF'
PODAGENT_ADMIN_API_KEY=<openssl rand -hex 32>
COUCHDB_USER=podagent
COUCHDB_PASSWORD=<openssl rand -hex 24>
PODAGENT_COUCHDB_PASSWORD=<same as COUCHDB_PASSWORD>

# Both required: OpenRouter is primary for every tier, Anthropic the fallback.
# The agent REFUSES TO BOOT if either is empty while that provider is in an
# active chain — see the note below before deploying.
PODAGENT_OPENROUTER_API_KEY=
PODAGENT_ANTHROPIC_API_KEY=

# Digests land inside the app directory, so one backup job captures everything.
DIGEST_DIR=./digests

# Loopback only. The console's LAN address is the container's own (NAS_LAN_IP
# below); this publish exists so the NAS host itself can reach the API, which
# it cannot reach directly, and it keeps working if the qnet address is ever
# removed.
AGENT_BIND=127.0.0.1:8080

# docker-compose.nas.yml's static qnet IP for the agent container -- Compose
# reads this from .env at parse time (${NAS_LAN_IP:-...} substitution), same
# file, no separate config. Pick a free address on your qnet subnet.
NAS_LAN_IP=10.0.0.2

# Memory here is tight and shared. large-v3-turbo wants ~2GB resident plus a
# 1.6GB first-run download; small.en is a fraction of that. Set as an env
# override rather than editing config.yaml, so the deployed config.yaml stays
# byte-identical to the repo's.
PODAGENT_ASR__MODEL=small.en
EOF

ssh -p $P $NAS "grep -c '^[A-Z]' $APP/.env"     # expect 10 assignments
```

> **You need an Anthropic API key before this will start.** `config.yaml` puts an
> `anthropic` endpoint in both tiers' fallback chains, and an active endpoint whose key is
> unset is a hard startup failure, not a warning:
>
> ```
> FATAL: invalid configuration
>   (root): Value error, an anthropic endpoint is active but
>   PODAGENT_ANTHROPIC_API_KEY is unset
> ```
>
> That is the intended behaviour — failing closed beats failing over to an
> unauthenticated provider. But it means "get an Anthropic key" is a prerequisite of this
> deployment, not a nice-to-have.
>
> If you would rather not have one, delete the `fallbacks:` block from both tiers in
> `config.yaml` before copying it up. Each tier keeps its OpenRouter primary and boots on
> that key alone — which is exactly what the generated `config.local.yaml` does for
> development.

---

## Build and ship the image

Copy `deploy.env.example` to `.deploy.env` (git-ignored) and fill in
`NAS_HOST`/`NAS_SSH_PORT` for your NAS -- `qnap/deploy.sh` and
`qnap/restore-nas.sh` auto-source it, so nothing needs exporting by hand or
editing in the tracked script. Then:

```bash
./qnap/deploy.sh
```

Cross-builds `linux/amd64`, streams it into `docker load` over one SSH pipe (nothing
touches disk as an intermediate on either end), verifies the image landed and reports its
architecture, then brings the stack up with both compose files.

`./qnap/deploy.sh --no-up` builds and ships without restarting — use it during the
migration below, where CouchDB has to come up *before* the agent.

---

## Migrating existing state from the Mac

Skip this section for a fresh install; `initial_lookback_days` will do the right thing.

### 1. Record what "correct" looks like, before you change anything

```bash
KEY=$(grep -E '^PODAGENT_ADMIN_API_KEY=' .env | cut -d= -f2-)
curl -fsS -H "X-API-Key: $KEY" http://127.0.0.1:8080/api/v1/status       > backups/before-migration-status.json
curl -fsS -H "X-API-Key: $KEY" http://127.0.0.1:8080/api/v1/search/status > backups/before-migration-search.json
jq '.episode_counts, ([.episode_counts[]] | add)' backups/before-migration-status.json
```

These counts are what tell you afterwards whether the restore was complete, and they are
unrecoverable once the Mac is off.

**Baseline captured 2026-08-04** — the numbers the NAS must reproduce:

| | |
|---|---|
| Total episodes | **885** (also the search index count — the two agree, which is itself a useful check) |
| `PUBLISHED` | 756 |
| `READY_FOR_DIGEST` | 45 |
| `SCORED_LOW` | 55 |
| `DROPPED` | 26 |
| `DIGEST_DIRECT` / `ERROR` | 2 / 1 |
| Feeds | 15 |
| `awaiting_digest` queue | 24 |
| Backfill | 14 shows complete, 780 episodes ingested, 155 archive files |

Both JSON files are saved under `backups/` — keep them until the NAS is verified.

### 2. Stop the Mac

```bash
pkill -f podcast-agent
docker start couchdb-podcast-local     # still needed, to take the backup
```

The CouchDB job lease (`joblock.py`) guards two processes against **one** database.
During a migration there are two databases, so it does not protect you — both hosts would
poll, summarise and bill independently.

### 3. Back up, and carry state across

Read the password from `.env` rather than typing it — a literal `...` gets you a
`curl: (22) … error: 401` and nothing else:

```bash
export COUCHDB_PASSWORD=$(grep -E '^PODAGENT_COUCHDB_PASSWORD=' .env | cut -d= -f2-)

./scripts/backup.sh ./backups
BK=$(ls -1t backups/podcast_agent-*.json.gz | head -1)

ssh -p $P $NAS "cat > $APP/backups/$(basename $BK)" < "$BK"
ssh -p $P $NAS "wc -c $APP/backups/$(basename $BK)"; wc -c "$BK"   # must match

COPYFILE_DISABLE=1 tar --no-xattrs -czf - -C data digests \
  | ssh -p $P $NAS "tar xzf - -C $APP"
```

**Both flags are load-bearing on macOS.** A plain `tar czf` here produces a working
transfer with two cosmetic-looking problems, only one of which is actually cosmetic:

- `--no-xattrs` stops bsdtar writing macOS extended attributes as pax headers. The NAS's
  tar does not know keywords like `com.apple.provenance` and prints
  `Ignoring unknown extended header keyword` **twice per file** — hundreds of lines of
  noise that bury any real error in the same stream.
- `COPYFILE_DISABLE=1` stops it emitting AppleDouble `._*` sidecar members. These are not
  noise: they arrive as real files. Without it this transfer landed 436 files where 209
  were sent — 227 of them `._`-prefixed junk sitting permanently in the digest tree.

Verify the count rather than trusting the absence of an error:

```bash
find data/digests -type f | wc -l
ssh -p $P $NAS "find $APP/digests -type f | wc -l"     # must match
```

If `._*` files did land, remove them with
`ssh -p $P $NAS "find $APP/digests -name '._*' -delete"`.

The digests move because CouchDB stores their paths **relative to `digest_dir`**;
`adopt_orphaned_digest_files` (`migrate.py:75`) reconciles them at first boot.

**Do not copy `data/work/search.db`.** It is a derived FTS5 cache, not a record — see
`search.py`, which is explicit that deleting it costs one rebuild. It is rebuilt below.

### 4. Restore, before the agent ever starts

`restore.sh` refuses to write into a database that already exists, and the agent creates
one at boot. So CouchDB comes up alone first:

```bash
ssh -p $P $NAS "cd $APP && $D compose -f docker-compose.yml -f docker-compose.nas.yml up -d couchdb-podcast"
```

Then restore from the Mac with `qnap/restore-nas.sh`:

```bash
./qnap/restore-nas.sh "$BK" podcast_agent
```

It prints a document count and an error count, and exits non-zero on any error.

**Why a dedicated script rather than `scripts/restore.sh`.** Two independent constraints
rule that one out here, and neither has a workaround:

- The NAS has **no python3**, so it cannot run there.
- This QNAP's sshd **refuses port forwarding**, so it cannot be run here against a
  tunnelled `127.0.0.1` either:

  ```
  channel 2: open failed: administratively prohibited: open failed
  curl: (56) Recv failure: Connection reset by peer
  ```

  `ssh -L` still binds the *local* port and looks like it worked — the failure only
  appears when something connects. Do not read a silent `ssh -L` as a working tunnel.

CouchDB is also published on the NAS's loopback only, so there is no LAN address to aim
at, and opening one for a restore would undo the main property of the network layout.

So the work splits along the line of what each machine actually has — the Mac has python3
and does the JSON transform, the NAS has curl and does the HTTP, with the prepared body
streamed between them over plain SSH stdin:

```
[Mac] gunzip → python3 transform → gzip ──ssh──> [NAS] gunzip → curl → CouchDB
```

The password is never transmitted: the remote side reads it from the `.env` already in the
app directory. The same pre-flight guard as `restore.sh` applies — it branches on the HTTP
status code, so a 401 aborts instead of being mistaken for "database does not exist".

Expect roughly **13 fewer documents than the backup reports**: `_design/*` documents are
skipped deliberately, and the agent recreates its Mango indexes at startup.

### 5. Check the stored console override

It lives in CouchDB, so it rides the restore and is deep-merged **over** `config.yaml` at
boot — and for `llm.tiers` it *replaces* rather than merges, so a stale override silently
beats the config you just shipped.

```bash
ssh -p $P $NAS "PW=\$(grep -E '^COUCHDB_PASSWORD=' $APP/.env | cut -d= -f2-); \
  curl -fsS -u \"podagent:\$PW\" http://127.0.0.1:5984/podcast_agent/control:settings" \
  | jq '.overrides | keys'
```

**Expected on this deployment: `["asr", "interest_profile"]` and nothing else.**

- `interest_profile` — **must** ride across. The console-tuned weights differ materially
  from `config.yaml` (`ot_ics` 3 vs 10, `cloud_security` present only here). Losing it
  silently changes what gets summarised and what reaches the digest.
- `asr` — pins `small.en`, which is what the NAS wants anyway.
- **`llm` must NOT be present.** One was cleared on 2026-08-04 (saved at
  `backups/console-override-2026-08-04.json`). It pinned tier-0 to Ollama with
  `allow_cloud_fallback: false`, which on the NAS means a chain of one endpoint at
  `127.0.0.1:11434` — the container's own loopback, where nothing listens. Triage would
  stop dead, and because deferral is the designed behaviour it would fail quietly.

If an `llm` key is back, clear it before starting the agent. Do it by clearing all
overrides and re-PUTting the others — `DELETE /api/v1/settings` is all-or-nothing, and
taking the whole document out is how you lose the interest profile without noticing.

### 6. Start the agent

```bash
ssh -p $P $NAS "cd $APP && $D compose -f docker-compose.yml -f docker-compose.nas.yml up -d"
ssh -p $P $NAS "cd $APP && $D compose logs -f podcast-agent"
```

---

## Verify

Run these from any machine on the **your LAN subnet LAN**, or from the NAS itself — qnet on
this NAS does not isolate the host the way standard macvlan would, so `10.0.0.2:8080`
answers there too (verified).

```bash
export KEY=<admin key> HOST=10.0.0.2:8080
```

> **Over a VPN this will appear to hang.** The TCP handshake completes and then nothing
> comes back. The container has its own identity on the LAN and uses the router as its
> gateway, so its replies only reach a VPN subnet if the router has a route back — and
> replies to the NAS's own address are not the same path. Test from the LAN, or run the
> checks over SSH on the NAS against `127.0.0.1:8080`.

| # | Check | Command |
|---|---|---|
| 0 | The LAN address actually attached | `ping -c1 10.0.0.2` |
| 1 | App is up, CouchDB reachable | `curl -fsS http://$HOST/healthz \| jq` |
| 2 | Data arrived intact | `curl -fsS -H "X-API-Key: $KEY" http://$HOST/api/v1/status \| jq` — compare with `/tmp/before.json` |
| 3 | `config.yaml` won, not an override | see below |
| 4 | Search rebuilt | `curl -fsS -X POST -H "X-API-Key: $KEY" http://$HOST/api/v1/search/rebuild` |
| 5 | Mounts are writable | `curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/digest?dry_run=true"` |
| 6 | State survives a restart | `docker compose restart`, then repeat 1–2 |

Step 0 matters because a qnet address that failed to attach looks exactly like an app that
did not start, and the two have nothing in common as fixes.

Step 3 — `active_chain` is what would actually be tried, after any stored override:

```bash
curl -fsS -H "X-API-Key: $KEY" "http://$HOST/api/v1/settings" \
  | jq '{chains: (.tiers|map_values(.active_chain)), overrides: .overrides, asr: .asr.model}'
```

Expect `openrouter/… → anthropic/…` per tier and `asr.model: "small.en"`.

Step 4 should report **885 episodes**, the same as the Mac's index. A materially lower
number means the restore dropped documents — investigate before doing anything else.
Allow a few minutes; it re-reads and decompresses every transcript attachment.

Step 2 compares against the table in *Record what "correct" looks like* above. `PUBLISHED`
(756) and the total (885) are the two that matter most.

Step 5 is the one that catches a bind-mount permission mistake: confirm files actually
appear under `$APP/digests`.

Also watch memory during the first ASR run — Home Assistant and the rest share this box:

```bash
ssh -p $P $NAS "free -m; $D stats --no-stream"
```

---

## Nightly backup

```bash
ssh -p $P $NAS "cat > $APP/.backup-env" <<'EOF'
COUCHDB_PASSWORD=<the password>
EOF
ssh -p $P $NAS "chmod 600 $APP/.backup-env"
```

Then add to the NAS crontab (QNAP: `crontab -e`, and it needs
`/etc/init.d/crond.sh restart` afterwards to survive):

```cron
30 3 * * * cd /share/Container/podcast-digest && . ./.backup-env && ./scripts/backup-nas.sh ./backups >> ./backups/backup.log 2>&1
```

`backup-nas.sh` refuses to publish a dump that fails any of four checks — gzip integrity,
`total_rows` present (catches auth failures), a complete JSON tail (catches a truncated
body), and a byte-size floor. A backup that looks fine but is not is worse than an obvious
failure, because you only find out when you need it.

The digests and the database now sit in one folder tree, so a QNAP Hybrid Backup Sync job
pointed at `/share/Container/podcast-digest` covers everything as well.

---

## Redeploying after a code change

```bash
./qnap/deploy.sh
```

Config-only changes need no rebuild — copy `config.yaml` up and restart:

```bash
ssh -p $P $NAS "cat > $APP/config.yaml" < config.yaml
ssh -p $P $NAS "cd $APP && $D compose -f docker-compose.yml -f docker-compose.nas.yml restart podcast-agent"
```

Settings changed from the console apply at the **next restart**, by design — swapping a
provider rebuilds the LLM router, and doing that underneath a summarisation in flight is a
bad trade. The console reports when a restart is pending.

> **If you register this as a Container Station *Application*** rather than driving it over
> SSH: its stored YAML must be **re-pasted**, not "Recreated", whenever a bind mount is
> added or changed. Recreate re-reads the stored definition, not the file on disk.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `exec format error` | Image built for the wrong architecture. `NAS_PLATFORM=linux/amd64`. |
| Container exits with `FATAL: invalid configuration` | Read the field paths it lists. `extra="forbid"` means a typo'd key is fatal by design. |
| `/api/v1/*` returns 503 | `PODAGENT_ADMIN_API_KEY` unset. It fails closed on purpose. |
| Startup refuses: "an anthropic endpoint is active but …" | A provider key is missing from `.env`. Both are required as shipped. |
| Console reachable from the NAS and LAN but not over a VPN | The container has its own LAN identity and uses the router as its gateway, so replies to a VPN subnet depend on the router having a route back. TCP connects, then no response. Use a LAN address, or reach it through the NAS. |
| Console unreachable from everywhere | `ping 10.0.0.2` first. If that fails the qnet attach failed; if it answers, check the container logs. |
| Permission denied writing digests/work | Bind-mount dirs owned by root because Docker created them. Remove, recreate as the SSH user, restart. |
| `stages_deferred` in a run summary | The tier's whole chain was unreachable — both providers. Work stayed queued; nothing is lost. |
| ASR killed mid-run | Memory. Check `free -m` against the other containers; `small.en` and `asr_concurrency: 1` are already the floor. |
| Backup cron silent, no file | `.backup-env` unreadable, or crond not restarted after `crontab -e`. Check `backups/backup.log`. |

---

## Rollback

The Mac is untouched until you delete it.

```bash
ssh -p $P $NAS "cd $APP && $D compose -f docker-compose.yml -f docker-compose.nas.yml down"
docker start couchdb-podcast-local
PODAGENT_CONFIG_FILE=config.local.yaml uv run podcast-agent
```

Keep the backup file and the Mac's CouchDB volume until the NAS has produced at least one
correct weekly digest.
