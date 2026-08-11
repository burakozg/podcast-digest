# Running on macOS (Apple Silicon)

**This is the development setup.** The deployment is the NAS —
see **[DEPLOY-NAS.md](DEPLOY-NAS.md)**. What follows is how to run the agent
natively on the Mac for development, and it is also the rollback target if the
NAS deployment has to be backed out.

`docker-compose.yml` still runs here unchanged (`docker compose config` parses;
`docker compose up couchdb-podcast` is a fine way to rehearse it).

    OpenRouter → Anthropic  ──┐   (model work, both tiers)
                              ├──►  podcast-agent  (uv run, port 8080)
    CouchDB (single container)┘

**Model work is cloud, here as on the NAS.** Ollama used to be the primary and
the reason the agent ran natively at all — Metal is unreachable from a Linux
container. Until there is a machine worth running a local model on, both tiers
go to OpenRouter with Anthropic behind them, and `config.local.yaml` no longer
rewrites the `llm` section: a local run that routed somewhere else would stop
rehearsing the deployment. `.env` therefore needs
`PODAGENT_OPENROUTER_API_KEY` and `PODAGENT_ANTHROPIC_API_KEY`.

The Ollama sections below are kept for when that machine exists; the models were
verified working on an M2 / 16 GB.

## One-time setup

```bash
# 1. CouchDB (the only container you need)
docker run -d --name couchdb-podcast-local -p 5984:5984 \
  -e COUCHDB_USER=podagent -e COUCHDB_PASSWORD=localdevpassword \
  -v podcast-couchdb-local:/opt/couchdb/data couchdb:3

# 2. Secrets. Running natively, the app needs the CouchDB password under its own
#    prefix — docker-compose does that mapping for you, nothing does it here.
cp .env.example .env
printf 'PODAGENT_COUCHDB_PASSWORD=localdevpassword\n' >> .env
# set PODAGENT_ADMIN_API_KEY in .env:  openssl rand -hex 32

# 3. Dependencies (uv fetches Python 3.12 itself)
uv sync --all-extras
```

`config.local.yaml` is **generated**, not hand-written:

```bash
uv run python scripts/make-local-config.py
```

Re-run that after any change to `config.yaml`. It was originally a hand-made copy,
which went stale the moment `config.yaml` grew a `backfill_mode` section — the
local run then fully summarised a daily news show that was meant to be indexed
only. Deriving it removes that failure mode.

It points at `127.0.0.1` for CouchDB, writes digests to `./data/digests`, and
uses the smaller ASR model. The `llm` section is copied through unchanged, so
local runs exercise the same provider chain the NAS will.

## Run it

```bash
export PODAGENT_CONFIG_FILE=config.local.yaml
uv run podcast-agent
```

**Bind it deliberately.** `api.host` defaults to `0.0.0.0`, which is right in a
container — compose decides what is published — and wrong here, where it means
every interface on the machine, with the admin key crossing the LAN in clear
text on every request. Running natively, pin it:

```bash
export PODAGENT_API__HOST=127.0.0.1   # this machine only; reach it over SSH or
                                      # Tailscale from elsewhere
```

Use the LAN IP instead (`192.168.1.x`) only if you want to open the console from
a phone or another machine, and only on a network you trust. Repeated wrong keys
from one address are throttled, which slows guessing but is not a substitute for
not listening where you do not have to.

**Stop the Mac sleeping under it.** On battery, macOS takes Maintenance Sleep
after a few idle minutes, and the agent goes with it: a model call that was
waiting for a reply sits there while the clock runs, then fails its timeout on
wake. It looks exactly like a hung backend — two 900-second timeouts here, one
against Ollama and one against OpenRouter, turned out to be a 940-second and a
956-second sleep, matching to the second. Scheduled firings are missed the same
way, and open connections are torn down, which is where the
`Server disconnected without sending a response` and `Unclosed client session`
lines come from.

```bash
caffeinate -is uv run podcast-agent    # -i no idle sleep, -s no system sleep
```

Nothing is lost when it does happen — every stage is resumable and work stays
queued — but a timed-out cloud call may still be billed, and the log fills with
failures that have nothing to do with the pipeline.

Then, in another shell:

```bash
export KEY=$(grep '^PODAGENT_ADMIN_API_KEY=' .env | cut -d= -f2)
export H="http://127.0.0.1:8080"

curl -sS $H/healthz | jq
curl -sS -X POST -H "X-API-Key: $KEY" "$H/api/v1/runs/ingest?wait=true" | jq .result
curl -sS -X POST -H "X-API-Key: $KEY" "$H/api/v1/runs/pipeline?wait=true" | jq .result
curl -sS -X POST -H "X-API-Key: $KEY" "$H/api/v1/runs/digest" | jq .result
open data/digests/*/*.md
```

Start small — the first pipeline run is the slow one:

```bash
PODAGENT_PIPELINE__INITIAL_LOOKBACK_DAYS=4 \
PODAGENT_PIPELINE__MAX_TRIAGE_PER_RUN=4 \
PODAGENT_PIPELINE__MAX_TRANSCRIPTS_PER_RUN=1 \
PODAGENT_PIPELINE__MAX_SUMMARIES_PER_RUN=1 \
PODAGENT_CONFIG_FILE=config.local.yaml uv run podcast-agent
```

## Model notes (learned the hard way on this machine)

**These apply to the local Ollama setup, which is currently not in use.** They
are kept because they are what the next machine's configuration should start
from, not because they describe how the agent runs today. The Ollama endpoints
they refer to are the commented-out blocks in `config.yaml` → `llm.tiers`.

**Qwen3.5 is a reasoning model.** Left alone it spends its entire token budget
in `thinking` and returns empty content — a trial call ran past 300 s with no
result. `extra_params: {think: false}` takes the same call to ~3 s. This is
already set in `config.local.yaml`, and `extra_params` passes any provider flag
straight through litellm.

**`qwen3.5-9b-longctx` (num_ctx 16384) is not usable on 16 GB.** 6.6 GB of
weights plus a 16k KV cache thrashes; a plain generation never returned. The
config uses `qwen3.5:9b` with `num_ctx: 8192`, and `pipeline.max_input_tokens`
is 5000 to stay inside that window with room for prompt and output. If you move
to a machine with more memory, raise both together — they must stay consistent.

**Both tiers use the same model deliberately.** Two different 9B models means
Ollama swaps ~6.6 GB in and out between Tier-0 and Tier-1 calls.

**Measured on M2/16 GB:** Tier-0 ~9-20 s per episode; Tier-1 map-reduce ~25-80 s
per call, so a long transcript (5 chunks + reduce) is roughly 4 minutes. ASR
with `small.en` is comfortably faster than real time. A full week's backlog is
best left to run in the background rather than watched.

## Stopping / cleaning up

```bash
pkill -f podcast-agent
docker stop couchdb-podcast-local          # docker rm -v ... to discard state
```

## Moving to the NAS later

Done — see **[DEPLOY-NAS.md](DEPLOY-NAS.md)** for the full runbook: what is
specific to that machine, how the image gets there without `scp`, how the
database is carried across, and how to verify it landed.

Use `config.yaml` there, not the generated `config.local.yaml`. With model work
in the cloud there is nothing to pull on the NAS and no GPU question to answer.
