# Podcast Digest Agent — Design & Architecture Document

**Version:** 1.0 (v1 scope)
**Status:** Ready for implementation
**Target implementer:** Claude Opus / Sonnet (agentic coding session)
**Owner:** (you)
**Date:** 2026-07-30

---

## 1. Purpose & Summary

A self-hosted background agent that monitors a curated list of cybersecurity podcasts (seeded from [awesome-cybersecurity-podcasts](https://github.com/TalEliyahu/awesome-cybersecurity-podcasts)), identifies episodes relevant to the owner's professional interests, produces **readable summaries** (the owner reads summaries instead of listening), and delivers a periodic digest as **Markdown files written to a synced folder** (Obsidian-compatible).

The owner does not want to rely solely on RSS descriptions to decide relevance, because descriptions are often thin or marketing-heavy. The system therefore uses a **tiered pipeline with confidence-based escalation**: cheap description-level triage first, and full transcript acquisition (published transcript or local Whisper ASR) when the description is insufficient to judge.

### v1 decisions (confirmed with owner)

| Decision | Choice |
|---|---|
| Digest delivery | Markdown files to a synced folder (Obsidian vault style) |
| Whisper/ASR escalation | **Included in v1** |
| State store | **CouchDB — new, dedicated instance** for this service (do NOT reuse the existing tasting-log CouchDB) |
| LLM access | Provider-agnostic via litellm: local (Ollama) and OpenRouter interchangeable per tier via config, with fallback chains |
| Runtime | Docker Compose stack on QNAP NAS (existing homelab pattern: macvlan network, Python services) |

---

## 2. Goals & Non-Goals

### Goals
1. Ingest RSS feeds for ~15–25 configured podcasts on a schedule; detect new episodes idempotently.
2. Classify every new episode with a cheap Tier-0 LLM pass producing: rough relevance, **description-informativeness confidence**, and routing decision.
3. Escalate low-confidence or high-priority episodes to transcript acquisition (published transcript → fallback to audio download + faster-whisper ASR).
4. Produce Tier-1 structured summaries + relevance scores against a configurable **interest profile**.
5. Generate a weekly digest (plus optional daily "hot items" note) as Markdown files with Obsidian-friendly frontmatter, written to a mounted output directory.
6. Support switching each tier's LLM between local (Ollama) and OpenRouter **by config only**, with automatic fallback, and record per-call cost/latency telemetry.
7. Be operable: health endpoint, manual-trigger endpoints, structured logs, safe restarts, no duplicate processing.

### Non-Goals (v1)
- No interactive web UI (no read/star/filter frontend). The Markdown output + a few JSON admin endpoints are the interface.
- No user accounts / multi-tenant support. Single user, LAN-only.
- No YouTube-only shows (skip shows without an audio RSS feed in v1; log them as unsupported).
- No embedding-based semantic search or vector DB. Prompt-based scoring only in v1.
- No automatic interest-profile learning from feedback. Profile is a hand-edited config file.

---

## 3. High-Level Architecture

```
                          ┌────────────────────────────────────────────┐
                          │  podcast-agent container (Python 3.12)     │
                          │                                            │
  RSS feeds  ────────────▶│  Ingestion (feedparser)                    │
  (internet)              │        │ new episodes                      │
                          │        ▼                                   │
                          │  Tier-0 Triage (litellm+instructor)  ──────┼──▶ Ollama container
                          │        │ route: DROP / DIGEST / ESCALATE   │      (local models)
                          │        ▼                                   │        │ fallback
                          │  Transcript Acquisition                    │        ▼
                          │   ├─ published transcript fetch            │    OpenRouter API
                          │   └─ audio download → faster-whisper ASR ──┼──▶ whisper container
                          │        │                                   │    (or in-process)
                          │        ▼                                   │
                          │  Tier-1 Summarize + Score                  │
                          │        │                                   │
                          │        ▼                                   │
                          │  Digest Generator (Jinja2 → Markdown)      │
                          │        │                                   │
                          │  FastAPI admin/health API (LAN only)       │
                          └────────┼───────────────────────────────────┘
                                   │
                 ┌─────────────────┼──────────────────┐
                 ▼                 ▼                  ▼
          CouchDB (new,      /data/digests      structured logs
          dedicated          (bind mount →      (stdout, JSON)
          instance)          synced folder)
```

### Containers (docker-compose)

| Service | Image / base | Purpose | Network |
|---|---|---|---|
| `podcast-agent` | Custom (python:3.12-slim) | FastAPI + APScheduler + pipeline | macvlan (static IP, LAN-only) |
| `couchdb-podcast` | `couchdb:3` | Dedicated state store | Internal Docker network only — **not** exposed on macvlan |
| `ollama` | `ollama/ollama` | Local LLM serving (optional — may point to an existing instance instead) | Internal Docker network |
| `whisper` | Custom (faster-whisper) **or** run in-process in podcast-agent | ASR for escalated episodes | Internal Docker network |

**Implementation note on Whisper:** prefer running faster-whisper **in-process** inside `podcast-agent` (library, not service) for v1 simplicity, executed in a worker thread/process pool with a concurrency limit of 1. Split into a separate container only if the image size or CPU contention becomes a problem. Make the ASR backend a small interface so this can change without touching pipeline code.

---

## 4. Data Flow & Pipeline Stages

### Stage 1 — Ingestion
- APScheduler cron job (default: every 6 hours, configurable).
- For each configured podcast: fetch RSS with `feedparser` + `httpx` (explicit timeout, retry w/ backoff, conditional GET using stored ETag/Last-Modified).
- For each entry: compute a stable episode ID = `sha256(podcast_slug + rss_guid)` (fall back to enclosure URL if GUID missing).
- If episode ID already exists in CouchDB → skip. Otherwise create an `episode` document with status `NEW` and the raw metadata (title, description/summary, published date, duration if present, enclosure URL, episode link, transcript URL if the feed exposes `<podcast:transcript>` per the Podcasting 2.0 namespace — check for it, several shows support it).
- Ingestion must be **idempotent and resumable**: crash mid-run → next run picks up where it left off based on document statuses.

### Stage 2 — Tier-0 Triage (every NEW episode)
- Input: title + cleaned description (strip HTML, truncate to ~2,000 chars) + podcast metadata (name, category, owner's per-show priority).
- One LLM call (small/cheap model) returning a **strict Pydantic-validated** structure:

```python
class Tier0Result(BaseModel):
    relevance_guess: int          # 0–10 vs interest profile
    confidence: int               # 0–10: is the description informative enough to judge?
    matched_interests: list[str]  # keys from the interest profile
    reasoning: str                # 1–2 sentences, for audit/log only
    route: Literal["DROP", "DIGEST_DIRECT", "ESCALATE"]
```

- Routing rules (applied in code, NOT trusted from the model — the model proposes, code disposes):
  - `confidence >= T_conf_high` AND `relevance_guess < T_rel_low` → **DROP** (status `DROPPED`, keep the doc for audit).
  - `confidence >= T_conf_high` AND `relevance_guess >= T_rel_high` → **ESCALATE** (we still want a real summary — the owner reads summaries, so relevant episodes always get Tier-1 treatment).
  - `confidence < T_conf_high` → **ESCALATE** (description too thin to judge — the core requirement).
  - Anything in the grey zone (`T_rel_low <= relevance_guess < T_rel_high` at high confidence) → **DIGEST_DIRECT**: include in digest as a one-liner "maybe interesting" item without a full summary, so nothing relevant is silently lost.
  - All thresholds in config; defaults: `T_conf_high = 7`, `T_rel_low = 4`, `T_rel_high = 7`.
- Per-show override: `always_escalate: true` for priority shows (full summary regardless of Tier-0).

### Stage 3 — Transcript Acquisition (ESCALATE only)
Ordered strategy, stop at first success:
1. **Feed-provided transcript** (`<podcast:transcript>` tag): fetch; support `text/plain`, `text/vtt`, `application/srt`, `application/json` (podcast namespace JSON). Normalize to plain text.
2. **Show-notes page scrape** only for shows explicitly configured with a `transcript_selector` (CSS selector) — do NOT attempt generic scraping.
3. **ASR fallback**: download enclosure audio (size cap, default 300 MB; stream to disk, never to memory) → faster-whisper (`large-v3-turbo` default, configurable), language hint `en`. Store transcript text.
- Concurrency limit: max 1 ASR job at a time; max 2 concurrent audio downloads. Queue the rest (episodes sit in status `AWAITING_TRANSCRIPT`).
- Failure handling: after `max_retries` (default 3, exponential backoff across scheduler runs) mark `TRANSCRIPT_FAILED` and **fall back to Tier-1 on description-only** with a flag `summary_basis: "description_only"` so the digest can label it honestly.
- Delete downloaded audio after successful transcription (configurable `keep_audio: false`). Transcripts are kept in CouchDB (as an attachment or gzipped field — see §6).

### Stage 4 — Tier-1 Summarize + Score
- Input: transcript (or description if fallback), title, show, interest profile.
- Long transcripts: if > `max_input_tokens` (config, default 24k tokens estimated), map-reduce: chunk → per-chunk bullet extraction → final synthesis call. Keep it simple: fixed-size chunking on paragraph boundaries.
- Output (Pydantic-validated, instructor with `max_retries=2`):

```python
class Tier1Result(BaseModel):
    relevance_score: int              # 0–10, final
    matched_interests: list[str]
    why_it_matters: str               # 1–2 sentences, personal to the profile
    summary_md: str                   # 150–400 word structured summary (Markdown)
    key_takeaways: list[str]          # 3–7 bullets
    entities: list[str]               # tools, CVEs, orgs, frameworks mentioned
    listen_anyway: bool               # true if audio adds value beyond the summary
    summary_basis: Literal["transcript", "published_transcript", "description_only"]
```

- Episodes with `relevance_score >= digest_threshold` (default 5) → status `READY_FOR_DIGEST`; below → `SCORED_LOW` (kept, auditable, appears in a collapsed "everything else" section of the digest — see §5).

### Stage 5 — Digest Generation
- Weekly job (default: Friday 06:00 Europe/Stockholm; also exposed as manual trigger endpoint).
- Collect all `READY_FOR_DIGEST` + `DIGEST_DIRECT` items since last digest → render via Jinja2 → write Markdown to `/data/digests/`.
- Mark included episodes `PUBLISHED` with a reference to the digest ID (so re-running never duplicates).

---

## 5. Digest Output Format (Obsidian-friendly)

One file per weekly digest: `/data/digests/YYYY/podcast-digest-YYYY-Www.md`
Optional per-episode notes (config flag `episode_notes: true`, default **false** in v1): `/data/digests/episodes/<podcast-slug>/<date>-<episode-slug>.md`, linked from the digest.

Digest file structure:

```markdown
---
type: podcast-digest
week: 2026-W31
generated: 2026-07-31T06:00:00+02:00
episodes_scanned: 34
episodes_summarized: 9
tags: [podcast-digest, cybersecurity]
---

# Podcast Digest — Week 31, 2026

## Top picks
### 🎙️ <Show> — <Episode title>  `9/10`
*Published 2026-07-28 · 62 min · basis: transcript · [episode link]*
**Why it matters:** <why_it_matters>
<summary_md>
**Key takeaways:**
- ...
**Interests matched:** OT/ICS · AI agent security

## Also relevant
(score 5–7, same structure, shorter)

## Maybe interesting (not summarized)
(DIGEST_DIRECT one-liners with links — description-based only, labeled as such)

## Everything else scanned
(collapsed table: show, title, score, route — full audit trail, one line each)
```

Rules for the implementer:
- All output MUST be valid Markdown; escape/strip anything from feeds that would break rendering.
- Filenames: ASCII slugs only, no spaces.
- Writes must be **atomic**: write to `*.tmp` in same directory, then `os.replace()`. The folder is watched by a sync client; partial files must never be visible.
- Never overwrite an existing digest file; if the file exists (manual re-run), suffix `-r2`, `-r3`.

---

## 6. Data Model (CouchDB)

Dedicated CouchDB instance, database name `podcast_agent`. Single-node, no clustering. Use `httpx` directly or `aiocouch` — keep the client thin; do not build an ORM.

### Document types (discriminated by `type` field)

**`podcast`** (one per show; seeded from config at startup — config is source of truth, doc stores runtime state):
```json
{
  "_id": "podcast:risky-business",
  "type": "podcast",
  "slug": "risky-business",
  "feed_url": "https://risky.biz/feeds/risky-business/",
  "etag": "...", "last_modified": "...",
  "last_polled_at": "...", "last_error": null, "consecutive_failures": 0
}
```

**`episode`**:
```json
{
  "_id": "episode:<sha256>",
  "type": "episode",
  "podcast_slug": "risky-business",
  "guid": "...", "title": "...", "link": "...",
  "description_raw": "...", "published_at": "...",
  "enclosure_url": "...", "duration_s": 3720,
  "status": "NEW | TRIAGED | AWAITING_TRANSCRIPT | TRANSCRIBED | TRANSCRIPT_FAILED | SUMMARIZED | READY_FOR_DIGEST | DIGEST_DIRECT | SCORED_LOW | DROPPED | PUBLISHED | ERROR",
  "tier0": { ...Tier0Result, "model": "...", "latency_ms": 812, "cost_usd": 0.0004 },
  "tier1": { ...Tier1Result, "model": "...", "latency_ms": ..., "cost_usd": ... },
  "transcript_source": "feed | scrape | asr | none",
  "digest_id": null,
  "attempts": {"transcript": 1, "tier0": 1, "tier1": 1},
  "created_at": "...", "updated_at": "..."
}
```
- Transcript text: store as CouchDB **attachment** (`transcript.txt.gz`, gzip) on the episode doc — keeps the JSON doc small and Mango-indexable.

**`digest`**:
```json
{
  "_id": "digest:2026-W31",
  "type": "digest",
  "period": {"from": "...", "to": "..."},
  "file_path": "2026/podcast-digest-2026-W31.md",
  "episode_ids": ["episode:..."],
  "stats": {"scanned": 34, "summarized": 9, "asr_runs": 4, "total_cost_usd": 0.31},
  "generated_at": "..."
}
```

**`llm_call`** (telemetry, one per LLM invocation — enables local-vs-OpenRouter economics over time):
```json
{
  "_id": "llmcall:<uuid>",
  "type": "llm_call",
  "tier": "tier0 | tier1_map | tier1_reduce",
  "provider": "ollama | openrouter", "model": "...",
  "episode_id": "...", "input_tokens": 0, "output_tokens": 0,
  "latency_ms": 0, "cost_usd": 0.0, "fallback_used": false,
  "validation_retries": 0, "ts": "..."
}
```

### Indexes (Mango)
- `(type, status)` — pipeline work queues.
- `(type, podcast_slug, published_at)` — digest assembly and dedupe.
- `(type, ts)` on `llm_call` — telemetry queries.

State transitions must be enforced in one module (`state.py`) with an explicit allowed-transition map; any illegal transition raises. Use CouchDB's MVCC (`_rev`) conflict on update as the concurrency guard — on 409, re-read and re-evaluate rather than force-overwrite.

---

## 7. LLM Abstraction Layer

### Requirements
1. Every tier's model is selected purely by config: provider (`ollama` / `openrouter` / `anthropic`), model name, base URL, params.
2. Automatic fallback chain per tier (e.g., local first → OpenRouter on failure/timeout).
3. All calls return Pydantic-validated structures with automatic retry-on-validation-failure.
4. All calls emit an `llm_call` telemetry document.

### Implementation
- **litellm** `Router` with model groups per tier:
  - Group `tier0`: e.g. `ollama/qwen3-8b` → fallback `openrouter/meta-llama/llama-3.2-3b-instruct`.
  - Group `tier1`: e.g. `ollama/qwen3.6-27b` → fallback `openrouter/qwen/qwen3.6-27b`.
- **instructor** on top of litellm for structured extraction (`response_model=`, `max_retries=2`).
- Wrap in a single module `llm/client.py` exposing exactly:
  ```python
  async def complete_structured(tier: str, system: str, user: str, response_model: type[T]) -> tuple[T, CallMeta]
  ```
  Nothing outside `llm/` may import litellm/instructor directly.
- Timeouts: tier0 60 s, tier1 300 s (local 27B on long context is slow). Fallback triggers on timeout, connection error, 5xx, or 2× validation failure.
- Token/cost accounting: use litellm's usage callbacks; for Ollama, cost = 0 but still record tokens and latency.
- Prompts live in `prompts/` as versioned files (`tier0_v1.md`, `tier1_v1.md`); the version string is logged with every call. The interest profile is injected into prompts from config — never hardcoded.

### Interest profile (config, hand-edited YAML)
```yaml
interest_profile:
  - key: ot_ics
    label: "OT/ICS security"
    description: "Industrial control systems, Purdue model, DCS/PLC/SCADA, OT incident response, OT/IT convergence"
    weight: 10
  - key: ai_agent_security
    label: "AI & agent security governance"
    description: "LLM/agent security, agentic risk, AI governance frameworks, prompt injection, model supply chain"
    weight: 10
  - key: microsoft_stack
    label: "Microsoft security ecosystem"
    description: "Entra, Purview, Defender, Sentinel, Fabric security"
    weight: 8
  - key: tprm
    label: "Third-party risk"
    description: "TPRM, supply chain risk, vendor assessment"
    weight: 7
  - key: leadership_policy
    label: "Security leadership, EU policy & regulation"
    description: "CISO topics, NIS2, CRA, DORA, EU cyber policy, consulting industry"
    weight: 6
```

---

## 8. Configuration

- **pydantic-settings**, layered: `config.yaml` (mounted, non-secret) + environment variables (secrets) + sensible defaults. Env prefix `PODAGENT_`.
- `config.yaml` sections: `podcasts` (list: slug, feed_url, priority, always_escalate, transcript_selector?), `interest_profile`, `scheduler` (cron expressions, timezone `Europe/Stockholm`), `pipeline` (thresholds, retries, caps), `llm` (tier groups, fallbacks, timeouts), `asr` (model, compute type, concurrency, size cap, keep_audio), `output` (digest dir, episode_notes flag), `couchdb` (url, db name), `api` (bind host/port, api key env ref), `logging` (level, format).
- Secrets (**env only, never in YAML, never logged**): `PODAGENT_OPENROUTER_API_KEY`, `PODAGENT_COUCHDB_PASSWORD`, `PODAGENT_ADMIN_API_KEY`.
- Config is validated at startup; invalid config = crash loudly with a clear message, do not start half-configured.
- Hot reload NOT required (restart container to apply config).

---

## 9. API Surface (FastAPI, LAN-only)

All under `/api/v1`, all except `/healthz` require header `X-API-Key: <PODAGENT_ADMIN_API_KEY>` (constant-time compare).

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Liveness: process up, CouchDB reachable, scheduler running. No auth. |
| `GET /api/v1/status` | Counts by episode status, last run times, queue depths, last errors |
| `POST /api/v1/runs/ingest` | Trigger ingestion now |
| `POST /api/v1/runs/pipeline` | Process pending work (triage/transcripts/summaries) now |
| `POST /api/v1/runs/digest` | Generate digest now (params: `since`, `dry_run`) |
| `POST /api/v1/episodes/{id}/retry` | Reset a failed episode to its last good state |
| `POST /api/v1/episodes/{id}/escalate` | Force-escalate a dropped/low-scored episode (owner override) |
| `GET /api/v1/episodes` | Paged list, filter by status/show — JSON, for curl debugging |
| `GET /api/v1/telemetry/costs` | Aggregated llm_call stats by provider/model/tier/day |

No public exposure: bind to the macvlan LAN IP only; no reverse proxy, no TLS termination in v1 (LAN-trusted network, consistent with existing homelab services). Document clearly that this must not be port-forwarded.

---

## 10. Non-Functional Requirements

### 10.1 Logging & Observability
- **structlog**, JSON output to stdout (Docker captures it; consistent with `docker logs` workflow).
- Every log line carries: `run_id` (per scheduler run), `episode_id` where applicable, `stage`, `duration_ms` where applicable.
- Log levels: lifecycle events INFO; per-episode routing decisions INFO (one line: episode, route, scores, confidence); external call failures WARNING (with retry count); pipeline-stopping errors ERROR.
- **Never log**: API keys, full transcripts, full LLM prompts/responses at INFO (allow full prompt/response at DEBUG behind a config flag `log_llm_io: false`).
- Metrics: keep it simple in v1 — the `llm_call` docs + `/status` + `/telemetry/costs` endpoints ARE the observability. No Prometheus in v1 (structure the code so a `/metrics` endpoint could be added later without refactoring).
- Each scheduler run ends with a one-line summary log: episodes seen/new/triaged/escalated/summarized/failed, total LLM cost, wall time.

### 10.2 Security
- **Threat model for v1:** LAN-only homelab service, single user; primary risks are (a) secret leakage, (b) processing untrusted internet content (feeds/pages/audio), (c) prompt injection from feed content, (d) supply chain.
- Secrets: env vars only; `.env` file excluded from git; never echoed in logs or error responses.
- **Untrusted input handling:**
  - Treat ALL feed content, show notes, scraped pages, and transcripts as untrusted data. Strip/sanitize HTML (bleach or equivalent) before storage and prompting.
  - **Prompt injection:** episode descriptions and transcripts are attacker-controllable text that gets placed into LLM prompts. Mitigations: (1) system prompts explicitly instruct the model that quoted content is data, not instructions; (2) all LLM outputs are schema-validated — free-text fields are rendered into Markdown with sanitization (strip HTML, no raw link auto-execution, escape Markdown control sequences in titles); (3) LLM output can never influence code paths beyond the enum/int fields defined in the schemas (routing is decided by code from validated numeric fields, never by free text).
  - Audio downloads: enforce content-length cap and content-type allowlist (`audio/*`), download to a quarantined tmp dir, never execute or parse beyond ffmpeg/whisper decode.
  - Outbound HTTP: explicit timeout on every request; redirect limit 5; only `http/https` schemes; optional domain allowlist derived from configured feeds (default on: enclosure/transcript URLs must share the feed's registrable domain OR be on a small allowlist of known CDNs — log and skip otherwise).
- API auth as in §9; CORS disabled; FastAPI docs (`/docs`) enabled but behind the same API key in v1.
- Containers: run as non-root UID, read-only root filesystem where possible, `no-new-privileges`, resource limits (memory limit on agent container to protect the NAS; ASR is the memory hog — cap and document).
- CouchDB: dedicated instance, admin credentials via env, bound to the internal Docker network only, not reachable from LAN.
- Supply chain: pin all Python deps with hashes (`uv` lockfile or `pip-tools`); pin base image digests; document an update cadence.

### 10.3 Reliability & Error Handling
- Every pipeline stage is idempotent and driven by document status — the scheduler can die at any point and the next run resumes safely.
- Retries with exponential backoff for: feed fetch, transcript fetch, audio download, LLM calls (on top of litellm fallback). Per-episode attempt counters with hard caps; exceeded → terminal `ERROR`/`TRANSCRIPT_FAILED` status + WARNING log, never infinite loops.
- A poison-pill episode (repeatedly crashing the worker) must not block the queue: catch per-episode exceptions, mark `ERROR` with the traceback stored on the doc, continue.
- Feed-level circuit breaker: after N consecutive poll failures for one podcast, back off to daily attempts and surface it in `/status`.
- Digest generation is transactional in effect: build full content in memory → atomic file write → only then mark episodes `PUBLISHED`. If marking fails midway, next digest run detects already-written file (digest doc exists) and completes the marking (reconciliation step).
- Clock/timezone: all storage in UTC ISO-8601; rendering in `Europe/Stockholm`.

### 10.4 Performance & Capacity (informational targets, not hard SLOs)
- Expected volume: 25–45 new episodes/week; ~5–15 escalations/week; ASR ≤ ~6 h audio/week worst case.
- Tier-0 latency budget: < 30 s/episode; Tier-1 < 10 min/episode including local 27B on long context.
- ASR throughput on CPU (NAS) will be slow (large-v3-turbo, int8: roughly ≥ 0.5× real-time on modern CPU; slower on the QNAP) — this is acceptable because everything is asynchronous and queued. Config allows pointing ASR at a remote Whisper endpoint later (e.g., the future Mac mini/GX10 box) via the ASR interface — design the interface now, implement `local` only.
- Backfill guard: on first run against a fresh DB, only process episodes published within `initial_lookback_days` (default 14) to avoid summarizing years of archives.

### 10.5 Maintainability & Testing
- Project layout:
  ```
  podcast_agent/
    main.py            # FastAPI app + scheduler startup
    config.py          # pydantic-settings models
    state.py           # status enum + transition map
    db/couch.py
    ingest/feeds.py
    triage/tier0.py
    transcripts/{acquire.py, asr.py}
    summarize/tier1.py
    digest/{generate.py, templates/}
    llm/client.py
    api/routes.py
    prompts/
  tests/
  ```
- Type hints everywhere; `ruff` + `mypy` in CI (a simple GitHub Actions workflow or local pre-commit is fine).
- Tests (pytest):
  - Unit: state transitions, routing logic (given Tier0Result → route), episode ID stability, digest rendering (golden-file tests), config validation, markdown sanitization.
  - Integration: pipeline against a **fixture RSS feed + canned LLM responses** (fake litellm layer — no live LLM in tests), CouchDB via testcontainers or a mocked thin client.
  - No tests may hit the network.
- All LLM prompts versioned in-repo; changing a prompt bumps its version string.

### 10.6 Data Retention & Privacy
- This system stores only public content + the owner's interest profile. No personal data beyond that.
- Retention (config): raw transcripts kept 180 days then attachment deleted (summary kept indefinitely); `llm_call` telemetry kept 365 days; audio files deleted immediately after ASR (default).
- Everything stays on-prem except: RSS/HTTP fetches, and LLM calls routed to OpenRouter when configured/fallback triggers. **Config flag `allow_cloud_fallback: true|false` per tier** — when false, fallback stays local-only and failures queue instead (owner's data-sovereignty preference must be a first-class switch, not an emergent behavior).

---

## 11. Scheduling (defaults)

| Job | Schedule | Notes |
|---|---|---|
| `ingest` | every 6 h | conditional GET, cheap |
| `pipeline` | every 30 min | processes any pending statuses; no-op if queue empty |
| `digest_weekly` | Fri 06:00 Europe/Stockholm | main deliverable |
| `retention_cleanup` | daily 04:00 | transcript/telemetry retention |

APScheduler `AsyncIOScheduler`, `max_instances=1` + `coalesce=True` per job (a slow run must never overlap with the next).

---

## 12. Seed Podcast List (initial config)

Seed from the awesome-cybersecurity-podcasts directory. The implementer must locate the actual RSS feed URL for each (the directory links homepages, not feeds). Initial set and priorities:

| Podcast | Priority | always_escalate | Rationale |
|---|---|---|---|
| Risky Business | high | true | Dense news + interviews, thin per-episode notes |
| CyberWire Daily | med | false | Daily volume; good descriptions; Tier-0 filter does the work |
| SANS Stormcast | med | false | Short dailies, transcript often available |
| Darknet Diaries | med | false | Published transcripts on site (configure transcript source) |
| Security Cryptography Whatever | high | true | Deep technical, irregular |
| Cloud Security Podcast by Google | high | true | AI security + cloud, owner-relevant |
| Microsoft Threat Intelligence Podcast | high | true | Microsoft stack relevance |
| The Defender's Advantage (Mandiant) | high | true | Threat intel depth |
| CISO Series Podcast | med | false | Leadership |
| Caveat | med | false | Policy/regulation |
| Smashing Security | low | false | Entertainment-leaning; strict filter |
| Hacking Humans | low | false | Social engineering |
| Defensive Security Podcast | med | false | Defender lessons |
| Unsupervised Learning | med | false | AI + security commentary |

(Remaining shows from the directory: include as `priority: low` if an RSS feed is trivially found; otherwise log as unsupported and move on.)

---

## 13. Implementation Plan (suggested milestones)

1. **M1 — Skeleton:** compose stack (agent + CouchDB), config loading, health endpoint, structlog, CouchDB client, state machine, ingestion writing `episode` docs. *Definition of done: new episodes appear in DB, idempotent re-runs.*
2. **M2 — Tier-0:** llm/client.py with litellm Router + instructor, tier0 prompt, routing logic, telemetry docs. *DoD: every NEW episode gets a validated route; unit tests for routing.*
3. **M3 — Digest (description-level):** digest generator + templates + atomic writes + PUBLISHED marking + manual trigger endpoint. *DoD: a real weekly digest file renders in Obsidian from DIGEST_DIRECT items — the system is already delivering value here.*
4. **M4 — Transcripts + ASR:** acquisition chain (feed transcript → configured scrape → faster-whisper), queueing, caps, cleanup. *DoD: an escalated episode ends up with a stored transcript from each of the three paths (test fixtures for the first two, one real ASR smoke run).*
5. **M5 — Tier-1:** summarizer incl. map-reduce for long transcripts, scoring, full digest with summaries. *DoD: end-to-end run produces the §5 digest with real summaries.*
6. **M6 — Hardening:** retries/circuit breakers, retention job, security items from §10.2, remaining admin endpoints, test coverage, README/runbook (deploy, rotate keys, add a podcast, tune thresholds).

---

## 14. Open Items (decide during implementation, low risk)

- Exact local model tags for tier0/tier1 (depends on what's pulled in Ollama at deploy time — config-only change).
- Whether Darknet Diaries transcript scraping is stable enough vs. just letting ASR handle it (implementer: try the configured-selector scrape; if brittle, drop to ASR).
- Per-episode notes files (config flag exists; default off — revisit after two weeks of digests).
- Remote ASR endpoint support (interface exists in v1; implementation deferred until the Mac mini / GX10 decision lands).
