# Podcast Digest Agent — Future Capabilities & Roadmap

**Version:** 1.0
**Status:** Ideas backlog — NOT in v1 scope
**Companion to:** `podcast-digest-agent-design.md` (v1 design)
**Owner:** (you)
**Date:** 2026-07-30

> **Status note (added by implementation, 2026-07-30):** Phases 2, 3 (A1) and B2
> are implemented. B1 — the read/triage interface with search and starring — is
> still open; the console built for B2 is operational, not a reading surface.
>
> Phases 2 and 3 (A1) are F3 was deliberately skipped — see the correction below. Phase 2 was
> A2, C2, E4, F5 in full plus E2's producer side. Items are marked below.
>
> **Correction to A1's premise:** this document warns that "many [feeds] truncate
> to last 50–100 episodes" and proposes PodcastIndex as the general fix. Measured
> against the 14 seeded feeds, only *one* truncates (Risky Business, back to
> 2025-02); the rest expose their full history — CyberWire to 2015, Stormcast to
> 2016. The archive was reachable from the feeds alone, so F3 was not needed and
> is not built.
>
> **The real constraint turned out to be transcription, not feed depth:** 3,914
> archive audio hours, of which only ~593 h ship transcripts. Backfill is
> therefore transcript-only by default, which makes a 12-month walk cost ~5 hours
> of local LLM and zero ASR jobs.
>
> (Original phase 2 note:) Phase 2 is implemented —
> A2, C2, E4, F5 in full, and E2 as the producer side only (`GET /api/v1/glance`;
> the display integration remains yours). Items are marked ✅ below. Everything
> from phase 3 onward is untouched and still an ideas backlog.
>
> One correction to the sequencing rationale, learned while building C2: the
> transcript-retention warning at the bottom of this document does not only gate
> backfill. It gates **re-scoring too**, which is now a live capability — so
> `retention.transcript_days` is already the binding constraint on how far back a
> profile change can be applied, before any backfill work starts.

Purpose: capture post-v1 directions so v1 architectural choices don't accidentally close doors. Each item notes what, why, rough complexity (S/M/L), and any v1 hooks it relies on. Nothing here obligates v1 beyond the hooks already present in the design doc.

---

## Theme A — Historical Backfill (owner-requested)

### A1. Batched archive import — "a month at a time" (M)

**✅ Implemented, transcript-only.** Per-podcast month cursor walking backwards,
`origin: backfill` isolating archive material from the routine pipeline and the
weekly digest, per-show `backfill_mode` (full / tier0_only / skip), stricter
archive threshold, per-show monthly files under `archive/<slug>/YYYY-MM.md`, and
a dry-run estimate built from real recorded latencies that refuses to spend
compute without `confirm=true`. ASR is structurally impossible during backfill
while `require_transcript` is set.

Also fixed a latent flaw this exposed: routine ingestion had no cutoff once a
show had any history, so the *second* poll would have pulled the whole feed page
(months of back catalogue) silently. Routine ingestion now only ever looks
forward; reaching backwards is exclusively this job's business.
Import back-catalog episodes for all configured podcasts in controlled batches rather than all at once.

- **Design sketch:**
  - New job type `backfill` with a cursor per podcast: `backfill_cursor: 2026-06` walking backwards month by month.
  - Each run processes exactly one month-window per podcast (or a global episode cap per run, e.g. 100), then stops. Triggered manually (`POST /runs/backfill?months=1`) or on a slow schedule (e.g. one month-window per night).
  - Backfilled episodes are flagged `origin: backfill` and **never enter the weekly digest** — they flow into separate archive digests (`/data/digests/archive/<podcast-slug>/YYYY-MM.md`) or per-episode notes, so the weekly signal stays clean.
  - Cost guardrails matter far more here than in steady-state: a pre-flight `dry_run` mode that counts episodes, estimates Tier-0/Tier-1/ASR volume and projected cost/wall-time from telemetry history, and requires an explicit confirm parameter before burning a weekend of NAS CPU on ASR.
  - Relevance thresholds should be **stricter for backfill** (e.g. digest_threshold 7 instead of 5) — old episodes must earn their summary; news-cadence shows (CyberWire Daily, Stormcast) should default to Tier-0-only in backfill since stale news has little value, while evergreen shows (Darknet Diaries, Security Cryptography Whatever) justify full treatment.
- **v1 hooks already in place:** idempotent episode IDs, `initial_lookback_days` guard (just parameterized), status-driven pipeline, telemetry for cost estimation.
- **Watch out for:** RSS feeds that only expose recent items (many truncate to last 50–100 episodes). Full archives may require the show's website or an Apple Podcasts/PodcastIndex lookup — a `podcastindex.org` API integration (free, open) is the clean general solution and also future-proofs feed discovery.

### A2. Archive-aware summarization context (S)

**✅ Implemented.** Tier-1 prompts moved to v2 (v1 retained, unedited); publication date and a human age note ("an archive episode" past 90 days) are injected, with instructions to anchor time-sensitive claims to the recording date.
When summarizing an old episode, inject its publication date prominently and instruct the model to frame it historically ("as of mid-2023...") so summaries don't read stale facts as current. Trivial prompt change, but easy to forget.

---

## Theme B — Interactive Web UI (owner-requested)

### B1. Read/triage interface (M)

**✅ Implemented (2026-08-01).** Browsing, reading and re-running were already
there. Added: starring (toggled from the row), read-marking as a timestamp
rather than a flag so time-to-read is recoverable, and a "wrong call" verdict in
*two* directions — `over` for a false positive, `under` for a false negative,
because a single flag conflates them and the second is the expensive kind. A
verdict stores the status, score and profile version it is disputing, so a later
re-score cannot silently rewrite what was being judged. All three are filterable
from the episode list.

Full-text search is a SQLite FTS5 sidecar in `work_dir`, not a Mango query:
`$regex` is an unindexable full scan and cannot reach a gzipped attachment,
which is where transcripts live. The index is a cache — rebuilt in full from the
database, thrown away without consequence, and a 409 rather than a 500 when it
does not exist yet.

Per-show priority editing turned out to already exist (`PATCH
/api/v1/podcasts/{slug}`, with a select on the Podcasts page); this document was
simply out of date about it.
Small self-hosted web app (single container, talks to the existing FastAPI) for browsing beyond what static Markdown offers:

- Episode list with filters (show, score, status, interest tag, date), full summary view, transcript view.
- Actions: mark read, star, "summarize anyway" (calls the existing force-escalate endpoint), "wrong call" flag (feeds Theme C), adjust per-show priority.
- Search across summaries/transcripts (CouchDB full-text via couchdb-lucene, or SQLite FTS5 sidecar index — decide then).
- Stack suggestion consistent with existing patterns: keep FastAPI as the only backend; UI as a static React/HTMX bundle served by the same container; PWA manifest so it installs on the phone like the tasting-log app. LAN-only remains fine; if remote access is ever wanted, put it behind the existing preferred approach (VPN/Mullvad-style, not port-forwarding).
- **v1 hooks:** all state already in CouchDB with clean status fields; API surface already exists; nothing to refactor.

### B2. Operations dashboard (S)

**✅ Implemented.** `GET /admin` — one self-contained page, no external requests,
polling `/status` and `/telemetry/costs`. Queue depths, per-show feed health and
circuit-breaker state, cost by provider, backfill progress per show, and controls
for every manual run. Also added the thing this exposed as missing: a persisted
start/pause control for the archive walk, defaulting to paused, with cooperative
stopping so a pause costs at most one in-flight episode.
Same UI, second tab: queue depths, per-show feed health/circuit-breaker state, cost per provider/model over time (chart over `llm_call` docs), ASR backlog. Essentially a visual skin over `/status` + `/telemetry/costs`.

---

## Theme C — Learning & Personalization

### C1. Feedback loop → threshold/profile tuning (M)

**◐ Phase 1 implemented (2026-08-01).** `GET /api/v1/insights/precision`, on the
Insights page. Per podcast and per interest: surfaced, starred, read, and both
flag directions, with suggestions carrying their numbers and the exact config
change. Nothing is applied, nothing is suggested below a sample floor, and the
report states that a missing star is weak evidence while an explicit flag is
not. A single "ranked too low" is reported without waiting for a sample, because
a false negative is the expensive kind. Phase 2 (few-shot injection) is untouched.

It needs a few weeks of use before it has anything to say — the signals it reads
only exist once you start starring and flagging.
Capture lightweight signals (read, starred, "wrong call", time-to-read) from B1 and use them:
- **Phase 1 (simple, no ML):** monthly report showing precision proxies per interest key and per show ("you starred 0 of 9 Smashing Security items — demote?"), with suggested config diffs the owner applies by hand. Human-in-the-loop, transparent, in character.
- **Phase 2:** few-shot injection — include 3–5 recent starred and rejected examples in the Tier-1 scoring prompt so the model calibrates to demonstrated taste rather than the static profile alone.

### C2. Interest profile versioning & drift detection (S)

**✅ Implemented.** The profile is hashed (stable under reordering, sensitive to any prompt-visible field) and stamped on every tier result. `/api/v1/status` reports `stale_episodes`; `POST /api/v1/runs/rescore` re-runs Tier-1 from stored transcripts. Published episodes are excluded so the digest on disk cannot disagree with the database.
Profile changes are already just config; add a changelog and re-score capability ("re-run Tier-1 scoring only, from stored transcripts, against profile v3") — cheap because transcripts are retained. Useful when the Cynode move (or any role change) shifts what "relevant" means.

---

## Theme D — Deeper Intelligence Over the Corpus

### D1. Cross-episode synthesis — weekly themes (M)

**✅ Implemented (2026-08-01).** `digest_themes_v1`, one call per digest over
the week's summaries, rendered as the digest's opening section and stored on the
digest document so the next week's "what's new" has something to compare
against. Only episodes with a real Tier-1 summary are fed in — a digest-direct
one-liner is a guess from a feed description, and including those would let the
opening assert things about episodes nobody read.

It can never cost the reader their digest: too few episodes, a model that is
down, or a response that validates to nothing all produce the same outcome, a
digest that opens with its episode summaries exactly as before.
A second-order LLM pass over the week's summaries: "what are the 3 themes across all shows this week, where do hosts disagree, what's new vs. last week." Becomes the digest's executive summary. High value-per-token since it reads summaries, not transcripts.

### D2. Entity & trend tracking (M)

**✅ Implemented (2026-08-01).** `GET /api/v1/entities` and `/entities/{key}`,
plus `POST /entities/notes` writing one Obsidian note per entity under
`entities/`, with wikilinks from episode notes so the graph has edges from both
ends. Aggregated on demand — no model call, no cache, nothing written to an
episode.

Canonicalisation is deliberately conservative, because the two errors are not
symmetric: over-merging fuses two unrelated entities into one timeline that
reads as evidence and nothing downstream can tell, while under-merging leaves
two rows a reader can judge. CVE identifiers are normalised properly; everything
else only has case, whitespace, articles and corporate suffixes stripped.
The Tier-1 schema already extracts `entities` (CVEs, threat actors, tools, frameworks). Aggregate them: entity timeline pages ("Volt Typhoon: mentioned in 6 episodes across 4 shows since March"), CVE watchlist intersections, and Obsidian-native output — one note per tracked entity with backlinks from episode notes, making the vault's graph view genuinely useful.

### D3. RAG — "chat with the archive" (L)
Embed transcripts/summaries (local embedding model, e.g. a BGE/GTE-class model on Ollama) into a vector store (Qdrant self-hosted fits the stack) and expose a Q&A endpoint/UI: "what have podcasts said about Purview DSPM this year?" This is the flagship justification for the local-LLM hardware beyond benchmarking — retrieval + generation fully on-prem. Depends on: transcript retention (extend the 180-day default), B1 for UI.

### D4. Speaker diarization + quote extraction (M)
Upgrade ASR path with diarization (whisperX / pyannote) so summaries can attribute positions ("Ptacek argued X, Connolly pushed back") and extract 1–2 verbatim-ish key quotes with timestamps + deep links (`?t=1234`) for the rare "actually listen to this 3 minutes" case. CPU cost is significant; gate to `always_escalate` shows.

---

## Theme E — Output & Delivery Extensions

### E1. Audio digest via Piper TTS (S–M)
Ironic but genuinely useful: render the weekly digest through the already-deployed Piper TTS into a private RSS feed (one MP3/week) hosted on the NAS — the digest becomes a podcast, listenable on the dog walk, on-prem end to end.

### E2. E-ink presence (S)

**⊘ Dropped (2026-08-01), owner's decision.** The Inky Frame integration is not
being pursued.

`GET /api/v1/glance` stays. It was the producer half of this and it is a small,
self-contained endpoint that anything glanceable can read — a phone widget, a
terminal prompt, a future display. Removing it would cost more than keeping it.
One glanceable line on the Inky Frame family calendar: "Podcast digest: 9 new summaries, top: Risky Biz 9/10". Reuses the existing SSE/display pipeline; strictly a teaser, not a reading surface.

### E3. Content-pipeline feed (M)

**✅ Implemented (2026-08-01).** `POST /api/v1/content/seeds` writes
`content-seeds.md` beside the digests; `GET` previews what qualifies without
spending a call. One pass over the summaries of episodes scoring above
`content.min_score` on `content.interests`, producing per-episode angles and
cross-episode threads.

Off by default, and it stays off until configured — a system that suggests what
to post unbidden is presumptuous in a way the rest of this is not. The prompt is
told to skip freely: fifteen mediocre angles are worth less than three real
ones, and an empty answer is rendered as the ordinary outcome it is. Episodes
are referenced by number rather than title, so every angle traces back to the
episode it came from; one that cannot be attributed is dropped rather than
rendered.
Bridge into the Medium/LinkedIn RSS content pipeline already being explored: a `content_seeds.md` output listing episodes scoring high on `ai_agent_security`/`ot_ics` with angle suggestions ("this Mandiant episode contradicts the common take on X — post idea"). Feeds Almedalen-style thought leadership, including the Agentic OT Risk Model narrative.

### E4. Notifications for exceptional items (S)

**✅ Implemented.** ntfy push at `min_score` (default 9), disabled by default, failures swallowed. Digest availability deliberately does not notify.
Push (ntfy self-hosted fits the stack) only when `relevance_score >= 9` — never for routine digest availability. Strict threshold so it stays rare and meaningful.

---

## Theme F — Platform & Operations Maturity

### F1. Remote ASR / inference offload (S — interface already exists)
Point the ASR interface and/or Ollama base URL at the future Mac mini M5 Pro or ASUS GX10 once purchased. Turns backfill (Theme A) from a week of NAS CPU into an overnight job, and unlocks D3/D4 practically. Explicitly anticipated in v1 §10.4.

### F2. Model evaluation harness (M)
Formalize what v1 telemetry enables: an A/B mode that runs the same episode through two Tier-1 configs (e.g. local Qwen3.6-27B vs. OpenRouter GLM-5.2) and stores both for side-by-side review in the UI. Produces the real-workload evidence for the hardware decision, and later for model upgrades. Include prompt-version A/B, not just model A/B.

### F3. Feed discovery & health (S)

**⊘ Skipped, premise did not hold.** Only 1 of 14 feeds truncates, so archive
depth never needed it. Still potentially useful for feed-change detection and
bulk show discovery; revisit then.
PodcastIndex API integration (also needed by A1) for: resolving homepage→RSS automatically when adding shows, detecting feed URL changes/redirects, and suggesting new shows by category — reducing dependence on the GitHub directory staying maintained.

### F4. Multi-profile support (M)
Second interest profile (e.g. a colleague, or a "Cynode-role" vs "PwC-role" profile) scored in the same Tier-1 pass (one extra schema field), separate digest outputs. Cheap at scoring time; mostly a config/output-routing exercise. Explicitly not multi-tenant auth — just multiple profiles for one household/team.

### F5. Backup & restore runbook (S)

**✅ Implemented.** `scripts/backup.sh` / `scripts/restore.sh` with attachments, rotation and a verified round-trip; restore refuses to merge into an existing database.
CouchDB continuous replication or scheduled `couchdb-dump` to the NAS backup target; digest folder is already covered by folder sync. Small, boring, should be done early in phase 2.

---

## Suggested sequencing (if everything above were pursued)

| Phase | Items | Rationale |
|---|---|---|
| 2 | ✅ F5, ⊘ E2, ✅ E4, ✅ A2, ✅ C2 | Done. E2 dropped; its `/glance` producer stays |
| 3 | ✅ A1, ⊘ F3 | Done. A1 shipped transcript-only with telemetry-based guardrails; F3 unnecessary (feeds are not truncated) |
| 4 | ✅ B2, ✅ B1 | Done. Browsing, reading, re-running, starring, read-marking, wrong-call verdicts, FTS5 search |
| 5 | ◐ C1, ✅ D1, ✅ D2 | D1 and D2 done. C1 phase 1 done; phase 2 (few-shot) open |
| 6 | F1, F2 → D3, D4, E1, ✅ E3 | E3 done early — it reuses D1's machinery. The rest is hardware- and corpus-dependent |

The single most important cross-cutting dependency: **extend transcript retention before starting backfill** — D2/D3 lose most of their value if archive transcripts were already purged at 180 days.

**✅ Resolved (2026-08-01).** `retention.transcript_days: 0` now means keep
indefinitely, and that is what the shipped config does. The measurement that
settled it: 160 transcripts held 2.3 MB gzipped, ~14 KB an episode, so the whole
archive is tens of megabytes. The storage this was protecting was never worth
the capability it cost.
