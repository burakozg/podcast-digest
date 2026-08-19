# Podcast Digest Agent

Self-hosted agent that monitors cybersecurity podcasts, decides which episodes
matter to *you*, summarises those, and writes a weekly Markdown digest into an
Obsidian-compatible synced folder. You read summaries instead of listening.

- **[architecture.md](architecture.md)** — how it is built and why
- **[RUNNING-ON-MAC.md](RUNNING-ON-MAC.md)** — running it natively on macOS
- [podcast-digest-agent-design.md](podcast-digest-agent-design.md) — the original
  v1 design, kept unedited as the historical record
- [podcast-digest-agent-roadmap.md](podcast-digest-agent-roadmap.md) — the ideas
  backlog, annotated with what shipped

---

## Why it isn't just an RSS filter

Episode descriptions are often thin or pure marketing, so filtering on them alone
silently loses good episodes — and you never learn what you missed.

The pipeline therefore judges **two** things per episode: how relevant it looks,
and *how informative the description was*. Only the second decides whether to pay
for a transcript. A vague description is never grounds to drop an episode; it is
grounds to go and find out.

```
RSS feeds ──► Ingestion ──► Tier-0 triage ──► route decided in code
                                                │
              ┌─────────────────────────────────┼──────────────────┐
              ▼                                 ▼                  ▼
           DROPPED                        DIGEST_DIRECT        ESCALATE
        (confidently                  (grey zone: listed     (relevant, or
         irrelevant,                   as a one-liner)        description too
         kept for audit)                                      thin to judge)
                                                                   │
                                          transcript acquisition ◄──┘
                                     feed transcript → configured scrape → ASR
                                                                   │
                                              Tier-1 summarise + score
                                                                   │
                                                   weekly Markdown digest
```

---

## Quick start

```bash
git clone <this repo> && cd podcast-digest
cp .env.example .env
openssl rand -hex 32   # → PODAGENT_ADMIN_API_KEY
openssl rand -hex 24   # → COUCHDB_PASSWORD
$EDITOR .env           # fill both, add the two model-provider keys below,
                       # and set DIGEST_DIR to your vault folder
$EDITOR config.yaml    # tune interest_profile — this drives everything

docker compose up -d --build
docker compose logs -f podcast-agent
```

**Model providers.** As shipped, both tiers run in the cloud —
OpenRouter as the primary with Anthropic behind it — because there is currently
no machine here worth running a local model on. `PODAGENT_OPENROUTER_API_KEY`
and `PODAGENT_ANTHROPIC_API_KEY` are therefore both required, and the service
refuses to boot without them rather than failing over unauthenticated.

To run locally instead, restore an Ollama primary in `config.yaml` (the blocks
are still there, commented, in `llm.tiers`), pull the models, and the keys stop
being needed:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull qwen3:8b    # tier0
docker compose exec ollama ollama pull qwen3:32b   # tier1
```

Then open the console at `http://<host>:8080/admin`, or kick a first run by hand:

```bash
export KEY=$(grep '^PODAGENT_ADMIN_API_KEY=' .env | cut -d= -f2)
export HOST=127.0.0.1:8080   # or <nas-ip>:8080

curl -fsS "http://$HOST/healthz" | jq
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/ingest?wait=true" | jq
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/pipeline?wait=true" | jq
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/digest?dry_run=true" | jq
```

On a fresh database only episodes from the last `initial_lookback_days` (default
14) are processed, so you won't summarise years of back catalogue.

Ollama runs as part of the stack only with `--profile ollama`. On a machine
whose GPU a Linux container cannot reach — an Apple Silicon Mac — run Ollama
natively and leave the profile out, pointing `api_base` at it.

**On a Mac** the service itself currently runs natively rather than in the
container: see [RUNNING-ON-MAC.md](RUNNING-ON-MAC.md). The compose stack is the
canonical deployment and runs unchanged there, so it can be rehearsed before it
is relied on.

### Local development

```bash
uv sync --all-extras          # ALWAYS --all-extras: a plain `uv sync` removes
                              # faster-whisper and silently disables ASR
uv run pytest                 # 1173 tests, no network, no CouchDB needed
./scripts/check.sh            # the full gate: format, lint, types, tests
```

---

## The consoles

**One word per thing.** The console names a stage for what it does, not for what
it is called in the source: **triage** (not Tier-0), **summarise** (not Tier-1),
**local transcription** / *transcribe locally* (not ASR), **routing** (not
dispatch). Config keys keep their real spelling — `llm.tiers.tier0`, the `asr`
section, `backfill_mode: tier0_only` — and the console shows the key beside the
plain-language label wherever you would edit it. A test enforces this on
rendered text, so the jargon cannot creep back in with the next page.


Eight pages sharing one navigation bar, each a single self-contained file with no
external requests.

| | |
|---|---|
| **`/admin`** | Operations: queue depths, feed health at a glance, LLM cost, and one-click runs of the scheduled jobs with an explanation of each |
| **`/admin/digests`** | Reading view: the latest digest rendered, and every earlier week, each with a *Read aloud* button |
| **`/admin/episodes`** | Episode browser, 50 per page, full-text search, star/read/wrong-call marking, per-episode Markdown export, with a legend explaining every status |
| **`/admin/podcasts`** | Podcasts: add, disable, and per-podcast ASR, priority, always-escalate, backfill mode and history window |
| **`/admin/insights`** | What your reading says: precision report per podcast and interest, suggested (never applied) config edits, most-discussed entities, and content seeds |
| **`/admin/backfill`** | Historical intake: start/pause the archive walk, per-podcast cursors and windows, cost estimates |
| **`/admin/logs`** | Activity: live log tail, kept warnings, job-run history, model calls and transcription |
| **`/admin/settings`** | Models per tier, provider switching, local transcription (ASR), routing thresholds and the interest profile |

The navigation is a left sidebar, defined once — markup and styling together —
and injected when a page is served, so the pages cannot disagree about what
exists or drift out of alignment. It collapses to a horizontal bar on a narrow
screen. Historical intake has its own page because it
is a long-running, opt-in operation — a switch that can consume days of compute
does not belong beside buttons that take seconds. Per-feed detail lives only on
the Podcasts page; the operations page keeps a single "is anything broken?"
signal rather than a second copy of the same table.

### Speech-to-text

Transcription runs **in the agent process, on the machine the agent runs on**,
using `faster-whisper` — a CTranslate2 build of OpenAI's Whisper. The audio is
downloaded to `output.work_dir`, transcribed, and deleted unless `keep_audio` is
set. Nothing is sent anywhere; there is a `remote` backend in the config schema
but it is a placeholder in v1.

It is the **last** transcript strategy, reached only when a podcast publishes no
transcript and no scrape selector is configured. The Transcripts column on
`/admin/podcasts` shows which of your podcasts actually need it.

Model, device, compute type, language, beam size and backend are editable from
`/admin/settings` (applied at the next restart, like every other setting there).
`max_audio_mb` and the concurrency caps stay in `config.yaml`: they protect the
machine rather than express a preference.

Transcription logs as it goes: `transcript.asr_start`, then faster-whisper's
own `Processing audio with duration…`, then an `asr.progress` line every 30
seconds with position, percent and realtime factor, and an `asr.complete` with
the overall speed. Watch it on `/admin/logs`.

On Apple Silicon `device: cpu` is correct — CTranslate2 has no Metal backend and
there is no CUDA — so `compute_type: int8` with a smaller model such as
`small.en` is the usual compromise. `faster-whisper` is an optional extra
(`pip install '.[asr]'`); the settings page says so plainly if it is missing,
rather than letting an episode fail to find out.

### Changing configuration from the console

`/admin/settings` edits the part of the configuration worth changing at runtime:
which model each tier uses and whether it may fall back to a cloud provider, the
routing thresholds, and the interest profile.

**Changes apply at the next restart**, and the page says so while any are
pending:

```bash
docker compose restart podcast-agent
```

That is deliberate. Swapping a provider rebuilds the LLM router and editing the
interest profile invalidates every cached prompt — doing either underneath a
summarisation in flight is a poor trade for a setting that changes a few times a
year.

A change is validated exactly as a config file is *before* it is stored, so
something that could not boot is refused immediately rather than at the next
restart. Storage, network binding, the security allowlist, output paths and the
schedule are **not** editable here: a typo in a browser should not be able to
make the service unreachable or unable to find its own database. Discarding
overrides returns every value to `config.yaml`.

The console pages are served **without** the API key, because a browser
navigation cannot send a header. They contain no data: each asks for the key,
holds it in `sessionStorage` for that tab, and every value on screen arrives
over the normal authenticated endpoints. Treat them like the rest of this
service — LAN-only, never port-forwarded.

### Browsing episodes

The operations console lists episodes 50 at a time, filtered by podcast, status and
score,
with two columns that answer most questions: whether a **summary** exists (and on what
basis) and whether a **transcript** does. Open any episode to read its summary,
takeaways and entities, and to re-run it. A published episode names the digest it
went into, linked straight to that week on `/admin/digests`.

**Summarising one episode.** `allow_asr=false` restricts acquisition to a
published transcript — seconds, free. `allow_asr=true` downloads and transcribes
locally if needed, which can take an hour. Either way a verdict is always
reached: with no transcript available the summary is made from the description
and labelled `description_only`.

```bash
curl -fsS -X POST -H "X-API-Key: $KEY" \
  "http://$HOST/api/v1/episodes/<id>/summarize?allow_asr=false" | jq
```

The call returns immediately and the work continues server-side; watch the
episode's status. `?wait=true` blocks instead, but a long transcript can outlast
any client timeout — the work completes either way, because it is owned by the
app rather than the request.

An episode already published in a digest is refused: rewriting only the database
would leave the written file disagreeing with it.

### Why a published episode may have no summary

`PUBLISHED` means "has appeared in a digest", not "has been summarised". The
grey-zone route is the common case: at high Tier-0 confidence but middling
relevance the episode is listed as a one-liner under *Maybe interesting* and
deliberately never transcribed. That is the choice that stops anything
possibly-relevant from vanishing without also paying to transcribe it. Those rows
read `no — indexed only (grey zone)`.

---

## Configuration

Two inputs, layered: **`config.yaml`** (non-secret, mounted read-only) and
**environment variables** (secrets). Env wins; `PODAGENT_<SECTION>__<KEY>`
overrides any nested value. Invalid config is a startup crash with a readable
message, never a half-configured run.

Podcasts can also be managed from the console, which stores **overrides** in the
database — see [Managing podcasts](#managing-podcasts).

### The interest profile

The part worth your attention. It is injected into every prompt, and both tiers
score against it:

```yaml
interest_profile:
  - key: ot_ics                       # stable id; appears in episode docs
    label: OT/ICS security            # shown in the digest
    description: >-                   # the model reads this — be specific
      Industrial control systems, Purdue model, DCS/PLC/SCADA,
      OT incident response, OT/IT convergence
    weight: 10                        # 1-10, relative importance
```

Vague descriptions produce vague scoring. Name technologies, standards and
sub-topics you care about.

### Thresholds worth knowing (`pipeline:`)

| Key | Default | Effect |
|---|---|---|
| `t_conf_high` | 7 | Below this Tier-0 confidence → always escalate. **Raise it if relevant episodes are being dropped.** |
| `t_rel_low` | 4 | Relevance below this (at high confidence) → drop |
| `t_rel_high` | 7 | Relevance at/above this → full summary |
| `digest_threshold` | 5 | Tier-1 score needed for a digest entry |
| `top_pick_threshold` | 8 | Score for "Top picks" vs "Also relevant" |
| `max_transcripts_per_run` | 6 | ASR jobs per 30-minute run — the main cost/time valve |

Tuning by symptom:

- **Too much noise in the digest** → raise `digest_threshold` (and
  `top_pick_threshold` with it; the config refuses an incoherent pair).
- **Missing episodes you'd have wanted** → raise `t_conf_high` toward 9 so more
  episodes get a transcript, and check the "Everything else scanned" table to see
  what was set aside and why.
- **ASR is saturating the NAS** → lower `max_transcripts_per_run`, turn
  `asr_enabled` off for lower-value podcasts, or set `asr.model: small.en`.

### Watching what it does

Each button on the Operations page carries **when that job last ran**, read from
stored run history rather than process memory — so it still answers the question
after a restart, which is exactly when the question gets asked. A job currently
running says so instead.

`/admin/logs` has three tabs:

- **Log stream** — the live tail, filterable by severity and by free text across
  every field, auto-refreshing every 5s. Held **in memory** and bounded to the
  last 2000 events, so it is empty right after a restart. That is deliberate:
  writing every log line to CouchDB means logging generates documents that
  generate logging, plus a retention job to clean up after it, for something
  whose value is almost entirely in the last few minutes.
- **Kept warnings** — warning and above, stored in the database and kept 30 days
  (`retention.log_days`), so a restart does not take the answer to "what went
  wrong" with it. Deliberately *not* a general log table: episode and feed
  failures already live on the episode and podcast documents with more
  structure than a log line, and are not duplicated here. What this captures is
  the events belonging to no episode and no run — a job dying, the database
  becoming unreachable, a backend missing at startup. Identical events written
  together collapse into one row with a count.
- **Scheduled jobs** — one row per job firing with its result summary. Stored in
  the database, kept 90 days (`retention.run_days`), pruned by the retention job.
- **Model work** — totals and per-model/provider/tier breakdown from the
  `llm_call` documents, and beneath it local transcription from the `asr_run`
  documents: audio hours, compute hours and realtime factor, by model and by
  podcast. Kept apart rather than folded together — a transcription has no
  tokens and no price, and averaging its minutes against a triage call's
  seconds would ruin both figures. `$0.0000` against an Ollama model is correct, not
  missing data.

Log events are captured *after* redaction, so a secret cannot reach the buffer
even though the buffer is served over the API — there is a test for exactly that,
and another that token counts survive redaction.

### What gets processed first

Routine stages queue **newest-first**. A digest is a weekly artefact, so an
episode published this morning is worth more than one from ten days ago, and
when capacity is short the fresh one should win.

**History waits for the present.** The archive walk does not start while any
recent-intake episode is still owed work, and a walk already running hands back
at its next episode boundary. That includes an episode queued for local
transcription — an archive month of transcript-only episodes would finish in
seconds while that one occupies a GPU for an hour, and it still waits, because
the ordering is recency rather than speed. The Historical intake page reports
*Waiting for recent episodes* with the list, so a deferred walk is not mistaken
for a broken one; it resumes on its own.

Two deliberate exceptions. `ERROR` episodes do not count as pending — one
document that needs a person must not halt the archive indefinitely. And
`POST /api/v1/runs/backfill?force=true` overrides the wait, so nobody is locked
out of their own machine.

An episode that does not finish before its week's digest runs is not stranded:
digest selection reaches back `pipeline.digest_catch_up_days` (default 30) for
episodes no digest has claimed, so a late finisher appears in the next digest —
under its own week's heading, marked **carried over**, never passed off as new.
Only an episode that takes longer than the catch-up window needs a manual
regenerate with a wider period (`POST /api/v1/runs/digest?since=…`).

### What historical intake collects

Routine polling **only looks forward** — on a fresh install it reaches back
`pipeline.initial_lookback_days` (14) and never further, by design. Everything
older comes from historical intake.

Historical intake **records every archive episode and triages it**, whether or
not it publishes a transcript. What a missing transcript costs is a *summary*,
not a place in the list.

**A podcast is archived only if you say so.** `backfill_mode` defaults to
`skip`, so adding a podcast never silently starts walking years of its back
catalogue — set it to *summarise* or *triage only* per podcast. Likewise
transcription: the archive transcribes a podcast only if that podcast has
*Transcribe locally* on. There is no global switch, because whether a back
catalogue is worth hours of CPU differs per podcast.

So for a podcast set to summarise:

- every archive episode is ingested and triaged;
- an episode with a transcript — published, or made locally — earns a full
  summary;
- an episode with neither is downgraded to a one-line index entry rather than
  failing acquisition and then spending the summarise step on its description
  alone: the most expensive stage for the least material.

So you see the whole back catalogue, scored, and decide case by case what to
summarise properly (the Episodes page can re-run any episode with or without
transcription). The dry-run estimate reports `indexed_only` separately from
`tier1_calls`, so the difference is visible before you commit.

### Re-walking the archive

The walk only ever moves **backwards**, so a month it has already passed is
unreachable: the cursor never returns to it. That matters whenever what the walk
collects changes — the months it crossed under the old rules keep whatever they
produced at the time, and widening the window does not help, because it extends
the far end while the gap is behind the cursor.

Per-podcast settings — archive mode, local transcription — are read **live** on
each run, so changing one on the Podcasts page applies to history already
ingested, not just to episodes walked afterwards.

**Pause stops more than the next round.** An archive episode is only ever
advanced inside a backfill run, so pausing freezes the ones already ingested
rather than letting them drain — no other job will look at them, because every
routine queue filters to routine origin. That mattered little when intake only
took transcript-bearing episodes; now that it records everything, a pause
strands the whole backlog until you press Start.

**Re-walk archive** on `/admin/backfill` (or `POST /api/v1/backfill/rewind?confirm=true`)
clears the cursors so the window is read again from the top. It deletes nothing:
ingestion is keyed on `sha256(slug + guid)` and create-if-absent, so a second
pass adds only episodes that were never recorded and leaves existing ones with
their status, summaries and digest entries intact.

### How far back the archive reaches

Set **per podcast**, in the History column on `/admin/podcasts`: 12, 24 or 36
months. Every podcast starts at 12 (`backfill.months`) and is raised
individually — an evergreen interview podcast can repay three years of back
catalogue where a daily news round-up does not repay one.

Widening only adds months to cover: cursors already recorded stay put, so a
podcast resumes where it stopped rather than starting over. Narrowing deletes
nothing already collected. Changes take effect at the next run, not the next
restart. Dry-run again after changing one — each step up is roughly another year
of archive for that podcast.

The Historical intake page shows each podcast's window in its progress table,
marked `default` where it simply inherits.

### Reading digests

`/admin/digests` renders the most recent digest and lists every earlier week
down the side, newest first. **View Markdown** toggles between the rendered view
and the source that landed in Obsidian.

Digests are read from the files on disk, not rebuilt from the database, so the
console cannot disagree with what was actually written — and a digest edited by
hand shows as edited. If the file has been moved or pruned the week is reported
as gone rather than silently rendering something else.

**Generating twice in one week produces two files, and both are listed.** The
generator never overwrites: a second run for the same ISO week writes
`podcast-digest-2026-W31-r2.md` beside the first, so nothing already synced to
your vault is rewritten. They are not two versions of one digest — each covers
only its own period, so the second contains what was published since the first
ran. The console lists every run, tagged `run 1` / `run 2`.

A digest covers **only what was published since the previous digest ended**, and
the rule holds even while historical intake is running. Archive episodes are
excluded on two independent grounds: they carry a `backfill` origin, and their
publication dates fall outside the window. They land in the per-podcast archive
files instead. Starting a backfill mid-week therefore cannot change that week's
digest — pinned by a test that generates the same week with and without a
five-episode archive haul and requires the output to be byte-identical.

Digest text is LLM output, which is downstream of podcast descriptions and
transcripts. Rendering it as HTML for the console is therefore done server-side
with raw HTML disabled *and* an allowlist filter applied to the result, so a
failure of either alone is not a hole.

### Managing podcasts

Easiest from `/admin/podcasts`: paste a feed URL, press **Check feed** — it
fetches and reports item count, audio enclosures and transcripts before you
commit — then add.

`config.yaml` remains the declared baseline for podcasts defined in it. Anything
changed in the console is stored as an **override** and marked as such, so a
value that no longer matches the file is always explainable, and one click
reverts it.

Each row also carries two things nobody has to maintain. The **description** is
the podcast's own blurb, captured from the feed at each poll, so it follows the
feed rather than a file — it is blank until the podcast has been polled once.
The **cadence** (`~daily`, `~weekly`, `~fortnightly`…) is the median gap between
recent episodes, measured rather than declared, so a weekly podcast that slips to
fortnightly reports it within a month or two. Read from the feed where possible
— what we hold is bounded by the lookback window — falling back to the episodes
held. It stays blank below three dated episodes rather than guessing from one gap.

The **Transcripts** column counts how many of the last 25 episodes *in the feed*
publish a transcript. That is the number that decides whether ASR is worth
turning on for a podcast: `all` means local transcription would be wasted
effort, `none` means every summary comes from ASR or the description alone.

**Nothing is ever deleted.** There is no way to remove a podcast or an episode, by
design: a deleted podcast would leave episodes with no way to explain where they
came from, and a deleted episode would take its summary with it. The only off
switch is **disable**, which stops new episodes being taken in and leaves every
existing summary in place and browsable. Episodes already queued when you disable
a podcast finish processing — the console shows how many.

To add a podcast in the file instead:

```yaml
  - slug: new-podcast                # lowercase, hyphens; permanent identifier
    name: New Podcast
    feed_url: https://example.com/feed.xml
    priority: med                    # high | med | low (prompt hint)
    always_escalate: false           # true = full summary regardless of Tier-0
    asr_enabled: false               # true = transcribe locally when no transcript
```

You need a **real RSS feed URL with audio enclosures** — not a homepage or an
Apple Podcasts link. Verify with:

```bash
curl -sSL "https://example.com/feed.xml" | head -40   # expect <rss>, <enclosure>
```

If the audio or transcript is served from a different registrable domain than the
feed, add that domain to `security.cdn_allowlist` or the fetch is refused. The
log line to look for is `ingest.url_rejected`.

All 14 seeded feeds were verified live at setup. Most serve audio from prefix/CDN
domains (`pdst.fm`, `prxu.org`, `mgln.ai`, `redcircle.com`, `podtrac.com`,
`blubrry.com`) and are allowlisted accordingly — don't prune that list without
re-checking.

**Every host in the chain needs an entry, not just the first.** Audio is reached
through analytics prefixers that redirect onward, and each hop is checked before
it is followed. `pdrl.fm` and `pscrb.fm` are on the list for exactly this reason:
no feed ever publishes them, but four shows redirect through them.

```bash
uv run python scripts/check-enclosure-chains.py   # every hop, every podcast
```

Run it after editing the allowlist or adding a podcast. It prints each chain and
exits non-zero naming any host that would be refused — which is worth doing
because the failure is silent: a rejected enclosure discards the episode, so the
symptom is a podcast that looks like it stopped publishing. Analytics prefixers chain: Click Here wraps a four-deep
`swap.fm → mgln.ai → podtrac → prxu.org` redirect, and only the **outermost**
host appears as the enclosure URL — so that is the one the allowlist must name.
A rejected enclosure discards the whole episode (a deliberate choice: nothing
is ingested on the strength of audio the guard refused), so a missing allowlist
entry looks like a silent, empty feed; `ingest.url_rejected` under kept
warnings on `/admin/logs` is the tell.

### ASR is per-podcast

`asr_enabled` decides whether the weekly pipeline may download and transcribe a
podcast's audio when it publishes no transcript. **Off by default** — ASR is the
expensive path and most podcasts do not repay it. The five `always_escalate` podcasts
ship with it on, because none of them publishes transcripts and they are the ones
that fill Top picks.

With it off, an episode with no published transcript is summarised from its
description and labelled `description_only`, so you can always see what you got.

Archive backfill needs `asr_enabled` before it will transcribe. One toggle should not be able to turn a
five-hour archive walk into a three-thousand-hour one.

### Switching a tier between local and cloud

Provider is config-only. Each tier has an ordered chain and its own timeout:

```yaml
llm:
  tiers:
    tier1:
      timeout_s: 300
      allow_cloud_fallback: true     # ← the data-sovereignty switch
      primary:
        provider: openrouter
        model: qwen/qwen3-32b
        max_tokens: 2000
      fallbacks:
        - provider: anthropic        # a second vendor, not a second model
          model: claude-sonnet-4-6
          max_tokens: 2000
```

Set `allow_cloud_fallback: false` and cloud endpoints are **removed from the
chain entirely** — work queues for a later run instead of leaving the LAN. That
is enforced at construction, not merely at call time.

**That switch is currently unusable, on purpose.** With no local model host,
both tiers are cloud end to end, so `false` would empty the chain — and the
config refuses to load rather than leave a tier that can never run. Restoring
the commented-out Ollama primary in `config.yaml` gives the switch its meaning
back. Note also that the fallback is a *different vendor*: a second OpenRouter
model does not survive OpenRouter itself being down or rejecting the key.

Two constraints govern which Anthropic model can go in a chain, both from this
codebase rather than the models: `temperature` is sent on every call, and the
current Sonnet/Opus generation rejects a non-default value with a 400; and those
models think by default, with thinking tokens counted against `max_tokens`, so a
summary budget would be spent reasoning. Hence Haiku 4.5 and Sonnet 4.6.

Costs and latency per provider/model/tier are recorded for every call:

```bash
curl -fsS -H "X-API-Key: $KEY" "http://$HOST/api/v1/telemetry/costs?days=30" | jq
```

---

## Digest output

One file per week: `<digest_dir>/YYYY/podcast-digest-YYYY-Www.md`, with Obsidian
frontmatter, an opening **This week** synthesis (cross-episode themes, built
from the week's summaries in one model call and skipped cleanly when the model
is down or the week is thin), then four sections — Top picks, Also relevant,
Maybe interesting (not summarized), and a collapsed "Everything else scanned"
audit table showing what was set aside and why. The synthesis opener can be
turned off with `pipeline.weekly_synthesis: false`. Entity mentions render as
`[[wikilinks]]`, so Obsidian's graph connects weeks that discuss the same CVE or
actor; `POST /api/v1/entities/notes` writes the matching note files under
`entities/`. Episodes that finished after their own week's digest appear under a
**carried over** marker rather than being passed off as new.

Writes are atomic (temp file + `os.replace`), so a sync client never sees a
partial file. An existing digest is **never overwritten**: a manual re-run writes
`-r2`, `-r3`. Every entry is labelled with its `basis` — `published transcript`,
`local transcription`, or `description only` — so you always know how much to
trust a summary.

See [`tests/fixtures/digest_golden.md`](tests/fixtures/digest_golden.md) for
exactly what it looks like.

Per-episode note files (`output.episode_notes: true`) are off by default.

### Reader marks, for something else to read

Stars and wrong-call flags are useful to the console and invisible to anything
else. Every Friday, half an hour after the digest, the marks made since the last
run are written to `signals/<week>.md` beside the digests: what you starred, what
you said was ranked too low, and what you said was not worth it — each with the
score the pipeline gave it, so a disagreement carries both numbers.

**Each file holds one period's new marks, not a running snapshot.** A mark stays
reported in the period it was made, and earlier files are never rewritten. That
keeps fifty near-identical files from accumulating and makes it obvious which
lines are new. Nothing is written in a week with no marks.

It is meant to be read by a model working over your vault, which is why the file
argues with itself in one specific way: it states plainly that an **unlisted
episode is not a rejection**. Most good episodes are never starred, and a reader
inferring dislike from absence would learn the exact opposite of the truth.

`POST /api/v1/signals/export` forces a run early. It writes only what is new, so
calling it twice does not repeat a mark.

---

## Operating it

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Liveness + CouchDB + scheduler. **No auth.** 503 when degraded |
| `GET /api/v1/status` | Counts by status, queue depths, feed health, backfill progress, last runs |
| `POST /api/v1/runs/ingest` | Poll feeds now (`?wait=true` to block) |
| `POST /api/v1/runs/pipeline` | Process pending work now |
| `POST /api/v1/runs/digest` | Generate a digest (`?since=`, `?until=`, `?dry_run=true`) |
| `POST /api/v1/backfill/rewind` | Clear archive cursors so the window is walked again |
| `POST /api/v1/runs/backfill` | Walk the archive (`?dry_run=true` by default, `?confirm=true` to spend) |
| `POST /api/v1/runs/rescore` | Re-score against the current interest profile (`?limit=`, `?force=`) |
| `POST /api/v1/runs/retention` | Run retention cleanup now |
| `GET/POST /api/v1/backfill/control` | Read, start or pause the archive walk |
| `GET /api/v1/episodes` | Paged list (`?status=`, `?podcast=`, `?starred=`, `?unread=`, `?flagged=`, score bounds) |
| `GET /api/v1/episodes/{id}` | One episode in full, including summary and traceback |
| `POST /api/v1/episodes/{id}/summarize` | Summarise one episode now (`?allow_asr=`, `?wait=`) |
| `POST /api/v1/episodes/{id}/retry` | Reset a failed episode to where it failed |
| `POST /api/v1/episodes/{id}/escalate` | Force full Tier-1 treatment for something dropped |
| `POST /api/v1/episodes/{id}/star` | Star or unstar |
| `POST /api/v1/episodes/{id}/read` | Mark read or unread (stored as a timestamp) |
| `POST /api/v1/episodes/{id}/feedback` | Flag the call as wrong (`over`/`under`) |
| `GET /api/v1/episodes/{id}/export` | The summary as a standalone Markdown file to share (409 if there is none) |
| `POST /api/v1/digests/{week}/narrate` | Read that week's digest aloud into an audio file beside it (`?force=`, `?wait=`) |
| `GET /api/v1/search` | Full-text search over summaries and transcripts |
| `GET /api/v1/search/status` · `POST .../sync` · `POST .../rebuild` | Search index state, incremental sync, full rebuild |
| `GET /api/v1/insights/precision` | Precision report over stars/reads/flags (`?days=`) |
| `GET /api/v1/entities` · `GET /api/v1/entities/{key}` | Named things across the corpus; one entity with its timeline |
| `POST /api/v1/entities/notes` | Write one Obsidian note per entity |
| `POST /api/v1/signals/export` | Mirror starred/flagged episodes into `reading-signals.md` |
| `GET /api/v1/content/seeds` | Preview which episodes qualify as writing material (no model call) |
| `POST /api/v1/content/seeds` | Find openings worth writing about → `content-seeds.md` |
| `GET /api/v1/digests` | Every digest generated, newest week first |
| `GET /api/v1/digests/{week}` | One digest: metadata, Markdown and rendered HTML |
| `GET/POST /api/v1/podcasts` | List or add podcasts |
| `PATCH /api/v1/podcasts/{slug}` | Override settings for one podcast |
| `DELETE /api/v1/podcasts/{slug}/overrides/{field}` | Revert one field to the `config.yaml` value |
| `POST /api/v1/podcasts/probe` | Check a feed URL before adding it |
| `GET /api/v1/logs` | Recent log events held in memory (level/text filters) |
| `GET /api/v1/logs/stored` | Warnings and errors kept in the database |
| `GET /api/v1/runs` | Job runs recorded, newest first |
| `GET /api/v1/runs/last` | When each job last ran — survives a restart |
| `GET /api/v1/telemetry/costs` | LLM cost/latency by provider, model, tier, day |
| `GET /api/v1/glance` | One-line headline for a small display |
| `GET /docs` | Swagger UI — behind the same API key |

Everything except `/healthz` and the console pages needs `X-API-Key`, compared in
constant time. Ten wrong keys from one address within five minutes gets that
address a `429` until the window drains — constant-time comparison stops timing
attacks and says nothing about volume. A correct key clears the count, so
mistyping it twice costs nothing. The counter is in memory only: an
unauthenticated request must not be able to make the service write to its
database.

**Bind it deliberately.** `api.host` defaults to `0.0.0.0`, which compose scopes
for you. Running natively — see [RUNNING-ON-MAC.md](RUNNING-ON-MAC.md) — set
`PODAGENT_API__HOST` to `127.0.0.1` or the LAN IP instead, rather than listening
on every interface.

### Schedule

| Job | Default |
|---|---|
| `ingest` | every 6 h (conditional GET, cheap) |
| `pipeline` | every 30 min (no-op when the queue is empty) |
| `digest_weekly` | Fri 06:00 Europe/Stockholm |
| `retention_cleanup` | daily 04:00 |
| `backfill` | every 20 min — **a no-op unless the archive walk is started** |
| `search_sync` | twice hourly (:15, :45) — incremental, no-op when nothing changed |

Each job is `max_instances=1` + `coalesce=True`, and manual triggers take the
same lock, so a slow run never overlaps the next firing.

### Reading the logs

Structured JSON on stdout, including third-party logging, so every line parses:

```bash
docker compose logs -f podcast-agent | jq -r 'select(.event=="tier0.routed")
  | "\(.route)\t\(.rule)\trel=\(.relevance) conf=\(.confidence)\t\(.title)"'

# One summary line per run
docker compose logs podcast-agent | jq 'select(.event | endswith("_summary"))'
```

Every line carries `run_id`; per-episode lines carry `episode_id` and `stage`.
API keys are redacted and transcripts truncated at the processor level, so a
stray log call cannot leak either. Full prompt/response logging is available
behind `llm.log_llm_io: true` at DEBUG only.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/api/v1/*` returns 503 "not configured" | `PODAGENT_ADMIN_API_KEY` unset. It fails closed on purpose |
| Container exits with "FATAL: invalid configuration" | Read the listed field paths; `extra="forbid"` means a typo'd key is fatal by design |
| `stages_deferred: ["triage"]` in a run summary | The tier's whole chain was unreachable — with the shipped cloud chain that means both OpenRouter and Anthropic failed (outage, bad key, no credit). Work stayed queued — nothing lost |
| Episodes stuck in `AWAITING_TRANSCRIPT` | Check `max_transcripts_per_run` against arrival rate, and whether the podcast has `asr_enabled` |
| Summaries say `description only` | That podcast has `asr_enabled: false` and publishes no transcript. Turn ASR on for it, or accept the label |
| `ingest.url_rejected` in logs | Audio/transcript host isn't the feed's domain and isn't allowlisted → add it to `security.cdn_allowlist`. Run `scripts/check-enclosure-chains.py` to see every host each podcast's audio actually passes through — **each hop** must be allowlisted, not just the first |
| `audio download rejected: '<host>' is neither…` | A host *inside* a redirect chain. Same fix; the script above names it |
| A feed stopped updating | `GET /api/v1/status` → `feeds[].circuit_open`. After 5 consecutive failures polling backs off to daily |
| Episode in `ERROR` | `GET /api/v1/episodes/{id}` for the traceback, then `POST .../retry` — it resumes where it failed |
| LLM timeouts at exactly the configured `timeout_s`, plus "Run time of job … was missed" | On a laptop, suspect sleep before suspecting the backend: the clock runs while nothing does. `pmset -g log \| grep Sleep` and compare. See [RUNNING-ON-MAC.md](RUNNING-ON-MAC.md) |
| Digest is empty | Episodes are probably still mid-pipeline (skipped, not lost). Check `queue_depths` |

---

## Archive backfill

Routine polling only ever looks *forward*. Reaching into the back catalogue is a
separate, deliberate job with its own thresholds, its own output, and a
confirmation step — it is the one operation that can consume days of compute by
accident.

**Always dry-run first.** The estimate comes from this deployment's own recorded
latencies, not generic guesses:

```bash
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/backfill?dry_run=true" | jq .result
curl -fsS -X POST -H "X-API-Key: $KEY" \
  "http://$HOST/api/v1/runs/backfill?dry_run=false&confirm=true" | jq .result
```

Without `confirm=true` a non-dry run is refused with a 400.

### Starting and pausing

The walk has a persisted run state, **paused by default**: a scheduled walk never
begins just because a config file was deployed.

```bash
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/backfill/control?paused=false"  # start
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/backfill/control?paused=true"   # pause
```

Once started, the `backfill` cron job advances one month-window per podcast per
firing until the window is exhausted. Pausing takes effect at the next podcast or
episode boundary, so the item in flight finishes rather than being discarded, and
the run reports `paused_mid_run: true` rather than implying it completed.

A manual run still proceeds while paused: the flag governs the *unattended* walk.

Progress lives in `GET /api/v1/status` under `backfill` — window, per-podcast
cursors, podcasts finished, episodes ingested, archive files written.

### Why it is affordable

~3,900 hours of archive audio, of which only ~590 h ship a published transcript;
transcribing the rest is weeks of NAS CPU. Backfill therefore **skips** episodes
without a transcript rather than queueing ASR it can never afford. Measured on
the 14 seeded feeds: one month-window is ~25 episodes and ~0.4 h of local LLM, so
the full 12 months is roughly 5 hours and **zero** transcription jobs.

### Per-podcast treatment

```yaml
  - slug: cyberwire-daily
    backfill_mode: tier0_only    # full | tier0_only | skip
```

`tier0_only` triages but never summarises — right for daily news, where a
two-year-old headline round-up earns an index line, not four minutes of LLM.
The seeded config sets it for CyberWire, Stormcast, Caveat and Hacking Humans.

Archive scoring uses `backfill.digest_threshold` (default 7, stricter than the
weekly 5): old material has to earn its summary. Summaries use archive-aware
prompts, so time-sensitive claims are anchored to the recording date rather than
presented as current.

### Output

`<digest_dir>/archive/<podcast-slug>/YYYY-MM.md`, one file per podcast-month, entirely
separate from the weekly digest — a 2019 episode can never appear in this week's
reading. A month is written only once every episode in it has been processed, and
files are never rewritten in place.

---

## Changing your interests later

The interest profile is hashed and that version stamped on every score, so it is
always answerable which profile produced a given verdict. After editing
`interest_profile`, `/api/v1/status` reports how many scored episodes predate the
change:

```json
"interest_profile": { "version": "b4863bc3334f", "stale_episodes": 12 }
```

Re-score them from stored transcripts — Tier-1 tokens only, no re-fetching and no
re-transcription:

```bash
curl -fsS -X POST -H "X-API-Key: $KEY" "http://$HOST/api/v1/runs/rescore?limit=50" | jq
```

Episodes already written into a digest are **not** re-scored: the Markdown on
disk would then disagree with the database. Re-scoring covers what has not yet
been published, and can promote a previously-low episode into the next digest or
demote one out of it.

**This depends on transcripts still existing.** `retention.transcript_days` is
therefore the real limit on how far back you can re-score. The shipped
`config.yaml` sets it to `0` — kept indefinitely — precisely because re-scoring,
search and every other feature built over the corpus reaches only as far back as
the transcripts do. (The code default without that line is 180 days.)

---

## Notifications

Off by default. When enabled, an episode scoring at or above `min_score`
(default 9) pushes to a self-hosted ntfy topic. Nothing else notifies — digest
availability deliberately does not, because a weekly notification stops being
read:

```yaml
notifications:
  enabled: true
  ntfy_url: http://ntfy.lan:8080
  topic: podcast-digest-alerts
  min_score: 9
```

Set `PODAGENT_NTFY_TOKEN` if your ntfy instance requires auth. Delivery failures
are logged and swallowed — the summary is already stored, and a dead notifier is
not a reason to fail a run.

---

## Backup and restore

The digests are Markdown in your synced vault and are already covered by that
sync. The state database is not, and losing it means re-fetching, re-transcribing
and re-summarising everything:

```bash
# Read it from .env — pasting a literal placeholder just yields curl error 401.
export COUCHDB_PASSWORD=$(grep -E '^PODAGENT_COUCHDB_PASSWORD=' .env | cut -d= -f2-)

./scripts/backup.sh /path/to/backups     # KEEP=14 by default
./scripts/restore.sh backups/podcast_agent-<stamp>.json.gz
```

Backups include the gzipped transcript attachments, which is what keeps a later
re-score cheap. `restore.sh` refuses to write into an existing database — it
restores under a new name so you verify before swapping, rather than discovering
afterwards that you merged two half-states. The restored database reports fewer
documents than the backup by exactly the number of Mango index design documents,
which the agent recreates at startup.

`scripts/backup.sh` needs `python3` to verify the dump before publishing it.
**The QNAP has no python3**, so the nightly job there runs `qnap/backup-nas.sh`
instead — same output, verified with curl/gzip/sed. The cron line and the
`.backup-env` it reads are in [DEPLOY-NAS.md](DEPLOY-NAS.md#nightly-backup).

On any host that does have python3:

```cron
30 3 * * * cd /path/to/podcast-digest && . ./.backup-env && ./scripts/backup.sh ./backups
```

`.backup-env` is a `chmod 600` file holding one line, `COUCHDB_PASSWORD=…`, so the
password is not sitting in the crontab.

### Moving to another machine

> **Moving to the QNAP NAS specifically?** Use
> **[DEPLOY-NAS.md](DEPLOY-NAS.md)** instead of this section. That machine has
> no `scp`, no `python3`, a `bash` that is really `sh`, and a `docker` that is
> not on `PATH` — enough differences that the generic steps below mislead more
> than they help.

The compose stack is the deployment, so a move is a copy plus a restore. Do it
in this order, and the corpus comes with you — transcript attachments ride the
backup, so nothing is re-fetched, re-transcribed or re-summarised:

```bash
# On the machine you are leaving
export COUCHDB_PASSWORD=$(grep -E '^PODAGENT_COUCHDB_PASSWORD=' .env | cut -d= -f2-)
./scripts/backup.sh ./backups
#   copy across: the backup file, .env, config.yaml

# On the new machine — this is the NEW host's password, not the old one's
$EDITOR .env                    # DIGEST_DIR → the synced vault folder there;
                                # both model-provider keys must be present
export COUCHDB_PASSWORD=$(grep -E '^COUCHDB_PASSWORD=' .env | cut -d= -f2-)
./scripts/restore.sh backups/podcast_agent-<stamp>.json.gz
docker compose up -d --build
curl -fsS http://<new-host>:8080/healthz | jq
```

Nothing needs pulling on the new machine: as shipped both tiers are cloud, so
the move carries no model weights. If you have restored an Ollama primary,
`docker compose --profile ollama up -d` and pull the models named in
`config.yaml` before the first run.

**Check the stored settings override before trusting the new host.** The console
can override `llm` (among other sections), and that override lives in CouchDB —
so it rides the restore and is applied at boot on top of `config.yaml`. A stale
override pointing at an Ollama host that no longer exists will quietly defeat
the config you just copied:

```bash
curl -fsS -H "X-API-Key: $KEY" "http://$HOST/api/v1/settings" \
  | jq '{overrides: .overrides.llm, chains: (.tiers | map_values(.active_chain))}'
```

`active_chain` is what the running process would actually try, in order, after
any stored override — so it is the answer to "did my config.yaml win?".

Then run one ingest and one pipeline by hand and compare `/api/v1/status`
against the old machine's before switching off the old one. `restore.sh` writes
to a new database name rather than over an existing one, so you verify before
swapping rather than discovering afterwards that two half-states were merged.

Remember the two things that are *not* in the backup: `config.local.yaml` is
generated and belongs to the machine that generated it, and the search index is
a derived cache — `POST /api/v1/search/rebuild` recreates it in seconds.

### Rotating keys

```bash
# Admin API key: no data impact
openssl rand -hex 32                      # update PODAGENT_ADMIN_API_KEY in .env
docker compose up -d podcast-agent

# CouchDB password: must change in both places at once
docker compose exec couchdb-podcast \
  curl -sS -X PUT "http://127.0.0.1:5984/_node/_local/_config/admins/podagent" \
  -u "podagent:$OLD" -d '"NEW_PASSWORD"'
# then update COUCHDB_PASSWORD in .env and:
docker compose up -d
```

---

## Security posture

**Threat model:** LAN-only single-user homelab service. The real risks are secret
leakage, processing untrusted internet content, prompt injection from feed
content, and supply chain. [architecture.md](architecture.md#8-untrusted-input)
covers the mechanisms; the operational rules are:

- **Do not port-forward this.** No TLS, no user accounts. CouchDB has no `ports:`
  and sits on an internal-only Docker network, unreachable from the LAN — that
  is the property the compose network layout exists for, and a test asserts it.
  Whoever can write to that database can change where model work is sent.
- The console pages are served without the key, but carry no data — every value
  arrives over the authenticated API. They are still LAN-only surfaces.
- **Where prompts and transcripts go is a deploy-time decision.** A tier's
  `api_base` may only name a host `config.yaml` already names, loopback, or the
  provider's own endpoint; the console cannot introduce a new one.
- **Outbound fetches are checked at every redirect hop**, not just at the URL a
  feed supplied, and a host resolving to a private address is refused — an
  allowlisted CDN cannot bounce the agent onto the LAN.
- **Container:** non-root UID 10001, read-only root filesystem, `cap_drop: ALL`,
  `no-new-privileges`, memory/CPU limits (ASR is the memory hog — 4 GB).
- **Supply chain:** base images pinned by digest, Python dependencies installed
  with `--require-hashes`. Refresh monthly:

  ```bash
  uv lock --upgrade
  uv export --format requirements-txt --extra asr --no-dev --no-emit-project \
    -o requirements.lock.txt
  uv run pytest && docker compose build
  ```

### Data retention

Transcripts are kept for `retention.transcript_days` — the shipped config sets
`0`, indefinitely, because search and re-scoring reach only as far back as the
transcripts do; **summaries are kept indefinitely, and episodes are never
deleted at all.** Telemetry is kept 365 days. Downloaded audio is deleted
immediately after transcription, and orphans from interrupted runs are swept
after 12 hours. Only public content plus your interest profile and reading
signals (stars, read marks, wrong-call flags) is stored.

---

## Known limitations

- Podcasts without an audio RSS feed (YouTube-only) are unsupported and logged as such.
- No semantic search or embeddings; scoring is prompt-based and search is
  full-text (FTS5), not similarity-based.
- The interest profile never changes itself. Stars, read marks and wrong-call
  flags feed a precision report that *suggests* config edits with its evidence;
  applying one is always your edit, never the system's.
- `asr.backend: remote` is a declared interface with no implementation yet;
  `local` is the working backend.
- ASR on QNAP CPU is slow (below real time). That is fine — everything is queued
  and asynchronous — but a week's escalations can take hours of wall time.
