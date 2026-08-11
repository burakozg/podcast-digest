# Architecture

How the agent is actually built, and why. This describes the system **as it
stands**; [`podcast-digest-agent-design.md`](podcast-digest-agent-design.md) is
the original v1 design and is left unedited as the historical record. Where the
two differ, this document says so and explains the reason —
[Deviations](#deviations-from-the-v1-design) collects them.

For operating instructions see [README.md](README.md).

---

## 1. The shape of the problem

Episode descriptions are unreliable. Some are a thorough rundown; many are a
sponsor read and a guest's job title. Filtering on them alone silently discards
good episodes, and the failure is invisible — you never learn what you missed.

Transcribing everything would solve that and is unaffordable: the fourteen
configured podcasts hold roughly 3,900 hours of archive audio, and local ASR on a
NAS runs at around real time.

So the system spends a little on everything and a lot on almost nothing. Tier-0
judges **two** numbers per episode — how relevant it looks, and *how informative
the description was* — and only the second decides whether to pay for a
transcript. A vague description is never grounds to drop an episode; it is
grounds to go and find out.

Everything else follows from that: the tiering, the routing table, the queues,
the cost guardrails.

---

## 2. Pipeline

```
                        ┌──────────────────────────────────────────┐
 RSS feeds ────────────▶│ Ingestion            episode docs = NEW  │
 (14 podcasts)          └──────────────┬───────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────────┐
                        │ Tier-0 triage   relevance + confidence   │
                        └──────────────┬───────────────────────────┘
                                       ▼
                          route decided in code, not by the model
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
        DROPPED                  DIGEST_DIRECT                ESCALATE
     confidently                grey zone: listed         relevant, OR the
     irrelevant,                as a one-liner,           description was too
     kept for audit             never transcribed         thin to judge
                                                               │
                                        transcript acquisition ◀┘
                              feed transcript → configured scrape → ASR
                                                               │
                                       ┌───────────────────────┴──────────┐
                                       ▼                                  ▼
                                  TRANSCRIBED                    TRANSCRIPT_FAILED
                                       └───────────┬──────────────────────┘
                                                   ▼
                                    Tier-1 summarise + score
                                                   │
                                 ┌─────────────────┴─────────────────┐
                                 ▼                                   ▼
                          READY_FOR_DIGEST                      SCORED_LOW
                                 └─────────────────┬─────────────────┘
                                                   ▼
                                       weekly Markdown digest
```

Archive backfill runs the same stages with different economics — see §7.

### Stage ownership

| Stage | Module | Reads | Writes |
|---|---|---|---|
| Ingestion | `ingest/feeds.py` | RSS | `episode` docs at `NEW` |
| Triage | `triage/tier0.py` + `triage/routing.py` | title, description | `tier0` block, `TRIAGED` |
| Dispatch | `triage/tier0.py` | stored route | `DROPPED` / `DIGEST_DIRECT` / `AWAITING_TRANSCRIPT` |
| Transcripts | `transcripts/` | feed transcript, page, audio | transcript attachment, `TRANSCRIBED` |
| Summarise | `summarize/tier1.py` | transcript or description | `tier1` block, classified status |
| Digest | `digest/generate.py` | claimed episodes | Markdown file, `digest` doc, `PUBLISHED` |

---

## 3. Invariants

These are the properties the design leans on. Each is enforced by a test, not by
discipline — the test name is given so the guarantee is traceable.

**The model proposes; code disposes.** Routing is computed from validated integer
fields only. `Tier0Result.route` — the model's own suggestion — is stored for
prompt evaluation and never acted on. A description that argues for its own
importance cannot change where it goes.
→ `test_routing.py::test_model_suggested_route_is_ignored`

**Every status change is legal or raises.** `state.py` holds one transition map;
`episodes.transition()` is the only writer. An illegal transition is a bug in a
stage and fails loudly rather than corrupting state.
→ `test_state.py::test_every_status_has_a_transition_entry`

**Provenance is code-set.** `summary_basis` records whether a summary came from a
transcript or only a description. It is deliberately *not* a field the model
returns, because untrusted output must not be able to relabel its own
trustworthiness — the digest shows that label to the reader.
→ `test_stages.py::test_basis_is_not_taken_from_the_model`

**A missing field matches nothing.** CouchDB's Mango holds no index entry for a
document that lacks a field, so *every* comparison against it fails — including
negative ones. `{"origin": {"$ne": "backfill"}}` does not match a document with
no `origin`, which is what a routine episode is. `MemoryStore` mirrors this
exactly, distinguishing absent from present-and-null, because the in-memory
store comparing `None != "backfill"` in Python is what let this selector pass
every test and match nothing in production. Excluding archive material goes
through `backfill.NOT_BACKFILL`, which carries the `$exists: false` arm.
→ `test_store.py::TestMangoSubset::test_ne_does_not_match_a_missing_field`,
  `test_backfill.py::TestRoutineEpisodesAreFoundAtAll`

**Recent episodes outrank history.** Every routine stage queues newest-first,
and the archive walk does not start — nor continue — while any non-archive
episode is still owed pipeline work. That includes one queued for local
transcription, even though a month of transcript-only archive items would finish
in a fraction of the time: the ordering is recency, not throughput. `ERROR` is
excluded from the check, so a single poisoned document cannot halt the archive
forever, and a manual run takes `force=true`.
→ `test_backfill.py::TestRecentWorkComesFirst`, `::TestQueueOrdering`

**The archive indexes everything and summarises selectively.** Historical intake
ingests every archive episode and triages it; whether it is then summarised
depends on having a transcript — published, or made because that podcast is set
to transcribe locally. An escalated episode with no
published transcript is downgraded to an index entry, because escalating it
would fail acquisition and then draw a Tier-1 call to summarise a description.
The two used to be conflated at ingestion, which silently discarded ~92% of the
archive and left a weekly podcast showing one episode.
→ `test_backfill.py::TestIndexedButNotSummarised`

**A weekly digest holds its own week, plus what finished late.** Selection
filters on three independent grounds: `digest_id` is null (claimed exactly once,
ever), `origin` is not `backfill`, and `published_at` falls inside the window.
The window's floor is not the period start but
`min(period_from, period_to - digest_catch_up_days)`: an episode that was still
mid-pipeline when its own week's digest ran is picked up by a later one — marked
**carried over**, never passed off as new — instead of falling outside every
window forever. (That stranding was real: 70 episodes before the fix.) Starting
a historical intake mid-week still cannot alter a digest — archive material goes
to the per-podcast archive files instead.
→ `test_digest.py::test_it_is_marked_carried_over_rather_than_passed_off_as_new`

**A job never overlaps itself, even across processes.** The in-process
`asyncio.Lock` covers the hourly case (a scheduled fire plus an impatient
click); a CouchDB lease document (`control:lock:<job>`, TTL 15 min, renewed by
heartbeat) covers what the lock cannot see — a second instance or a CLI run.
Acquisition is a create-if-absent write, so exactly one of a simultaneous pair
wins. Stale leases from a dead local process are reclaimed at boot; shutdown
drains running jobs before the store closes so the release write can land.
→ `test_joblock.py::test_exactly_one_of_a_simultaneous_pair_wins`

**Digest HTML reaches the console inert.** Digest text is LLM output, which is
downstream of feed descriptions and transcripts, and Markdown escaping says
nothing about HTML safety. Rendering runs with raw HTML disabled *and* filters
the result against a tag allowlist, so neither a preset change nor an allowlist
bug is sufficient on its own. Digest file paths come off documents and are
resolved strictly inside the configured digest directory.
→ `test_digest_read.py::TestMarkdownRendering`, `::TestPathConfinement`

**Nothing is deleted.** No code path deletes an episode document. Retention
expires transcript *attachments* and telemetry; the episode and its summary
persist. Podcasts are disabled, never removed, so every episode's origin stays
explainable.
→ `test_api.py::test_no_podcast_can_be_deleted`

**One vendor boundary.** Only `llm/` imports litellm or instructor. Stages depend
on the `StructuredLLM` protocol, which keeps provider churn contained and lets
the whole pipeline be tested without loading a vendor SDK.
→ `test_boundaries.py::test_nothing_outside_llm_imports_litellm_or_instructor`

**Resumable everywhere.** Every stage is driven by document status, so the
process can die at any point and the next run continues. Per-episode exceptions
are recorded on the document and the run carries on.
→ `test_pipeline_e2e.py::test_poison_pill_does_not_block_the_queue`

**Routine ingestion only looks forward.** Reaching backwards is the backfill
job's business alone, and that job is capped, estimated and confirmed first.
→ `test_backfill.py::test_regular_ingest_never_walks_backwards`

---

## 4. Data model

Single CouchDB database, documents discriminated by `type`.

| Type | Id | Holds |
|---|---|---|
| `podcast` | `podcast:<slug>` | Poll state (etag, failures), feed `description`, console `overrides`, backfill cursor |
| `episode` | `episode:<sha256>` | Metadata, `tier0`, `tier1`, status, transcript attachment |
| `digest` | `digest:<YYYY-Www>` | Weekly digest: period, file path, episode ids, `marking_complete` |
| `archive` | `archive:<slug>:<YYYY-MM>` | Archive month: same, per podcast-month |
| `llm_call` | `llmcall:<uuid>` | One row per LLM invocation: tokens, latency, cost, fallback |
| `run` | `run:<uuid>` | One row per job firing: job, time, result summary |
| `asr_run` | `asrrun:<uuid>` | One row per local transcription: audio and compute seconds, model, device |
| `log` | `log:<uuid>` | Kept warnings/errors (30 days); the live tail stays in memory |
| `control` | `control:backfill` | Whether the unattended archive walk is paused |
| `control` | `control:settings` | Console-edited configuration overlay (§6b) |
| `control` | `control:lock:<job>` | Cross-process job lease: holder host/pid, expiry, heartbeat |

Episodes additionally carry reader signals written by the console: `starred`,
`read_at` (a timestamp, not a flag) and a `feedback` block recording an explicit
"this call was wrong" verdict alongside what the pipeline had judged at the
time. They feed the insights report (§7b) and are never read by the pipeline
itself.

Episode ids are `sha256(podcast_slug + guid)` — stable forever for a given
(podcast, guid) pair. That single choice is what makes ingestion idempotent: two
concurrent runs seeing the same episode cannot both create it, because insertion
is create-if-absent on that id.

Transcripts are gzipped attachments rather than fields, keeping the JSON body
small and Mango indexes fast.

### Querying

CouchDB requires the sort fields of a Mango query to be a **prefix of a real
index**. Sorting by `published_at` alone fails with `no_usable_index` even when an
index on `(type, published_at)` exists. Every sorted query therefore pins `type`
in its selector and leads the sort with it, via `db.base.typed_sort()`.

This was found the hard way: the in-memory test double sorted anything happily,
so every `/episodes` query passed in tests and returned HTTP 500 in production.
`MemoryStore` now enforces the same rule, so an unsortable query fails in tests
too.

Index selection is **pinned, not left to the planner**: `db/base.py` resolves
which declared index a selector can use and passes it as `use_index`, so a
CouchDB upgrade cannot silently change a query plan from an index scan to a full
scan. A selector no declared index can serve raises `StoreError` — in tests,
via the same check in `MemoryStore` — rather than degrading quietly in
production.

**`use_index` names a design document, not an index.** CouchDB names a design
document after a hash of its contents unless told otherwise, so pinning by index
name asked for a document that did not exist: the pin was ignored and the
planner chose for itself, correctly enough that nothing ever failed. The only
symptom was one warning per query shape saying the document "does not contain a
valid index for this query" — which reads like a bad index rather than a missing
document. Indexes are therefore created with `ddoc` equal to their name, and
hash-named duplicates from earlier deployments are dropped at startup. Worth
knowing when debugging this: `_explain` resolves a bare name where `_find` does
not, so a query can look healthy under `_explain` and still be unpinned.
→ `test_store.py::TestPinningAnIndexActuallyBinds`

---

## 5. The LLM boundary

`llm/client.py` is the only module that knows litellm exists. Everything else
depends on:

```python
async def complete_structured(tier, system, user, response_model) -> tuple[T, CallMeta]
```

**Fallback is walked here, not delegated.** The design requires failover on
"timeout, connection error, 5xx, **or 2× validation failure**". litellm's Router
can only fail over on exceptions, so it cannot express the validation-count
trigger. The Router is used as transport, with one deployment group registered
per endpoint (`tier0`, `tier0__fb0`, …) so a call can target an exact endpoint,
and the chain is walked in our code.

**Classification unwraps to the root cause.** instructor wraps *every* failure —
transport included — in `InstructorRetryException`. Taking that at face value
made timeouts count as validation failures, retrying a dead endpoint instead of
failing over. `_root_cause()` unwraps before deciding.

**Cloud endpoints are removed, not skipped.** With `allow_cloud_fallback: false`
a tier's cloud endpoints are absent from the chain entirely, so no code path can
route to them. Work queues instead of leaving the LAN.

**A failed endpoint is stepped over briefly.** The chain is re-walked from the
primary on every call, so a backend that is down costs a full timeout per
episode — which is why a deferred stage drained so slowly once it returned. A
*transport* failure marks that endpoint for 60 seconds; a validation failure does
not, because a model emitting bad JSON is still answering. When every endpoint is
cooling the whole chain is walked anyway: the cooldown exists to avoid a wasted
timeout and must never turn "slow" into "unavailable".
→ `test_llm_client.py::TestEndpointCooldown`

Prompts are versioned files (`prompts/tier0_v1.md`). Changing behaviour means
adding `_v2`, never editing a shipped version, so a recorded `prompt_version` in
telemetry always identifies what actually ran. Tier-1 is on v2 (archive-aware
framing); `tier1_map` remains v1 because it did not change.

---

## 6. The podcast list

The v1 design made `config.yaml` the source of truth for podcasts. That was right
while editing a file was the only way to change one, but `config.yaml` is mounted
**read-only** in the container, so a management console cannot write to it.

As built: **config.yaml is the declared baseline; the database holds overrides
and additions.** `podcasts.PodcastRegistry` merges them and is refreshed at the
start of every run, so a console change takes effect on the next run rather than
the next restart.

- A field edited in the console stops tracking `config.yaml` **for that podcast and
  that field only**. Each record reports which fields are overridden, so a value
  that no longer matches the file is always explainable, and reverting is one
  request.
- Podcasts added in the console exist only in the database (`source: "console"`).
- Podcasts are **disabled, never deleted**. Deletion would leave episodes whose
  origin cannot be explained, and for a config-defined podcast it would be undone
  at the next startup anyway.

### Per-podcast ASR

`asr_enabled` decides whether the routine pipeline may transcribe a podcast's audio
when it publishes no transcript. Off by default; on for the five `always_escalate`
podcasts, none of which publishes transcripts and which between them fill Top picks.

Archive backfill requires **both** `asr_enabled` **and**
should not be able to turn a five-hour archive walk into a three-thousand-hour
one.

---

## 6b. Console-editable configuration

The same read-only-file constraint applies to configuration as to the podcast list.
A `control:settings` document holds a deep-merged override layer, restricted to
LLM tiers, routing thresholds, the interest profile, ASR and notifications.

Storage, network binding, security allowlist, output paths and the schedule are
**not** overridable: a typo in a browser should not be able to make the service
unreachable or unable to find its own database.

**Where model work is sent is not console-editable either.** A tier's
`api_base` decides where every prompt and transcript goes, so whoever can set it
receives everything the pipeline reads. An override's endpoints must name a host
that `config.yaml` already names, loopback, or the provider's own endpoint;
anything else is refused with instructions to edit the file and restart. The
allowed set is derived from the *file* baseline rather than the running
configuration — otherwise a stored override, once applied at boot, would appear
in the baseline and authorise itself from then on. It is checked again in the
lifespan, because the document is writable by anything with database access.
→ `test_config.py::TestModelEndpointsAreConfinedToTheFile`

Overrides apply at **startup**, in two phases — the database connection comes
from the file, and the overrides come from the database, so settings are built
once from the file, then rebuilt with the stored overlay. A stored override that
no longer validates logs loudly and is ignored rather than bricking the service.

Changes are validated by rebuilding the entire `Settings` object before anything
is written, so a change that could not boot is refused at the point of saving.

Not applied live, deliberately: swapping a provider rebuilds the LLM router and
editing the interest profile invalidates every cached prompt. Doing either
underneath an in-flight summarisation is a poor trade for a setting that changes
a few times a year.

---

## 7. Archive backfill

Routine polling looks forward; the archive is walked backwards by a separate job
with its own thresholds, output and confirmation step.

- A per-podcast **month cursor** on the podcast document walks backwards one
  window per run, so a run is bounded, resumable and stoppable. The floor is
  computed from a **pinned anchor month** recorded on the first run, not from
  "now": without the anchor, a podcast added later had its window drift forward
  with the calendar, so the walk stopped short of the months it was asked for
  while the tracker reported it complete.
- Episodes carry `origin: "backfill"`, which excludes them from the routine
  queues and the weekly digest entirely — a 2019 episode cannot appear in this
  week's reading.
- **Transcript-only by default.** An episode with no published transcript is
  skipped rather than queued, enforced structurally by `allow_asr=False` at the
  acquisition boundary rather than by policy anyone can forget.
- `backfill_mode` per podcast: `full`, `tier0_only` (index line, no summary — right
  for daily news) or `skip`.
- A **dry run** estimates cost from this deployment's own recorded latencies and
  escalation rate; with no history it says so rather than inventing a number.
  Spending compute requires `confirm=true`.
- Pausing is cooperative: checked between podcasts and between episodes, so a pause
  costs at most the item in flight.

Output is one file per podcast-month under `archive/<slug>/YYYY-MM.md`. A month is
written only once every episode in it has been processed, since these files are
never rewritten in place.

**A measurement that changed the plan.** The roadmap assumed most feeds truncate
to the last 50–100 episodes and proposed a PodcastIndex integration to reach past
that. Measured against the fourteen configured feeds, exactly one truncates; the
rest expose their full history. The integration was unnecessary and is not built.
The real constraint was transcription hours, not feed depth.

---

## 7b. Built over the corpus

Once the corpus existed, several features became cheap because they read what
the pipeline already paid for. They share two design rules: **derived, never
authoritative** — each can be deleted and rebuilt, and no pipeline decision
reads any of them — and **summaries, not transcripts**, so each is one model
call (or none) regardless of how many episodes it covers.

**Full-text search** (`search.py`) — a SQLite FTS5 sidecar at
`work_dir/search.db` over summaries and transcripts. Synced incrementally twice
an hour by signature hash (zero attachment reads when nothing changed);
rebuildable atomically from scratch. Explicitly a cache: it going stale degrades
a search box and nothing else.

**Weekly synthesis** (`digest/synthesis.py`) — the digest's opening section:
cross-episode themes with episode references, built from the week's summaries in
one call. Skipped, not failed, when the model is down or the week is thin
(< 3 episodes); themes from recent weeks are fed back so the opener does not
repeat itself.

**Entities** (`entities.py`) — named things (CVEs, vendors, groups) aggregated
from Tier-1 output across the corpus, with canonicalisation (CVE formats, `The`
prefixes, corporate suffixes) so "Volt Typhoon" and "VOLT TYPHOON" are one
thing. Digest mentions become `[[entity]]` wikilinks; on request, one Obsidian
note per entity is written under `entities/` with its episode timeline. No model
call — it is pure aggregation.

**Reader signals and the precision report** (`feedback.py`, `insights.py`) —
star / mark-read / "this call was wrong", written by the console onto episode
documents, and a report that reads them back: per-podcast and per-interest
precision with suggested config edits. Three refusals define it: nothing below a
minimum sample (8), an unstarred episode is only ever weak evidence, and
**nothing is ever applied automatically** — every suggestion is a diff for a
human, carrying its numbers. `enabled: false` is the strongest demotion it will
ever propose; deletion is never suggested, per the nothing-is-deleted invariant.

**Content seeds** (`content.py`) — "is there anything here worth writing
about?": one call over recent high-scoring summaries producing per-episode
angles and multi-episode threads, written to `content-seeds.md`. Off until
configured with the interests you actually write about. Episodes are referenced
by list number, never by title, and an angle whose reference does not resolve is
dropped rather than rendered — an unattributable claim is exactly what this must
not produce.

---

## 8. Untrusted input

Feed content, scraped pages, transcripts and LLM output are all treated as
hostile.

- **Sanitisation happens at validation time**, inside the Pydantic models, so no
  rendering path can forget it. HTML is stripped; `<script>`/`<style>` bodies are
  removed wholesale rather than flattened to text.
- **Prompt injection** is mitigated in three layers: system prompts declare the
  fenced content untrusted data; all output is schema-validated; and model output
  cannot influence control flow, because routing derives from validated integers
  only.
- **Structure cannot be hijacked.** Model-authored headings are demoted and
  frontmatter delimiters neutralised, so a summary cannot restructure the digest.
- **Feed XML is parsed with `defusedxml`** — stdlib ElementTree is vulnerable to
  entity-expansion DoS.
- **Outbound fetches** are scheme-checked, redirect-capped, byte-capped
  mid-stream (against a lying `Content-Length`), content-type-checked, and
  restricted to the feed's registrable domain or the CDN allowlist. The guard
  fails closed. Audio streams to a quarantined directory, never to memory.

The allowlist governs where a feed's *audio and transcripts* may come from, not
where the feed itself lives, so a self-hosted feed on a LAN address works.

**Redirects are walked in `net.py`, not by httpx.** Checking only the URL a
fetch starts at leaves the guard inspecting one host while the transport
connects to another — up to five times, silently. Enclosure chains really are
four deep through analytics prefixers (`swap.fm → mgln.ai → podtrac → prxu.org`),
and any host in such a chain can answer with a `Location` of its choosing. Every
request goes out with `follow_redirects=False` and each hop is re-checked before
it is followed; a resumed download re-walks from the original URL, because CDN
targets are signed and short-lived. Feed-supplied targets pass the full guard at
every hop; the feed URL itself keeps every check but the CDN allowlist, since
feeds legitimately move between domains.
→ `test_net.py::TestEveryRedirectHopIsChecked`

**A name that resolves inward is refused.** The allowlist reasons about names;
this is the arm that reasons about where a name actually goes. Each hop's host
is resolved and rejected if *any* answer is private, loopback, link-local,
reserved or multicast — every answer, because otherwise which address wins is
decided by the resolver's ordering. Exempt: a host equal to the feed's own, so
self-hosting a feed beside the agent still works, and everything when
`enforce_domain_allowlist` is off, so the escape hatch turns the guards off
together rather than half of them. Accepted residual: the address is checked and
connected to separately (TOCTOU); closing that needs a pinning transport.
→ `test_net.py::TestPrivateAddressesAreRefused`

---

## 9. Observability

Structured JSON on stdout, everything through one processor chain — including
third-party stdlib logging from uvicorn and httpx, because a single plain-text
line interleaved with JSON breaks `docker logs | jq`.

Every line carries `run_id`; per-episode lines carry `episode_id` and `stage`.
Secrets are redacted at the processor level, and bulk content is truncated.

Redaction is precise, not eager: an early version matched `token` inside
`input_tokens` and silently destroyed the per-call token counts that the cost
telemetry exists to collect.

The `llm_call` documents are the observability substrate. `/status`,
`/telemetry/costs` and the backfill cost estimate are all views over them.

---

## 10. Testing

1,173 tests. No test reaches the network, a real CouchDB, or an LLM.

- Storage is `MemoryStore`, which mirrors CouchDB's MVCC conflicts, collation
  order and index requirements — a double that is *more* strict than production
  in the places where being lax would let bugs through.
- HTTP is `respx`-mocked; fixture feeds mirror real publisher quirks.
- The LLM is faked at the `complete_structured` boundary, with `test_llm_client.py`
  covering the real client's failover machinery directly against a stubbed
  transport.

**Where bugs escaped.** Every defect found by running the system for real lived
in a seam the tests stubbed: logging configuration nothing called, a Mango
constraint the double didn't model, an exception path only the vendor SDK
produced. Each fix landed with the regression test that would have caught it.

---

## 11. Deviations from the v1 design

| Design | As built | Why |
|---|---|---|
| §6 config.yaml is the source of truth for podcasts | config is the baseline; DB holds overrides and additions | config.yaml is read-only in the container, so a console cannot write to it |
| §4 `summary_basis` is a Tier-1 output field | code-set, not model-returned | untrusted output must not relabel its own provenance |
| §7 litellm Router fallbacks | Router as transport, chain walked in our code | Router cannot express "fail over after 2× validation failure" |
| §14 PodcastIndex needed for archive depth | not built | measured: 13 of 14 feeds expose full history |
| §10.4 routine ingest capped only on a fresh database | cutoff is the oldest episode already seen | otherwise the second poll swept months of back catalogue silently |
| §2 no interactive UI in v1 | seven console pages | operational need; the reading surface is still Obsidian |

---

## 12. Layout

```
podcast_agent/
  main.py          app assembly, lifespan, entrypoint
  config.py        layered settings; all validation lives here
  settings_store.py console override layer: deep-merge, allowlisted sections
  podcasts.py      registry merging config with database overrides
  state.py         status enum + the allowed-transition map
  models.py        LLM response schemas (sanitising validators)
  sanitize.py      untrusted-content handling
  net.py           outbound guards, byte caps, domain allowlist
  episodes.py      guarded status transitions
  joblock.py       cross-process job leases (control:lock:*)
  scheduler.py     APScheduler wiring + graceful job draining
  search.py        SQLite FTS5 sidecar over summaries and transcripts
  feedback.py      reader signals: star, read, wrong-call verdicts
  insights.py      precision report over the signals; suggests, never applies
  entities.py      named-thing aggregation, wikilinks, entity notes
  content.py       content seeds: openings worth writing about
  retention.py     transcript/telemetry expiry, audio sweeping
  migrate.py       startup document migrations
  notify.py        exceptional-item push
  logstore.py      kept warnings in CouchDB; logbuffer.py is the live tail
  db/              Store protocol, CouchDB client, in-memory double
  ingest/          RSS polling, idempotent episode creation
  triage/          tier0.py (the call) + routing.py (the decision)
  transcripts/     acquire, normalize, asr, stage
  summarize/       tier1.py + chunking.py (map-reduce)
  digest/          generate.py, synthesis.py, archive.py, read.py, templates
  backfill/        ingest, process, estimate, control
  llm/             the ONLY place litellm/instructor are imported
  api/             routes, podcast management, auth, console pages
  prompts/         versioned prompt files
```
