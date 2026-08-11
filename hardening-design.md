# Hardening design — security mitigations and pipeline improvements

Implementation spec for the findings of the 2026-08 architecture and security
review. Written to be executed by an implementing agent without further
discussion; where an item changes observable behaviour it says so and the change
is **approved by this document**.

Threat model context (unchanged): LAN-only, single-user, self-hosted, never
port-forwarded. The service's inputs of concern are RSS feeds, podcast CDNs,
scraped pages and LLM output — all treated as hostile. See
[architecture.md §8](architecture.md#8-untrusted-input).

---

## Ground rules for the implementer

- The gate is `./scripts/check.sh` (`ruff format --check`, `ruff check`,
  `mypy podcast_agent`, `pytest -q`). Every item lands green.
- Work on a new branch off `fix/pipeline-debt-batch`.
- `StrictModel` (`extra="forbid"`) — any new config key must be added to the
  model *and*, if console-editable, to the allowlists in `settings_store.py`.
  Default to **not** console-editable for anything in this document: these are
  security settings, and §6b's rule is that a browser typo must not weaken them.
- `MemoryStore` must mirror any behaviour a test depends on. If an item adds a
  query shape, `db/base.py` index resolution must accept it (`check_indexable`
  fails in tests otherwise — that is by design).
- No test may reach the network; HTTP is `respx`-mocked.
- Console text obeys the vocabulary tests in `tests/test_console_assets.py`
  (say "podcast" not "show"; no Tier-0/Tier-1/ASR in prose; config keys only
  inside `<code>`).
- Tests are named for the behaviour they pin, in the style of the existing
  suite (`test_the_first_taker_wins`, not `test_lock_1`).
- Nothing is deleted, ever — no item below may remove an episode, podcast or
  summary.

Priorities: **P0** before the next deploy; **P1** soon after; **P2** when
convenient; **P3** optional.

## Status — implemented 2026-08-01, branch `hardening/security-batch`

| | Item | Outcome |
|---|---|---|
| H1 | Guard-checked redirect hops | ✅ |
| H2 | Private-address check per hop | ✅ |
| H3 | Auth throttling + bind guidance | ✅ |
| H4 | `api_base` confined to the file baseline | ✅ (400, not 422 — see below) |
| H5 | Console escaping audit + regression test | ✅ (audit found no live hole) |
| H6 | Archive transcription budget | ✅ (own `backfill` key — see below) |
| H8 | LLM endpoint cooldown | ✅ |
| H9 | Bounded cloud concurrency | **Not done, deliberately** — see below |
| H10 | Bridge networking, compose canonical | ✅ |

**Deviations, and why.**

- **H6 uses `backfill.max_transcripts_per_run`, not `pipeline`'s.** The spec
  said to reuse the pipeline knob. `BackfillConfig`'s own docstring says its
  caps are separate from `pipeline` throughout, and reaching across sections
  would have been the first break of that. Same fix, invariant intact.
- **H4 returns 400, not 422.** Every other `OverrideRejected` in the API is a
  400; matching the codebase beat matching the spec on a status code.
- **H9 is not implemented.** The gate is that it must not complicate the
  deferral semantics disproportionately, and it does: `_stage_summarize`'s
  contract is per-episode error isolation plus "first `LLMUnavailable` stops the
  stage with the queue intact", and running episodes concurrently makes both
  nondeterministic — which episode's failure wins, and how many were already
  in flight when it did. Against that, the benefit here is close to zero: this
  deployment runs Ollama on one GPU, where concurrency thrashes rather than
  helps, and cloud is a fallback rather than the normal path. Worth revisiting
  only if a cloud tier becomes the primary — F1's remote inference would be the
  moment.

---

## H1 (P0) — Every redirect hop passes the URL guard

**Problem.** `UrlGuard.check` vets only the URL the fetch *starts* at
(`transcripts/acquire.py:170,202,246`, `ingest/feeds.py:368`). `build_client()`
sets `follow_redirects=True` (`net.py:87`), so the five permitted hops are
followed inside httpx and never re-checked. Enclosure chains are routinely
multi-hop (`swap.fm → mgln.ai → podtrac → prxu.org`); a compromised or
redirect-selling host on the allowlist can bounce the agent to
`http://192.168.x.x/...` or the CouchDB container, and the response lands in a
parser or on disk. This is the one place untrusted feed data chooses where the
agent connects.

**Design.**

- In `net.py`, add a redirect-walking layer used by `fetch_text`,
  `download_to_file`/`_stream_into` and the feed poll: issue each request with
  `follow_redirects=False`; on a 3xx, resolve `Location` against the current
  URL (`urljoin`), run the check below, count the hop against `MAX_REDIRECTS`,
  and continue. More than `MAX_REDIRECTS` hops → `UrlRejected`.
- The per-hop check depends on who supplied the starting URL:
  - **Feed-supplied targets** (enclosures, transcript URLs, scraped pages):
    every hop passes the full `guard.check(hop, related_to=feed_url)` — scheme,
    registrable domain vs. feed domain, CDN allowlist.
  - **Owner-supplied targets** (the feed URL itself, ntfy): hops pass the
    scheme check and H2's address check only, not the domain allowlist —
    the owner chose the URL, and feeds legitimately redirect across domains.
- `Range` resumption (`_stream_into`) must re-walk the chain from the original
  URL on each resume attempt, not resume against a remembered final hop —
  redirect targets are frequently signed and short-lived.
- Keep `build_client(follow_redirects=True)` available for CouchDB's client
  only if it needs it (it sets its own); the shared outbound client used with
  guarded fetches must not silently auto-follow. The simplest shape: the walker
  passes `follow_redirects=False` per-request, so no client-level change is
  needed.

**Consequence to document (README troubleshooting):** every host in a redirect
chain must now individually pass the guard, not just the outermost. All four
hosts of the known Click Here chain are already allowlisted. A rejection now
names the failing *hop*.

**Tests.** respx chains: allowlisted → allowlisted succeeds; allowlisted →
unlisted domain rejected at the hop; allowlisted → `http://127.0.0.1/` rejected;
hop count over `MAX_REDIRECTS` rejected; `https → http` downgrade on a hop is
allowed only if the hop passes the same checks (scheme check permits http;
assert no *other* scheme); resumption re-walks from the original URL.

---

## H2 (P1) — Private-address check at each hop (DNS-rebinding hardening)

**Problem.** The guard reasons about names; an allowlisted name may resolve to
`10.0.0.0/8`, `172.16/12`, `192.168/16`, `127/8`, `169.254/16`, or a ULA.

**Design.**

- In `net.py`, add `def resolve_public(host: str) -> None` (or similar): resolve
  via `socket.getaddrinfo`, and raise `UrlRejected` if **any** answer is
  private, loopback, link-local, unspecified or reserved (`ipaddress` module —
  `is_private or is_loopback or is_link_local or is_reserved or is_unspecified`).
- Called from the H1 hop check, for feed-supplied and owner-supplied targets
  alike, **except**: a host equal to the configured feed URL's host is exempt
  (preserves the documented self-hosted-LAN-feed shortcut,
  `net.py:124`), and the check is skipped entirely when
  `security.enforce_domain_allowlist: false` (the existing escape hatch).
- New config: nothing. This rides the existing switch; a separate knob invites
  turning off the wrong half.
- Accepted residual: check-then-connect TOCTOU (a second resolution at connect
  time could differ). Note it in the module docstring; IP-pinning transports
  are out of scope for this pass.
- Tests must not resolve real names: monkeypatch the resolver. Cover: public A
  record passes; private A record rejected; mixed public+private rejected;
  feed-host exemption; enforce-off bypass.

---

## H3 (P2) — Throttle failed authentication + bind guidance

**Problem.** `require_api_key` compares in constant time but accepts unlimited
attempts, and `api.host` defaults to `0.0.0.0` (`config.py:343`) — correct in
the container (compose scopes it), LAN-wide when run natively on the Mac.

**Design.**

- In `api/auth.py`: an in-memory sliding-window counter of failed attempts per
  client IP (`request.client.host`). After 10 failures in 5 minutes, further
  attempts from that IP get `429` with `Retry-After`, until the window drains.
  A successful auth clears the IP's counter. Constants in the module, not
  config. Memory-bounded (cap tracked IPs at ~1024, evict oldest).
- The counter is process-local by design — do not put this in CouchDB; an
  unauthenticated path must not be able to generate database writes.
- `/healthz` is unaffected (it never had auth).
- README + RUNNING-ON-MAC: one paragraph — when running natively, set
  `PODAGENT_API__HOST` to the machine's LAN IP (or `127.0.0.1` plus a
  tunnel/Tailscale) rather than relying on the `0.0.0.0` default.
- Tests: 11th failure within the window is 429; success resets; a second IP is
  unaffected; 429 body does not reveal whether the key was close.

---

## H4 (P1) — Console overrides cannot re-point model traffic off-LAN

**Problem.** `control:settings` may override LLM tier endpoints, including
`api_base`. Anyone who can write that document (console key, or CouchDB access
if its port is ever exposed) can silently redirect every transcript and prompt
to an arbitrary "model" endpoint. This is the highest-leverage escalation in
the system.

**Design.**

- At override **save time** (`api/settings.py` PUT) and again at **apply time**
  (`load_settings(overrides=...)` in `main.py` lifespan — belt and braces, the
  document can be written by other means): every `api_base` appearing in the
  merged settings must have a host from an allowed set, built from the
  file-based configuration:
  - every `api_base` host already present in `config.yaml`'s `llm` section;
  - `localhost` / `127.0.0.1` (the Ollama default);
  - hosts implied by each configured provider's default endpoint (e.g.
    `openrouter.ai` when a tier uses the `openrouter` provider).
- A save with a host outside the set is refused with a 422 naming the host and
  saying how to authorise it: *add it to `config.yaml` and restart* — new
  egress destinations require file-level (deploy-time) authority.
- An already-stored override that fails the check at apply time follows the
  existing invalid-override path (`main.py:136`): log loudly, run on the file.
- Tests: same-host override accepted; new-host override 422 with the host
  named; stored bad override falls back at boot and logs; provider-default
  host accepted.

---

## H5 (P1) — Console DOM-injection audit, then a test that keeps it fixed

**Problem.** `md_to_safe_html` is double-defended, but the eight console pages
also render API JSON (titles, log lines, entity names) with their own JS. One
`innerHTML` interpolation of an unescaped episode title exposes the
`sessionStorage` key. The pages have not been audited for this.

**Design.**

- Audit every `innerHTML` / `insertAdjacentHTML` / `outerHTML` /
  `document.write` use in `podcast_agent/api/static/*.html`.
- Introduce (or standardise on) a single `esc()` text-escaping helper in the
  shared page scaffolding; every `${...}` interpolation inside markup-building
  template literals must pass through `esc()` unless the value is one of a
  small set of server-sanitised fields (the digest HTML from
  `md_to_safe_html` is the notable one — it is *already* HTML and must not be
  double-escaped; inject it via one clearly-named path).
- Add a rule to `tests/test_console_assets.py`, in the spirit of the existing
  vocabulary tests: for each page, every `${` occurring inside a template
  literal assigned to `innerHTML`/`insertAdjacentHTML` must begin `${esc(` or
  use an explicitly allowlisted expression (maintained as a short list in the
  test, each entry justified by a comment). A regex approximation is
  acceptable; the test exists to stop *regressions*, the audit is what
  establishes the baseline.
- Fix whatever the audit finds. Findings that are already-safe constants need
  no change, only allowlisting.

---

## H6 (P0) — The archive transcription batch gets its own cap

**Problem.** `backfill/process.py:181` sizes the `_transcribe` batch with
`max_summaries_per_run` — a copy-paste that silently couples archive ASR
throughput to the summary budget.

**Design.** Use `settings.pipeline.max_transcripts_per_run` (the knob that
already means exactly this for the routine pipeline). No new config. Two-line
change plus a test asserting the transcribe stage's batch honours
`max_transcripts_per_run` and ignores `max_summaries_per_run`.

---

## H8 (P2) — Cooldown for failing LLM endpoints

**Problem.** The fallback chain is re-walked from the primary on every call
(`llm/client.py`); a down primary costs a full timeout per episode, which is
what makes a deferred stage slow to drain.

**Design.**

- In the client: a process-local map `endpoint_key → monotonic time of last
  transport failure`. When walking the chain, skip an endpoint whose failure is
  younger than 60 s — **unless every endpoint in the chain is cooling down**,
  in which case walk the full chain anyway (a cooldown must never convert
  "slow" into "unavailable").
- Only transport-class failures (`_TRANSPORT_ERRORS`) set the timestamp;
  validation failures do not — a model emitting bad JSON is not a dead
  endpoint.
- A success clears the endpoint's entry. Log skips at debug, one info line the
  first time an endpoint enters cooldown.
- Constant in the module; not config.
- Tests drive the existing stubbed transport: second call within the window
  skips straight to the fallback; all-cooling still attempts the primary;
  success clears; validation failure does not trigger cooldown.

---

## H9 (P3, optional) — Bounded concurrency for cloud summarisation

`_stage_summarize` is serial. For a cloud tier, an `asyncio.Semaphore(3)` over
per-episode summarise calls would cut wall time; for single-GPU Ollama it would
only thrash. Gate on the tier's *primary* provider being a cloud provider, keep
strict serialism otherwise, preserve per-episode error isolation and the
`LLMUnavailable` stage-deferral semantics (first failure cancels the remaining
tasks and defers the stage). Implement last; skip if it complicates the
deferral logic disproportionately.

---

## H10 (P2) — One deployment story: bridge networking, compose canonical

**Decision (owner, 2026-08).** The permanent home is a NAS/server, arriving
later; the Mac runs the service natively in the interim. Compose is therefore
the canonical deployment; the native mode is documented as the interim, not a
peer. This item makes the compose stack portable *now* so the eventual move is
a rehearsed copy, not a first boot.

**Problem.** `docker-compose.yml` uses a macvlan network to give the agent its
own LAN IP. Docker Desktop does not support macvlan, which is what forked the
Mac into a parallel native universe dragging four artifacts behind it:
`RUNNING-ON-MAC.md`, the generated `config.local.yaml`, `requirements.lock.txt`
(image-only), and the `host: 0.0.0.0` default whose safety assumes compose
scoping (H3). Every one of those is a place where a change to one deployment
story silently misses the other. macvlan's only benefit was the dedicated IP;
nothing in the system needs it.

**Design.**

- Replace the macvlan `lan` network with an ordinary bridge and publish the
  agent's port (`8080:8080`). Drop `AGENT_LAN_IP`. The agent is reached at
  `<host>:8080` instead of its own IP.
- **Preserve the property that mattered:** `couchdb-podcast` keeps no `ports:`
  and stays on the internal network only — unreachable from the LAN. Assert
  this in a comment beside the service, since it is now the *only* thing the
  network layout is protecting.
- Make the `ollama` service a compose **profile** (e.g. `--profile ollama`):
  on the NAS it runs as today; on a Mac, Ollama stays native for Metal and the
  profile is simply not started. Document the one-line `api_base` implication.
- Update the compose header comments (they currently explain macvlan) and the
  README quick start. `RUNNING-ON-MAC.md` shrinks to: interim mode, Ollama is
  native for the GPU, and the compose stack is verifiable on the Mac with
  `docker compose config` + a CouchDB-only bring-up.
- Keep `config.local.yaml` and the native mode working unchanged — the interim
  host still uses them. They are retired on migration day, not before.
- **Migration-day checklist** (append to the README's backup section, so it
  lives next to the scripts it uses): `scripts/backup.sh` on the Mac →
  copy `.env`, `config.yaml`, backup file → `restore.sh` on the NAS →
  set `DIGEST_DIR` to the synced vault mount → pull the Ollama models →
  `docker compose up -d` → `healthz`, then one manual ingest+pipeline run and
  compare `/status` counts against the Mac's. Transcript attachments ride the
  backup, so nothing is re-fetched or re-transcribed.

**Tests.** Compose is not exercised by pytest; the verifiable parts are:
`docker compose config` parses (a CI-less repo can still assert the YAML is
valid in a unit test via `yaml.safe_load` on the file and the absence of a
`ports:` key under `couchdb-podcast`), and the README/compose comments agree
on the network story (extend `tests/test_console_assets.py`'s
documentation-drift approach only if cheap — do not build a compose linter).

---

## Explicitly out of scope

- **TLS / reverse proxy** — operational, not code; the README's "do not
  port-forward" rule stands.
- **Database backups** — already covered by `scripts/backup.sh` +
  `scripts/restore.sh` and documented in the README.
- **Prometheus `/metrics`** — the `run`/`llm_call` documents already answer the
  questions; add only if something starts graphing them.
- **IP-pinning custom transport** (H2 residual) — revisit if the threat model
  changes.
- **Rejected enclosures keep discarding the episode.** Proposed (create the
  episode without its audio, heal on re-poll) and **declined by the owner,
  2026-08** — the current behaviour is a deliberate decision, not an oversight.
  Do not "fix" it. The operational tell for a missing allowlist entry remains
  `ingest.url_rejected` in the kept warnings, and the README documents it.

## Suggested order

H6 → H1 → H2 → H4 → H5 → H3 → H8 → (H9). H6 first because it is two
lines; H1/H2 together because H2's check runs inside H1's hop walker.
H10 is independent of the rest and can land any time; doing it before H3
is mildly preferable, since H3's bind guidance reads differently once
compose is the single canonical story.
