# COGL — P2P Swarm Inference for Frontier MoE Models

Research project: can a volunteer P2P network of consumer GPUs / Apple M-series machines
serve a ~2.8T-parameter MoE model, with each peer holding an expert slice?

- Master plan (frozen, never edit): `docs/plan/swarm-moe-research-plan.v1.md`
- Plan review & proposed amendments: `docs/plan-review.md`
- Live status of every phase/test: `ROADMAP.md`
- Durable results ledger: `FINDINGS.md`
- Decision log: `DECISIONS.md`
- Lab notebook (one file per day): `notebook/YYYY/YYYY-MM-DD.md`, index in `notebook/INDEX.md`
- Experiments: `experiments/TNN-short-name/` (README = question, method, kill criterion, results)
- Remote: `git@github.com:ubaydullohpulat/COGL-T1.git` (branch `main`); commit and push at the end of each session (researcher asked for all code to live there)

## Standing constraints (update in DECISIONS if they change)
- Hardware: Apple M4 Max, 128 GB unified memory, **~300 GB free disk** — never plan to hold all of K3 (1.56 TB) or all proxies at once
- Network: ~72 Mbps down / 30 Mbps up (measured 2026-09-15)
- K3 can't be served locally. Testing strategy = proxy portfolio + layer-streamed real K3 (see `experiments/P00-proxy-selection/README.md`, D-005)
- Any download > 1 GB: state name, source, size and disk impact, and get the researcher's OK first

---

## Lab notebook protocol — MANDATORY every session

You (Claude) are the notebook keeper for this research. The researcher must be able to open
the notebook months from now and know exactly what was tried, what worked, what failed, and why.
**No session ends without its notebook entry being complete.**

### 1. At session start
1. Get today's date from the shell (`date +%Y-%m-%d`) — never guess it.
2. Read, in order: `ROADMAP.md` (the "Current step" block), the last 2 entries listed in
   `notebook/INDEX.md`, and any open questions in the most recent entry.
3. Open today's file `notebook/YYYY/YYYY-MM-DD.md`. If it doesn't exist, create it from
   `notebook/_template.md`. If it exists (second session that day), append a new
   `## Session N` block — never overwrite earlier sessions.
4. Tell the researcher in 2–4 lines: where we are, what the last session left open, and the
   proposed goal for this session. Record the agreed goal in the entry.

### 2. During the session — log as you go, not only at the end
- Every attempt goes in "What we tried": what, how (command / config / script path / model +
  revision), outcome. **Failed attempts and dead ends are logged with the reason** — negative
  results are results.
- Every number goes in the Results table with an **evidence type**:
  - `MEASURED` — we ran it; link the script/log/output file
  - `COMPUTED` — derived by arithmetic from other numbers; link the calc
  - `CITED` — from a paper/report/config; give exact source (URL, section, table, file@revision)
  - `ASSUMED` — a modelling input we chose; say why
  - `GUESS` — a prior with no support; must be replaced later
- Never present a `GUESS`/`ASSUMED` value as a finding. Never invent citations, model specs or
  numbers; if a source can't be verified, mark it `UNVERIFIED`.
- Reproducibility for any run: seed, library versions, model name + revision/hash, hardware,
  dataset/prompt-set version. Put raw outputs under the experiment folder, not in the notebook.
- If something contradicts an earlier entry, do not edit the old entry — add a dated
  **Correction** note in today's entry and fix `FINDINGS.md`.

### 3. At session end (before saying "done")
1. Complete today's entry: Summary, Tried, Achieved, Results, Decisions, Open questions, **Next step**.
2. Update `ROADMAP.md`: status of touched tests, and the "Current step" block.
3. Durable result (a number/conclusion others will build on) → add a row to `FINDINGS.md`.
4. A choice that changes scope/method/plan → add an entry to `DECISIONS.md` (the v1 plan
   file stays frozen; amendments live in DECISIONS).
5. Add/refresh the one-line summary for today in `notebook/INDEX.md`.
6. Finish by telling the researcher: what we achieved today, and the concrete next step.

### 4. Kill gates are explicit
Whenever a test with a kill criterion completes, write a **Gate verdict** in the notebook
(PASS / FAIL / GRAY + the numbers vs threshold) and record it in `DECISIONS.md`.
Do not soften a failed gate; surface it plainly and let the researcher decide.

### Status vocabulary (ROADMAP)
`NOT STARTED` · `IN PROGRESS` · `BLOCKED (reason)` · `DONE — PASS` · `DONE — FAIL` · `DONE — GRAY` · `SKIPPED (reason)`
