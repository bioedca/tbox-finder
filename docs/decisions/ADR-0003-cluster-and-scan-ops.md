# ADR-0003 — Cluster & scan ops

- **Status:** Accepted (user sign-off recorded 2026-07-08, CLAUDE.md §7 item 2)
- **Date:** 2026-07-08
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR)
- **Delegated from:** PRD §7.2, §15, §18.2, §18.3
- **Supersedes / superseded by:** none
- **Related:** ADR-0001 (architecture & aims — the non-circularity principle that a compute-forced cut must never masquerade as a finding, D3), ADR-0002 (environment & ML stack — the conda-lock envs these jobs activate; the D7 continued-pretraining pass), ADR-0004 (split & leakage policy — the GATE-4 readiness a sweep-prune must not compromise, D7), ADR-0005 (non-circular eval — the Stage-1 recall floor this ADR must never relax), ADR-0006 (validation decision rule & tiering — the GATE-3 scan-plan-execution sub-gate that D5's minimum-viable scan defines + the per-clade scanned-coverage table + calibrated-negative rollup a down-scope feeds)

This ADR pins the **cluster/SLURM operating contract, the compute-governance rules, and the discovery-scan execution plan**. It owns the operational envelope of the two most expensive activities in the project — GPU training/sweeps and the genome-scale discovery scan — under a cluster that has **no accounting quota** (so contention + patience is the binding limit, not QOS) and **no reliable job-exit record** (`sacct` off → artifact-based verification). Two P0 **numeric** thresholds are pinned here (the genome-array shard-failure-tolerance fraction, D8; the sole-GPU-node-outage hold window, D9); the **compute ceiling** and **patience budget** are pinned as a **determination rule** with their numeric values **explicitly frozen at the pre-P5 sizing-benchmark gate** (§7.2), because the binding number needs the P2-trained scanner's Stage-1 candidate density, which cannot exist at P0. It does **not** own any gate threshold that touches the scientific claim (those are ADR-0004/0005/0006); accordingly the scientific-claim gate-default **blinded-freeze** discipline (owned by ADR-0005 per §18.3; framed in ADR-0001 D4) does **not** apply to the ops thresholds here — but changing any of them still requires ADR sign-off (CLAUDE.md §7 item 2).

---

## Context

The `amlab` cluster (lab `mondragonlab`, user `ecc1695`) is the only machine that can run GPU training/scanning and large downloads (PRD §15; CLAUDE.md §9). Its constraints are load-bearing for every operational decision here:

- **Topology.** `ssh two` → login node `two.amlab` (96c/256 GB, outbound internet via ProxyJump `zero`). Compute is the **`gpu` partition** — nodes `one` + `two`, each **8× RTX A4000 (16 GB VRAM)**, 96c/256 GB. Node `one` is **often down**, so `two`'s 8 GPUs are the *de facto* reliable capacity. `zero` is a shared gateway with **no GPU**; the `compute` partition is `zero`-only. Storage: node-local `/tmp/$USER-$SLURM_JOB_ID` (~600 GB, fast) for build; shared HOME `/exports/people/mondragonlab/ecc1695` is **87% full** (~737 GB free).
- **No accounting.** No `--account`/QOS; `sinfo`/`squeue`/`sbatch` work but **`sacct` is off**. There is therefore **no quota ceiling from the scheduler** — the only real limit on a long campaign is *cluster contention and our own patience* — and **no scheduler-side exit record** — so a job's success must be proven from its **artifacts** (per-shard `SHARD_OK` markers + zero-byte `.err`), not from `sacct` (CLAUDE.md §9.3 step 8).
- **No container runtime.** No apptainer/singularity/docker/podman; environments are the pinned `conda-lock` envs on the user miniconda3 (ADR-0002 D1). tcsh is the default login shell → compound remote commands use `bash -lc`/heredoc (CLAUDE.md §9.2).

The two activities this ADR governs are the project's cost centres: (1) **GPU training/sweeps** — the §11 six-axis Hydra multirun plus a probe-triggered continued-pretraining pass; (2) the **genome-scale discovery scan** (P5) — the single largest job, running the P2 scanner over GTDB species-reps + RefSeq + MAGs (+ an Archaea stretch), the output of which feeds GATE-3. Because the scan's breadth *is* the flagship scientific result (PRD §1), a compute-forced cut must **never** be allowed to masquerade as a biological finding of bounded distribution (PRD §7.2, §18.2; ADR-0001 D3). This ADR pins the governance that keeps that guarantee mechanical.

---

## Decision

### D1. Cluster & SLURM contract (locked)

All GPU work runs under the CLAUDE.md §9.3 SSH submit-ack protocol against the following fixed contract:

- **Partition/resources:** `--partition=gpu`, GPUs via `--gres=gpu:a4000:<N>` (N ≤ 8 realistically, since `two` is the reliable node). **Never** `--partition=compute`, never node `zero` (no GPU there). **Never** `--account`/`--qos` (accounting disabled). **Never** `--nodelist` / hard node-pin — `sbatch -p gpu` and let the scheduler place the job (it lands on `two` when `one` is down); hard-pinning `one` would strand jobs during its frequent outages (D9).
- **Verification is artifact-based (`sacct` off).** Every job writes success-path markers only: per-shard `SHARD_OK` (or a single `DONE` marker for non-array jobs) and per-shard `.err` that is **zero bytes on success**. Job exit is confirmed by (a) `squeue -j <id>` empty, (b) all `SHARD_OK` present, (c) all `.err` zero-byte — any missing → CLAUDE.md §7 stop-and-ask; **no auto-resubmit** (CLAUDE.md §9.3 step 8 / §13). The dev-log stanza records `verification: artifact-based (sacct off)`.
- **Storage discipline.** Build in node-local `/tmp/$USER-$SLURM_JOB_ID`; move **only the final artifact** to shared HOME / the DVC scratch remote (D3); clean `/tmp` on exit. Do not build large derived data directly in the 87%-full HOME (PRD §15; §18.2 shared-HOME risk).
- **Shell.** tcsh login shell → pure binary calls (`ssh two sbatch …`) are shell-agnostic; short compounds wrap in `bash -lc '…'`; multi-line/nested scripts use stdin-heredoc; sbatch bodies start `#!/bin/bash` with `set -euo pipefail` and source the miniconda3 `conda.sh` before activating the pinned env (CLAUDE.md §9.2/§9.3).

### D2. Snakemake executor CPU-only; GPU via hand-authored `sbatch` (locked)

- The **local/CPU DAG** (data → eval → validate → figures) runs via `snakemake --cores` (PRD §16). CPU rules may use the Snakemake SLURM executor if ever needed for large CPU steps.
- **GPU stages (train, scan) are NEVER submitted through the Snakemake SLURM executor.** They are **hand-authored `sbatch` jobs** submitted through the CLAUDE.md §9.3 one-shot-ack protocol. Rationale (PRD §16; CLAUDE.md §9.3/§13): the SLURM executor submits programmatically and **auto-retries failed jobs**, which would bypass the `sbatch --test-only` preflight, the one-shot human `submit` ack, and the **no-auto-resubmit** rule — the exact guarantees D8/D9 and CLAUDE.md §9.3 depend on. GPU stages still **declare** their resources (`a4000`, partition `gpu`, runtime, mem) in the DAG so dependencies track and CLAUDE.md §3.2 (rule = env) is honored; Hydra drives in-job config.

### D3. Scratch / DVC-remote path + estimate-anchored DVC-capacity rule (locked; numeric verified in P0-10)

- The **DVC remote is the cluster scratch** (`dvc-ssh`), holding DVC-tracked `data/interim/`, `data/processed/`, and model checkpoints (PRD §16; CLAUDE.md §5.2). `.dvc/config` is per-machine and not committed.
- **Capacity rule (the estimate-anchored DVC-capacity rule).** Before any phase-gate `dvc push`, the remote's confirmed **free capacity must be ≥ the P5-estimated artifact volume** — *not* merely "larger than the 87%-full HOME" (PRD §15). If capacity is tight, the **retention-pruning fallback** applies: prune superseded interim artifacts / stale checkpoints (keeping every phase-gate-tagged artifact and its `provenance.json`) to recover headroom. If **no** volume ≥ the estimate exists even after pruning, that is a **CLAUDE.md §7 stop-and-ask** — never a silent skip of `dvc push`.
- **P0 vs P0-10 split.** This ADR pins the **rule**; the **path + free-capacity verification** (confirming a concrete volume ≥ the P5 estimate) is the mechanical P0-10 step, which **depends on this ADR**. The P5 artifact-volume estimate is itself produced at the pre-P5 sizing gate (D6); until then P0-10 verifies against a conservative provisional estimate and is re-verified when the sizing gate freezes the real number.

### D4. Corpus set, de-duplication priority & sharding (locked)

- **Corpus set (PRD §7.2):** GTDB species-representatives of the one governing GTDB release; targeted under-sampled clades (Gram-negatives/Proteobacteria, CPR/Patescibacteria, candidate phyla); full RefSeq prokaryotic complete genomes; tiered-confidence MAGs; environmental metagenomes (capped); **Archaea/DPANN as a pre-registered stretch** (own per-corpus prior + FDR null, §7.2) — the same six tiers D5 prioritizes. Each corpus is pinned by accession list + checksum + release version into `data/external/` provenance (immutable; CLAUDE.md §10.2/§11).
- **De-duplication (locked priority):** the union of pinned scan accession lists is de-duplicated on a **canonical genome/assembly accession** with the stated priority **GTDB-rep > RefSeq-complete > MAG**, *before* sharding. A genome shared across corpora (e.g. a GTDB species-rep that is also a RefSeq complete genome) is scanned — and counted against the compute ceiling (D6) — **exactly once**, retained under its highest-priority corpus label. The candidate table is keyed on `(accession, coordinates)`; **dedup counts are `log`ged** (PRD §7.2). Priority rationale: GTDB species-reps give the taxonomically-balanced backbone and the governing-release novelty frame (§7.2); RefSeq-complete outranks MAG on assembly quality; MAGs are noisiest (§7.2 MAG-confidence floor). The de-dup key is scoped to **accessioned assembled genomes** (the three-level GTDB-rep/RefSeq-complete/MAG priority above); **environmental metagenomes** (D5 tier 6) are not accession-dedup'd but bounded by their version-tagged sharding cap.
- **Sharding / streaming:** an explicit **shard-and-stream** model into node-local `/tmp/$USER-$SLURM_JOB_ID` (~600 GB) with a stated **disk-high-water + egress budget**; each corpus carries a **version-tagged cap** (environmental metagenomes otherwise effectively unbounded). Shards emit `SHARD_OK` on success only (D1, D8).

### D5. Corpus scan priority, minimum-viable scan & the breadth-preserving feasibility lever (locked)

- **Pre-registered corpus scan priority** (retention order under budget pressure, highest first): **(1)** GTDB species-representative set of the governing release (taxonomically-balanced backbone); **(2)** targeted under-sampled clades (Gram-neg/Proteobacteria, CPR/Patescibacteria, candidate phyla); **(3)** RefSeq prokaryotic complete genomes; **(4)** tiered-confidence MAGs; **(5)** Archaea/DPANN stretch; **(6)** environmental metagenomes (capped). Priority protects taxonomic **breadth** first.
- **Minimum-viable scan (the GATE-3 scan-plan-execution floor).** The minimum-viable scan = tiers **(1) + (2)** at **un-down-sampled breadth** — the full governing-release GTDB species-representative set (taxonomically balanced) **plus** the targeted under-sampled-clade set. Its **execution to spec** is the mandatory GATE-3 "scan-plan execution" sub-gate, whose definition is pinned in **ADR-0006** (the mandatory-vs-non-blocking GATE-3 split is ADR-0001 D3; the per-corpus→GATE-3 rollup is ADR-0006). Tiers (3)–(6) are **breadth-extending**, not part of the floor.
- **Feasibility lever (breadth-preserving down-scope).** If the pre-P5 sizing gate (D6) shows the planned scan **exceeds the compute ceiling**, the **default lever cuts coverage breadth-preservingly**: sub-sample **depth** within the abundant tiers (RefSeq, MAGs, environmental) while **retaining** the taxonomically-balanced (1) + targeted-under-sampled (2) breadth; **every cap is `log`ged** and every calibrated-negative PASS ships a **per-clade scanned-coverage table** (ADR-0006). The lever **never** relaxes the Stage-1 recall floor (ADR-0005) and **never** silently drops a clade.
- **Below minimum-viable = stop-and-ask.** A scan that cannot execute tiers (1)+(2) to spec within the ceiling is a **CLAUDE.md §7 stop-and-ask** (present ceiling + observed requirement + options), **never** a silent down-scope — so a budget-forced cut can never masquerade as a calibrated-negative biological finding (PRD §7.2, §18.2; ADR-0001 D3).

### D6. Compute-ceiling determination rule + patience budget (locked as RULE; numeric frozen at the pre-P5 sizing gate)

The **per-corpus + aggregate upper GPU-hour / wall-clock ceiling** is the denominator of the §7.2 "exceeds capacity" feasibility test. With no scheduler QOS quota, this ceiling *is* the binding limit on the campaign (contention + patience). Its **numeric value cannot be a P0 number** — it needs the P2-trained scanner's Stage-1 candidate density (§7.2/§15). This ADR therefore pins the **determination rule + the patience-budget policy**; the numbers **freeze at the pre-P5 sizing-benchmark gate** (§7.2) and are signed off there.

- **Sizing benchmark (inputs, §7.2).** On a ~100-genome sample spanning divergent clades, measure: Caduceus windows/sec/GPU `w`; RiNALMo candidates/sec/GPU `r`; Stage-1 candidates/Mbp `ρ`; and the low-order-confidence / **both-strand carry-through fraction** `φ` (§6, concentrated on divergent loci).
- **Determination rule (the estimator).** For a corpus of total sequence `T` Mbp (post-dedup, D4):
  - Stage-1 windows ≈ `T·10⁶ / stride` (stride per ADR-0001 D1; Caduceus-PS is RC-equivariant → one pass covers both strands). Stage-1 GPU-hours = windows / `w` / 3600.
  - Stage-2 candidates ≈ `T · ρ · (1 + φ)` (the `φ` fraction carried on both strands). Stage-2 GPU-hours = candidates / `r` / 3600.
  - **Per-corpus required GPU-hours** = Stage-1 + Stage-2; **aggregate** = Σ corpora. **Wall-clock** = GPU-hours / (parallel A4000 count `G`, `G` ≤ 8 on `two`).
  The **ceiling** is a pre-registered `(per-corpus, aggregate)` pair of **`(GPU-hours, wall-clock-days)`** bounds; the feasibility gate emits **pass/fail** by comparing the estimator's required GPU-hours/wall-clock against these bounds. Exceed → D5 breadth-preserving lever; sub-minimum-viable → §7 stop-and-ask.
- **Patience budget (policy).** Because contention (not quota) binds, the ceiling's **wall-clock-days** component is a **pre-registered patience bound** on sustained shared-`gpu`-partition occupancy — how many days of campaign wall-clock we pre-commit to before a §7 re-plan. It is set **alongside** the GPU-hour ceiling at the sizing gate (same sign-off), from the *measured* per-genome cost and the *observed* partition contention (neither knowable at P0). The patience budget composes with — but is distinct from — the transient-outage hold window (D9): D9 governs *interruptions*, the patience budget governs *total acceptable duration*.
- **Freeze + governance.** Ceiling and patience-budget numbers are frozen at the sizing gate and recorded in the P5 dev-log; a later change needs ADR sign-off (§7 item 2). They are **ops** bounds, not scientific-claim thresholds, so the gate-default blinded-freeze discipline (ADR-0005) does not apply — but the D5 guarantee (a cut never masquerades as a finding) makes their honest application load-bearing.

### D7. Training/sweep GPU-hour budget (locked as RULE; numeric frozen at first measurement)

- **Scope.** The §11 six-axis Hydra multirun (P2) + the probe-triggered continued-pretraining / NT-multispecies pass (ADR-0002 D7) carry their **own aggregate GPU-hour budget**, separate from the scan ceiling (D6).
- **Determination rule.** budget ≈ (sweep-grid cardinality × per-run GPU-hours) + (continued-pretraining pass GPU-hours), capped by a pre-registered aggregate bound. PRD §15 delegates this budget to this ADR without fixing its freeze point; this ADR **determines** the freeze point to be **first measurement**. Per-run GPU-hours are first measurable at the P1 smoke-training gate (PRD §18.1 P1 "smoke run reproducible"), which is *before* the pre-P5 sizing gate that binds the scan ceiling (D6) — so this budget's number freezes at **P1**, not the sizing gate. The P1 smoke per-run cost is extrapolated to the P2 six-axis multirun's full-scale runs where the cost is actually incurred; a material P1→P2 divergence **re-freezes** the number (a re-freeze is an ADR-sign-off change). Until measured, P2 training is planned against a conservative provisional bound.
- **Governance.** If the measured sweep would exceed the aggregate bound, the default lever **prunes the sweep grid** (drop low-information axes/levels; `log` every cut) rather than under-training the production scanner; a prune that would compromise GATE-4 readiness (ADR-0004) is a §7 stop-and-ask. Number change → ADR sign-off (§7 item 2).

### D8. Genome-scale-array shard-failure-tolerance rule (locked; fraction NUMERIC)

The genome-scale scan is a large `sbatch --array` job under the CLAUDE.md §9.3 **no-auto-resubmit** contract. A **bounded shard-failure-tolerance rule** distinguishes *transient* from *substantive* failure:

- **Transient (batch-re-submit once):** if the failed shards are **≤ 2% of the array's total shards** **AND** not concentrated by a single error signature or a single corpus, they may be **batch-re-submitted exactly once** via a **single CLAUDE.md §9.3 `submit` ack** (one preflight, one human ack, no loop).
- **Substantive (stop-and-ask):** **> 2%** failed shards, **OR** failures concentrated in one corpus / one error signature (a systematic bug or env problem, regardless of fraction), **OR** any shard that fails a **second** time → **CLAUDE.md §7 stop-and-ask** (surface the failure set; do not resubmit). This preserves the §9.3 guarantee that a failing pipeline cannot be silently retried into apparent success.
- **Numeric:** the batch-re-submittable fraction is pinned at **≤ 2%** — approximately the expected transient node-preemption / IO-hiccup rate over a multi-thousand-shard array, above which a systematic cause (a code/env bug, a bad corpus) is more likely than coincidence. It is a pre-registered ops threshold (not a scientific-claim gate → no blinded-freeze); a change needs ADR sign-off (§7 item 2). Every re-submit and every failure classification is `log`ged into the scan provenance.

### D9. Sole-GPU-node-outage contingency / patience-hold rule (locked; hold window NUMERIC)

Node `one` is often down and `two` is the sole reliable GPU node — an extended `two` outage would otherwise strand the campaign (PRD §15; §18.2 risk).

- **No hard node-pin** (D1): jobs `sbatch -p gpu` and are scheduled onto whichever GPU node is up.
- **`one` downtime is expected** and is **never itself an escalation trigger** — the scheduler simply places jobs on `two`.
- **Hold, don't cancel.** If `two` is unavailable (outage, or the `gpu` partition has zero schedulable A4000s), **queued jobs are held, not cancelled** — SLURM schedules them when the node returns; there is **no auto-resubmit** (§9.3). Polling continues at the §9.3 cadence.
- **Numeric hold window:** the campaign tolerates a **sustained `two` outage of up to 72 h** before it becomes a **CLAUDE.md §7 stop-and-ask** (surface the outage; decide: keep holding, or re-plan / seek alternate compute). 72 h spans a weekend outage plus a business day of cluster-admin turnaround before a re-plan is warranted; it is a pre-registered ops threshold (not a scientific-claim gate → no blinded-freeze); a change needs ADR sign-off (§7 item 2).

### D10. Delegations (values NOT pinned here / pinned elsewhere)

- **Numeric compute ceiling + patience-budget days** → frozen at the pre-P5 sizing gate (D6), recorded in the P5 dev-log.
- **Numeric training/sweep budget** → frozen at P1 first measurement (D7).
- **DVC-remote path + concrete free-capacity number** → verified in P0-10 against the P5 estimate (D3).
- **Governing GTDB release + the pinned per-corpus accession lists/counts** → pinned in the P0 data-foundation steps (§7.2), not here; this ADR governs how those corpora are de-duplicated, prioritized, sharded, and bounded.
- **Stage-1 recall floor, `cmsearch` operating point, GATE-1/2 machinery** → ADR-0005 (this ADR must never relax the recall floor, D5).
- **Validation tiers, the per-clade coverage table's role in the per-corpus→GATE-3 rollup, the calibrated-negative PASS rule** → ADR-0006.

---

## Consequences

- **Positive.** The two cost centres (train/sweep, scan) have a single governance contract; a compute-forced scan cut is mechanically prevented from masquerading as a biological finding (D5 minimum-viable floor + breadth-preserving lever + stop-and-ask); the no-auto-resubmit guarantee survives both shard failures (D8) and node outages (D9) with pre-registered numeric bounds; the compute ceiling has a real estimator (D6) whose number is honestly deferred to the only point it can be measured.
- **Costs / constraints.** Deferring the ceiling number to the sizing gate means P0-10 verifies DVC capacity against a provisional estimate and must re-verify at the sizing gate. Forbidding the Snakemake SLURM executor for GPU (D2) means every GPU submission is a hand-authored, human-acked round-trip — deliberate friction that buys the §9.3 guarantees. The ≤ 2% (D8) and 72 h (D9) numbers are engineering judgments, revisable only via ADR sign-off.
- **This ADR pins two P0 numbers** (D8 ≤ 2% shard fraction; D9 72 h hold window) and **defers the rest** (ceiling, patience-days, training budget, DVC capacity) to their first measurable points — each with a determination rule pinned now.

## Cross-reference impact list

- **PRD:** §7.2 (corpora, de-dup priority, sizing/feasibility lever, minimum-viable scan, per-clade coverage table), §15 (cluster topology, DVC capacity rule, compute ceiling + patience budget, training/sweep budget, shard-failure + node-outage rules), §16 (Snakemake executor CPU-only), §18.1 (P0 compute-ceiling-rule deliverable; P5 sizing gate), §18.2 (shared-HOME / scan-exceeds-budget / node-outage / scan-array-partial-failure risks), §18.3 (ADR-0003 delegation line).
- **Sibling ADRs:** ADR-0001 (D5 minimum-viable scan feeds the GATE-3 scan-plan-execution sub-gate; no-relax-recall honors ADR-0001 D3), ADR-0002 (D1 envs are the ADR-0002 conda-lock envs; D7 continued-pretraining pass is ADR-0002 D7), ADR-0004 (D7 sweep-prune must not compromise GATE-4 readiness), ADR-0005 (Stage-1 recall floor D5 must never relax; gate-default blinded-freeze discipline lives here), ADR-0006 (GATE-3 scan-plan-execution sub-gate definition + per-clade scanned-coverage table + calibrated-negative rollup).
- **imp.md:** P0-09 (this step); P0-10 (DVC path+capacity verify, depends on this D3); P5 (sizing gate freezes D6/D7 numbers; scan array runs under D8/D9).
- **CLAUDE.md:** §5.2 (DVC at phase gates), §7 items 2/4 (ADR sign-off; gate/stop-and-ask), §9 (cluster protocol — D1/D2 restate its contract), §9.3 (submit-ack, artifact-based verification, no-auto-resubmit — D8/D9 depend on it), §10.3 (a down-scope never fabricates a calibrated-negative).
- **Cards / paper (release-bound):** dataset card (per-clade scanned-coverage table + de-dup counts), `paper/manuscript.qmd` (scan-plan-execution + any breadth-preserving cap disclosed as-found).
- **Phase-0 exit (2026-07-12):** the phase-gate `dvc push` of the seven P0 artifacts (67 MB, run at finalization) is governed by the **D3** estimate-anchored DVC-capacity rule (delegated numeric verified in P0-10) — capacity was pre-verified, so the push did not surprise-fail (`dvc status --cloud` → in sync); else the retention-pruning fallback / §7 stop-and-ask would apply.

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-08 (bioedca), CLAUDE.md §7 item 2.

---

## Amendments

### Amendment A1 — Governing GTDB release pin (P0-13, 2026-07-09)

Fills the D10 delegation *"Governing GTDB release … pinned in the P0 data-foundation steps (§7.2), not here."* The **one governing GTDB release** (PRD §7.2; referenced by the D1/D3/D4/D5 corpus definitions and the §13.2 P6 placement DB) is pinned to:

- **GTDB R232 (Release 11-RS232, released 2026-04-15).** This is the release the reusable tboxevo taxonomy maps were built against (tboxevo ADR-0001) — cross-checked here: GTDB's own `bac120_taxonomy_r232.tsv` MD5 (`4d42e137…`) equals the value tboxevo independently pinned.
- **Contingency satisfied → r220 fallback NOT invoked.** PRD §7.2 makes the tboxevo-matching release contingent on an available GTDB-Tk reference package; `gtdbtk_r232_data.tar.gz` (56.63 GB, GTDB-Tk ≥ 2.7.0) is published, so R232 — not the pre-registered r220 fallback — governs. Placement (P6) uses the same release, so **no §13.2 release-to-release crosswalk is required**.
- **Species-representative count (pinned to the release's published value):** bac120 **189,801** + ar53 **10,122** = **199,923** (GTDB `stats/r232`, accessed 2026-07-09; independently re-verified as the `sp_clusters_r232.tsv` row count in the P0-13 count gate — fail-loud on drift, CLAUDE.md §10.3).
- **License:** GTDB data are CC-BY-SA-4.0 (`gtdb.ecogenomic.org/downloads`, accessed 2026-07-09).

Provenance (release id, base URL, GTDB-Tk package + version floor, license, per-file authoritative MD5s, staged crosswalk SHA-256 + byte count, published + counted species-rep values) is written to `data/external/gtdb/provenance.json` by `workflow/rules/data.smk::pin_gtdb_release` → `src/tbox_finder/taxonomy.py::pin_release`. The ~49 MB species-rep crosswalk + the on-demand genome→lineage taxonomy/metadata tables are `data/external/`-immutable, re-fetched by that checksummed rule (CLAUDE.md §5.2), never DVC-tracked.

**Impact.** Binds all §7.2 novelty determinations + the union prior (P0-14), the frozen governing-release taxonomy for TaxId re-placement (P0-15), the §9.2 split clade labels (P0-22), and the P6 GTDB-Tk placement DB (§13.2). No change to any locked decision (D1–D10); this only supplies the D10-delegated value.

- **User sign-off:** ☑ recorded 2026-07-09 (bioedca), CLAUDE.md §7 items 1/2 (Q-question release choice + ADR amendment approved as drafted).

### Amendment A2 — Training/sweep GPU-hour budget frozen (P2-06, 2026-07-17)

Fills the D7 / D10 delegation *"Numeric training/sweep budget → frozen at first measurement (D7)"* and **executes D7's determination rule** — `budget ≈ (sweep-grid cardinality × per-run GPU-hours) + (continued-pretraining pass GPU-hours), capped by a pre-registered aggregate bound` — now that both inputs it needs exist: a measured per-run cost (P2-05) and a decided grid cardinality (P2-06). No locked decision (D1–D10) changes; D7's rule is unchanged and this amendment only supplies the number it always deferred, plus a cross-reference correction (below).

**Freeze point — reconciled to *first Stage-1 measurement* (P2-05), enacted here (P2-06).** D7's Decision text (line 78) and D10 (line 101) fix the freeze point at *"P1 first measurement"*: the P1 smoke gate was the anticipated first point a per-run cost could be extrapolated. In fact the P1 deliverables (PRD §18.1 "smoke run reproducible") were the RiNALMo/Caduceus capability probes and a reproducibility smoke — they established that a Stage-1 step *runs* and is deterministic, but did **not** time a full-scale Stage-1 training run over the real fold, so no per-run GPU-hour figure was extrapolated there (P2-05 recorded `freezes_adr0003_d7_budget: false`, validator-enforced, deferring the freeze by design). The **first full-scale per-run measurement is P2-05's sizing smoke** (SLURM job 569, 2026-07-17). D7's freeze-point rule — *numeric frozen at first measurement* — therefore resolves to **P2-05**, and the freeze is **enacted at P2-06**, the first step at which the second input D7's rule requires (the grid cardinality) is decided. This is a refinement of the freeze *point* (P1 → first-actual-measurement = P2-05), not a change to the determination *rule*.

**Measured inputs.**
- **Per-run cost (P2-05, job 569; the ckpt-ON basis point `ws8_bs8_ckpt1`, which is the config the sweep runs):** a 10-epoch full-fine-tune, DDP×8, batch 8, 1024-nt windows, gradient-checkpointing ON, over the **full 8,303-window `nested_train` fold = 1.6129 GPU-h/run** (`total_windows` 83,030; aggregate throughput 114.4 win/s; DDP×8 efficiency 98.1%). Scope caveats carried from P2-05: **fp32** (`train_stage1` exposes no precision knob and ADR-0002 A7 disables TF32 — so this is *not* the bf16/TF32-native throughput PRD §10.3 names as a card property), and `step_seconds` **includes** the per-step full-parameter grad-finiteness scan (P2-06 retains it — γ=0.5 is the exact fractional-γ regime P2-02 found a silent NaN gradient in — so every figure below is a conservative *upper* bound on that axis, the right direction for a budget).
- **Sweep grid (user decision 2026-07-17):** γ∈{0.5,1,2,3} × `optim.lr`∈{3e-5,1e-4,3e-4} × `class_weight_alpha`∈{0,0.25,0.5} = **36 points**. Window/stride held at the ADR-0005 D3+A3 1024/512 geometry; RC-combination + CRF deferred to P2-12 (per `train_stage1.py`'s code-emitted `swept_by` map). The two dead-on-arrival members of the stale PRD §11 "six-axis" enumeration (LoRA rank — moot under ADR-0002:129 full-fine-tune; Stage-2 aux-loss weight — a P3 axis) are excluded.
- **Two P2-06-specific per-point adjustments to the P2-05 figure.** (1) A sweep point trains on the **`inner_train` fold** (7,472 records, the D5 `nested_train` fold minus the P2-06a cluster-grouped selection-val carve), not the full 8,303 — training scales ~linearly with fold length: 1.6129 × (7,472/8,303) = **1.451 GPU-h/point**. (2) Each point additionally **evaluates on the 830-record selection-val rung** (P2-06a) — 3,334 windows forwarded / 1.96M positions scored, forward-only, rank-0 only, plus a 2000-replicate homology-cluster block bootstrap — **≤ 0.065 GPU-h/point** at the (conservative, forward-slower-than-real) 14.3 win/s/GPU rate.

**Frozen numbers.**
- **Point estimate ≈ 55 GPU-h** (active device-hours): 36 × (1.451 train + 0.065 eval) = 54.6 GPU-h. Under the conservative *un-rescaled* full-fold basis (36 × 1.6129, training-only) it is **≈ 58 GPU-h**; charging all 8 GPUs of the node through the rank-0 eval phase it is **≈ 68 GPU-h**. The three agree to within the accounting convention; **≈ 58 GPU-h is adopted as the pre-registered point estimate** (the conservative middle).
- **Pre-registered aggregate sweep bound: ≤ 90 GPU-h.** This is the D7 feasibility ceiling for the sweep. Headroom above the ~68 GPU-h reserved estimate covers: a bounded **≤ 2 % D8 batch re-submit** (≤ 1 of 36 points ≈ +1.6 GPU-h), the reserved-node eval accounting, and the fact that **`epochs = 10` is an assumption, not a pin** — GPU-hours scale linearly in it, so a production epoch count that lands modestly higher does not breach the bound. A measured sweep that would **exceed 90 GPU-h** triggers D7's governance lever (prune low-information grid levels; a prune that would compromise GATE-4 readiness is a §7 stop) — but see the finding below.
- **Continued-pretraining (CPT) pass — excluded from this freeze.** D7's rule includes the ADR-0002 D7 probe-triggered continued-pretraining / NT-multispecies pass in the aggregate. That pass is **probe-triggered (P3+) and not yet scoped**, so its GPU-hours cannot be measured now; fabricating a number would violate CLAUDE.md §10.3. It is therefore **not** in the 90 GPU-h bound. When/if the probe triggers and the pass is scoped, its first-measured cost is **added to the aggregate at a re-freeze** (an ADR-sign-off change, D7 governance) — the same freeze-at-first-measurement rule applied to a second cost centre.

**Finding — the D5-style prune lever is not triggered (recorded, not a silent skip).** At ~55–68 GPU-h against a quota-free `gpu` partition (8× A4000 on `two`; the sweep is ~8.5 wall-h run serially on one node, ~4.3 wall-h if `one` is also up — well inside the D6 patience frame), the sweep is nowhere near a ceiling that would force pruning the user-decided 36-point grid. Per D7/D5 this is a *finding to log*, not a reason to skip the freeze.

**Cross-reference correction (fixes D7's internal contradiction).** The Cross-reference impact list's imp.md line read *"P5 (sizing gate freezes D6/D7 numbers …)"*, which lumps the D7 sweep budget with the D6 scan ceiling at the pre-P5 sizing gate — contradicting D7 (line 78) and D10 (line 101), which freeze **D7 at first measurement**, *before* and *independent of* the P5 sizing gate that binds **D6**. The Decision section is normative; the cross-ref was stale. It is corrected to: **"P5 (sizing gate freezes the D6 ceiling + patience-days numbers; the D7 training/sweep budget froze at first Stage-1 measurement — P2-05 measured, P2-06 A2 enacted; scan array runs under D8/D9)."** (The D6 numbers remain deferred to the P5 sizing gate, unchanged.)

**Impact.** Binds the P2-06 sweep's feasibility gate and the CLAUDE.md §9.3 submit-ack cost estimate; recorded in the P2-06 dev-log and the sweep-summary stanza, and referenced by `slurm/p2/sweep_stage1.sbatch` + `train_stage1.py` (the grad-finiteness-scan retention note). Does **not** touch any scientific-claim threshold (ADR-0004/0005/0006) — an ops budget, so the gate-default blinded-freeze discipline does not apply (D7 governance), but the ADR-0001 D3 / D5 guarantee (a compute-forced cut never masquerades as a finding) makes its honest application load-bearing: a grid *prune* under this budget would be `log`ged, and the sweep selects a **config**, never bounds a biological result.

- **User sign-off:** ☑ recorded 2026-07-17 (bioedca), CLAUDE.md §7 item 2 (a number change → ADR sign-off). Approved as drafted; the P2-06 authoring commit lands the budget, and the SLURM submit-ack remains a separate §9.3 gate.
