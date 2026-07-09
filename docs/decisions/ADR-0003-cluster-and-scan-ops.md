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

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-08 (bioedca), CLAUDE.md §7 item 2.
