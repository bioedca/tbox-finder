# ADR-0002 — Environment & ML stack

- **Status:** Accepted (user sign-off recorded 2026-07-08, CLAUDE.md §7 item 2)
- **Date:** 2026-07-08
- **Deciders:** bioedca (project owner)
- **Phase:** P0 (seed ADR)
- **Delegated from:** PRD §3.1 (CLAUDE.md), §10.1–10.3, §15, §18.3
- **Supersedes / superseded by:** none
- **Related:** ADR-0001 (architecture & aims — pins the D5 swapped-backbone integrity clause and the D6 branch-B3 backbone-fallback route this ADR supplies the *values* for); ADR-0003 (cluster/scan ops, compute ceilings); ADR-0004 (GATE-4 default); ADR-0005 (GATE-1/GATE-2 machinery + the RNA-FM-swap ECE/recall margins for conditions (c)/(d)); ADR-0006 (validation rollup)

This is the **environment & ML-stack ADR.** It pins the **per-env conda-lock policy + CI image**, the **load-bearing ML/data/infernal version stack** (with a named version + rationale for every load-bearing pin), the **CUDA/sm_86 kernel pins + pure-PyTorch fallbacks**, the **RiNALMo commit + parity numerics** (the numeric parity margin **N**, its magnitude rationale, and the fallback phasing), the **RNA-FM-swap re-clearing conditions** (the swapped backbone re-clears the *absolute* GATE-1/GATE-2 gates and re-runs the P4→P5 go/no-go), and the **Caduceus prokaryotic-transfer two-part go/no-go + backbone fallbacks**. Every version fact herein was verified against authoritative sources (PyPI JSON, `download.pytorch.org`, GitHub release asset lists, the Caduceus HF config/modeling files, the RiNALMo paper) on **2026-07-08** — the assistant knowledge cutoff is Jan 2026, so *no version was taken from memory* (CLAUDE.md §3.3). The **empirical sm_86 wheel/build-availability confirmation** is deferred to **P0-06**, which amends this ADR with the observed result (PRD §10.3).

---

## Context

The stack is **load-bearing and never-silently-bumped** (CLAUDE.md §3.1): the reproducibility of every training/eval/scan artifact — and therefore the defensibility of the GATE-1 generalization claim — depends on an exactly-pinned, mutually-compatible environment that also *runs on the cluster's actual hardware*. The runtime target is fixed and non-negotiable (PRD §15):

- **Hardware:** NVIDIA RTX A4000 = **Ampere, compute capability `sm_86`, 16 GB VRAM**; CUDA 12.x toolkit; NVIDIA driver **590.x** (CUDA-13-capable, hence forward-compatible with any 12.x runtime).
- **No container runtime on the cluster** (no apptainer/singularity/docker/podman) → environments are **conda-lock** on the existing user miniconda3, one lockfile per env (CLAUDE.md §3.1). The CI **Docker image is the archival/portability artifact, not the cluster runtime.**
- **Two model backbones drive the stack**, and each carries a *hypothesis-to-validate* rather than an assumption: **Stage-1 Caduceus** (BiMamba/MambaDNA, RC-equivariant, single-nucleotide DNA; `trust_remote_code=True` from `kuleshov-group/caduceus-*`; **human-reference pretrained** → prokaryotic-transfer go/no-go) [arXiv:2403.03234; PMID:40567809], and **Stage-2 RiNALMo-giga** (650M RNA LM, FlashAttention-2; accessed via the transformers-native `multimolecule` mirror; **no published T-box result** → parity + transfer gates) [PMID:40593636; DOI:10.1038/s41467-025-60872-5; PMC12219582].

Two facts about the current package ecosystem (2026-07-08) shape every pin below and are the reason "pin the newest of everything" is **wrong** here:

1. **A `trust_remote_code` ceiling on `transformers`.** Caduceus is loaded as Hub remote code; `transformers` 5.0 removed `is_torch_fx_available` (PR #37234), which broke a broad class of remote-code modeling files (issue #44561). The Caduceus + `mamba_ssm` remote path is unvalidated on v5. **`transformers` is capped at the last 4.x release, 4.57.5**, which cascades the whole HF stack down to the Jan-2026 4.x epoch (`huggingface_hub` < 1.0, `tokenizers` ≤ 0.23.0, era-matched `accelerate`/`peft`/`deepspeed`/`datasets`).
2. **An `sm_86`-wheel ceiling on `torch`.** The three CUDA-kernel packages the backbones need — `mamba-ssm`, `causal-conv1d` (Caduceus), `flash-attn` (RiNALMo) — do **not** all publish prebuilt Ampere `sm_86`/CUDA-12 wheels at the same `torch` version, and the newest `mamba-ssm` (2.3.x = Mamba-3) makes a Hopper/Blackwell-oriented kernel stack a *hard* runtime dependency of unverified Ampere-import-safety. The `torch` anchor is therefore set by the **kernel ecosystem**, not by `torch` itself (D3).

The **`pytorch-cuda` conda metapackage is dead** — PyTorch stopped publishing its Anaconda `pytorch` channel after 2.5 (issue #138506). PRD §15's "GPU env pins `pytorch-cuda` matching driver 590 / CUDA 12.x" describes *intent* (a CUDA-12.x GPU PyTorch matching the driver); the **mechanism** is now a pinned `pip:` block against the CUDA-12.x wheel index inside the conda-lock env (D1/D3). This is a mechanism update within PRD §15's stated intent, not a scope change, and is flagged here per the CLAUDE.md precedence rule.

---

## Decision

### D1. Per-env conda-lock policy (locked)

- **Five environments, one lockfile each** (CLAUDE.md §3.1): `envs/{data,infernal,ml,viz,app}.yml`, each with its own `envs/<name>.conda-lock.yml` solved `-p linux-64` — **never** a single aggregate lock. Built by P0-05.
- **Python 3.12** across all five envs. This is **forced, not stylistic**: `numpy` 2.5.1 and `scipy` 1.18.0 both raise `requires-python >= 3.12`, so 3.11 would strand the data/viz stack on an older numpy/scipy *and* desync its numpy ABI from the `ml` env; the `ml` kernel wheels cover `cp312`; the infra floor (`snakemake`) is 3.11. 3.12 satisfies every pin and keeps one numpy ABI across envs. (3.13 is unnecessary bleeding edge.)
- **GPU-PyTorch mechanism (supersedes the literal `pytorch-cuda` wording of PRD §15).** The `pytorch-cuda` metapackage is deprecated; the Anaconda `pytorch` channel ended at 2.5. In the `ml` env, keep Python/BLAS/numpy/etc. as conda deps and pin `torch` **and the three CUDA-kernel packages** in the env's `pip:` block against the CUDA-12.x PyTorch wheel index (`--extra-index-url https://download.pytorch.org/whl/cu128`); `conda-lock` captures pip deps in the lock. This guarantees the kernel wheels **ABI-match the exact `torch` build** (all `manylinux_2_28` = C++11-ABI TRUE; the kernel wheels must be the `cxx11abiTRUE` variants). Pulling `torch` from conda-forge instead silently breaks this ABI match — do not.
- **Adding a dependency later** = edit the yml, regenerate that env's lock, re-run the affected golden/eval test, commit yml+lock together (CLAUDE.md §3.1). Never ad-hoc `pip install`/`conda install`; never `conda env update`/`pip install -U`.

### D2. The pinned stack (locked; every load-bearing pin named + rationalized)

Versions are the **latest mutually-compatible** stable releases as of 2026-07-08, chosen for compatibility over recency. `numpy 2.5.1` is pinned **identically** in the `data` and `ml` envs (one numpy-2 ABI shared by torch/multimolecule/pandas/scipy/sklearn/biopython — never let any dep pull a numpy-1 build).

**`ml` env (GPU; training / eval / scan) — the load-bearing ML stack.** GPU/kernel pins are D3.

| Package | Version | Source | Rationale (load-bearing) |
|---|---|---|---|
| `python` | 3.12 | conda-forge | numpy/scipy floor; kernel wheels cover cp312 |
| `torch` | **2.7.1 (+cu128)** | pip (cu128 index) | anchor set by the kernel ecosystem — see **D3** and **Considered options** |
| `mamba-ssm` | **2.2.6.post3** | pip (source-built) | Mamba-2, no Hopper deps — **D3** |
| `causal-conv1d` | **1.6.2.post1** | pip (prebuilt `cu12torch2.7`) | Caduceus depthwise-conv accelerator — **D3** |
| `flash-attn` | **2.8.3.post1** | pip (prebuilt `cu12torch2.7`) | FA2 = the Ampere line — **D3** |
| `transformers` | **4.57.5** | conda-forge | **ceiling** — last 4.x; Caduceus `trust_remote_code` unvalidated on v5 (config declares 4.38.1; v5 removed `is_torch_fx_available`, issue #44561). 4.57.0 was yanked → .5 |
| `tokenizers` | 0.22.2 | conda-forge | newest **stable** ≤ transformers' `<=0.23.0` cap (0.23.0 was skipped; 0.23.1 exceeds it) |
| `huggingface_hub` | 0.35.3 | conda-forge | last 0.x; transformers 4.57.5 pins `<1.0,>=0.34.0` (hub 1.x = the v5 line) |
| `safetensors` | 0.7.0 | conda-forge | era-matched to the 4.57 epoch (>=0.4.3 required; 0.8.0 is the v5 era) |
| `accelerate` | 1.12.0 | conda-forge | FSDP `FULL_SHARD` + DeepSpeed ZeRO-3 plugins for the §10.3 full-FT fallback; era-matched, no transformers cap |
| `peft` | 0.18.1 | conda-forge | `LoraConfig(target_modules="all-linear")` + grad-checkpointing (§10.3 LoRA default); era-matched, no transformers cap |
| `deepspeed` | 0.18.5 | conda-forge | ZeRO-3 full-FT fallback (states shard to ~1.3 GB/GPU ×8); JIT-builds ops vs local CUDA 12.x (needs in-env nvcc) |
| `datasets` | 4.5.0 | conda-forge | era-matched; coupled to HF only via `huggingface_hub` (satisfied) |
| `multimolecule` | **0.0.9** | pip | the RiNALMo/RNA-FM transformers-native mirror; **era-matched to transformers 4.57.5** (avoid 0.1.0/0.2.0 which may assume v5 internals) — **D5** |
| `wandb` | 0.28.0 | conda-forge | offline mode (`WANDB_MODE=offline`) + login-node `wandb sync`; pulls `pydantic>=2.6` |
| `hydra-core` / `omegaconf` | 1.3.2 / 2.3.0 | conda-forge | run-config driver; `omegaconf<2.4` cap (do **not** use the 1.4.0/2.4.0 dev pre-releases) |
| `numpy` | 2.5.1 | conda-forge | shared numpy-2 ABI (identical to `data` env) |
| `cuda-nvcc` + `ninja` + `packaging`/`setuptools>=61` | CUDA 12.8 line | nvidia/conda-forge | toolchain to source-build `mamba-ssm` for `sm_86` and JIT DeepSpeed ops (D3/D4) |

**`data` env (CPU DAG / ingest / labels / splits / metrics).** `python` 3.12; `numpy` 2.5.1, `scipy` 1.18.0, `pandas` 3.0.3, `polars` 1.42.1, `scikit-learn` 1.9.0, `biopython` 1.87 (all conda-forge, numpy-2 native); `snakemake` 9.23.1 + `snakemake-executor-plugin-slurm` 2.7.1 (bioconda; the SLURM executor is used for **CPU DAG bookkeeping only** — GPU jobs are hand-authored sbatch, PRD §16); `dvc` 3.67.1 + `dvc-ssh` 4.3.0 (conda-forge; `dvc-ssh` 4.x is **sshfs/asyncssh**-based, not paramiko — install via the `dvc[ssh]` extra so the two stay `>=4,<5` in lockstep); `hydra-core` 1.3.2 / `omegaconf` 2.3.0.

**`infernal` env (baseline + folding).** `infernal` **1.1.5** (bioconda) — the `cmsearch`/`cmbuild`/`cmcalibrate` suite that is the **Stage-0 baseline the GATE-1 recall claim is measured against**; its version string is load-bearing for the recall-vs-`cmsearch` metric — do not float it. `viennarna` 2.7.2 (**bioconda only** — not on conda-forge; SWIG Python bindings). `biopython` 1.87, `numpy` 2.5.1.

**`viz` / `app` envs (non-load-bearing toolchains).** The plotting/Quarto (`viz`) and Svelte-atlas build/serve (`app`) toolchains are **not** part of the load-bearing ML-reproducibility surface; their exact pins are settled at **P0-05** build time. Where they touch numerical data they share the **`numpy` 2.5.1** pin for ABI consistency.

**Pin-coupling notes (watch when bumping):** `snakemake` 9.23.1 ↔ `snakemake-executor-plugin-slurm` 2.7.1 both target the `snakemake-interface-executor-plugins` 9.x line and co-resolve — re-check they still land on one 9.x interface release if either is bumped. `hydra-core` needs `omegaconf<2.4` (also pulled transitively unpinned by `dvc`) — keep `omegaconf<2.4` on any bump. Channel skew: PyPI is ahead of conda-forge for `hydra-core` (1.3.4) / `omegaconf` (2.3.1); for a pure-conda lock use the conda-forge versions above.

### D3. Torch/CUDA anchor + `sm_86` kernel pins & pure-PyTorch fallbacks (locked; empirical check → P0-06)

**Anchor: `torch` 2.7.1, CUDA-12.x build (`cu128`), C++11-ABI TRUE (`manylinux_2_28`).** CUDA 12.8 is the safe 12.x choice on driver 590.x (Ampere `sm_86` is in `torch`'s default `TORCH_CUDA_ARCH_LIST` for every 12.x build; driver 590 is forward-compatible to CUDA 13). torchvision/torchaudio are **not required** (both backbones are token-sequence models); add only if a later step demonstrably imports them, at the co-released version from the same index.

**Kernel pins (all `sm_86`-capable on CUDA 12.x):**
- **`flash-attn` 2.8.3.post1** — prebuilt Ampere wheel exists: `flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-linux_x86_64.whl` (Dao-AILab GitHub release; `sm_86` covered by the always-compiled `compute_80` gencode). **FA2 is the correct line for Ampere** — `flash-attn` 3.x (SM90/Hopper-only) and the `fa4` betas (Blackwell/Hopper) do **not** target `sm_86` and merely auto-fall-back to FA2 there.
- **`causal-conv1d` 1.6.2.post1** — prebuilt `cu12torch{2.6..2.10}` Ampere wheels; **optional** accelerator (Mamba falls back to `nn.Conv1d` without it).
- **`mamba-ssm` 2.2.6.post3** — the **last pre-Mamba-3 release**, keeping the classic CUDA `selective_scan` kernel (gencode `compute_80`/`sm_80` + `sm_87` = native Ampere) with minimal deps. Its prebuilt `cu12` wheels cap at `torch2.5`, so at the `torch` 2.7 anchor it is **source-built** (nvcc 12.x via in-env `cuda-nvcc`, `ninja`, `--no-build-isolation`; compiles `sm_86` cleanly). This one source build is captured/verified at P0-06 and baked into the CI image (D4) for reproducibility. **Mamba-3 (`mamba-ssm` 2.3.x) is rejected** — it makes `quack-kernels` (nvidia-cutlass-dsl CuTe-DSL), `tilelang`, and `apache-tvm-ffi` **hard** runtime deps of unverified Ampere-import-safety, a load-bearing liability for **zero Caduceus benefit** (Caduceus is Mamba-2).

**Pure-PyTorch fallbacks (confirmed in source; P0-06 records the throughput cost, PRD §10.3):**
- **Mamba selective-scan:** `Mamba(use_fast_path=False)` → `selective_scan_ref`/`mamba_inner_ref` in `mamba_ssm/ops/selective_scan_interface.py`, skipping the fused kernel; `causal-conv1d` degrades to `nn.Conv1d`. So a missing kernel is a **speed cost, not a capability loss**. The measured genome-scan throughput cost of the fallback is recorded in this ADR at P0-06.
- **FlashAttention-2:** `torch.nn.functional.scaled_dot_product_attention` (SDPA)/eager. **Important:** the *reference* `lbcb-sci/RiNALMo` repo has **no** SDPA fallback (it calls flash-attn unconditionally), but the **`multimolecule` HF port we use provides the SDPA/eager path** — so accessing RiNALMo through the `multimolecule/rinalmo-giga` mirror (D5) makes flash-attn an **accelerator, not a hard requirement**. This is an additional reason to prefer the mirror.

**Empirical gates:** the **`sm_86` wheel/build-availability check is P0-06** (folded back into this ADR as an amendment with the observed wheel tags + the source-build result); the **kernel import/forward smoke gate is P1** (PRD §10.3). If P0-06 finds the `mamba-ssm` source build unreliable on the cluster, the **pre-registered fallback is Considered-option A** below (torch 2.5, all-prebuilt wheels) — an ADR-0002 amendment, not a silent switch.

#### Considered options for the anchor (the sign-off decision)

- **Option B — `torch` 2.7.1 + cu128 (RECOMMENDED, adopted above).** Prebuilt `sm_86` wheels for `flash-attn` + `causal-conv1d`; **one** source build (`mamba-ssm` 2.2.6.post3); clean Mamba-2 kernels with no Hopper deps; modern torch; **all three kernel packages have exact, confirmed, named versions**.
- **Option A — `torch` 2.5.1 + cu124 (pre-registered fallback).** **Zero source builds** — prebuilt `sm_86` wheels for all three (`flash-attn` 2.8.3.post1 `cu12torch2.5`, `mamba-ssm` 2.2.6.post3 `cu12torch2.5`, `causal-conv1d` **1.5.x** `cu12torch2.5`). Marginally more reproducible / no cluster-nvcc dependency, but `torch` is ~20 months old and the `causal-conv1d` 1.5.x exact patch must be reconfirmed at env-lock. **Adopted only if** P0-06 shows the Option-B `mamba-ssm` source build is unreliable on the cluster.
- **Option C — `torch` 2.10 + `mamba-ssm` 2.3.x (REJECTED).** All-prebuilt at the newest torch, but drags the Hopper/Blackwell `quack-kernels`/`tilelang`/`apache-tvm-ffi` hard-dep stack of unverified Ampere-import-safety into a load-bearing `sm_86` env — unacceptable risk for zero benefit.

### D4. CI Docker image (locked)

`docker/Dockerfile` builds a **CUDA-12.8 + PyTorch** image in CI (GHCR), from a `nvidia/cuda:12.8.*-cudnn-devel-ubuntu` base (the `-devel` base supplies nvcc for the `mamba-ssm` source build), installing the `ml` env from its committed `conda-lock.yml`. It is the **archival/portability/reviewer artifact and the path to a future apptainer/udocker run — not the cluster runtime** (no container runtime on the cluster, D1). Built on schedule/release (CLAUDE.md §8.6), not per-push.

### D5. RiNALMo commit + parity numerics (locked; the delegated margin **N** is authoritative here, D4/§2.3 carve-out)

- **Commit to RiNALMo, parity-gated (PRD §10.2).** Stage-2 is **RiNALMo-giga (650M)**, accessed via the transformers-native mirror **`multimolecule/rinalmo-giga`** (`multimolecule` **0.0.9**); the official `lbcb-sci/RiNALMo` weights (Zenodo 15043668) are the fallback. RiNALMo ingests **sequence only** (no structure-input channel). A **P1 parity step** confirms the mirror reproduces RiNALMo's published **ArchiveII nine-family leave-one-family-out (inter-family) secondary-structure generalization** on the same LOFO splits **before** committing.
- **Parity criterion + the numeric margin N.** Primary criterion (PRD §10.2) = "mirror inter-family F1 within the **published SD** on the same LOFO splits." **The RiNALMo paper reports no per-split SD** (only per-family mean F1 in Fig. 4a + per-structure box-plots in Fig. 4b), so the primary criterion is **not directly usable** and the pinned **±N pp fallback governs**:
  - **N = 2 pp (±0.02 F1)**, applied to the **mean inter-family F1 (0.72)** and to **each of the eight stable held-out families** (5S 0.88, SRP 0.70, tRNA 0.93, tmRNA 0.80, RNase P 0.80, Group I intron 0.66, 16S 0.74, 23S 0.85).
  - **Telomerase carve-out:** the **telomerase-RNA** family (published F1 **0.12**, near the failure floor, high-variance — the acknowledged outlier) is **excluded from the strict parity gate** (or given a widened ±5–10 pp band). A same-weights mirror can swing several pp there on benign seed noise; gating parity on telomerase alone would manufacture false failures. Parity is decided on the **mean + the eight stable families at ±2 pp**.
  - **Magnitude rationale (documented per D4/§2.3, blinded-frozen at P0).** The published values are printed at 0.01 (1 pp) resolution → 1 pp is the irreducible reporting granularity; the mirror loads the **same** weights, so the only expected divergence is fine-tuning-run stochasticity (head init, shuffle seed) + reimplementation numerical drift (tokenizer mapping, attention/precision path) — ~1 pp on a family-*mean* F1 (which averages over many structures and is far tighter than the wide per-structure box-plot IQRs). 2 pp cleanly covers 1 pp rounding + ~1 pp noise **without admitting a meaningful accuracy gap**. The huge 0.12–0.93 across-family spread is **biological family-difficulty variation, not mirror noise**, and must not set N. A >2 pp shift on the mean or on multiple stable families is treated as a **real reimplementation discrepancy to investigate**, not accepted. Source: RiNALMo, `PMID:40593636; DOI:10.1038/s41467-025-60872-5; PMC12219582` (Fig. 4a/4b), accessed 2026-07-08.
- **The `transformers`-4.57.5 ceiling ↔ `multimolecule` tension (flagged for sign-off).** Caduceus caps `transformers` at 4.57.5 (D2); `multimolecule` pins **no** `transformers` version, so 0.0.9 is era-matched to that epoch and the **P1 parity gate runs under exactly `transformers` 4.57.5 + `multimolecule` 0.0.9**. **Contingency:** if a *required* future `multimolecule` hard-requires `transformers ≥ 5` while Caduceus still needs ≤ 4.57.5, Stage-1 (DNA) and Stage-2 (RNA) can no longer share one `transformers` in a single `ml` env → the pre-registered resolution is an **env split** (`envs/ml-dna.yml` + `envs/ml-rna.yml`), an ADR-0002 amendment under sign-off — **not** a silent pick.
- **P1 forward-throughput probe = advisory-only (PRD §10.2).** A RiNALMo forward-throughput probe (candidates/sec/GPU on the pretrained checkpoint) runs alongside the parity gate to surface the latency risk early, but the **binding** latency decision is frozen at the **P5 sizing gate** (condition (b) below); a crude pre-P3 "clearly-hopeless" reading is a **CLAUDE.md §7 stop-and-ask, not an auto-switch**.

### D6. RiNALMo → RNA-FM fallback phasing + swapped-backbone re-clearing (locked; (c)/(d) margins delegated to ADR-0005)

Stage-2 falls back to **RNA-FM** (`multimolecule/rnafm`, ~100M; drop-in via the same `multimolecule` interface, so sunk cost is bounded) **only** on a phase-assigned trigger (PRD §10.2):
- **(a) [P1 parity gate]** the D5 parity gate fails;
- **(b) [VRAM at P1 / §10.3; genome-scale latency frozen at the P5 sizing gate]** RiNALMo-giga cannot meet the A4000 16 GB-VRAM + genome-scale-latency budget after PEFT/bf16;
- **(c) [P3–P4]** post-calibration leave-clade-out ECE is worse than RNA-FM's by a **pre-registered numeric margin — pinned in ADR-0005**, not here;
- **(d) [P4]** RiNALMo transfers worse on the primary discovery metric (leave-clade-out recall@matched-precision / AUPRC) by a **pre-registered margin — pinned in ADR-0005**, even if ECE matches.

**Swapped-backbone integrity (mirrors ADR-0001 D5).** **If the swap fires, the substituted RNA-FM Stage-2 must clear the *absolute* GATE-1 (+10 pp point / +5 pp CI-floor) and GATE-2 (in-distribution ECE ≤ 0.05) thresholds and re-pass the P4→P5 go/no-go on its own P3/P4 numbers before entering the GATE-3 scan** — the shipped backbone's gates are **never inherited from RiNALMo**. The covariance model is kept as an **orthogonal cross-validator, never the sole Stage-2** (PRD §10.2). (The absolute GATE-1/GATE-2 threshold *values* are authoritative in ADR-0005; this ADR pins only that the swapped backbone must re-clear them and re-run the go/no-go.)

### D7. Caduceus prokaryotic-transfer two-part go/no-go + backbone fallbacks (locked)

The public Caduceus checkpoints are **human-reference pretrained**, while the discovery scan is fully prokaryotic/archaeal/MAG — so Stage-1 cross-domain transfer is a **hypothesis to validate, symmetric with RiNALMo's parity gate** (PRD §10.1). A **two-part P1 go/no-go runs before P2 commits training compute:**
1. **Cheap frozen-embedding binary linear-separability pre-filter** — held-out bacterial/archaeal T-box vs GC-matched prokaryotic background (an advisory screen: is there above-chance linear separability in the frozen embeddings?).
2. **Binding per-nucleotide fine-tune segmentation smoke F1** on held-out prokaryotic loci, matching the GATE-4-graded task — because a binary probe can pass while the embeddings lack the positional structure for segmentation, **or** fail while full fine-tuning succeeds. **This part is binding**; the pre-filter is not.

**Go/no-go rule.** *Go* iff both are positive: the pre-filter shows above-chance separability **and** the segmentation smoke recovers the three core elements (Stem I, Specifier, Antiterminator) at per-nt F1 **clearly above a background-only baseline** on held-out prokaryotic loci — a directional "the backbone can learn prokaryotic T-box boundaries" signal, **not** the GATE-4 bar (that is graded at P2 on the full split). A **borderline** read is a **CLAUDE.md §7 stop-and-ask**, not an auto-decision (the exact small-N smoke threshold is set with the P1 fixture, since it is a go/no-go and not a blinded-frozen gate).

**Fallbacks on weak transfer, in order (PRD §10.1/§10.3):** (1) full fine-tuning (already the Stage-1 default); (2) an **optional GTDB-prokaryotic continued-pretraining pass** before T-box fine-tuning (budgeted SLURM, probe-triggered); (3) a **prokaryotic/multispecies DNA backbone** (e.g. **Nucleotide-Transformer multispecies**) as a Stage-1 ablation/fallback. A **transfer-driven** (not merely capacity-driven) **GATE-4 failure at P2 re-triggers these same continued-pretraining / NT-multispecies fallbacks** — this is the concrete backbone-fallback ladder that **ADR-0001 D6 branch B3** routes to before any terminal pivot (a capacity/transfer method-gate failure is *not* auto-routed to the GATE-1-failure deliverable).

### D8. Never-silently-bump discipline + amendment procedure (locked)

- The `torch`/kernel/`transformers`/`multimolecule`/RiNALMo-stack pins are **load-bearing** (CLAUDE.md §3.1, §13). No `conda env update`, no `pip install -U`, no silent bump. A bump requires **user approval + an ADR-0002 amendment listing which steps re-validate** (CLAUDE.md §3.1).
- **Two bellwethers to watch** (bump *triggers*, not licenses to bump silently): (i) **`flash-attn`** is the binding torch-ceiling package — a `flash-attn` release adding `torch>2.10` prebuilt Ampere wheels is the trigger to *consider* re-anchoring the whole GPU group; (ii) the **`transformers`-4.57.5 ↔ `multimolecule`** tension (D5) — a required `multimolecule` needing `transformers ≥ 5` triggers the env-split contingency.
- **Re-validation on any bump** (CLAUDE.md §8.5): re-run the kernel import/forward smoke (P1), the RiNALMo parity gate (D5), and the affected golden/eval gates before the amended lock is committed.

### D9. Delegations (values NOT pinned here)

- **ADR-0005:** the RNA-FM-swap ECE/recall trigger **margins** for conditions (c)/(d) (D6); the **absolute GATE-1/GATE-2 threshold values** the swapped backbone must re-clear.
- **ADR-0004:** the GATE-4 per-element F1 default (0.80) the P2 segmentation is graded on.
- **ADR-0003:** the per-corpus/aggregate compute ceilings + training/sweep GPU-hour budget that bound the continued-pretraining pass (D7) and the scan.
- **P0-06 (amends this ADR):** the empirical `sm_86` wheel/build-availability result + the measured pure-PyTorch-fallback throughput cost + the Option-A/B anchor confirmation.

---

## Consequences

- **Positive.** Every training/eval/scan artifact is reproducible from an exactly-pinned, mutually-compatible, `sm_86`-runnable stack; the two ecosystem ceilings (`transformers`←Caduceus, `torch`←kernels) are made explicit so no downstream step silently bumps into a broken combination; the RiNALMo parity margin, its magnitude rationale, and the RNA-FM fallback phasing are pre-registered and blinded-frozen; the swapped backbone can never inherit RiNALMo's gates; the Caduceus go/no-go and backbone-fallback ladder give ADR-0001's B3 branch concrete values.
- **Costs / constraints.** The `transformers`-4.57.5 ceiling forgoes v5 features and creates a latent env-split risk (D5) if `multimolecule` moves to v5. The `torch`-2.7 anchor requires **one source build** (`mamba-ssm`) with an in-env CUDA toolchain, with Option A (all-prebuilt, older torch) as the pre-registered fallback if the build is unreliable. The never-bump discipline trades late flexibility for reproducibility integrity.
- **Empirical items still open (by design).** The `sm_86` wheel/build availability (P0-06), the kernel import/forward smoke (P1), the RiNALMo parity result (P1), and the Caduceus transfer go/no-go (P1) are confirmed at their named gates; this ADR pins the *targets, decision rules, and fallbacks*, not their empirical outcomes.

## Cross-reference impact list

- **PRD:** §3.1 (env policy in CLAUDE.md), §10.1 (Caduceus backbone + transfer go/no-go + checkpoint map), §10.2 (RiNALMo commit/parity + RNA-FM conditions a–d), §10.3 (fine-tuning footprint + `sm_86` kernels/fallbacks), §15 (compute/env; `pytorch-cuda`-mechanism update flagged), §18.1 (P0/P1 exit gates), §18.3 (ADR index).
- **Sibling ADRs:** ADR-0001 (D5 swapped-backbone integrity + D6 branch-B3 backbone fallbacks — this ADR supplies their values); ADR-0003 (compute ceilings bounding continued-pretraining + scan); ADR-0004 (GATE-4 default); ADR-0005 ((c)/(d) RNA-FM margins + absolute GATE-1/GATE-2 values); ADR-0006 (rollup).
- **imp.md:** P0-04 (this step); P0-05 (builds the five envs from these pins); P0-06 (empirical `sm_86` check + fallbacks — amends this ADR); P0-07 (CI Docker image); P1 (kernel smoke, Caduceus go/no-go, RiNALMo parity + throughput probe).
- **CLAUDE.md:** §3.1 (conda-lock + never-bump), §3.2 (rule=env), §3.3 (context7 before pinning), §7 item 2 (this sign-off) + item 4 (borderline go/no-go stop-and-ask), §8.5 (re-validate on bump), §10.1 (parity citation carried).
- **Cards / paper (release-bound):** model card (backbone versions, LoRA/FSDP config, parity result, Caduceus transfer outcome); `docker/Dockerfile` + `envs/*.conda-lock.yml` (the reproducibility artifacts); `paper/manuscript.qmd` (methods: pinned stack + `cmsearch` 1.1.5 baseline).

## Sign-off

- **User sign-off:** ☑ recorded 2026-07-08 (bioedca), CLAUDE.md §7 item 2 — Option B anchor adopted as-is (torch 2.7.1+cu128, `mamba-ssm` 2.2.6.post3 source-built; Option A all-prebuilt is the pre-registered P0-06 fallback).
