# P0-10 — DVC-remote path pin + scratch-capacity verification record

**Date:** 2026-07-08 · **Step:** imp.md P0-10 · **Rule:** ADR-0003 D3 (estimate-anchored
DVC-capacity rule) · **PRD:** §15, §16, §5.2 (CLAUDE.md) · **Verification:** login-node `df`
(artifact/`df`-based; not a SLURM job, no submit-ack) · **Compute:** LOCAL + `ssh two`

This is the committed capacity record required by ADR-0003 D3 ("the **path + free-capacity
verification** … is the mechanical P0-10 step"). The `dvc-ssh` remote definition itself lives in
per-machine `.dvc/config.local` (gitignored, CLAUDE.md §5.2) and is **not** committed.

---

## 1. Pinned DVC remote

| field | value |
|---|---|
| remote name | `cluster-scratch` (DVC default) |
| URL | `ssh://ecc1695@two/onedata/mondragonlab/ecc1695/tbox-finder/dvc-remote` |
| filesystem | `/onedata` — `beegfs_nodev` (shared parallel FS, reachable from login node `two` **and** the `gpu` compute nodes) |
| auth | ssh-agent (ED25519); **no** `keyfile` in config.local — see §4 |
| config location | `.dvc/config.local` (per-machine, gitignored; both `/.dvc/config` and `/.dvc/config.local` are in the repo `.gitignore`) |

Chosen filesystem rationale: `/onedata` is a **shared** beegfs volume (so a scan job on compute
node `two` can write a final artifact that a `dvc push` from either the login node or the laptop
resolves to the same store), it is **user-owned** at `/onedata/mondragonlab/ecc1695` (writable
without escalation), and it carries the **most free headroom** of the lab-accessible volumes. It is
distinct from the 88 %-full HOME (`/scr/exports`, 691 GB free), which PRD §15 explicitly rules
insufficient as a capacity basis.

## 2. Capacity verification (the ADR-0003 D3 gate)

`ssh two df -B1 /onedata` (2026-07-08):

| quantity | bytes | human |
|---|---:|---:|
| total | 159,369,571,860,480 | 145 TiB (159.4 TB) |
| used (80 %) | 126,486,394,699,776 | 115 TiB (126.5 TB) |
| **free** | **32,883,177,160,704** | **29.9 TiB (32.9 TB)** |

**Provisional P5-estimated artifact volume: 2 TB** (derivation in §3).

**Gate:** free capacity **32.9 TB ≥ 2 TB estimate** → **PASS**, with ≈ **16× headroom**. No
retention-pruning fallback needed; no CLAUDE.md §7 stop-and-ask triggered (the Stop fires only when
free < estimate and pruning cannot close the gap — ADR-0003 D3).

## 3. Provisional P5 artifact-volume estimate — derivation

ADR-0003 D3/D6: the **real** P5 artifact volume freezes at the pre-P5 sizing-benchmark gate (it
needs the P2 scanner's Stage-1 candidate density, unknowable at P0). Until then P0-10 verifies
against a **conservative provisional** estimate. This is an **ops** engineering estimate, not a
scientific-claim threshold — so no ADR-0005 blinded-freeze and no §10.1 evidence gate applies; it is
re-verified (and, if it moves, re-signed-off) when the sizing gate freezes the real number.

DVC tracks `data/interim/`, `data/processed/`, and model checkpoints (CLAUDE.md §5.2). The
multi-TB **raw genome corpora** (GTDB reps + RefSeq + MAGs) are **not** in this budget — ADR-0003
D1/D3 stream them to node-local `/tmp/$USER-$SLURM_JOB_ID` and discard them; only final derived
artifacts move to the remote. Conservative (generous) component budget:

| component (DVC-tracked) | provisional |
|---|---:|
| Model checkpoints — Stage-1 Caduceus scanner (production + RC/context/head + class-II-CM-naive ablations + hard-negative-mining iterations), Stage-2 RiNALMo fine-tune, continued-pretraining pass | ~300 GB |
| Stage-1 genome-scan candidate tables (interim; higher-recall, pre-Stage-2) | ~500 GB |
| Stage-2 processed / calibrated candidate table (`data/processed`) | ~200 GB |
| Training + eval datasets — tokenized windows for the §11 six-axis sweep, hard-negative rounds, synthetic class-II recovery set, §9.1 decoy/negative corpora | ~500 GB |
| Alignments, split-assignment tables, union-prior reconciliation, calibration + eval artifacts, figure sources | ~200 GB |
| **Subtotal** | **~1.7 TB** |
| **Rounded up (margin) → provisional estimate** | **2 TB** |

Even a 3× overrun of this estimate (6 TB) clears the 32.9 TB free with > 5× headroom.

## 4. dvc-ssh connectivity smoke (through the ProxyJump)

The remote is reached via the `~/.ssh/config` `Host two` alias, which ProxyJumps through
`zero.biochem.northwestern.edu`; dvc-ssh (asyncssh 2.23.0) resolves connection params in the order
DVC-params → URL → **ssh config** → defaults, so it honors the alias + ProxyJump.

- A throwaway file was `dvc add`-ed and `dvc push -r cluster-scratch`-ed → **"1 file pushed"**; the
  object landed on the cluster at `dvc-remote/files/md5/33/7bf06…`. The probe (file, `.dvc`
  pointer, local cache, and remote object) was then fully removed — nothing throwaway is committed
  and the remote dir is left empty for real artifacts.
- **Durable finding:** the local key `~/.ssh/id_ed25519` is passphrase-encrypted. Setting `keyfile`
  in config.local makes dvc-ssh import the key file directly → *"Passphrase must be specified to
  import encrypted private keys"* and the push fails. The fix is to **omit `keyfile`** and let
  asyncssh use the **ssh-agent** (`allow_agent` default true), exactly as interactive `ssh two`
  does. config.local therefore carries only `url` (no `keyfile`, no secret).

## 5. Post-merge note (per-machine config.local)

`.dvc/config.local` is gitignored and lives only in this worktree; after squash-merge + worktree
removal it must be re-created in the primary checkout (it does not travel with the merge). Recreate
with, from the repo root:

```
dvc remote add --local -d cluster-scratch \
  ssh://ecc1695@two/onedata/mondragonlab/ecc1695/tbox-finder/dvc-remote
```

(no `keyfile` — rely on the ssh-agent, §4). The committed scaffold (`.dvc/.gitignore`, `.dvcignore`)
marks the repo DVC-initialized; the remote is per-machine by design (CLAUDE.md §5.2).

## 6. Re-verification trigger

Re-run this verification when the **pre-P5 sizing gate** (ADR-0003 D6) freezes the real P5
artifact-volume number. If the frozen number exceeds the then-current free capacity and the
retention-pruning fallback cannot close the gap, that is a **CLAUDE.md §7 stop-and-ask** (ADR-0003
D3) — never a silent skip of the phase-gate `dvc push`.
