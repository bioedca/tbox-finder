"""Closed allow-list of the pinned **Stage-2 RNA** backbones (ADR-0002 D5/D6 + **A15**).

**Why this module exists.** Until P3-17 the Stage-2 backbone identity was two bare constants
in :mod:`tbox_finder.eval.rinalmo_parity` (``REPO_ID`` / ``REVISION``),
``lora_harness.load_rinalmo_backbone`` hard-coded ``RiNALMoModel``, and three validators
*refused* any other ``repo_id``. That is the right posture for a **production** pin, and it is
kept — but it made ADR-0002 **D6**'s pre-registered RNA-FM comparator unreachable, and PRD
§10.2's condition (c) needs one to compare against. Worse, it would have been unreachable in
the silent way: ``conf/model/rinalmo_stage2.yaml`` is a *record*, not an input, so a
``model.repo_id=multimolecule/rnafm`` override composes cleanly, is refused-or-ignored, and the
run reports the code constants — the P2-11 ``label_source`` footgun class (CLAUDE.md §10.3).

This module is the RNA-side twin of :mod:`tbox_finder.models.backbone_registry` (the Stage-1
Caduceus allow-list added by ADR-0002 A14) and follows it deliberately: selection goes through
a **closed** allow-list, an unknown key or an unpinned revision **raises**, and the parameter
count is **measured at load** and checked against the recorded expectation rather than asserted
from the PRD.

**Torch-free, and load-bearing twice over** (the A14 rationale, restated because it is the
reason the file looks like this):

* ``Stage2TrainConfig.__post_init__`` can reject an unknown ``backbone`` at **compose** time on
  the login node — before ``sbatch``, before a GPU node, before a 2.5 GB download — instead of
  dying after the queue wait (the job-669 class);
* the **bare CI tier** and the torch-free ``tbox-finder-data`` env can both lock the allow-list.

So the multimolecule model class is carried as a **string** and resolved by the loader; nothing
here imports ``torch``, ``transformers`` or ``multimolecule``.

**The production entry is single-sourced, not re-declared.** ``rinalmo-giga``'s repo id,
revision and hidden size are *imported* from :mod:`tbox_finder.eval.rinalmo_parity` — the module
P1-13's parity gate proved faithful (ADR-0002 A9). Re-typing them here would let the shipped
pin and this table drift apart silently, which is the whole failure mode the file exists to
prevent.

**Provenance of the numbers.** Revisions verified against the Hugging Face API on 2026-08-21.
Each ``expected_param_count`` is the **backbone** parameter count — what
``<Class>.from_pretrained(..., add_pooling_layer=False)`` actually instantiates, i.e. the
``model.*`` tensors, *excluding* the ``lm_head`` / ``ss_head`` a Stage-2 arm drops. It was
obtained by summing the checkpoint's own ``model.safetensors`` header shapes, a method
**validated** by reproducing the independently measured live count for ``rnafm``
(99,111,680, from ``sum(p.numel())`` on a CPU load under ``multimolecule`` 0.2.0 on
2026-08-21) **to the integer** before it was trusted for the giga checkpoint. Neither number is
copied from the PRD: PRD §10.2's "650M" and "~100M" are round labels, and
:attr:`RnaBackbone.prd_label` carries them separately so a table can print both without one
standing in for the other.

**⚠ The two entries do not run under the same environment, and that is recorded per-entry.**
ADR-0002 A15: ``multimolecule`` 0.1.0 (``envs/ml-rna.yml``, the pin every shipped Stage-2
artifact was produced under) **cannot load** ``multimolecule/rnafm`` at all — its
``configuration_rnafm.py`` demands ``vocab_size == 26`` for the nucleotide variant while the
published checkpoint declares 28 — and no older checkpoint revision exists to pin around it.
A15 therefore added ``envs/ml-rnafm.yml`` (``multimolecule`` 0.2.0) rather than bumping
``ml-rna`` and firing CLAUDE.md §8.5 across every published RiNALMo number.
:attr:`RnaBackbone.env_lock` carries which lock an arm runs under so a run's provenance names
the environment it actually used, and so the **P3-18 condition-(c) comparison can state the
one-minor-version difference instead of inheriting it silently**.

**What this module does NOT do.** It does not fire the D6 swap. Adding ``rnafm`` here makes the
comparator *buildable*; whether it *replaces* RiNALMo is ADR-0005 D17's margin, measured at
P3-18, and a CLAUDE.md §7 decision on top of that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# Single-source the production pin from the module P1-13's parity gate proved faithful
# (ADR-0002 A9). `rinalmo_parity` is bare-importable — its multimolecule import is lazy — so
# this costs the torch-free tier nothing.
from tbox_finder.eval.rinalmo_parity import D_MODEL, N_LAYERS, REPO_ID, REVISION

#: Length of a git commit SHA — the only revision form a pin may take (never ``main``; D2).
_SHA_LEN = 40


@dataclass(frozen=True)
class RnaBackbone:
    """One pinned Stage-2 RNA backbone: identity + the facts a run must re-derive.

    ``expected_param_count`` is an *expectation to check a measurement against*, never a value
    to report in place of one — the A14 contract, restated because the temptation is identical.

    ``hidden_size`` matters here for the same reason ``d_model`` does on the Stage-1 side: the
    seven Stage-2 heads are ``nn.Linear(d_model, …)``, and RNA-FM is **640** against
    RiNALMo-giga's **1280**, so a head sized to the wrong constant would mismatch the hidden
    state. :func:`tbox_finder.stage2.model._hidden_size` measures it off the live module; this
    field is what that measurement is checked against.
    """

    key: str
    repo_id: str
    revision: str
    #: The ``multimolecule`` class name, carried as a **string** so this module stays
    #: torch-free; :func:`tbox_finder.train.lora_harness.load_rna_backbone` resolves it.
    model_class: str
    hidden_size: int
    n_layer: int
    n_heads: int
    #: ``config.max_position_embeddings``. NOT the tokenizer's context cap: the shared
    #: :mod:`tbox_finder.stage2.tokenizer` mirror pins 1024 (1022 nucleotides + 2 specials),
    #: which is ``<=`` both entries' value, so the existing ceiling stays conservative for
    #: both. Carried so an entry that *lowered* it could not slip in unnoticed.
    max_position_embeddings: int
    position_embedding_type: str
    #: Embedding rows. Both entries are **28** — the same table
    #: :data:`tbox_finder.stage2.tokenizer.VOCAB` mirrors — which is what makes the two arms
    #: comparable on identical tokenisation (ADR-0002 A15, asserted by a test rather than
    #: assumed).
    vocab_size: int
    #: Measured (see the module docstring); the loader compares, never asserts.
    expected_param_count: int
    #: The conda lock an arm using this backbone runs under (ADR-0002 A15). The two differ.
    env_lock: str
    #: Whether the training path can RUN with gradient checkpointing enabled — a property of
    #: the port, measured, not a preference.
    #:
    #: ⚠ This is not "does it help". On ``rinalmo-giga`` checkpointing runs and is a measured
    #: **no-op** (saving ratio 0.9986 across 36 flagged modules, P3-06); on ``rnafm`` it
    #: **raises**, because that port adds its ABSOLUTE position embeddings in place
    #: (``modeling_rnafm.py:728``, ``embeddings += position_embeddings``) while checkpointing
    #: forces the embedding output to be a leaf that requires grad. RiNALMo is rotary, so it
    #: never reaches that branch — which is why nothing in this repo hit it until P3-17's
    #: SLURM job 1370 died in its sizing leg. Isolated to a 2x2 on a cluster A4000
    #: (2026-08-21): rnafm x on -> RuntimeError, rnafm x off -> OK, rinalmo x both -> OK.
    gradient_checkpointing_usable: bool
    #: Why, in one line, for the report that has to explain an absent measurement.
    gradient_checkpointing_note: str
    role: str
    #: Size as PRD §10.2 words it, for a human-readable table column. A label, not a count.
    prd_label: str


#: The shipped Stage-2 backbone (PRD §6/§10.2; ADR-0002 D5/A9). **Unchanged by A15.**
PRODUCTION_BACKBONE = "rinalmo-giga"
#: The ADR-0002 D6 / PRD §10.2 ablation comparator (P3-17). Building it is not swapping to it.
COMPARATOR_BACKBONE = "rnafm"

#: Closed allow-list (ADR-0002 D5/D6 + A15). Adding an entry is an ADR amendment, not config.
BACKBONES: Mapping[str, RnaBackbone] = MappingProxyType(
    {
        PRODUCTION_BACKBONE: RnaBackbone(
            key=PRODUCTION_BACKBONE,
            repo_id=REPO_ID,
            revision=REVISION,
            model_class="RiNALMoModel",
            hidden_size=D_MODEL,
            n_layer=N_LAYERS,
            n_heads=20,
            max_position_embeddings=1024,
            position_embedding_type="rotary",
            vocab_size=28,
            expected_param_count=649_239_051,
            env_lock="envs/ml-rna.conda-lock.yml",
            gradient_checkpointing_usable=True,
            gradient_checkpointing_note=(
                "runs, but is a measured NO-OP on this port: the encoder loop never calls "
                "_gradient_checkpointing_func, so 36 modules carry a flag nothing reads "
                "(peak 3.1640 vs 3.1595 GiB, saving ratio 0.9986; P3-06 sizing, job 1051)"
            ),
            role="production",
            prd_label="650M",
        ),
        COMPARATOR_BACKBONE: RnaBackbone(
            key=COMPARATOR_BACKBONE,
            repo_id="multimolecule/rnafm",
            revision="7d6e73ad3b48e042b378f9a788a56ccb4d573a27",
            model_class="RnaFmModel",
            hidden_size=640,
            n_layer=12,
            n_heads=20,
            max_position_embeddings=1026,
            position_embedding_type="absolute",
            vocab_size=28,
            expected_param_count=99_111_680,
            env_lock="envs/ml-rnafm.conda-lock.yml",
            gradient_checkpointing_usable=False,
            gradient_checkpointing_note=(
                "RAISES: this port adds its absolute position embeddings IN PLACE "
                "(modeling_rnafm.py:728) and checkpointing makes the embedding output a leaf "
                "requiring grad -> 'a leaf Variable that requires grad is being used in an "
                "in-place operation'. Measured on a cluster A4000 2026-08-21 (SLURM job 1370, "
                "then isolated to a 2x2 against rotary rinalmo-giga, which is unaffected)"
            ),
            role="comparator",
            prd_label="~100M",
        ),
    }
)

#: Stable ordering for tables and CLI help — production first.
BACKBONE_KEYS: tuple[str, ...] = tuple(BACKBONES)


def resolve_backbone(key: str) -> RnaBackbone:
    """Resolve an allow-list key → its :class:`RnaBackbone`, or raise.

    Fail-closed by construction: the allow-list is the *only* way to name a Stage-2 backbone,
    so an unknown or misspelled key cannot degrade to "load the default" — which is exactly how
    a comparator ends up fine-tuning the production model and reporting a green gate.
    """
    if not isinstance(key, str):
        raise ValueError(f"backbone must be a string key; got {key!r} ({type(key).__name__})")
    spec = BACKBONES.get(key)
    if spec is None:
        raise ValueError(
            f"unknown Stage-2 backbone {key!r}; the ADR-0002 A15 allow-list is closed: "
            f"{BACKBONE_KEYS}. Adding a backbone is an ADR amendment (D5/D6/D8 — pins are "
            "added, never bumped silently), not a config override."
        )
    return spec


def require_pinned_revision(key: str, revision: str | None) -> RnaBackbone:
    """Reject any revision but ``key``'s pinned one; return the resolved backbone.

    ``revision=None`` means "the pinned one" — the caller did not ask for anything else — so it
    resolves rather than raising. Any *other* value raises: for ``rinalmo-giga`` because P1-13
    proved **that** revision faithful and a different one inherits no such proof (ADR-0002
    D5/A9), and for ``rnafm`` because D2 admits only an immutable commit sha, never ``main``.
    Re-pinning is a code change plus re-sign-off, never a runtime argument.
    """
    spec = resolve_backbone(key)
    if revision is not None and revision != spec.revision:
        raise ValueError(
            f"revision {revision!r} != the pinned {spec.revision!r} for backbone {spec.key!r} "
            f"({spec.repo_id}). Re-pinning needs a code change + ADR sign-off (ADR-0002 "
            "D2/D5/A9/A15)."
        )
    return spec


def backbone_for_repo_id(repo_id: str) -> RnaBackbone | None:
    """The allow-list entry whose ``repo_id`` matches, or ``None``.

    Used to **re-derive** which backbone a checkpoint belongs to from the checkpoint's own
    recorded evidence — a PEFT ``adapter_config.json`` records ``base_model_name_or_path`` and
    nothing else — rather than defaulting a missing axis to "production" on faith, which is
    what would let an RNA-FM adapter be applied to a RiNALMo base and score somebody else's
    logits. The A14 ``checkpoint_for_repo_id`` precedent, for the same reason.
    """
    for spec in BACKBONES.values():
        if spec.repo_id == repo_id:
            return spec
    return None


def backbone_summary(spec: RnaBackbone) -> dict[str, object]:
    """The JSON-safe identity block a run report echoes for its **resolved** backbone."""
    return {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_class": spec.model_class,
        "hidden_size": int(spec.hidden_size),
        "n_layer": int(spec.n_layer),
        "n_heads": int(spec.n_heads),
        "max_position_embeddings": int(spec.max_position_embeddings),
        "position_embedding_type": spec.position_embedding_type,
        "vocab_size": int(spec.vocab_size),
        "expected_param_count": int(spec.expected_param_count),
        "env_lock": spec.env_lock,
        "gradient_checkpointing_usable": bool(spec.gradient_checkpointing_usable),
        "gradient_checkpointing_note": spec.gradient_checkpointing_note,
        "role": spec.role,
        "prd_label": spec.prd_label,
    }
