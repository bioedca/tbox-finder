"""The pre-flight junction gate, end-to-end on a smoke fixture (P2-10d′-b, A7 pin 7).

`tests/unit/test_junction_probe.py` tests the gate's *rules* against hand-written reports.
This runs the whole path — build the three arms, featurise, cross-validate, re-derive the
clauses — so a change that breaks the measurement rather than the rule is caught too.

Smoke-sized by necessity: the real substrate (`mining_pool_v0.parquet`) is DVC-tracked and
absent from a CI checkout, so the full-scale measurement is a step artifact
(`reports/p2/junction_control.json`) and this tier proves the machinery on a fixture whose
arms have the same construction.

The fixture is built so the gate's two directional clauses are **both** non-trivial:
hosts are pseudo-random ACGT, so the background→background control differs from plain
windows in nothing a k-mer can see; the decoy inserts are GC-skewed, so the decoy arm is
genuinely separable and the must-fire power clause has something to find.
"""

from __future__ import annotations

import hashlib

import pytest

from tbox_finder.data.embedding import embed_decoy_rows, junction_control
from tbox_finder.eval.junction_probe import junction_clauses, junction_measurement

pytest.importorskip("sklearn", reason="the pre-flight probe is a scikit-learn model")

WINDOW = 256
SEED = 20260721
# Per arm. NOT a free parameter: the k=4 featuriser has 256 columns, so at n=90 the
# cross-validated probe overfits and separates ANY two disjoint sets of exchangeable
# windows — two *unspliced* host sets measured 0.576 while the single null draw happened
# to land at 0.504, and the band derived from that one draw understated the noise. The
# junction arm then read 0.66 with nothing about the junction having changed. At 600 the
# overfitting is gone (unspliced-vs-unspliced 0.489) and the junction arm sits at
# 0.510-0.540 across three seeds. The shipped measurement runs at n = 2,804-4,999.
N_HOSTS = 600


def _window(tag: str, weights: str = "ACGT") -> str:
    raw = hashlib.shake_256(tag.encode()).digest(WINDOW)
    return "".join(weights[b % len(weights)] for b in raw)


def _hosts(prefix: str, n: int) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"{prefix}{i}:lead",
            "sequence": _window(f"{prefix}:{i}"),
            "source_record_id": f"{prefix}{i}",
        }
        for i in range(n)
    ]


def _decoy_rows(n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(n):
        pool = "structured_rna" if i % 2 else "dinuc_shuffled"
        # GC-skewed inserts: a real decoy is compositionally distinct from its host
        # (measured AUROC 0.71-1.00 in ADR-0005 A7), and a fixture whose insert were
        # indistinguishable would make the must-fire clause unsatisfiable for reasons
        # that have nothing to do with the code under test.
        rows.append(
            {
                "decoy_id": f"d{i}",
                "pool": pool,
                "sequence": _window(f"decoy:{i}", weights="GCGCGCAT")[: 70 + (i % 30)],
                "masked": False,
            }
        )
    for pool in ("gc_background", "leader_decoy"):
        rows.append(
            {"decoy_id": f"{pool}_x", "pool": pool, "sequence": "ACGTACGT", "masked": False}
        )
    return rows


@pytest.fixture(scope="module")
def measurement() -> dict[str, object]:
    # Disjoint host sets per arm. Sharing them makes each spliced window the pair of its
    # own unspliced counterpart, and a cross-validated classifier that memorises a host
    # then scores its pair in the opposite direction — which drove every junction AUROC
    # BELOW chance the first time this was measured.
    embed_hosts = _hosts("emb", N_HOSTS)
    control_hosts = _hosts("ctl", N_HOSTS)
    plain = [str(h["sequence"]) for h in _hosts("plain", N_HOSTS)]
    null_reference = [str(h["sequence"]) for h in _hosts("null", N_HOSTS)]

    rows = _decoy_rows(N_HOSTS)
    decoy, report = embed_decoy_rows(rows, embed_hosts, seed=SEED, window=WINDOW, unique_hosts=True)
    assert report["n_embedded"] == N_HOSTS
    # Paired control (verifies matchedness) and AUROC control (disjoint hosts) - the two
    # roles cannot be played by one arm; see junction_measurement's docstring.
    matched_control = junction_control(decoy, embed_hosts, seed=SEED, window=WINDOW)
    decoy_b, _ = embed_decoy_rows(rows, control_hosts, seed=SEED, window=WINDOW, unique_hosts=True)
    control = junction_control(decoy_b, control_hosts, seed=SEED, window=WINDOW)
    return junction_measurement(
        plain=plain,
        null_reference=null_reference,
        control=control,
        matched_control=matched_control,
        decoy=decoy,
        host_sequences={str(h["candidate_id"]): str(h["sequence"]) for h in embed_hosts},
        k=4,
        seed=SEED,
    )


def test_the_preflight_gate_passes_on_a_matched_construction(measurement: dict) -> None:
    clauses = junction_clauses(measurement)
    assert all(clauses.values()), (clauses, measurement)


def test_the_junction_arm_sits_at_the_measured_null(measurement: dict) -> None:
    """The claim R2 rests on: a real-DNA-into-real-DNA splice leaves no compositional
    trace. Stated against the *measured* null, not against 0.5."""
    from tbox_finder.eval.junction_probe import deviation

    band = deviation(measurement["auroc_null_unspliced_vs_unspliced"]) + 3.0 * float(
        measurement["null_stderr"]
    )
    assert deviation(measurement["auroc_junction_control_vs_plain"]) <= band


def test_the_probe_has_power_on_this_fixture(measurement: dict) -> None:
    """MUST FIRE. Without it, 'the junction is invisible' is indistinguishable from 'the
    probe sees nothing at all'."""
    assert junction_clauses(measurement)["probe_can_discriminate"] is True


def test_the_arms_are_matched_end_to_end(measurement: dict) -> None:
    assert measurement["arms_matched"] is True
    detail = measurement["arms_matched_detail"]
    assert detail["n_pairs"] == N_HOSTS
    assert detail["n_control_unspliced"] == 0
    assert detail["n_host_unresolved"] == 0


def test_overlapping_null_arms_are_refused(measurement: dict) -> None:
    """A shared window between the two null arms makes the null optimistic, which widens
    the band the junction arm has to sit inside — failing open."""
    from tbox_finder.eval.junction_probe import JunctionProbeError

    shared = [str(h["sequence"]) for h in _hosts("plain", 4)]
    with pytest.raises(JunctionProbeError, match="both null arms"):
        junction_measurement(
            plain=shared,
            null_reference=shared,
            control=[],
            matched_control=[],
            decoy=[],
            host_sequences={},
            k=4,
            seed=SEED,
        )
