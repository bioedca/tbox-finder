"""Unit test for the ADR-0004 overlap-prevalence measurement core.

`scripts/measure_overlap_prevalence.py::compute_overlaps` is a metric calculator
(CLAUDE.md §8.1) feeding the ADR-0004 D1 precedence table, so its overlap and
directional-containment arithmetic is locked here against a small synthetic
fixture with hand-computed expectations. pandas/numpy are gated (importorskip)
so the module stays collectable in the bare CI env, matching the repo pattern
(e.g. `test_priors.py`). The full-corpus read (`measure`) is exercised at ADR
authoring time, not in CI (the corpus is a gitignored DVC input).
"""

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_overlap_prevalence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_overlap_prevalence", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture_df():
    """Two hand-built records exercising nesting, partial overlap, and absence.

    Coordinates are inclusive-nt in a per-record local frame.

    Record 0 (canonical class-I geometry):
      Stem_I [10,60]  Specifier [30,32] (⊂ Stem_I)  Antiterminator [70,110]
      Terminator [100,180] (∩ Antiterm = 100..110 = 11 nt)
      Discriminator [74,77] (⊂ Antiterminator)
    Record 1 (Specifier absent via -1 sentinel; Discriminator absent via NaN):
      Stem_I [5,40]  Specifier [-1,-1]  Antiterminator [50,80]
      Terminator [70,140] (∩ Antiterm = 70..80 = 11 nt)  Discriminator NaN
    """
    rows = [
        {
            "s1_start": 10,
            "s1_end": 60,
            "codon_start": 30,
            "codon_end": 32,
            "stem2_region_start": float("nan"),
            "stem2_region_end": float("nan"),
            "stem3_start": float("nan"),
            "stem3_end": float("nan"),
            "antiterm_start": 70,
            "antiterm_end": 110,
            "term_start": 100,
            "term_end": 180,
            "discrim_start": 74,
            "discrim_end": 77,
        },
        {
            "s1_start": 5,
            "s1_end": 40,
            "codon_start": -1,
            "codon_end": -1,
            "stem2_region_start": float("nan"),
            "stem2_region_end": float("nan"),
            "stem3_start": float("nan"),
            "stem3_end": float("nan"),
            "antiterm_start": 50,
            "antiterm_end": 80,
            "term_start": 70,
            "term_end": 140,
            "discrim_start": float("nan"),
            "discrim_end": float("nan"),
        },
    ]
    return pd.DataFrame(rows)


def _pair(report, a, b):
    for r in report["pairs"]:
        if r["A"] == a and r["B"] == b:
            return r
    raise AssertionError(f"pair {a}-{b} not in report")


def test_presence_counts_treat_negative_and_nan_as_absent():
    mod = _load_module()
    rep = mod.compute_overlaps(_fixture_df())
    assert rep["n_records"] == 2
    pres = rep["presence"]
    assert pres["Stem_I"]["n"] == 2
    assert pres["Specifier"]["n"] == 1  # record 1 has the -1 sentinel
    assert pres["Discriminator"]["n"] == 1  # record 1 has NaN
    assert pres["Antiterminator"]["n"] == 2
    assert pres["Stem_II"]["n"] == 0


def test_specifier_contained_in_stem_i():
    mod = _load_module()
    rep = mod.compute_overlaps(_fixture_df())
    r = _pair(rep, "Stem_I", "Specifier")
    assert r["n_both"] == 1  # only record 0 has both
    assert r["n_overlap"] == 1
    assert r["prev_pct"] == 100.0
    assert r["B_in_A"] == 1 and r["B_in_A_pct"] == 100.0  # Specifier ⊂ Stem_I
    assert r["A_in_B"] == 0
    assert r["median_bp"] == 3.0  # codon 30..32 inclusive


def test_antiterminator_terminator_partial_overlap():
    mod = _load_module()
    rep = mod.compute_overlaps(_fixture_df())
    r = _pair(rep, "Antiterminator", "Terminator")
    assert r["n_both"] == 2
    assert r["n_overlap"] == 2 and r["prev_pct"] == 100.0
    # both records overlap by exactly 11 inclusive-nt (100..110 and 70..80)
    assert r["median_bp"] == 11.0
    # neither is contained in the other
    assert r["A_in_B"] == 0 and r["B_in_A"] == 0


def test_discriminator_contained_in_antiterminator():
    mod = _load_module()
    rep = mod.compute_overlaps(_fixture_df())
    r = _pair(rep, "Antiterminator", "Discriminator")
    assert r["n_both"] == 1  # record 1 Discriminator absent
    assert r["prev_pct"] == 100.0
    assert r["B_in_A"] == 1 and r["B_in_A_pct"] == 100.0  # Discriminator ⊂ Antiterm
    assert r["median_bp"] == 4.0  # 74..77 inclusive


def test_disjoint_pair_reports_zero():
    mod = _load_module()
    rep = mod.compute_overlaps(_fixture_df())
    r = _pair(rep, "Stem_I", "Terminator")
    assert r["n_overlap"] == 0
    assert r["prev_pct"] == 0.0
    assert r["median_bp"] is None
