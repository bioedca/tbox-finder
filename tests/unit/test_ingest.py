"""Unit tests for src/tbox_finder/ingest.py (P0-12).

Two tiers:

- **stdlib tier** — the controlled per-record hashing (``_row_token`` /
  ``record_hash`` / ``records_digest``) uses only the standard library, so these
  tests run in the bare CI ``test`` env (``pyproject`` ``dependencies = []``) and
  lock the hash contract there. Golden hex values are pinned so an accidental
  change to the serialisation is caught even where pandas is absent.
- **pandas tier** — the cleaning contract is exercised on small synthetic frames,
  guarded by ``importorskip`` so it runs locally / in the ``data`` conda env and
  skips (green) in the bare CI env.
"""

from __future__ import annotations

import math

import pytest

from tbox_finder import ingest


# --------------------------------------------------------------------------- #
# stdlib tier — hash contract (runs everywhere, incl. the bare CI test env)
# --------------------------------------------------------------------------- #
def test_row_token_missing_and_scalars() -> None:
    assert ingest._row_token(None) == ingest._KIND_NA
    assert ingest._row_token(float("nan")) == ingest._KIND_NA
    assert ingest._row_token(1.5) == "F1.5"
    assert ingest._row_token(-2.0) == "F-2.0"
    assert ingest._row_token(3) == "I3"
    assert ingest._row_token(0) == "I0"
    assert ingest._row_token(True) == "B1"
    assert ingest._row_token(False) == "B0"
    assert ingest._row_token("ACGU") == "SACGU"
    # a string equal to the missing tag is disambiguated by the S kind tag
    assert ingest._row_token(ingest._KIND_NA) == "S" + ingest._KIND_NA
    # embedded comma + newline (why the parse gate exists) survives verbatim
    assert ingest._row_token("x,y\nz") == "Sx,y\nz"


def test_row_token_demotes_object_with_item() -> None:
    class FakeNpFloat:
        # mimics a non-NaN numpy scalar: equal to itself (so `x != x` is False,
        # i.e. not treated as missing) and `.item()` returns a plain python float
        def item(self) -> float:
            return 2.5

        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

        __hash__ = None  # type: ignore[assignment]

    assert ingest._row_token(FakeNpFloat()) == "F2.5"


def test_record_hash_is_deterministic_and_order_sensitive() -> None:
    a = ingest.record_hash(("Name_A", 1.5, None, "ACGU"))
    assert a == ingest.record_hash(("Name_A", 1.5, None, "ACGU"))
    assert a != ingest.record_hash(("ACGU", 1.5, None, "Name_A"))  # order matters
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)


def test_record_hash_nan_equals_none_token() -> None:
    # NaN and None both canonicalise to the missing token, so they hash equal
    assert ingest.record_hash((float("nan"),)) == ingest.record_hash((None,))


def test_record_hash_framing_is_unambiguous() -> None:
    # length-prefixed framing: ("ab","") must not collide with ("a","b")
    assert ingest.record_hash(("ab", "")) != ingest.record_hash(("a", "b"))
    # a genuine string equal to the missing tag must not collide with a missing cell
    assert ingest.record_hash((ingest._KIND_NA,)) != ingest.record_hash((None,))
    # a string "1.5" must not collide with the float 1.5 (kind tags differ)
    assert ingest.record_hash(("1.5",)) != ingest.record_hash((1.5,))


def test_row_token_exotic_type_is_disambiguated_from_string() -> None:
    from decimal import Decimal

    # a Decimal("1.5") str()s to "1.5" but must not collide with the string "1.5"
    assert ingest._row_token("1.5") == "S1.5"  # str path unchanged (golden-stable)
    assert ingest._row_token(Decimal("1.5")) != ingest._row_token("1.5")
    assert ingest._row_token(Decimal("1.5")).startswith("SDecimal:")


def test_hash_contract_golden_values() -> None:
    # Pinned so a serialisation change is caught in the bare CI env (P0-12).
    recs = [
        ("Name_A", 1.5, None, "ACGU"),
        ("Name_B", float("nan"), 3, "GG"),
        ("x,y\nz", -2.0, True, 0),
    ]
    expected = [
        "5d26df4d741f982778a6fb8256c3ac263ad8d697e06172da24bcaf1bdb3601da",
        "775a712d554c99f1bebabc721bc2ebbe20b6879aa788a9d1fbf44bbaa3bf66af",
        "deef6c31215b809121e01c4db979e79f2be02529e75b1732b3f1a2c50ac7876b",
    ]
    got = [ingest.record_hash(r) for r in recs]
    assert got == expected
    assert (
        ingest.records_digest(got)
        == "a408eb8b30461c5ce210028d4d900aa9eb03064a07e6664fcee4b21a62943a7d"
    )


def test_records_digest_depends_on_order() -> None:
    h1, h2 = "a" * 64, "b" * 64
    assert ingest.records_digest([h1, h2]) != ingest.records_digest([h2, h1])


def test_expected_count_constants() -> None:
    # The gate values (PRD §4/§7.1) are load-bearing — pin them.
    assert ingest.EXPECTED_RECORDS == 23535
    assert ingest.EXPECTED_RAW_COLS == 107
    assert ingest.EXPECTED_NAMED_COLS == 106
    assert ingest.EXPECTED_RAW_COLS - ingest.EXPECTED_NAMED_COLS == 1


# --------------------------------------------------------------------------- #
# pandas tier — cleaning contract (skips in the bare CI env)
# --------------------------------------------------------------------------- #
def _synthetic_raw():
    pd = pytest.importorskip("pandas")
    # 3 records, leading unnamed index col → 107-analogue (here 8 raw cols)
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "Name": ["a", "b", "c"],
            "Regulation": ["Transcriptional", "Unknown", "Translational"],
            "type": ["Transcriptional", "Translational", "Translational"],
            "phylum": ["Firmicutes", "Firmicutes", None],
            "Tbox_start": [0, 5, 9],
            "discrim_end": [-2, 7, 0],
            "s1_start": [0, 0, 3],
        }
    )


def test_clean_drops_unnamed_index_and_normalises_sentinels() -> None:
    pytest.importorskip("pandas")
    out = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    assert "Unnamed: 0" not in out.columns
    assert out.shape == (3, 7)
    # 0 → NaN in coord cols; the real value 5/9/3 preserved
    assert math.isnan(out["Tbox_start"].iloc[0]) and out["Tbox_start"].iloc[1] == 5
    assert out["s1_start"].iloc[0] != out["s1_start"].iloc[0]  # NaN
    assert out["s1_start"].iloc[2] == 3
    # -2 → NaN only in discrim_end; note discrim_end is NOT in the zero-sentinel
    # list, so its literal 0 (row 2) is preserved as 0.0
    assert math.isnan(out["discrim_end"].iloc[0])
    assert out["discrim_end"].iloc[1] == 7
    assert out["discrim_end"].iloc[2] == 0


def test_row_token_pandas_na_is_missing() -> None:
    pd = pytest.importorskip("pandas")
    # pd.NA (nullable dtypes) must hash as missing, not str(pd.NA) == "<NA>"
    assert ingest._row_token(pd.NA) == ingest._KIND_NA


def test_clean_fails_loud_on_nonnumeric_coordinate() -> None:
    pytest.importorskip("pandas")
    raw = _synthetic_raw()
    # non-numeric, non-sentinel coord value → must not silently coerce to NaN
    raw["Tbox_start"] = [0, "junk", 9]  # object column from the start (no dtype warning)
    with pytest.raises(ValueError, match="non-numeric coordinate"):
        ingest.clean(raw, expect_records=None, expect_named_cols=None)


def test_clean_rejects_bogus_regulation_label() -> None:
    pytest.importorskip("pandas")
    raw = _synthetic_raw()
    raw.loc[0, "Regulation"] = "Riboswitchy"
    with pytest.raises(ValueError, match="Regulation has unexpected labels"):
        ingest.clean(raw, expect_records=None, expect_named_cols=None)


def test_clean_preserves_unknown_regulation() -> None:
    pytest.importorskip("pandas")
    out = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    assert "Unknown" in set(out["Regulation"])


def test_assert_parse_gate_flags_wrong_shape() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2]})  # 2×1, not the expected shape
    with pytest.raises(ValueError, match="parse-gate FAIL"):
        ingest.assert_parse_gate(df)
    # correct shape for a small fixture passes
    ingest.assert_parse_gate(df, expected_records=2, expected_raw_cols=1)


def test_hash_identity_matches_and_mismatches() -> None:
    pytest.importorskip("pandas")
    clean = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    hashes = ingest.compute_record_hashes(clean)
    ident = ingest.hash_identity(hashes, clean)
    assert ident["identical"] and ident["pct_identity"] == 100.0
    # perturb one record → identity drops below 100%
    perturbed = clean.copy()
    perturbed.loc[0, "Name"] = "MUTATED"
    ident2 = ingest.hash_identity(hashes, perturbed)
    assert not ident2["identical"]
    assert ident2["n_matching"] == 2 and ident2["pct_identity"] < 100.0


def test_hash_identity_detects_schema_drift() -> None:
    pytest.importorskip("pandas")
    clean = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    hashes = ingest.compute_record_hashes(clean)
    # same values + positions, one column renamed → values hash identical but the
    # schema differs; columns_match must catch it and fail `identical`
    renamed = clean.rename(columns={"Name": "NAME"})
    ident = ingest.hash_identity(hashes, renamed, our_columns=list(clean.columns))
    assert ident["n_matching"] == len(hashes)  # values still match positionally
    assert ident["columns_match"] is False
    assert not ident["identical"]
    # matching schema passes the column gate
    ident_ok = ingest.hash_identity(hashes, clean, our_columns=list(clean.columns))
    assert ident_ok["columns_match"] is True and ident_ok["identical"]


def test_record_hashes_ignore_existing_hash_column() -> None:
    pytest.importorskip("pandas")
    clean = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    h_before = ingest.compute_record_hashes(clean)
    withcol = ingest.add_record_hashes(clean)
    assert ingest.RECORD_HASH_COL in withcol.columns
    # hashing the frame-with-hash-column excludes that column → same hashes
    assert ingest.compute_record_hashes(withcol) == h_before


def test_class_split_report_maps_type_to_class_i_ii() -> None:
    pytest.importorskip("pandas")
    clean = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    rep = ingest.class_split_report(clean)
    assert rep["available"]
    assert rep["class_col"] == "type"
    assert rep["class_I"] == 1  # one Transcriptional
    assert rep["class_II"] == 2  # two Translational
    assert rep["ratio_I_to_II"] == 0.5


def test_phylum_report_counts_and_dominance() -> None:
    pytest.importorskip("pandas")
    clean = ingest.clean(_synthetic_raw(), expect_records=None, expect_named_cols=None)
    rep = ingest.phylum_report(clean)
    assert rep["available"]
    assert rep["counts"]["Firmicutes"] == 2
    assert rep["counts"]["NaN"] == 1
    assert rep["dominant_phylum"] == "Firmicutes"


def test_run_ingest_end_to_end_and_gate(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    raw = _synthetic_raw()
    csv = tmp_path / "sample.csv"
    # write with the faithful leading-empty header (',Name,...') → re-read 8 cols
    raw2 = raw.copy()
    raw2.columns = [""] + list(raw2.columns[1:])
    raw2.to_csv(csv, index=False)

    # canonical = the cleaned synthetic frame written to parquet
    canonical = ingest.clean(raw, expect_records=None, expect_named_cols=None)
    canon_pq = tmp_path / "canonical.parquet"
    canonical.to_parquet(canon_pq, engine="pyarrow", index=False)

    report = ingest.run_ingest(
        raw_csv=csv,
        canonical_parquet=canon_pq,
        out_interim=tmp_path / "interim.parquet",
        out_processed=tmp_path / "processed.parquet",
        out_report=tmp_path / "report.json",
        out_provenance=tmp_path / "prov.json",
        env_lock=None,
        expect_records=3,
        expect_raw_cols=8,
        require_identity=True,
    )
    assert report["hash_identity"]["identical"]
    assert report["parse_gate"]["cleaned_columns"] == 7
    # interim carries the hash column; processed does not
    interim = pd.read_parquet(tmp_path / "interim.parquet")
    processed = pd.read_parquet(tmp_path / "processed.parquet")
    assert ingest.RECORD_HASH_COL in interim.columns
    assert ingest.RECORD_HASH_COL not in processed.columns
    assert (tmp_path / "prov.json").exists()


def test_run_ingest_raises_on_identity_shortfall(tmp_path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    raw = _synthetic_raw()
    csv = tmp_path / "sample.csv"
    raw2 = raw.copy()
    raw2.columns = [""] + list(raw2.columns[1:])
    raw2.to_csv(csv, index=False)
    # canonical differs from the ingest → identity < 100% → gate must raise
    canonical = ingest.clean(raw, expect_records=None, expect_named_cols=None)
    canonical.loc[0, "Name"] = "DIFFERENT"
    canon_pq = tmp_path / "canonical.parquet"
    canonical.to_parquet(canon_pq, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="hash-identity gate FAIL"):
        ingest.run_ingest(
            raw_csv=csv,
            canonical_parquet=canon_pq,
            out_interim=tmp_path / "i.parquet",
            out_processed=tmp_path / "p.parquet",
            out_report=tmp_path / "r.json",
            out_provenance=None,
            env_lock=None,
            expect_records=3,
            expect_raw_cols=8,
            require_identity=True,
        )
