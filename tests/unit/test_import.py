"""P0-01: the package imports and exposes a version."""


def test_import_tbox_finder():
    import tbox_finder

    assert tbox_finder.__version__
