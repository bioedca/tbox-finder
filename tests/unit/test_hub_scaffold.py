"""P0-11: the W&B + Hugging Face scaffold is bare-env-safe and secret-free.

Four guards on the tracking/release scaffold (imp.md P0-11 Validation gate):
  1. `import tbox_finder.hub` succeeds with `huggingface_hub` ABSENT — the release
     helper defers every third-party import, so it loads in the bare CI test env
     (the provenance.py discipline, CLAUDE.md §11).
  2. HF auth is ENV-ONLY: `resolve_token()` reads `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
     (HF_TOKEN wins) and returns `None` when neither is set — never a committed value.
  3. W&B offline is the compute-node default (`mode: offline` in the Hydra config), and
     the login-node `wandb_sync` rule exists, runs an `ml` env, and calls `wandb sync`.
  4. NO secret is committed to the public repo (CLAUDE.md §4): the scaffold files carry
     no HF-token / W&B-key literal.

Stdlib-only (no PyYAML) so it runs in any CI test env.
"""

import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_CONF = REPO_ROOT / "conf" / "tracking" / "wandb.yaml"
SETUP_SMK = REPO_ROOT / "workflow" / "rules" / "setup.smk"
HUB_PY = REPO_ROOT / "src" / "tbox_finder" / "hub.py"

# Secret literals that must never appear in a public-repo file (CLAUDE.md §4):
#   - a real HF token (`hf_` + ~34 chars); `hf_api`/`HfApi` are far too short to match
#   - an explicit `WANDB_API_KEY = <value>` / `WANDB_API_KEY: <value>` assignment
#     (referring to the env-var *name* in prose is fine — no `:`/`=` value follows)
#   - a bare 40-hex run (a W&B key / sha1-length literal); our files carry none
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bWANDB_API_KEY\s*[:=]\s*['\"]?\w"),
    re.compile(r"\b[0-9a-f]{40}\b"),
)


def test_hub_imports_without_huggingface_hub():
    """The helper loads in a bare env and exposes the release entrypoints."""
    hub = importlib.import_module("tbox_finder.hub")
    for name in ("resolve_token", "push_folder", "pull_snapshot", "push_model", "push_dataset"):
        assert callable(getattr(hub, name)), f"tbox_finder.hub.{name} missing/not callable"
    assert "HF_TOKEN" in hub.TOKEN_ENV_VARS
    assert hub.TOKEN_ENV_VARS[0] == "HF_TOKEN", "HF_TOKEN must take precedence"


def test_resolve_token_is_env_only(monkeypatch):
    """`resolve_token` reads the env (HF_TOKEN first) and returns None when unset."""
    hub = importlib.import_module("tbox_finder.hub")
    for var in hub.TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert hub.resolve_token() is None

    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "legacy-value")
    assert hub.resolve_token() == "legacy-value"

    monkeypatch.setenv("HF_TOKEN", "canonical-value")
    assert hub.resolve_token() == "canonical-value", "HF_TOKEN must win over the legacy var"


def test_wandb_offline_is_the_default():
    """The Hydra W&B config defaults to offline mode (compute nodes have no outbound)."""
    text = WANDB_CONF.read_text()
    assert re.search(r"^mode:\s*offline\s*$", text, re.MULTILINE), "conf must set `mode: offline`"


def test_wandb_sync_rule_present_and_uses_ml_env():
    """The login-node sync rule exists, runs an `ml` conda env, and calls `wandb sync`."""
    text = SETUP_SMK.read_text()
    assert re.search(r"^rule\s+wandb_sync\s*:", text, re.MULTILINE), "rule wandb_sync missing"
    assert "wandb sync" in text, "the rule must invoke `wandb sync`"
    assert re.search(r"envs/ml-(dna|rna)\.yml", text), "sync rule must declare an `ml` conda env"


def test_no_committed_secrets():
    """No HF token / W&B key literal in any scaffold file (public repo — CLAUDE.md §4)."""
    offenders = []
    for path in (WANDB_CONF, SETUP_SMK, HUB_PY):
        text = path.read_text()
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                offenders.append(f"{path.name}: matches {pat.pattern!r}")
    assert not offenders, f"possible committed secret(s): {offenders}"
