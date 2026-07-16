"""Stage-1 / Stage-2 training entry points (P1 onward).

Heavy dependencies (``torch``, ``transformers``, ``hydra``, ``wandb``) are imported
**lazily inside functions** so this package imports in a bare environment (the CI Tier-1
path); only the GPU/training tiers pull the ML stack.
"""

from __future__ import annotations

__all__ = ["lora_harness", "repro", "stage1_smoke"]
