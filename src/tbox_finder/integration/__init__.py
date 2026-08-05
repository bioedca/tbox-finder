"""Cross-stage integration harnesses.

:mod:`tbox_finder.integration.two_stage` (P3-14) composes the Stage-1 inference operators
(:mod:`tbox_finder.infer`) with the calibrated Stage-2 re-ranker into the single path PRD §6
specifies and the P5 genome-scale scan array reuses. Nothing here re-derives an operator; the
package exists so the *composition* has one home and one regression fixture.
"""

from __future__ import annotations

__all__ = ["two_stage"]
