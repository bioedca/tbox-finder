"""tbox_finder.models — Stage-1 / Stage-2 model backbones + heads.

Heavy imports (``torch`` / ``transformers`` / ``mamba_ssm``) are performed **lazily
inside functions**, so this package imports cleanly in a CPU-only / no-torch context
(a bare CI test env), matching the ``tbox_finder.kernels`` discipline (ADR-0002 A2 C2:
the Caduceus modeling class hard-imports ``selective_scan_cuda``, so a top-level import
would drag the CUDA-kernel stack into every import site).
"""
