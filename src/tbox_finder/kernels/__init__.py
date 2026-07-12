"""Stage-1 CUDA-kernel utilities (Caduceus / Mamba DNA backbone).

The load-bearing kernels are ``mamba-ssm`` (selective-scan) and ``causal-conv1d``,
pinned as prebuilt Ampere ``sm_86`` wheels in ``envs/ml-dna.yml`` (ADR-0002 D2/A2).
Heavy imports (``torch``, ``mamba_ssm``, ``causal_conv1d``) are performed lazily
inside functions so this package imports cleanly in a CPU-only / no-torch context
(CI, the report-schema unit test).
"""
