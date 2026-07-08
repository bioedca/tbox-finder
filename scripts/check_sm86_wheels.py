#!/usr/bin/env python3
"""sm_86 (Ampere) wheel-availability probe for the ml-env CUDA kernels — tbox-finder P0-06.

Empirically answers, for each of the three CUDA-kernel packages the backbones need
(``mamba-ssm``, ``causal-conv1d``, ``flash-attn``), whether a **prebuilt Ampere
``sm_86`` / CUDA-12 / cp312 / linux_x86_64 wheel** exists at the ADR-0002 D3 torch
anchor — classifying each as ``wheel-present`` / ``source-build-required`` /
``unavailable`` — so the ADR-0002 amendment records an *observed* matrix rather than
an asserted one (PRD §10.3; CLAUDE.md §10.2 "source defensible data, don't fabricate").

Ground truth, not model recall: the prebuilt Ampere wheels for these three packages are
published as **GitHub release assets** (not on PyPI, which carries only the sdist), so the
probe reads the GitHub Releases REST API for the wheel tags and the PyPI JSON API for
sdist (source-buildability). ``sm_86`` itself is not a wheel tag — Ampere ``sm_86`` is
covered by the always-compiled ``compute_80`` (``sm_80``) gencode in every ``cu12`` build
of these kernels (ADR-0002 D3), so wheel availability is keyed on ``cu12`` + ``torch<anchor>``
+ ``cp312`` + ``linux_x86_64`` + the C++11-ABI variant.

This is an **availability/metadata probe only** — it imports no GPU code and needs no GPU.
The actual import/forward smoke on the cluster A4000 is P1 (ADR-0002 D3).

Network: PyPI JSON API + GitHub Releases REST API. Set ``GITHUB_TOKEN`` (or ``GH_TOKEN``)
to raise the GitHub rate limit from 60/h (unauthenticated) to 5000/h; the probe works
unauthenticated too. Read-only, public endpoints — no secret is embedded or logged.

Usage:
    python scripts/check_sm86_wheels.py              # human table + JSON to stdout
    python scripts/check_sm86_wheels.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# --- The pinned ml-env kernels (envs/ml.yml / ADR-0002 D2/D3) --------------------------
# repo = the GitHub repo whose Releases host the prebuilt Ampere wheels.
KERNELS: list[dict[str, str]] = [
    {
        "pypi": "flash-attn",
        "dist": "flash_attn",
        "repo": "Dao-AILab/flash-attention",
        "version": "2.8.3.post1",
    },
    {
        "pypi": "causal-conv1d",
        "dist": "causal_conv1d",
        "repo": "Dao-AILab/causal-conv1d",
        "version": "1.6.2.post1",
    },
    {
        "pypi": "mamba-ssm",
        "dist": "mamba_ssm",
        "repo": "state-spaces/mamba",
        "version": "2.2.6.post3",
    },
]

# Target runtime profile (the cluster A4000, PRD §15) + the two ADR-0002 D3 torch anchors.
TARGET_PY = "cp312"  # python=3.12
TARGET_PLAT = "linux_x86_64"
TARGET_CUDA_MAJOR = "12"  # cu12*; sm_86 rides the compute_80 gencode
ANCHOR_B = "2.7"  # Option B (adopted): torch 2.7.1 + cu128
ANCHOR_A = "2.5"  # Option A (pre-registered fallback): torch 2.5.1 + cu124

USER_AGENT = "tbox-finder-sm86-probe/1.0 (+P0-06 ADR-0002)"

# Parse a PEP-427 wheel filename: {dist}-{ver}-{pytag}-{abitag}-{plattag}.whl
# where {ver} carries the local segment, e.g. 2.8.3.post1+cu12torch2.7cxx11abiTRUE
_WHEEL_RE = re.compile(
    r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<ver>[0-9][^-]*)"
    r"-(?P<py>cp\d+)-(?P<abitag>[^-]+)-(?P<plat>[A-Za-z0-9_]+)\.whl$"
)
# Parse the CUDA-kernel local-version segment, e.g. cu12torch2.7cxx11abiTRUE
_LOCAL_RE = re.compile(
    r"cu(?P<cuda>\d+)torch(?P<torch>\d+\.\d+(?:\.\d+)?)cxx11abi(?P<abi>TRUE|FALSE)"
)


@dataclass
class WheelTag:
    torch: str  # "2.7"
    torch_minor: str  # "2.7" (major.minor for anchor matching)
    cuda: str  # "12"
    abi: str  # "TRUE" / "FALSE"
    filename: str


@dataclass
class KernelResult:
    pypi: str
    version: str
    repo: str
    matching_wheels: list[WheelTag] = field(default_factory=list)
    torch_minors_with_wheels: list[str] = field(default_factory=list)
    sdist_on_pypi: bool = False
    pypi_version_exists: bool = False
    github_tag: str | None = None
    notes: list[str] = field(default_factory=list)

    def anchor_status(self, anchor: str) -> dict:
        """wheel-present / source-build-required / unavailable at a torch minor anchor."""
        hits = [w for w in self.matching_wheels if w.torch_minor == anchor]
        abis = sorted({w.abi for w in hits})
        if hits:
            status = "wheel-present"
        elif self.sdist_on_pypi or self.torch_minors_with_wheels:
            # Compiles from sdist; wheels exist at other torch lines but not this anchor.
            status = "source-build-required"
        else:
            status = "unavailable"
        return {
            "torch_anchor": anchor,
            "status": status,
            "abis_present": abis,
            "wheel_files": [w.filename for w in hits],
        }


class ProbeError(RuntimeError):
    """An API/transport error that must NOT be silently read as 'not found'.

    A rate-limit (403/429), outage (5xx), or connection failure (status 0) would otherwise
    flip a kernel's verdict to source-build-required/unavailable on unobserved data — a wrong
    result that looks like a real one. The probe fails loud instead (CLAUDE.md §10.3).
    """


def _get(url: str, token: str | None, accept: str = "application/json") -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:  # an HTTP status reached us (404/403/429/5xx …)
        return exc.code, exc.read()
    except urllib.error.URLError as exc:  # transport failure (DNS/connection/timeout)
        return 0, str(exc.reason).encode()


def github_release_assets(
    repo: str, version: str, token: str | None
) -> tuple[str | None, list[str]]:
    """Return (matched_tag, [asset_filenames]) for the release matching `version`.

    Genuine absence (both tag spellings 404 -> (None, [])) is distinguished from an
    API/transport error (0/403/429/5xx -> ProbeError), so a rate-limit or outage can never
    masquerade as "no prebuilt wheel" and silently flip a verdict (§10.3).
    """
    rel_id: int | None = None
    tag_name: str | None = None
    for tag in (f"v{version}", version):
        code, body = _get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", token)
        if code == 200:
            rel = json.loads(body)
            rel_id, tag_name = rel["id"], rel["tag_name"]
            break
        if code == 404:
            continue  # this tag spelling is absent; try the next form
        raise ProbeError(f"GitHub {repo} releases/tags/{tag} -> HTTP {code}: {body[:200]!r}")
    if rel_id is None:
        return None, []  # genuinely no release for this version (both tag forms 404)
    assets: list[str] = []
    page = 1
    while True:
        c, b = _get(
            f"https://api.github.com/repos/{repo}/releases/{rel_id}/assets"
            f"?per_page=100&page={page}",
            token,
        )
        if c != 200:  # incomplete asset list -> could drop a wheel and misclassify
            raise ProbeError(f"GitHub {repo} assets page {page} -> HTTP {c}: {b[:200]!r}")
        batch = json.loads(b)
        assets.extend(a["name"] for a in batch)
        if len(batch) < 100:
            break
        page += 1
    return tag_name, assets


def pypi_files(pkg: str, version: str) -> tuple[bool, bool]:
    """Return (version_exists, has_sdist) from the PyPI JSON API.

    404 is genuine absence; any other non-200 (0/5xx/rate-limit) raises ProbeError rather
    than being misread as "version does not exist" (§10.3).
    """
    code, body = _get(f"https://pypi.org/pypi/{pkg}/{version}/json", token=None)
    if code == 404:
        return False, False
    if code != 200:
        raise ProbeError(f"PyPI {pkg}=={version} -> HTTP {code}: {body[:200]!r}")
    data = json.loads(body)
    has_sdist = any(u.get("packagetype") == "sdist" for u in data.get("urls", []))
    return True, has_sdist


def classify_assets(
    pypi: str,
    version: str,
    repo: str,
    asset_names: list[str],
    *,
    sdist_on_pypi: bool,
    pypi_version_exists: bool,
    github_tag: str | None,
) -> KernelResult:
    """Pure, network-free core: turn a release's asset filenames into a KernelResult.

    Keeps only cp312 / linux_x86_64 / cu12 wheels whose base version equals the pinned
    version; NGC-container torch tags (e.g. torch25.06) and other torch lines are recorded
    but the anchor verdicts key on the upstream torch minor (2.7 / 2.5). Split out so the
    classification that drives the ADR verdict is unit-tested without a live network.
    """
    res = KernelResult(
        pypi=pypi,
        version=version,
        repo=repo,
        sdist_on_pypi=sdist_on_pypi,
        pypi_version_exists=pypi_version_exists,
        github_tag=github_tag,
    )
    minors: set[str] = set()
    for name in asset_names:
        wm = _WHEEL_RE.match(name)
        if not wm or wm.group("py") != TARGET_PY or wm.group("plat") != TARGET_PLAT:
            continue
        lm = _LOCAL_RE.search(wm.group("ver"))
        if not lm or lm.group("cuda")[:2] != TARGET_CUDA_MAJOR:
            continue
        if wm.group("ver").split("+", 1)[0] != version:
            continue
        torch_minor = ".".join(lm.group("torch").split(".")[:2])
        minors.add(torch_minor)
        res.matching_wheels.append(
            WheelTag(
                torch=lm.group("torch"),
                torch_minor=torch_minor,
                cuda=lm.group("cuda"),
                abi=lm.group("abi"),
                filename=name,
            )
        )
    res.torch_minors_with_wheels = sorted(minors, key=lambda s: tuple(int(x) for x in s.split(".")))
    if not pypi_version_exists:
        res.notes.append("pinned version NOT found on PyPI — reconfirm the pin")
    if not github_tag:
        res.notes.append("no GitHub release matched the pinned version tag")
    return res


def probe(kernel: dict[str, str], token: str | None) -> KernelResult:
    pypi_version_exists, sdist_on_pypi = pypi_files(kernel["pypi"], kernel["version"])
    tag, assets = github_release_assets(kernel["repo"], kernel["version"], token)
    return classify_assets(
        kernel["pypi"],
        kernel["version"],
        kernel["repo"],
        assets,
        sdist_on_pypi=sdist_on_pypi,
        pypi_version_exists=pypi_version_exists,
        github_tag=tag,
    )


def build_matrix(token: str | None) -> dict:
    kernels = []
    for k in KERNELS:
        r = probe(k, token)
        kernels.append(
            {
                "pypi": r.pypi,
                "version": r.version,
                "repo": r.repo,
                "github_tag": r.github_tag,
                "pypi_version_exists": r.pypi_version_exists,
                "sdist_on_pypi": r.sdist_on_pypi,
                "cp312_cu12_linux_torch_minors_with_wheels": r.torch_minors_with_wheels,
                "option_B_torch_2_7": r.anchor_status(ANCHOR_B),
                "option_A_torch_2_5": r.anchor_status(ANCHOR_A),
                "notes": r.notes,
            }
        )
    return {
        "probe": "check_sm86_wheels.py (tbox-finder P0-06)",
        "target_profile": {
            "python": TARGET_PY,
            "platform": TARGET_PLAT,
            "cuda_major": TARGET_CUDA_MAJOR,
            "sm_86_via": "compute_80 gencode (ADR-0002 D3)",
        },
        "torch_anchors": {"option_B_adopted": ANCHOR_B, "option_A_fallback": ANCHOR_A},
        "github_authenticated": bool(token),
        "sources": [
            "https://pypi.org/pypi/<pkg>/<version>/json",
            "https://api.github.com/repos/<repo>/releases",
        ],
        "kernels": kernels,
    }


def render_table(matrix: dict) -> str:
    lines = ["", "sm_86 (Ampere) wheel-availability matrix — cp312 / cu12 / linux_x86_64", "=" * 78]
    for k in matrix["kernels"]:
        b, a = k["option_B_torch_2_7"], k["option_A_torch_2_5"]
        lines.append(f"\n{k['pypi']} {k['version']}   (repo {k['repo']}, tag {k['github_tag']})")
        lines.append(
            f"  torch minors with prebuilt wheels: "
            f"{k['cp312_cu12_linux_torch_minors_with_wheels'] or '(none)'}"
            f"   sdist on PyPI: {k['sdist_on_pypi']}"
        )
        lines.append(
            f"  Option B (torch 2.7): {b['status']:<22} " f"abi={b['abis_present'] or '-'}"
        )
        lines.append(
            f"  Option A (torch 2.5): {a['status']:<22} " f"abi={a['abis_present'] or '-'}"
        )
        for n in k["notes"]:
            lines.append(f"  ! {n}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", help="also write the JSON matrix to PATH")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    matrix = build_matrix(token)
    print(render_table(matrix))
    payload = json.dumps(matrix, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(payload + "\n")
        print(f"[wrote JSON matrix → {args.json}]")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
