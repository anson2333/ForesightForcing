from __future__ import annotations

import sys
from pathlib import Path


def ensure_vggt_importable(vggt_src_path: str | None = None) -> None:
    """Expose the VGGT source tree as the top-level package `vggt`.

    Resolution order, first hit wins:
      1. An explicit `vggt_src_path` (config / CLI override).
      2. The vendored checkout at `third_party/vggt`.
      3. A sibling checkout next to the repository root.

    A `vggt` package that is already importable (pip install) is left alone --
    this function only prepends search paths, so an installed distribution keeps
    priority unless a path is inserted ahead of it.
    """
    if "vggt" in sys.modules:
        return

    candidates: list[Path] = []
    if vggt_src_path:
        root = Path(vggt_src_path).expanduser().resolve()
        # Accept either the repo root (containing `vggt/`) or the package dir.
        candidates.extend([root, root.parent])

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root / "third_party" / "vggt")
    candidates.append(repo_root.parent / "vggt")

    for path in candidates:
        if (path / "vggt" / "models" / "vggt.py").is_file():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return


def ensure_vggt_omega_importable(vggt_src_path: str | None = None) -> None:
    """Expose a VGGT-Omega checkout as the top-level ``vggt_omega`` package.

    VGGT-Omega intentionally uses a different package name from the original
    VGGT. Keeping a separate resolver prevents an installed ``vggt`` package
    from shadowing Omega and lets a run pin the exact source tree used for its
    teacher checkpoint.
    """
    if "vggt_omega" in sys.modules:
        return

    candidates: list[Path] = []
    if vggt_src_path:
        root = Path(vggt_src_path).expanduser().resolve()
        candidates.extend([root, root.parent])

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root / "third_party" / "vggt-omega")
    candidates.append(repo_root / "third_party" / "vggt_omega")
    candidates.append(repo_root.parent / "vggt-omega")
    candidates.append(repo_root.parent / "vggt_omega")

    for path in candidates:
        if (path / "vggt_omega" / "models" / "vggt_omega.py").is_file():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return

    searched = ", ".join(str(path) for path in candidates)
    raise ImportError(
        "Could not import VGGT-Omega: no checkout containing "
        f"vggt_omega/models/vggt_omega.py was found. Searched: {searched}"
    )
