"""Windows 原生 DLL 搜索路径（PyTorch / ONNX Runtime / Python）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _site_packages_roots() -> list[Path]:
    roots: list[Path] = []
    for base in (Path(sys.prefix), Path(sys.executable).resolve().parent):
        sp = base / "Lib" / "site-packages"
        if sp.is_dir():
            roots.append(sp)
    try:
        import site

        for p in site.getsitepackages():
            roots.append(Path(p))
    except Exception:
        pass
    uniq: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def native_dll_dirs(python_exe: str | None = None) -> list[Path]:
    """返回应加入 DLL 搜索路径的目录。"""
    dirs: list[Path] = []
    py = Path(python_exe or sys.executable).resolve()
    dirs.append(py.parent)

    rel_subdirs = (
        "torch/lib",
        "onnxruntime/capi",
        "numpy.libs",
        "scipy.libs",
        "pandas.libs",
    )
    for sp in _site_packages_roots():
        for sub in rel_subdirs:
            d = sp / sub.replace("/", os.sep)
            if d.is_dir():
                dirs.append(d)

    uniq: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def configure_native_dll_paths(python_exe: str | None = None) -> list[str]:
    """配置 os.add_dll_directory 与 PATH，缓解 WinError 1114 (c10.dll)。"""
    if os.name != "nt":
        return []

    added: list[str] = []
    add_fn = getattr(os, "add_dll_directory", None)
    for d in native_dll_dirs(python_exe):
        s = str(d)
        if add_fn is not None:
            try:
                add_fn(s)
            except OSError:
                pass
        added.append(s)

    if added:
        prefix = os.pathsep.join(added)
        os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")
    return added
