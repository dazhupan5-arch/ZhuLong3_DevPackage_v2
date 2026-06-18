"""
路径解析：兼容开发模式与 PyInstaller 打包 EXE。
安装目录只读时，日志/DB/用户 config 写入 %APPDATA%\\ZhuLong\\
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) is True


def install_dir() -> Path:
    """EXE 或项目根目录（只读资源）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # zhulong/utils/paths.py → 向上查找含 config.json 的仓库根
    here = Path(__file__).resolve()
    for base in (here.parent.parent.parent, here.parent.parent):
        if (base / "config.json").is_file():
            return base
    return here.parent.parent


def appdata_dir() -> Path:
    """用户可写目录。"""
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    path = Path(base) / "ZhuLong"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    env = os.environ.get("ZHULONG_LOGS_DIR")
    if env:
        path = Path(env)
    else:
        path = appdata_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return appdata_dir() / "trading.db"


def models_dir() -> Path:
    return install_dir() / "models"


def resolve_model_path(rel: str | Path, *, root: Path | None = None) -> Path:
    """models/... 先查安装目录，缺失时回退 AppData（Program Files 只读部署）。"""
    p = Path(rel)
    if p.is_absolute():
        return p
    base = root or install_dir()
    install = base / p
    if install.is_file():
        return install
    appdata = appdata_dir() / p
    if appdata.is_file():
        return appdata
    return install


def writable_data_dir() -> Path:
    """用户可写 data/（%APPDATA%\\ZhuLong\\data），安装目录只读时使用。"""
    path = appdata_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_bundled_data_path(rel: str | Path) -> Path:
    """安装目录 data/ 资源；缺失时回退 AppData 副本（升级/只读安装目录）。"""
    p = Path(rel)
    if p.is_absolute():
        return p
    install = install_dir() / p
    if install.is_file():
        return install
    writable = resolve_writable_data_path(p)
    if writable.is_file():
        return writable
    return install


def resolve_writable_data_path(rel: str | Path) -> Path:
    """
    将 config 中 data/... 相对路径解析到 AppData。
    例：data/agent_state.json → %APPDATA%\\ZhuLong\\data\\agent_state.json
    """
    p = Path(rel)
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        sub = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    else:
        sub = p
    out = writable_data_dir() / sub
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def resolve_runtime_path(rel: str | Path, *, root: Path | None = None) -> Path:
    """
    运行时相对路径解析（安装目录只读时安全）：
      logs/...  → %APPDATA%\\ZhuLong\\logs\\...
      data/...  → %APPDATA%\\ZhuLong\\data\\...
      其他      → root（默认 install_dir）/...
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "logs":
        sub = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
        out = logs_dir() / sub
        if sub.suffix:
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
        return out
    if p.parts and p.parts[0] == "data":
        return resolve_writable_data_path(p)
    return (root or install_dir()) / p


def resolve_writable_log_path(rel: str | Path) -> Path:
    """logs/... 或 cognition 等日志相对路径 → AppData logs。"""
    p = Path(rel)
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if p.parts and p.parts[0] == "logs":
        return resolve_runtime_path(p)
    return logs_dir() / p


def data_dir() -> Path:
    """安装目录 data/（只读资源）；运行时写入请用 writable_data_dir()。"""
    return install_dir() / "data"


def macro_events_path() -> Path:
    return data_dir() / "macro_events.csv"


def model_dir_for_symbol(symbol: str) -> Path:
    return models_dir() / symbol


def config_search_paths() -> list[Path]:
    """config.json 查找顺序（见 DELIVERY.md §5.1）。"""
    return [
        install_dir() / "config.json",
        appdata_dir() / "config.json",
    ]


def resolve_agent_config_path(cfg_rel: str | Path, root: Path | None = None) -> Path:
    """config_agent.json 优先 AppData，与 warmup/C# AgentConfigSync 一致。"""
    p = Path(cfg_rel)
    base = root or install_dir()
    if p.is_absolute() and p.is_file():
        return p.resolve()
    name = p.name if p.name else "config_agent.json"
    candidates = [
        appdata_dir() / name,
        appdata_dir() / p,
        base / p,
        base / "config" / "config_agent.json",
        appdata_dir() / "config_agent.json",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return (base / p).resolve()


def default_config_template() -> dict[str, Any]:
    for candidate in (install_dir() / "config.json", Path(__file__).resolve().parent.parent.parent / "config.json"):
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("未找到 config.json 模板")


def resolve_config_path() -> Path:
    for p in config_search_paths():
        if p.is_file():
            return p
    target = appdata_dir() / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(default_config_template(), f, indent=2, ensure_ascii=False)
    return target


def map_symbol(standard: str, mapping: dict[str, str]) -> str:
    return mapping.get(standard, standard)


def broker_symbol(standard: str, mapping: dict[str, str]) -> str:
    return map_symbol(standard, mapping)
