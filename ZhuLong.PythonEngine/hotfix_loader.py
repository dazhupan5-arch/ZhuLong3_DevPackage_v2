"""AppData 热更新：MetaPathFinder 优先加载 AppData 补丁，不 eager import 重型模块。"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

# 与 install_post_setup.ps1 热更新列表一致
_HOTFIX_MODULES: dict[str, str] = {
    "zhulong.utils.json_safe": "zhulong/utils/json_safe.py",
    "zhulong.utils.py_syntax_gate": "zhulong/utils/py_syntax_gate.py",
    "zhulong.engine.agent_engine": "zhulong/engine/agent_engine.py",
    "zhulong.engine.runtime_config": "zhulong/engine/runtime_config.py",
    "zhulong.agent.trading_agent": "zhulong/agent/trading_agent.py",
    "zhulong.agent.horizon_predictor": "zhulong/agent/horizon_predictor.py",
    "zhulong.agent.knowledge_net_kn2": "zhulong/agent/knowledge_net_kn2.py",
    "zhulong.agent.kn2_location_labels": "zhulong/agent/kn2_location_labels.py",
    "zhulong.agent.knowledge_net": "zhulong/agent/knowledge_net.py",
    "zhulong.agent.tick_brief": "zhulong/agent/tick_brief.py",
    "zhulong.agent.cognition": "zhulong/agent/cognition.py",
    "zhulong.agent.trader_mind": "zhulong/agent/trader_mind.py",
    "zhulong.agent.structure_service": "zhulong/agent/structure_service.py",
    "zhulong.agent.rl_agent": "zhulong/agent/rl_agent.py",
    "zhulong.utils.paths": "zhulong/utils/paths.py",
}

_FINDER_INSTALLED = False


class _AppDataHotfixFinder(importlib.abc.MetaPathFinder):
    def __init__(self, appdata: Path) -> None:
        self._appdata = appdata

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        rel = _HOTFIX_MODULES.get(fullname)
        if not rel:
            return None
        fpath = (self._appdata / rel.replace("/", os.sep)).resolve()
        if not fpath.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, fpath)


def apply_appdata_hotfixes() -> int:
    """注册 MetaPathFinder；返回可热更新模块数量（非 eager import）。"""
    global _FINDER_INSTALLED
    appdata = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
    if not appdata.is_dir():
        return 0
    if _FINDER_INSTALLED:
        return sum(1 for rel in _HOTFIX_MODULES.values() if (appdata / rel.replace("/", os.sep)).is_file())
    sys.meta_path.insert(0, _AppDataHotfixFinder(appdata))
    _FINDER_INSTALLED = True
    return sum(1 for rel in _HOTFIX_MODULES.values() if (appdata / rel.replace("/", os.sep)).is_file())


def verify_hotfix_syntax_readonly() -> None:
    """只读 ast 校验 AppData 热更新文件（不写 __pycache__）。"""
    import ast

    appdata = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
    if not appdata.is_dir():
        return
    for rel in _HOTFIX_MODULES.values():
        path = (appdata / rel.replace("/", os.sep)).resolve()
        if path.is_file():
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
