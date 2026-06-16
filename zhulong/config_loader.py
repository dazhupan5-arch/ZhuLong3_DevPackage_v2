"""配置加载与保存。"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from zhulong.utils.paths import resolve_config_path

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        logger.info("配置已保存: %s", self._path)

    def update(self, patch: dict[str, Any]) -> None:
        self._data = _deep_merge(self._data, patch)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> Config:
    path = resolve_config_path()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info("已加载配置: %s", path)
    return Config(data, path)
