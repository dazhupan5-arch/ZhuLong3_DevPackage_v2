"""可写路径解析测试。"""

from __future__ import annotations

import os

from zhulong.utils.paths import appdata_dir, resolve_writable_data_path


def test_resolve_writable_data_path_under_appdata():
    p = resolve_writable_data_path("data/agent_state.json")
    assert p.is_absolute()
    assert appdata_dir() in p.parents or p.parent == appdata_dir() / "data"
    assert "agent_state.json" in p.name
    assert os.environ.get("APPDATA", "") in str(p)
