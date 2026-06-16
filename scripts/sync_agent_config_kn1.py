#!/usr/bin/env python3
"""将安装包 config_agent.json 中 KN1 关键字段同步到 AppData。"""

from __future__ import annotations

import json
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong\config\config_agent.json")
APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "config_agent.json"


def main() -> int:
    if not INSTALL.is_file():
        print(f"缺少安装配置: {INSTALL}")
        return 1
    if not APPDATA.is_file():
        print(f"缺少 AppData 配置: {APPDATA}")
        return 1

    inst = json.loads(INSTALL.read_text(encoding="utf-8-sig"))
    user = json.loads(APPDATA.read_text(encoding="utf-8-sig"))
    changed = False

    kn2_en = (inst.get("kn2") or {}).get("enabled")
    if kn2_en is not None and user.get("kn2", {}).get("enabled") != kn2_en:
        user.setdefault("kn2", {})["enabled"] = kn2_en
        changed = True
        print(f"kn2.enabled -> {kn2_en}")

    in_dim = (inst.get("knowledge_net") or {}).get("input_dim")
    if in_dim is not None and user.get("knowledge_net", {}).get("input_dim") != in_dim:
        user.setdefault("knowledge_net", {})["input_dim"] = in_dim
        changed = True
        print(f"knowledge_net.input_dim -> {in_dim}")

    if changed:
        APPDATA.write_text(json.dumps(user, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入 {APPDATA}")
    else:
        print("AppData 已与安装包 KN1 字段一致，无需更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
