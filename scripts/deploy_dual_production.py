#!/usr/bin/env python3
"""部署 XAUUSD v12 + USOIL v1 双品种实机模型与配置。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def deploy_xau() -> None:
    from scripts.deploy_v12_production import main as v12_main  # noqa: WPS433

    v12_main()


def deploy_oil() -> None:
    from scripts.deploy_oil_v1_production import deploy_to

    deploy_to(ROOT, ROOT / "models" / "USOIL")


def main() -> int:
    print("== deploy XAUUSD v12 ==")
    deploy_xau()
    print("== deploy USOIL v1 ==")
    deploy_oil()
    print("双品种部署完成。请确认 config/config_oil_v1.json 中 broker_symbol 与 MT5 市场报价一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
