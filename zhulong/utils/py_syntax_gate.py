"""Python 源文件语法门禁：只读解析，不向安装目录写 __pycache__。"""

from __future__ import annotations

import ast
from pathlib import Path


def verify_python_syntax(path: Path) -> None:
    """解析 .py 源码；失败抛 SyntaxError/ValueError，不写 .pyc。"""
    text = path.read_text(encoding="utf-8-sig")
    ast.parse(text, filename=str(path))


def verify_python_syntax_many(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            verify_python_syntax(path)
