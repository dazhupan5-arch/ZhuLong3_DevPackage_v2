#!/usr/bin/env python3
"""训练专用日志：写入 data/training/，不依赖 PowerShell 重定向。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_train_logging(log_path: Path | None = None, level: str = "INFO") -> Path:
    root = Path(__file__).resolve().parents[1]
    if log_path is None:
        log_path = root / "data" / "training" / "train_until_accepted.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger_root = logging.getLogger()
    logger_root.setLevel(log_level)
    logger_root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_path,
        maxBytes=20 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger_root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger_root.addHandler(ch)

    return log_path


def flush_train_logs() -> None:
    for h in logging.getLogger().handlers:
        h.flush()
