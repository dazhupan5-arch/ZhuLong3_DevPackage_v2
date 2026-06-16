"""Parquet 读取兼容层（避免同事机 pyarrow/pandas 版本不一致导致推理失败）。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_parquet_safe(path: Path) -> pd.DataFrame | None:
    """读取 Parquet；失败时返回 None，由调用方降级（CSV / 在线 VMD）。"""
    if not path.is_file():
        return None

    # 1) pyarrow 底层读表（绕过 pandas extension 注册问题）
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        df = table.to_pandas(timestamp_as_object=True, split_blocks=True, self_destruct=True)
        if not df.empty:
            return _normalize_index(df)
    except Exception as ex:
        logger.warning("pyarrow.read_table 失败 %s: %s", path, ex)

    # 2) pandas 默认引擎
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if not df.empty:
            return _normalize_index(df)
    except Exception as ex:
        logger.warning("pd.read_parquet(pyarrow) 失败 %s: %s", path, ex)

    # 3) fastparquet（若已安装）
    try:
        df = pd.read_parquet(path, engine="fastparquet")
        if not df.empty:
            return _normalize_index(df)
    except Exception as ex:
        logger.warning("pd.read_parquet(fastparquet) 失败 %s: %s", path, ex)

    return None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()
