#!/usr/bin/env python3
"""将桌面/MT5 导出的 M5 CSV 增量合并进训练数据（只追加新时间戳，重叠以 source 为准）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv


def _target_has_header(path: Path) -> bool:
    peek = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    return bool(peek and peek[0].lower().startswith("time"))


def _write_m5(df, path: Path, *, with_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index()
    if with_header:
        reset = out.reset_index()
        reset.columns = ["time", "open", "high", "low", "close", "volume"]
        reset.to_csv(path, index=False)
    else:
        lines: list[str] = []
        for ts, row in out.iterrows():
            date = ts.strftime("%Y.%m.%d")
            time = ts.strftime("%H:%M")
            lines.append(
                f"{date},{time},{row['open']},{row['high']},{row['low']},{row['close']},{int(row['volume'])}"
            )
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_supplement(source: Path, target: Path, *, dry_run: bool = False) -> int:
    if not source.is_file():
        print(f"源文件不存在: {source}")
        return 1
    if not target.is_file():
        print(f"目标不存在，将从源文件初始化: {target}")
        existing = None
    else:
        existing = load_vendor_csv(target)

    incoming = load_vendor_csv(source)
    with_header = _target_has_header(target) if target.is_file() else False

    if existing is None or existing.empty:
        merged = incoming.sort_index()
        print(f"init rows={len(merged)} range={merged.index.min()} .. {merged.index.max()}")
    else:
        old_max = existing.index.max()
        new_ix = incoming.index[incoming.index > old_max]
        overlap = incoming.index.intersection(existing.index)
        merged = pd.concat([existing, incoming]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        print(f"target max={old_max}  source max={incoming.index.max()}")
        print(f"append={len(new_ix)}  overlap_update={len(overlap)}  total={len(merged)}")

    if dry_run:
        print(f"[dry-run] would write {target} rows={len(merged)}")
        return 0

    _write_m5(merged, target, with_header=with_header)
    print(f"OK: {target} rows={len(merged)} range={merged.index.min()} .. {merged.index.max()}")
    return 0


def main() -> int:
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(description="桌面 M5 CSV → 训练目录增量补齐")
    parser.add_argument("--source", required=True, help="桌面/MT5 导出的 CSV")
    parser.add_argument("--target", required=True, help="项目内训练 CSV 路径")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    tgt = _ROOT / args.target if not Path(args.target).is_absolute() else Path(args.target)
    return merge_supplement(src, tgt, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
