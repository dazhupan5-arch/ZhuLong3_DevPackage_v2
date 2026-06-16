#!/usr/bin/env python3
"""Quick check of v4 label distribution."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.labels import LabelConfig, generate_labels
from zhulong.training.lgb.splits import split_indices

m5_path = _ROOT / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv"
m5 = load_vendor_csv(m5_path)
cfg = LabelConfig(horizon=6, gain_threshold=0.0045)
lab = generate_labels(m5, config=cfg)
splits = split_indices(lab.index)

for name, ix in (("all", lab.index), ("train", splits.train), ("val", splits.val), ("test1", splits.test1)):
    sub = lab.loc[ix, "label"]
    n = len(sub)
    print(
        f"{name}: long={(sub==1).sum()} ({100*(sub==1).sum()/n:.1f}%) "
        f"short={(sub==-1).sum()} ({100*(sub==-1).sum()/n:.1f}%) "
        f"flat={(sub==0).sum()} ({100*(sub==0).sum()/n:.1f}%)"
    )
