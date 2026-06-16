import numpy as np
import pandas as pd
from pathlib import Path

d = np.load("d:/trae_projects/ZhuLong_3/data/training_data.npz", allow_pickle=True)
print(f"rows={len(d['close'])}")
print(f"struct_shape={d['struct'].shape}")
print(f"keys={list(d.keys())}")

t = pd.to_datetime(d["time"])
print(f"time_range: {t.min()} ~ {t.max()}")
print(f"years: {sorted(t.year.unique())}")

# check knowledge net
kn = Path("d:/trae_projects/ZhuLong_3/models/knowledge_net.pth")
print(f"knowledge_net.pth: {kn.stat().st_size} bytes, exists={kn.exists()}")
kn2 = Path("d:/trae_projects/ZhuLong_3/models/knowledge_net.onnx")
print(f"knowledge_net.onnx: {kn2.stat().st_size if kn2.exists() else 'N/A'}, exists={kn2.exists()}")
