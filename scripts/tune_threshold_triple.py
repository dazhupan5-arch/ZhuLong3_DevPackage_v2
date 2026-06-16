#!/usr/bin/env python3

"""三分类阈值调优（v11 / v13 triple v3）。"""



from __future__ import annotations



import argparse

import json

import sys

from pathlib import Path



_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



import joblib

import pandas as pd

import xgboost as xgb



from zhulong.training.lgb.splits import split_indices

from zhulong.training.v11.train import tune_triple_thresholds





def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", default="XAUUSD")

    parser.add_argument("--version", choices=["v11", "v13_triple", "v3"], default="v3")

    parser.add_argument("--model", default="")

    parser.add_argument("--labels", default="data/training/XAUUSD_triple_v3.csv")

    parser.add_argument("--target-precision", type=float, default=0.55)

    args = parser.parse_args()



    root = _ROOT

    if args.version in ("v13_triple", "v3"):

        from zhulong.training.lgb.data_io import load_vendor_csv

        from zhulong.training.v13.triple import tune_precision_thresholds



        model_dir = root / "models" / args.symbol / "triple_barrier"

        model_path = Path(args.model) if args.model else model_dir / "xgb_triple_v3.json"

        meta = joblib.load(model_dir / "params_v13_triple.pkl")

        cols = meta["feature_columns"]

        feats = pd.read_parquet(root / "data" / "training" / "v13" / args.symbol / "features.parquet")

        lab = pd.read_csv(root / args.labels, index_col=0, parse_dates=True)

        m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")

        aligned = feats.join(lab[["label"]], how="inner").dropna(subset=["label"])

        va_ix = split_indices(aligned.index).val.intersection(aligned.index)

        model = xgb.XGBClassifier()

        model.load_model(str(model_path))

        proba = model.predict_proba(aligned.loc[va_ix, cols])

        y_va = aligned.loc[va_ix, "label"].values.astype(int)

        long_thr, short_thr, sweep, best = tune_precision_thresholds(

            proba, y_va, va_ix, m5, target_precision=args.target_precision, max_signals_per_day=6.0

        )

        print(

            f"selected thr={long_thr:.2f} wprec={best.get('weighted_precision', 0):.3f} "

            f"long={best.get('long_precision', 0):.3f} short={best.get('short_precision', 0):.3f}"

        )

        out = root / "data" / "training" / "reports" / "v13_triple" / args.symbol / "threshold_tune_v3.json"

        out.parent.mkdir(parents=True, exist_ok=True)

        out.write_text(json.dumps({"best": best, "sweep": sweep}, indent=2), encoding="utf-8")

        print(f"report -> {out}")

        return 0



    meta = joblib.load(root / "models" / args.symbol / "v11" / "v11_meta.pkl")

    cols = meta["feature_columns"]

    feats = pd.read_parquet(root / "data" / "training" / "v8" / args.symbol / "features.parquet")

    lab = pd.read_csv(root / "data" / "training" / f"{args.symbol}_labeled_triple.csv", index_col=0, parse_dates=True)

    aligned = feats.join(lab, how="inner")

    va_ix = split_indices(aligned.index).val.intersection(aligned.index)



    model = xgb.XGBClassifier()

    model.load_model(str(root / "models" / args.symbol / "v11" / "xgb_triple.json"))

    proba = model.predict_proba(aligned.loc[va_ix, cols])

    y_va = aligned.loc[va_ix, "label"].values.astype(int)



    best, sweep = tune_triple_thresholds(proba, y_va, va_ix, target_precision=args.target_precision)

    print(f"selected thr={best.long_thr:.2f} wprec={best.weighted_precision:.3f} sig/day={best.signals_per_day:.1f}")

    for r in sweep:

        print(r)



    out = root / "data" / "training" / "reports" / "v11" / args.symbol / "threshold_tune_v11.json"

    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps({"best": best.__dict__, "sweep": sweep}, indent=2), encoding="utf-8")

    print(f"report -> {out}")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


