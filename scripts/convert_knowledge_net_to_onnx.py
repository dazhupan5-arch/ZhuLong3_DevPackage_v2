#!/usr/bin/env python3
"""将 KnowledgeNet (.pth) 导出为 ONNX，供 onnxruntime 快速推理。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401 — Windows 下须最先加载

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import KnowledgeNetInference, _knowledge_net_class
from zhulong.agent.training_utils import load_training_config, resolve_symbol_paths


class _ExportWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(x)


def export_onnx(model_path: Path, out_path: Path | None = None) -> Path:
    meta_path = model_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    input_dim = int(meta.get("input_dim", 30))
    hidden_dim = int(meta.get("hidden_dim", 64))
    embed_dim = int(meta.get("embed_dim", 32))
    num_res_blocks = int(meta.get("num_res_blocks", 2))

    KnCls, torch_mod = _knowledge_net_class(num_res_blocks=num_res_blocks)
    net = KnCls(input_dim, hidden_dim, embed_dim, num_res_blocks=num_res_blocks)
    state = torch_mod.load(model_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    wrapper = _ExportWrapper(net)
    wrapper.eval()
    dummy = torch_mod.randn(1, input_dim, dtype=torch_mod.float32)
    dst = out_path or model_path.with_suffix(".onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch_mod.onnx.export(
            wrapper,
            dummy,
            str(dst),
            input_names=["struct_features"],
            output_names=["logits", "probs", "embedding"],
            dynamic_axes={"struct_features": {0: "batch"}, "probs": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
    except TypeError:
        torch_mod.onnx.export(
            wrapper,
            dummy,
            str(dst),
            input_names=["struct_features"],
            output_names=["logits", "probs", "embedding"],
            dynamic_axes={"struct_features": {0: "batch"}, "probs": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
        )
    return dst


def benchmark(model_path: Path, onnx_path: Path, scaler_path: Path | None) -> None:
    import numpy as np

    meta_path = model_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    input_dim = int(meta.get("input_dim", 30))
    x = np.random.randn(1, input_dim).astype(np.float32)
    kn_pt = KnowledgeNetInference(model_path, scaler_path=scaler_path)
    t0 = time.perf_counter()
    for _ in range(20):
        kn_pt.predict(x)
    pt_ms = (time.perf_counter() - t0) * 1000 / 20

    kn_onnx = KnowledgeNetInference(onnx_path, scaler_path=scaler_path)
    t0 = time.perf_counter()
    for _ in range(20):
        kn_onnx.predict(x)
    onnx_ms = (time.perf_counter() - t0) * 1000 / 20

    print(f"PyTorch 单样本: {pt_ms:.2f} ms")
    print(f"ONNX 单样本:    {onnx_ms:.2f} ms (目标 <5 ms)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(args.symbol, cfg)
    model_path = Path(args.model) if args.model else paths["knowledge_model"]
    if not model_path.is_file():
        print(f"模型不存在: {model_path}")
        return 1

    out = Path(args.out) if args.out else model_path.with_suffix(".onnx")
    print(f"导出 ONNX: {model_path} → {out}")
    export_onnx(model_path, out)
    print(f"已保存 {out} ({out.stat().st_size / 1024:.1f} KB)")

    if not args.no_benchmark:
        benchmark(model_path, out, paths["knowledge_scaler"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
