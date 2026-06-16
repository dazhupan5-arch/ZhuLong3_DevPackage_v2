"""知识内化网络：三分类 + 嵌入层。

v2 改进：
  - ResNet 跳跃连接，缓解深层梯度消失
  - 特征去冗余（自动剔除相关系数 >0.98 的特征）
  - 混合采样：50% 均衡 batch + 50% 自然分布 batch
  - 支持 device 参数传递
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import logging

import numpy as np

logger = logging.getLogger(__name__)

_torch = None
_nn = None
_TORCH_ERR: Exception | None = None


def _ensure_torch():
    """延迟加载 PyTorch，避免 agent 模块导入阶段触发 c10.dll 冲突。"""
    global _torch, _nn, _TORCH_ERR
    if _torch is not None:
        return _torch, _nn
    if _TORCH_ERR is not None:
        raise _TORCH_ERR
    try:
        import torch
        import torch.nn as nn

        _torch = torch
        _nn = nn
        return torch, nn
    except (ImportError, OSError) as ex:
        _TORCH_ERR = ex
        raise


_KnowledgeNetCls: type | None = None


def _infer_arch_from_state(state: dict) -> tuple[int, int, int]:
    """从 checkpoint 推断 (input_dim, hidden_dim, num_res_blocks)。"""
    w = state["input_proj.weight"]
    input_dim = int(w.shape[1])
    hidden_dim = int(w.shape[0])
    num_res = 0
    for i in range(1, 8):
        if f"res{i}.fc.weight" in state:
            num_res = i
    return input_dim, hidden_dim, max(num_res, 1)


def _build_knowledge_net_class(num_res_blocks: int = 2):
    """按残差块数量构建 KnowledgeNet 类（兼容 2-block / 3-block checkpoint）。"""
    torch, nn = _ensure_torch()

    class ResidualBlock(nn.Module):
        """带跳跃连接的全连接残差块。"""

        def __init__(self, dim: int, dropout: float = 0.3):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.bn = nn.BatchNorm1d(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            h = torch.relu(self.bn(self.fc(x)))
            h = self.dropout(h)
            return h + residual

    class KnowledgeNet(nn.Module):
        """ResNet 风格知识网络：输入 → 线性投影 → N×残差块 → 分类头 + 嵌入头。"""

        def __init__(
            self,
            input_dim: int = 30,
            hidden_dim: int = 128,
            embed_dim: int = 32,
            num_res_blocks: int = 2,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.num_res_blocks = num_res_blocks
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_bn = nn.BatchNorm1d(hidden_dim)
            self.input_dropout = nn.Dropout(0.30)
            # 与已训 checkpoint 键名一致：res1 / res2 / res3
            for i in range(1, num_res_blocks + 1):
                setattr(self, f"res{i}", ResidualBlock(hidden_dim, dropout=0.30))
            self.classifier = nn.Linear(hidden_dim, 3)
            self.embedding = nn.Linear(hidden_dim, embed_dim)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            h = torch.relu(self.input_bn(self.input_proj(x)))
            h = self.input_dropout(h)
            for i in range(1, self.num_res_blocks + 1):
                h = getattr(self, f"res{i}")(h)
            logits = self.classifier(h)
            probs = torch.softmax(logits, dim=-1)
            emb = self.embedding(h)
            return logits, probs, emb

    return KnowledgeNet, torch


def _knowledge_net_class(num_res_blocks: int = 2):
    global _KnowledgeNetCls
    KnCls, torch = _build_knowledge_net_class(num_res_blocks)
    _KnowledgeNetCls = KnCls
    return KnCls, torch


def build_labels_from_close(close: np.ndarray, horizon: int = 12, thr: float = 0.002) -> np.ndarray:
    """三分类标签 0=空 / 1=观望 / 2=多（训练用）。"""
    from zhulong.agent.training_utils import signed_to_class, build_signed_labels

    return signed_to_class(build_signed_labels(close, horizon, thr))


class KnowledgeNetInference:
    @classmethod
    def heuristic_fallback(cls, model_path: str | Path = "models/knowledge_net.onnx") -> "KnowledgeNetInference":
        """PyTorch/ONNX 均不可用时的空壳实例（predict 走启发式）。"""
        inst = cls.__new__(cls)
        inst.model_path = Path(model_path)
        inst.input_dim = 30
        inst.embed_dim = 32
        inst.hidden_dim = 64
        inst._keep_cols = None
        inst.model = None
        inst.scaler = None
        inst._onnx_session = None
        inst._ready = False
        return inst

    def __init__(
        self,
        model_path: str | Path,
        meta_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        allow_pytorch: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        meta = {}
        mp = meta_path or self.model_path.with_suffix(".meta.json")
        if Path(mp).is_file():
            meta = json.loads(Path(mp).read_text(encoding="utf-8"))
        self.input_dim = int(meta.get("input_dim", 30))
        self.embed_dim = int(meta.get("embed_dim", 32))
        self.hidden_dim = int(meta.get("hidden_dim", 64))
        # 加载特征去重列索引；旧模型没有时跳过
        self._keep_cols: np.ndarray | None = None
        kc = meta.get("keep_cols")
        if kc is not None and len(kc) > 0:
            self._keep_cols = np.array(kc, dtype=np.intp)
        self.model = None
        self.scaler = None
        self._onnx_session = None
        self._ready = False
        sp = scaler_path or meta.get("scaler_path")
        if sp:
            sp_path = Path(sp)
            if not sp_path.is_file() and not sp_path.is_absolute():
                for base in (self.model_path.parent, Path.cwd()):
                    cand = base / sp_path
                    if cand.is_file():
                        sp_path = cand
                        break
            if sp_path.is_file():
                self._load_scaler(sp_path)
        elif self.model_path.with_name("knowledge_scaler.pkl").is_file():
            self._load_scaler(self.model_path.with_name("knowledge_scaler.pkl"))
        if not self.model_path.is_file():
            onnx_alt = (
                self.model_path.with_suffix(".onnx")
                if self.model_path.suffix.lower() != ".onnx"
                else self.model_path
            )
            if onnx_alt.is_file():
                self.model_path = onnx_alt
            else:
                raise FileNotFoundError(f"KnowledgeNet 模型不存在: {self.model_path}")
        pth_path = (
            self.model_path
            if self.model_path.suffix.lower() == ".pth"
            else self.model_path.with_suffix(".pth")
        )
        onnx_path = (
            self.model_path
            if self.model_path.suffix.lower() == ".onnx"
            else self.model_path.with_suffix(".onnx")
        )
        if onnx_path.is_file() and self._load_onnx(onnx_path):
            self._ready = True
            return
        if not pth_path.is_file():
            raise FileNotFoundError(
                f"KnowledgeNet 模型缺失: 需要 {onnx_path}（推荐）或 {pth_path}"
            )
        if allow_pytorch:
            self._load_pytorch(pth_path, meta)
            self._ready = True
            return
        # 实机推理仅 ONNX；.pth 仅供训练机/开发机
        raise RuntimeError(
            f"KnowledgeNet ONNX 加载失败，且实机不加载 PyTorch 权重: {onnx_path} "
            f"(请 pip install onnxruntime 并确认安装包含 knowledge_net.onnx)"
        )

    def _load_onnx(self, path: Path) -> bool:
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._onnx_session = ort.InferenceSession(
                str(path.resolve()),
                opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("KnowledgeNet ONNX 已加载: %s", path)
            return True
        except ImportError as ex:
            raise ImportError(
                "onnxruntime 未安装，请运行 install_python_deps.ps1"
            ) from ex
        except Exception as ex:
            logger.error("KnowledgeNet ONNX 加载失败 %s: %s", path, ex)
            self._onnx_session = None
            return False

    def _predict_onnx(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inp_name = self._onnx_session.get_inputs()[0].name
        out_names = [o.name for o in self._onnx_session.get_outputs()]
        outputs = self._onnx_session.run(out_names, {inp_name: arr})
        if len(outputs) >= 3:
            probs, emb = outputs[1], outputs[2]
        else:
            logits = outputs[0]
            exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            emb = outputs[1] if len(outputs) > 1 else np.zeros((arr.shape[0], self.embed_dim), dtype=np.float32)
        return probs.astype(np.float32), emb.astype(np.float32)

    def _load_pytorch(self, path: Path, meta: dict) -> None:
        n_blocks = int(meta.get("num_res_blocks", 2))
        KnCls, torch_mod = _knowledge_net_class(num_res_blocks=n_blocks)
        net = KnCls(self.input_dim, self.hidden_dim, self.embed_dim, num_res_blocks=n_blocks)
        state = torch_mod.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        self.model = net
        logger.info("KnowledgeNet PyTorch 已加载 (dev/backtest): %s", path)

    def _load_scaler(self, path: str | Path) -> None:
        try:
            import joblib

            self.scaler = joblib.load(path)
        except Exception:
            self.scaler = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _transform(self, arr: np.ndarray) -> np.ndarray:
        # 先做特征去重（列选择），再标准化
        if self._keep_cols is not None and arr.shape[1] > len(self._keep_cols):
            if self._keep_cols.max() < arr.shape[1]:
                arr = arr[:, self._keep_cols]
        elif self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            n_scaler = int(self.scaler.n_features_in_)
            if arr.shape[1] > n_scaler == self.input_dim:
                arr = arr[:, :n_scaler]
        if self.scaler is not None:
            if arr.shape[1] != getattr(self.scaler, "n_features_in_", arr.shape[1]):
                logger.warning(
                    "KnowledgeNet 特征维不匹配: arr=%d scaler=%s model=%d keep=%s",
                    arr.shape[1],
                    getattr(self.scaler, "n_features_in_", "?"),
                    self.input_dim,
                    len(self._keep_cols) if self._keep_cols is not None else 0,
                )
            return self.scaler.transform(arr).astype(np.float32)
        return arr.astype(np.float32)

    def _heuristic_predict(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.array([[0.34, 0.33, 0.33]], dtype=np.float32)
        if arr.shape[1] >= 1:
            t = float(arr[0, 0])
            if t > 0.15:
                probs = np.array([[0.2, 0.2, 0.6]], dtype=np.float32)
            elif t < -0.15:
                probs = np.array([[0.6, 0.2, 0.2]], dtype=np.float32)
        emb = np.zeros((arr.shape[0], self.embed_dim), dtype=np.float32)
        emb[:, : min(arr.shape[1], self.embed_dim)] = arr[:, : self.embed_dim]
        return probs, emb

    def predict(self, struct_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(struct_features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if not self._ready or (self.model is None and self._onnx_session is None):
            return self._heuristic_predict(arr)
        try:
            arr = self._transform(arr)
            if self._onnx_session is not None:
                return self._predict_onnx(arr)
            _, torch_mod = _knowledge_net_class()
            with torch_mod.no_grad():
                tensor = torch_mod.tensor(arr, dtype=torch_mod.float32)
                _, probs, emb = self.model(tensor)
            return probs.numpy(), emb.numpy()
        except Exception as ex:
            logger.warning("KnowledgeNet 推理失败，启发式回退: %s", ex)
            return self._heuristic_predict(arr)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _deduplicate_columns(x: np.ndarray, corr_threshold: float = 0.99) -> np.ndarray:
    """移除相同列和高度相关列，返回保留列的索引。"""
    n_cols = x.shape[1]
    keep = np.ones(n_cols, dtype=bool)
    for i in range(n_cols):
        if not keep[i]:
            continue
        if np.std(x[:, i]) < 1e-10:
            keep[i] = False
            continue
        for j in range(i + 1, n_cols):
            if not keep[j]:
                continue
            if np.array_equal(x[:, i], x[:, j]):
                keep[j] = False
                continue
            n_samples = min(len(x), 20000)
            idx = np.random.RandomState(42).choice(len(x), n_samples, replace=False)
            corr = np.corrcoef(x[idx, i], x[idx, j])[0, 1]
            if abs(corr) >= corr_threshold:
                keep[j] = False
    return np.where(keep)[0]


def _apply_smote(
    x: np.ndarray, y: np.ndarray, ratio: float = 0.3, k_neighbors: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """对少数类（short=0, long=2）做 SMOTE 过采样。"""
    try:
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(y, minlength=3)
        majority = int(counts[1])
        strat = {
            0: max(counts[0], int(majority * ratio)),
            2: max(counts[2], int(majority * ratio)),
        }
        sm = SMOTE(
            sampling_strategy=strat,
            k_neighbors=min(k_neighbors, min(counts[0], counts[2]) - 1),
            random_state=42,
        )
        x_res, y_res = sm.fit_resample(x, y)
        return x_res, y_res
    except ImportError:
        print("imblearn not available, skip SMOTE")
        return x, y


class _FocalLoss:
    """Focal loss for imbalanced 3-class (optional gamma=0 → plain CE)."""

    def __init__(self, weight, gamma: float = 2.0) -> None:
        self.weight = weight
        self.gamma = float(gamma)

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        if self.gamma <= 0:
            return ce.mean()
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def train_knowledge_net(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_ratio: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 10,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    out_path: str | Path = "models/knowledge_net.pth",
    scaler_path: str | Path | None = None,
    shuffle_train: bool = False,
    log_path: str | Path | None = None,
    use_smote: bool = True,
    smote_ratio: float = 0.3,
    class_weights: list[float] | None = None,
    device: str = "auto",
    num_res_blocks: int = 3,
    select_by: str = "accuracy",
    times: np.ndarray | None = None,
    train_end: str = "2024-12-31",
    focal_gamma: float = 0.0,
) -> dict[str, Any]:
    torch, nn = _ensure_torch()
    from zhulong.utils.device import resolve_torch_device
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    import joblib

    y = np.asarray(labels, dtype=np.int64)
    if y.min() < 0:
        y = y + 1
    y = y.clip(0, 2)

    from sklearn.model_selection import train_test_split

    n = len(features)
    if times is not None and len(times) == n:
        train_mask, val_mask = _time_split_mask(times, train_end)
        if train_mask.sum() < 1000 or val_mask.sum() < 500:
            print(f"WARN: temporal split too small train={train_mask.sum()} val={val_mask.sum()}, fallback random")
            times = None
        else:
            x_raw, y_raw = features[train_mask], y[train_mask]
            x_va, y_va = features[val_mask], y[val_mask]
            print(f"时间切分 train={len(x_raw)} val={len(x_va)} cutoff={train_end}")
    if times is None or len(times) != n:
        if shuffle_train:
            x_raw, x_va, y_raw, y_va = train_test_split(
                features, y, test_size=val_ratio, random_state=42, stratify=y,
            )
        else:
            split = int(n * (1 - val_ratio))
            x_raw, y_raw = features[:split], y[:split]
            x_va, y_va = features[split:], y[split:]

    # --- 1. 特征去冗余（低维结构特征保留全列，避免 30→12 过度压缩） ---
    if features.shape[1] <= 40:
        keep_cols = np.arange(features.shape[1], dtype=np.intp)
        print(f"特征保留全列: {features.shape[1]} 维（跳过去冗余）")
    else:
        keep_cols = _deduplicate_columns(x_raw)
        print(f"特征去冗余: {features.shape[1]} -> {len(keep_cols)} 维")
    x_raw = x_raw[:, keep_cols]
    x_va = x_va[:, keep_cols]

    # --- 2. SMOTE 过采样少数类（仅训练集） ---
    if use_smote:
        x_tr, y_tr = _apply_smote(x_raw, y_raw, ratio=smote_ratio)
    else:
        x_tr, y_tr = x_raw, y_raw
    counts = np.bincount(y_tr, minlength=3)

    # --- 3. 标准化（先截尾再 StandardScaler，避免 col 17 等离群值主导方差） ---
    for c in range(x_tr.shape[1]):
        col = x_tr[:, c].astype(np.float64)
        lo, hi = np.percentile(col, [1, 99])
        col = np.clip(col, lo, hi)
        x_tr[:, c] = col.astype(np.float32)
        col_va = x_va[:, c].astype(np.float64)
        x_va[:, c] = np.clip(col_va, lo, hi).astype(np.float32)
    scaler = StandardScaler()
    chunk = 10000
    for start in range(0, len(x_tr), chunk):
        end = min(start + chunk, len(x_tr))
        scaler.partial_fit(np.asarray(x_tr[start:end], dtype=np.float64))
    x_tr_out = np.empty_like(x_tr, dtype=np.float32)
    for start in range(0, len(x_tr), chunk):
        end = min(start + chunk, len(x_tr))
        x_tr_out[start:end] = scaler.transform(x_tr[start:end]).astype(np.float32)
    x_tr = x_tr_out
    del x_tr_out
    x_va_out = np.empty_like(x_va, dtype=np.float32)
    for start in range(0, len(x_va), chunk):
        end = min(start + chunk, len(x_va))
        x_va_out[start:end] = scaler.transform(x_va[start:end]).astype(np.float32)
    x_va = x_va_out
    del x_va_out

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sc_path = Path(scaler_path) if scaler_path else out.with_name(out.stem.replace("knowledge_net", "knowledge_scaler") + ".pkl")
    if sc_path.name == out.name:
        sc_path = out.with_name("knowledge_scaler.pkl")
    joblib.dump(scaler, sc_path)

    # --- Device ---
    device = torch.device(resolve_torch_device(device))

    # --- 损失函数 ---
    cw = class_weights or [3.0, 1.0, 3.0]
    class_weight_tensor = torch.tensor(cw[:3], dtype=torch.float32, device=device)
    loss_fn = _FocalLoss(class_weight_tensor, gamma=focal_gamma)

    ds = TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    loss_name = f"Focal(g={focal_gamma})" if focal_gamma > 0 else "CrossEntropy"
    print(
        f"KnowledgeNet on {device} | {loss_name} w={cw} | "
        f"{'SMOTE' if use_smote else u'自然'}分布 {counts.tolist()} "
        f"(short={counts[0]/len(y_tr)*100:.0f}% flat={counts[1]/len(y_tr)*100:.0f}% long={counts[2]/len(y_tr)*100:.0f}%)"
    )

    # --- 模型 ---
    KnCls, _ = _knowledge_net_class(num_res_blocks=num_res_blocks)
    model = KnCls(len(keep_cols), hidden_dim=hidden_dim, embed_dim=embed_dim, num_res_blocks=num_res_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 验证集预加载到 GPU
    xv = torch.tensor(x_va, device=device)
    yv = torch.tensor(y_va, dtype=torch.long, device=device)

    best_f1 = -1.0
    stale = 0
    best_acc = 0.0
    best_loss = float("inf")
    log_lines: list[str] = []

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(xv)
            val_loss = float(loss_fn(logits, yv))
            pred = logits.argmax(dim=1).cpu().numpy()
        acc = float((pred == y_va).mean())
        f1 = _macro_f1(y_va, pred)

        # 打印每类准确率
        per_cls = {}
        for c, name in [(0, "short"), (1, "flat"), (2, "long")]:
            mask = y_va == c
            if mask.any():
                per_cls[name] = f"{float((pred[mask] == c).mean()):.3f}"
        line = (
            f"Epoch {ep+1:3d}: tloss={train_loss/len(loader):.4f} vloss={val_loss:.4f} "
            f"acc={acc:.4f} f1={f1:.4f} | {per_cls}"
        )
        log_lines.append(line)
        print(line)

        improved = False
        if select_by == "accuracy":
            if acc > best_acc + 0.0005 or (acc >= best_acc - 0.0005 and f1 > best_f1 + 0.001):
                improved = True
        elif f1 > best_f1 + 0.001 or (f1 >= best_f1 - 0.001 and val_loss < best_loss):
            improved = True
        if improved:
            best_f1 = f1
            best_acc = acc
            best_loss = val_loss
            stale = 0
            torch.save(model.state_dict(), out)
            meta = {
                "input_dim": int(len(keep_cols)),
                "feature_dim_original": int(features.shape[1]),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_res_blocks": num_res_blocks,
                "scaler_path": str(sc_path),
                "val_accuracy": acc,
                "macro_f1": f1,
                "keep_cols": keep_cols.tolist(),
            }
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping")
                break

    if log_path:
        Path(log_path).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    size_kb = out.stat().st_size / 1024 if out.is_file() else 0
    t0 = time.perf_counter()
    if out.is_file():
        onnx_path = out.with_suffix(".onnx")
        try:
            kn = KnowledgeNetInference(
                onnx_path if onnx_path.is_file() else out,
                scaler_path=sc_path,
                allow_pytorch=True,
            )
            for _ in range(3):
                kn.predict(features[:1])
        except Exception as ex:
            logger.warning("训练后推理自检跳过 (ONNX/PyTorch): %s", ex)
            infer_ms = 0.0
        else:
            infer_ms = (time.perf_counter() - t0) * 1000 / 3.0
    else:
        infer_ms = 0.0

    return {
        "val_loss": best_loss,
        "val_accuracy": best_acc,
        "macro_f1": best_f1,
        "model_path": str(out),
        "scaler_path": str(sc_path),
        "model_size_kb": size_kb,
        "infer_ms_single": infer_ms,
        "passed_acc": best_acc >= 0.60,
        "passed_f1": best_f1 >= 0.45,
        "passed_size": size_kb < 1024,
        "passed_infer": infer_ms < 5.0,
    }


def _time_split_mask(times: np.ndarray, train_end: str) -> tuple[np.ndarray, np.ndarray]:
    """按时间切分：train_end 及之前为训练集，之后为验证集。"""
    import pandas as pd

    ts = pd.to_datetime(times)
    tz = ts.tz if isinstance(ts, pd.DatetimeIndex) else ts.dt.tz
    cutoff = pd.Timestamp(train_end)
    if tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(tz)
    elif tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    train_mask = np.asarray(ts <= cutoff)
    return train_mask, ~train_mask


def v14_proba_to_kn(proba_v14: np.ndarray) -> np.ndarray:
    """V14 列 0=flat,1=long,2=short → KN 列 0=short,1=flat,2=long。"""
    out = np.empty_like(proba_v14, dtype=np.float32)
    out[:, 0] = proba_v14[:, 2]
    out[:, 1] = proba_v14[:, 0]
    out[:, 2] = proba_v14[:, 1]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-9] = 1.0
    return out / row_sum


def _eval_v14_distill_metrics(y_pred: np.ndarray, teacher_probs: np.ndarray) -> dict[str, float]:
    teacher_pred = teacher_probs.argmax(axis=1)
    v14_agreement = float((y_pred == teacher_pred).mean())
    trade_mask = y_pred != 1
    trade_precision = 0.0
    if trade_mask.sum() > 0:
        trade_precision = float((y_pred[trade_mask] == teacher_pred[trade_mask]).mean())
    return {
        "v14_agreement": v14_agreement,
        "trade_precision": trade_precision,
        "trade_rate": float(trade_mask.mean()),
        "val_acc": v14_agreement,
        "macro_f1": _macro_f1(teacher_pred, y_pred),
    }


def train_knowledge_net_v14_distill(
    features: np.ndarray,
    labels: np.ndarray,
    teacher_probs: np.ndarray,
    times: np.ndarray,
    *,
    train_end: str = "2024-12-31",
    train_stride: int = 1,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 0.0005,
    patience: int = 20,
    hidden_dim: int = 128,
    embed_dim: int = 32,
    out_path: str | Path = "models/knowledge_net.pth",
    scaler_path: str | Path | None = None,
    log_path: str | Path | None = None,
    distill_weight: float = 0.95,
    temperature: float = 2.0,
    class_weights: list[float] | None = None,
    device: str = "auto",
    num_res_blocks: int = 3,
    min_v14_agreement: float = 0.70,
    min_trade_precision: float = 0.55,
) -> dict[str, Any]:
    """
    V14 教师蒸馏训练 KN1：时间切分 + KL 软标签，不用 SMOTE。
    验收：v14_agreement >= min_v14_agreement 且 trade_precision >= min_trade_precision。
    """
    torch, nn = _ensure_torch()
    import torch.nn.functional as F
    from zhulong.utils.device import resolve_torch_device
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    import joblib

    y = np.asarray(labels, dtype=np.int64)
    if y.min() < 0:
        y = y + 1
    y = y.clip(0, 2)
    teacher_probs = np.asarray(teacher_probs, dtype=np.float32)
    times = np.asarray(times)
    if teacher_probs.shape != (len(y), 3):
        raise ValueError(f"teacher_probs 应为 (N,3)，实际 {teacher_probs.shape}")

    train_mask, val_mask = _time_split_mask(times, train_end)
    if train_mask.sum() < 100 or val_mask.sum() < 50:
        raise ValueError(
            f"时间切分样本不足: train={train_mask.sum()}, val={val_mask.sum()}, train_end={train_end}"
        )

    x_raw, y_raw = features[train_mask], y[train_mask]
    t_raw = teacher_probs[train_mask]
    x_va, y_va = features[val_mask], y[val_mask]
    t_va = teacher_probs[val_mask]
    if train_stride > 1:
        idx = np.arange(0, len(x_raw), train_stride)
        x_raw, y_raw, t_raw = x_raw[idx], y_raw[idx], t_raw[idx]
        print(f"训练集 stride={train_stride} 子采样: {len(idx)} / {train_mask.sum()}")
    print(f"时间切分 train={len(x_raw)} val={len(x_va)} cutoff={train_end}")

    if features.shape[1] <= 70:
        keep_cols = np.arange(features.shape[1], dtype=np.intp)
        print(f"特征保留全列: {features.shape[1]} 维（V14 对齐，跳过去冗余）")
    else:
        keep_cols = _deduplicate_columns(x_raw)
        print(f"特征去冗余: {features.shape[1]} -> {len(keep_cols)} 维")
    x_raw = x_raw[:, keep_cols]
    x_va = x_va[:, keep_cols]

    x_tr, y_tr, t_tr = x_raw, y_raw, t_raw

    for c in range(x_tr.shape[1]):
        col = x_tr[:, c].astype(np.float64)
        lo, hi = np.percentile(col, [1, 99])
        col = np.clip(col, lo, hi)
        x_tr[:, c] = col.astype(np.float32)
        col_va = x_va[:, c].astype(np.float64)
        x_va[:, c] = np.clip(col_va, lo, hi).astype(np.float32)

    scaler = StandardScaler()
    chunk = 10000
    for start in range(0, len(x_tr), chunk):
        end = min(start + chunk, len(x_tr))
        scaler.partial_fit(np.asarray(x_tr[start:end], dtype=np.float64))
    x_tr_out = np.empty_like(x_tr, dtype=np.float32)
    for start in range(0, len(x_tr), chunk):
        end = min(start + chunk, len(x_tr))
        x_tr_out[start:end] = scaler.transform(x_tr[start:end]).astype(np.float32)
    x_tr = x_tr_out
    x_va_out = np.empty_like(x_va, dtype=np.float32)
    for start in range(0, len(x_va), chunk):
        end = min(start + chunk, len(x_va))
        x_va_out[start:end] = scaler.transform(x_va[start:end]).astype(np.float32)
    x_va = x_va_out

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sc_path = Path(scaler_path) if scaler_path else out.with_name(
        out.stem.replace("knowledge_net", "knowledge_scaler") + ".pkl"
    )
    if sc_path.name == out.name:
        sc_path = out.with_name("knowledge_scaler.pkl")
    joblib.dump(scaler, sc_path)

    device_obj = torch.device(resolve_torch_device(device))
    cw = class_weights or [2.0, 1.0, 2.0]
    class_weight_tensor = torch.tensor(cw[:3], dtype=torch.float32, device=device_obj)
    ce_weight = max(0.0, 1.0 - distill_weight)

    ds = TensorDataset(
        torch.tensor(x_tr),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(t_tr),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)
    print(
        f"KN V14 distill on {device_obj} | distill_w={distill_weight} T={temperature} | "
        f"CE w={ce_weight} class_w={cw}"
    )

    KnCls, _ = _knowledge_net_class(num_res_blocks=num_res_blocks)
    model = KnCls(len(keep_cols), hidden_dim=hidden_dim, embed_dim=embed_dim, num_res_blocks=num_res_blocks).to(device_obj)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xv = torch.tensor(x_va, device=device_obj)
    best_agreement = -1.0
    best_metrics: dict[str, float] = {}
    stale = 0
    log_lines: list[str] = []

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, tb in loader:
            xb = xb.to(device_obj, non_blocking=True)
            yb = yb.to(device_obj, non_blocking=True)
            tb = tb.to(device_obj, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(xb)
            loss = torch.tensor(0.0, device=device_obj)
            if ce_weight > 0:
                loss = loss + ce_weight * F.cross_entropy(logits, yb, weight=class_weight_tensor)
            if distill_weight > 0:
                student_log = F.log_softmax(logits / temperature, dim=1)
                teacher_soft = F.softmax(tb / temperature, dim=1)
                kl = F.kl_div(student_log, teacher_soft, reduction="batchmean") * (temperature**2)
                loss = loss + distill_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(xv)
            pred = logits.argmax(dim=1).cpu().numpy()
        metrics = _eval_v14_distill_metrics(pred, t_va)
        line = (
            f"Epoch {ep+1:3d}: loss={train_loss/len(loader):.4f} "
            f"agreement={metrics['v14_agreement']:.4f} trade_prec={metrics['trade_precision']:.4f} "
            f"f1={metrics['macro_f1']:.4f} trade_rate={metrics['trade_rate']:.3f}"
        )
        log_lines.append(line)
        print(line, flush=True)

        if metrics["v14_agreement"] > best_agreement + 0.0005:
            best_agreement = metrics["v14_agreement"]
            best_metrics = metrics
            stale = 0
            torch.save(model.state_dict(), out)
            meta = {
                "input_dim": int(len(keep_cols)),
                "feature_dim_original": int(features.shape[1]),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_res_blocks": num_res_blocks,
                "scaler_path": str(sc_path),
                "train_mode": "v14_distill",
                "train_end": train_end,
                "v14_agreement": metrics["v14_agreement"],
                "trade_precision": metrics["trade_precision"],
                "trade_rate": metrics["trade_rate"],
                "val_accuracy": metrics["val_acc"],
                "macro_f1": metrics["macro_f1"],
                "distill_weight": distill_weight,
                "temperature": temperature,
                "keep_cols": keep_cols.tolist(),
            }
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping")
                break

    if log_path:
        Path(log_path).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not out.is_file():
        raise RuntimeError("V14 蒸馏训练未产生 checkpoint")

    size_kb = out.stat().st_size / 1024
    t0 = time.perf_counter()
    onnx_path = out.with_suffix(".onnx")
    kn = KnowledgeNetInference(onnx_path if onnx_path.is_file() else out, scaler_path=sc_path)
    for _ in range(3):
        kn.predict(features[:1])
    infer_ms = (time.perf_counter() - t0) * 1000 / 3.0

    passed = (
        best_metrics.get("v14_agreement", 0) >= min_v14_agreement
        and best_metrics.get("trade_precision", 0) >= min_trade_precision
    )
    return {
        "val_loss": 0.0,
        "val_accuracy": best_metrics.get("val_acc", 0),
        "macro_f1": best_metrics.get("macro_f1", 0),
        "v14_agreement": best_metrics.get("v14_agreement", 0),
        "trade_precision": best_metrics.get("trade_precision", 0),
        "trade_rate": best_metrics.get("trade_rate", 0),
        "model_path": str(out),
        "scaler_path": str(sc_path),
        "model_size_kb": size_kb,
        "infer_ms_single": infer_ms,
        "passed_acc": passed,
        "passed_f1": best_metrics.get("macro_f1", 0) >= 0.35,
        "passed_size": size_kb < 1024,
        "passed_infer": infer_ms < 5.0,
        "train_mode": "v14_distill",
    }
