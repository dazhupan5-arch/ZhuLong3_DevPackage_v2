"""训练设备选择：GPU 优先，不可用时回退 CPU。"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_name() -> str:
    try:
        import torch

        return str(torch.cuda.get_device_name(0))
    except Exception:
        return ""


def resolve_torch_device(pref: str = "auto") -> str:
    """返回 ``cuda`` 或 ``cpu``（供 PyTorch 使用）。"""
    pref = (pref or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if _torch_cuda_available():
        if pref in ("auto", "cuda", "gpu"):
            name = _gpu_name()
            if name:
                log.info("PyTorch device: cuda (%s)", name)
            return "cuda"
        log.warning("Requested %s but only auto/cuda/gpu supported with CUDA — using cuda", pref)
        return "cuda"
    if pref in ("cuda", "gpu"):
        log.warning(
            "CUDA unavailable — install NVIDIA driver and CUDA-enabled PyTorch. Using CPU."
        )
    return "cpu"


def resolve_sb3_device(pref: str = "auto") -> str:
    """stable-baselines3 设备：``cuda`` / ``cpu``。"""
    pref = (pref or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "cuda", "gpu"):
        return "cuda" if _torch_cuda_available() else "cpu"
    return "cpu"


def resolve_structure_n_jobs(pref: int | str = 0) -> int:
    """结构特征并行 worker 数。0=自动（多核），1=单线程。"""
    if isinstance(pref, str):
        pref = pref.strip().lower()
        if pref in ("auto", ""):
            return 0
        pref = int(pref)
    return int(pref)


def describe_gpu_status() -> dict[str, Any]:
    """诊断 GPU / CUDA 是否可用于训练。"""
    info: dict[str, Any] = {
        "torch_imported": False,
        "cuda_available": False,
        "device_name": "",
        "torch_version": "",
        "cuda_version": "",
        "recommended_torch_device": "cpu",
        "recommended_sb3_device": "cpu",
    }
    try:
        import torch

        info["torch_imported"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = _gpu_name()
            info["cuda_version"] = str(torch.version.cuda or "")
            info["recommended_torch_device"] = "cuda"
            info["recommended_sb3_device"] = "cuda"
    except Exception as ex:
        info["error"] = str(ex)
    return info


def print_gpu_status() -> int:
    """打印 GPU 状态；CUDA 可用返回 0，否则返回 1。"""
    s = describe_gpu_status()
    print("=== GPU / CUDA 检查 ===")
    if not s.get("torch_imported"):
        print("PyTorch: 未安装或无法导入")
        if s.get("error"):
            print(f"  错误: {s['error']}")
        return 1
    print(f"PyTorch: {s['torch_version']}")
    print(f"CUDA available: {s['cuda_available']}")
    if s["cuda_available"]:
        print(f"GPU: {s['device_name']}")
        print(f"CUDA: {s['cuda_version']}")
        print("KnowledgeNet / PPO 将使用 GPU。")
        return 0
    print("GPU: 不可用")
    print("请安装 NVIDIA 驱动，并安装 CUDA 版 PyTorch：")
    print("  py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("StructureAnalyzer 仍为 CPU 算法（pandas/sklearn），无法用 GPU 加速。")
    return 1
