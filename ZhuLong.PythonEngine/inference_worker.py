#!/usr/bin/env python3
"""常驻推理 Worker — stdin/stdout NDJSON RPC，复用 AgentEngine 缓存。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent
_install = os.environ.get("ZHULONG_INSTALL_DIR")
_appdata = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
_path_candidates: list[Path] = []
if _appdata.is_dir():
    _path_candidates.append(_appdata)
_path_candidates.append(_ENGINE)
if _install and Path(_install).is_dir():
    _path_candidates.append(Path(_install))
for p in reversed(_path_candidates):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

_ROOT = Path(_install) if _install and Path(_install).is_dir() else Path(__file__).resolve().parent.parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
    _install = os.environ.get("ZHULONG_INSTALL_DIR")
    if _install and os.path.isdir(_install):
        add_fn = getattr(os, "add_dll_directory", None)
        if add_fn is not None:
            try:
                add_fn(_install)
            except OSError:
                pass
        os.environ["PATH"] = _install + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

from hotfix_loader import apply_appdata_hotfixes  # noqa: E402

apply_appdata_hotfixes()

from inference_cli import (  # noqa: E402
    _cmd_agent_record_signal,
    _cmd_agent_record_trade,
    _cmd_agent_tick,
    _cmd_agent_validate,
    _cmd_agent_warmup,
)

_REAL_STDOUT = sys.stdout


def _handle_agent_warmup(req: dict) -> dict:
    """Horizon ONNX + AgentEngine 全栈热加载（Horizon/KN2/RL，开机即就绪）。"""
    return _cmd_agent_warmup(req)


_HANDLERS = {
    "ping": lambda req: {"ok": True, "worker": True, "pid": os.getpid()},
    "shutdown": lambda req: {"ok": True, "shutdown": True},
    "agent_warmup": _handle_agent_warmup,
    "agent_tick": _cmd_agent_tick,
    "agent_validate": _cmd_agent_validate,
    "agent_record_trade": _cmd_agent_record_trade,
    "agent_record_signal": _cmd_agent_record_signal,
}


def _dispatch(req: dict) -> dict:
    cmd = str(req.get("cmd") or "")
    fn = _HANDLERS.get(cmd)
    if fn is None:
        return {"ok": False, "error": f"unknown cmd: {cmd}"}
    return fn(req)


def _write_response(resp: dict, req_id) -> None:
    from zhulong.utils.json_safe import dumps_strict, json_safe

    safe = json_safe(resp)
    if req_id is not None:
        safe["id"] = req_id
    _REAL_STDOUT.write(dumps_strict(safe) + "\n")
    _REAL_STDOUT.flush()


class _RpcStdout:
    """NDJSON RPC：库 print 重定向 stderr；仅合法 JSON 行才转发 stdout。"""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    @staticmethod
    def _forward_json_line(real, line: str) -> None:
        stripped = line.strip()
        if not stripped.startswith("{"):
            sys.stderr.write(stripped + "\n")
            sys.stderr.flush()
            return
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            sys.stderr.write(f"[stdout-filter] non-json: {stripped[:240]}\n")
            sys.stderr.flush()
            return
        real.write(stripped + "\n")
        real.flush()

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not line.strip():
                continue
            self._forward_json_line(self._real, line)
        return len(s)

    def flush(self) -> None:
        tail = self._buf.strip()
        if tail:
            self._forward_json_line(self._real, tail)
            self._buf = ""
        self._real.flush()
        sys.stderr.flush()

    def reconfigure(self, **kwargs):  # type: ignore[no-untyped-def]
        fn = getattr(self._real, "reconfigure", None)
        if fn is not None:
            fn(**kwargs)


def main() -> int:
    global _REAL_STDOUT
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    _REAL_STDOUT = sys.stdout
    sys.stdout = _RpcStdout(_REAL_STDOUT)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            resp = _dispatch(req)
            _write_response(resp, req_id)
            if resp.get("shutdown"):
                return 0
        except Exception as ex:
            err = {"ok": False, "error": str(ex), "trace": traceback.format_exc()}
            _write_response(err, req_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
