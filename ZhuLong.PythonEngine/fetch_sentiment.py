#!/usr/bin/env python3
"""LLM 宏观情绪离线分析 → AppData/data/sentiment.json"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 旧版 config 中的失效域名
LEGACY_API2D_HOSTS = ("api.api2d.com",)

# 按优先级尝试的 OpenAI 兼容端点
DEFAULT_LLM_ENDPOINTS = (
    "https://oa.api2d.net/v1/chat/completions",
    "https://openai.api2d.net/v1/chat/completions",
    "https://stream.api2d.net/v1/chat/completions",
)


def resolve_data_dir() -> Path:
    env = os.environ.get("ZHULONG_DATA_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    p = Path(appdata) / "ZhuLong" / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_secret_file(name: str) -> str:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    path = Path(appdata) / "ZhuLong" / "secrets" / name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_config() -> dict:
    for cfg_path in (
        Path(os.environ.get("APPDATA", "")) / "ZhuLong" / "config.json",
        ROOT / "config.json",
    ):
        if cfg_path.is_file():
            with cfg_path.open(encoding="utf-8") as f:
                return json.load(f)
    return {}


def fetch_gold_silver_ratio() -> float:
    return 82.0


def normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return DEFAULT_LLM_ENDPOINTS[0]
    for host in LEGACY_API2D_HOSTS:
        if host in u:
            return DEFAULT_LLM_ENDPOINTS[0]
    if u.endswith("/v1"):
        return u.rstrip("/") + "/chat/completions"
    if "/chat/completions" not in u:
        return u.rstrip("/") + "/v1/chat/completions"
    return u


def iter_llm_endpoints(configured: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in (configured, *DEFAULT_LLM_ENDPOINTS):
        n = normalize_base_url(u)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def call_llm(base_url: str, api_key: str, model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是宏观情绪分析师，只输出 JSON，不要 markdown。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(base_url, headers=headers, json=body, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def main() -> int:
    cfg = load_config()
    macro = cfg.get("macro", {}) or {}
    sent_cfg = macro.get("sentiment", {}) or {}
    api_key = read_secret_file("llm_api_key.txt") or (
        sent_cfg.get("api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    ).strip()
    configured_url = sent_cfg.get("base_url") or os.environ.get("ZHULONG_LLM_BASE_URL") or ""
    model = sent_cfg.get("model") or "gpt-3.5-turbo"
    out_path = resolve_data_dir() / Path(sent_cfg.get("json_path") or "data/sentiment.json").name

    gs_ratio = fetch_gold_silver_ratio()
    existing: dict = {}
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        gs_ratio = float(existing.get("gold_silver_ratio", gs_ratio))

    if not api_key:
        print("警告: 未配置 LLM API Key，写入占位 sentiment.json")
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "gold_silver_ratio": gs_ratio,
            "xauusd_sentiment": existing.get("xauusd_sentiment", 0.5),
            "usoil_sentiment": existing.get("usoil_sentiment", 0.5),
            "overall_sentiment": existing.get("overall_sentiment", 0.5),
            "llm_summary": "未配置 LLM，使用占位情绪分。",
        }
    else:
        prompt = f"""分析 XAUUSD（黄金）与 USOIL（原油）短期宏观情绪。
当前金银比约 {gs_ratio:.1f}。
请输出 JSON：
{{
  "gold_silver_ratio": {gs_ratio},
  "xauusd_sentiment": 0.0到1.0,
  "usoil_sentiment": 0.0到1.0,
  "overall_sentiment": 0.0到1.0,
  "llm_summary": "一句话中文总结"
}}"""
        last_err = ""
        raw = None
        for endpoint in iter_llm_endpoints(configured_url):
            try:
                print(f"  LLM 尝试: {endpoint}")
                raw = call_llm(endpoint, api_key, model, prompt)
                print(f"  LLM 成功: {endpoint}")
                break
            except requests.HTTPError as ex:
                last_err = f"{endpoint}: HTTP {ex.response.status_code if ex.response else ex}"
                print(f"  LLM 失败: {last_err}")
            except Exception as ex:
                last_err = f"{endpoint}: {ex}"
                print(f"  LLM 失败: {last_err}")

        if raw is None:
            print(f"警告: LLM 全部失败 ({last_err})，写入占位 sentiment.json")
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "gold_silver_ratio": gs_ratio,
                "xauusd_sentiment": existing.get("xauusd_sentiment", 0.5),
                "usoil_sentiment": existing.get("usoil_sentiment", 0.5),
                "overall_sentiment": existing.get("overall_sentiment", 0.5),
                "llm_summary": f"LLM 不可用，已保留占位（{last_err}）",
            }
        else:
            parsed = parse_llm_json(raw)
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **parsed,
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
