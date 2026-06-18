"""分层归因分析 + 调参建议。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zhulong.attribution.schema import normalize_regime


@dataclass
class LayerBin:
    label: str
    count: int = 0
    wins: int = 0
    pnl_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.pnl_sum / self.count if self.count else 0.0


@dataclass
class TuneSuggestion:
    key: str
    reason: str
    action: str
    priority: str = "medium"

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "reason": self.reason, "action": self.action, "priority": self.priority}


@dataclass
class AttributionReport:
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    horizon_bins: list[LayerBin] = field(default_factory=list)
    regime_bins: list[LayerBin] = field(default_factory=list)
    gate_bins: list[LayerBin] = field(default_factory=list)
    kn2_bins: list[LayerBin] = field(default_factory=list)
    confidence_bins: list[LayerBin] = field(default_factory=list)
    loss_attribution: list[dict[str, Any]] = field(default_factory=list)
    tune_suggestions: list[TuneSuggestion] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        def _bins(rows: list[LayerBin]) -> list[dict[str, Any]]:
            return [
                {
                    "label": b.label,
                    "count": b.count,
                    "win_rate": round(b.win_rate, 4),
                    "avg_pnl_pct": round(b.avg_pnl, 4),
                }
                for b in rows
            ]

        return {
            "generated_at": self.generated_at,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "profit_factor": round(self.profit_factor, 2),
            "horizon_bins": _bins(self.horizon_bins),
            "regime_bins": _bins(self.regime_bins),
            "gate_bins": _bins(self.gate_bins),
            "kn2_bins": _bins(self.kn2_bins),
            "confidence_bins": _bins(self.confidence_bins),
            "loss_attribution": self.loss_attribution,
            "tune_suggestions": [s.to_dict() for s in self.tune_suggestions],
        }


class AttributionEngine:
    """分析带 attribution_json 的已平仓交易。"""

    def __init__(self, min_samples: int = 5) -> None:
        self.min_samples = min_samples

    def analyze(self, rows: list[dict[str, Any]]) -> AttributionReport:
        report = AttributionReport(generated_at=datetime.now(timezone.utc).isoformat())
        if not rows:
            return report

        pnls = [float(r.get("pnl_percent") or 0) for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        report.total_trades = len(rows)
        report.win_rate = wins / len(rows)
        report.avg_pnl_pct = sum(pnls) / len(rows)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        report.profit_factor = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        h_map: dict[str, LayerBin] = {}
        r_map: dict[str, LayerBin] = {}
        g_map: dict[str, LayerBin] = {}
        k_map: dict[str, LayerBin] = {}
        c_map: dict[str, LayerBin] = {}
        loss_layers: dict[str, int] = defaultdict(int)

        for row in rows:
            pnl = float(row.get("pnl_percent") or 0)
            is_win = pnl > 0
            snap = self._parse_snap(row.get("attribution_json") or row.get("attribution"))
            self._add_bin(h_map, f"horizon={snap.get('horizon_direction', '?')}", is_win, pnl)
            self._add_bin(r_map, f"regime={snap.get('cognition_regime', 'unknown')}", is_win, pnl)
            fr = str(snap.get("filter_reason") or "none")
            self._add_bin(g_map, f"gate={fr or 'none'}", is_win, pnl)
            kn2 = "veto" if snap.get("kn2_should_trade") is False and snap.get("kn2_shadow_mode") else (
                "allow" if snap.get("kn2_should_trade") else "na"
            )
            self._add_bin(k_map, f"kn2={kn2}", is_win, pnl)
            conf = float(snap.get("horizon_confidence") or snap.get("cognition_confidence") or 0)
            cbin = self._conf_bin(conf)
            self._add_bin(c_map, cbin, is_win, pnl)
            if not is_win:
                loss_layers[self._primary_loss_layer(snap, pnl)] += 1

        report.horizon_bins = sorted(h_map.values(), key=lambda b: -b.count)
        report.regime_bins = sorted(r_map.values(), key=lambda b: -b.count)
        report.gate_bins = sorted(g_map.values(), key=lambda b: -b.count)
        report.kn2_bins = sorted(k_map.values(), key=lambda b: -b.count)
        report.confidence_bins = sorted(c_map.values(), key=lambda b: cbin_label_sort(b.label))
        report.loss_attribution = [
            {"layer": k, "loss_count": v, "share": round(v / max(sum(loss_layers.values()), 1), 3)}
            for k, v in sorted(loss_layers.items(), key=lambda x: -x[1])
        ]
        report.tune_suggestions = self._suggest(report, rows)
        return report

    @staticmethod
    def _parse_snap(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _add_bin(m: dict[str, LayerBin], label: str, is_win: bool, pnl: float) -> None:
        b = m.setdefault(label, LayerBin(label=label))
        b.count += 1
        b.wins += int(is_win)
        b.pnl_sum += pnl

    @staticmethod
    def _conf_bin(conf: float) -> str:
        if conf < 0.55:
            return "conf=0.50-0.55"
        if conf < 0.65:
            return "conf=0.55-0.65"
        if conf < 0.75:
            return "conf=0.65-0.75"
        if conf < 0.85:
            return "conf=0.75-0.85"
        return "conf=0.85-1.00"

    @staticmethod
    def _primary_loss_layer(snap: dict[str, Any], pnl: float) -> str:
        h = str(snap.get("horizon_direction", "flat"))
        sig = str(snap.get("signal_direction", ""))
        if sig in ("buy", "sell") and h not in ("long", "short"):
            return "horizon_flat"
        if h == "long" and sig == "sell":
            return "horizon_vs_signal"
        if h == "short" and sig == "buy":
            return "horizon_vs_signal"
        regime = normalize_regime(str(snap.get("cognition_regime", "")))
        pos = float(snap.get("pos_in_range", 0.5))
        if regime == "ranging" and sig == "buy" and pos > 0.65:
            return "location_high_long"
        if regime == "ranging" and sig == "sell" and pos < 0.35:
            return "location_low_short"
        if str(snap.get("filter_reason", "")):
            return "execution_gate"
        if str(snap.get("rl_raw_action", "")) != str(snap.get("final_action", "")):
            return "rl_override"
        return "rl_execution"

    def _suggest(self, report: AttributionReport, rows: list[dict[str, Any]]) -> list[TuneSuggestion]:
        out: list[TuneSuggestion] = []
        for b in report.regime_bins:
            if b.count >= self.min_samples and b.label == "regime=ranging" and b.win_rate < 0.42:
                out.append(
                    TuneSuggestion(
                        "execution_gates.structure_location_gate",
                        f"震荡盘胜率 {b.win_rate:.1%} (n={b.count})",
                        "保持或收紧 structure_location_gate",
                        "high",
                    )
                )
        for b in report.horizon_bins:
            if b.count >= self.min_samples and "flat" not in b.label and b.win_rate < 0.45:
                out.append(
                    TuneSuggestion(
                        "horizon_min_confidence",
                        f"{b.label} 胜率 {b.win_rate:.1%}",
                        "运行 calibrate_horizon_v16.py 或提高 min_confidence",
                        "medium",
                    )
                )
        for b in report.confidence_bins:
            if b.count >= self.min_samples and "0.55-0.65" in b.label and b.win_rate < 0.45:
                out.append(
                    TuneSuggestion(
                        "rl_inference.min_confidence_for_trade",
                        f"中低置信区间胜率 {b.win_rate:.1%}",
                        "提高 min_confidence_for_trade 至 0.58+",
                        "medium",
                    )
                )
        rl_loss = sum(1 for r in rows if float(r.get("pnl_percent") or 0) < 0)
        rl_layer = next((x for x in report.loss_attribution if x["layer"] in ("rl_execution", "rl_override")), None)
        if rl_layer and rl_layer.get("share", 0) > 0.45 and rl_loss >= self.min_samples:
            out.append(
                TuneSuggestion(
                    "meta_finetune",
                    f"RL 层亏损占比 {rl_layer['share']:.0%}",
                    "触发 weekly_finetune / PPO 继续训练",
                    "high",
                )
            )
        return out

    def save_report(self, report: AttributionReport, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p


def cbin_label_sort(label: str) -> int:
    order = ["0.50", "0.55", "0.65", "0.75", "0.85"]
    for i, o in enumerate(order):
        if o in label:
            return i
    return 99
