"""LightGBM 验收标准（v1 / v2 临时 / final 最终）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LgbAcceptanceThresholds:
    precision: float = 0.75
    recall: float = 0.20
    f1: float = 0.30
    long_short_prec_gap: float = 0.15
    oos_win_rate: float = 0.70
    oos_avg_rr: float = 1.5
    max_daily_signals: int = 5
    stress_max_drawdown: float = 0.15
    stress_max_consec_loss: int = 8


def thresholds_v1() -> LgbAcceptanceThresholds:
    return LgbAcceptanceThresholds(
        precision=0.60,
        recall=0.10,
        f1=0.15,
        oos_win_rate=0.55,
        oos_avg_rr=1.2,
        max_daily_signals=8,
        stress_max_drawdown=0.25,
    )


def thresholds_v2() -> LgbAcceptanceThresholds:
    """v2 临时上线测试门槛。"""
    return LgbAcceptanceThresholds(
        precision=0.40,
        recall=0.15,
        f1=0.20,
        oos_win_rate=0.50,
        oos_avg_rr=1.0,
        max_daily_signals=12,
        stress_max_drawdown=0.35,
        stress_max_consec_loss=12,
    )


def thresholds_final() -> LgbAcceptanceThresholds:
    return LgbAcceptanceThresholds()


def thresholds_v4() -> LgbAcceptanceThresholds:
    """v4 实盘模拟门槛。"""
    return LgbAcceptanceThresholds(
        precision=0.55,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.52,
        oos_avg_rr=1.2,
        max_daily_signals=12,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v42() -> LgbAcceptanceThresholds:
    """v4.2：60min 预测窗口，持仓≤4h。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.50,
        oos_avg_rr=1.3,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v5() -> LgbAcceptanceThresholds:
    """v5 二分类做多 vs 非多。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.20,
        f1=0.25,
        oos_win_rate=0.52,
        oos_avg_rr=1.3,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v51() -> LgbAcceptanceThresholds:
    """v5.1：gain=0.20%，回测带 60min 冷却。"""
    return LgbAcceptanceThresholds(
        precision=0.45,
        recall=0.10,
        f1=0.15,
        oos_win_rate=0.50,
        oos_avg_rr=1.3,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v6() -> LgbAcceptanceThresholds:
    """v6：盈亏对齐标签 + 60min 持仓 + 冷却。"""
    return LgbAcceptanceThresholds(
        precision=0.55,
        recall=0.20,
        f1=0.30,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v61() -> LgbAcceptanceThresholds:
    """v6.1：盈亏标签 + 120min 持仓（24 根 M5）。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.15,
        f1=0.20,
        oos_win_rate=0.52,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v7() -> LgbAcceptanceThresholds:
    """v7：LSTM 端到端 OHLCV 序列。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.20,
        f1=0.25,
        oos_win_rate=0.52,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v8() -> LgbAcceptanceThresholds:
    """v8：多尺度分解 + XGB/LGB 集成。"""
    return LgbAcceptanceThresholds(
        precision=0.55,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.52,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.15,
        stress_max_consec_loss=99,
    )


def thresholds_v9() -> LgbAcceptanceThresholds:
    """v9：双分类集成 + 趋势过滤 + 移动止损。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.25,
        stress_max_consec_loss=99,
    )


def thresholds_v11() -> LgbAcceptanceThresholds:
    """v11：三分类 XGBoost。"""
    return LgbAcceptanceThresholds(
        precision=0.55,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.25,
        stress_max_consec_loss=99,
    )


def thresholds_v15() -> LgbAcceptanceThresholds:
    """V15：OOS 优先 + 适度 val 精度（与当前数据分布匹配）。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.15,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.45,
        max_daily_signals=8,
        stress_max_drawdown=0.25,
        stress_max_consec_loss=99,
    )


def thresholds_v13_triple() -> LgbAcceptanceThresholds:
    """v13 三重屏障 XGBoost（v3 对齐 SL/TP + 趋势过滤）。"""
    return LgbAcceptanceThresholds(
        precision=0.55,
        recall=0.30,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.20,
        stress_max_consec_loss=99,
    )


def thresholds_v13() -> LgbAcceptanceThresholds:
    """v13：简化标签 + 反转特征 XGBoost 三分类。"""
    return LgbAcceptanceThresholds(
        precision=0.60,
        recall=0.40,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.25,
        stress_max_consec_loss=99,
    )


def thresholds_v12() -> LgbAcceptanceThresholds:
    """v12：双向不对称后处理。"""
    return LgbAcceptanceThresholds(
        precision=0.53,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.55,
        oos_avg_rr=1.5,
        max_daily_signals=8,
        stress_max_drawdown=0.30,
        stress_max_consec_loss=99,
    )


def thresholds_oil_v1() -> LgbAcceptanceThresholds:
    """USOIL v1：原油三分类 XGBoost。"""
    return LgbAcceptanceThresholds(
        precision=0.50,
        recall=0.0,
        f1=0.0,
        oos_win_rate=0.52,
        oos_avg_rr=1.4,
        max_daily_signals=8,
        stress_max_drawdown=0.35,
        stress_max_consec_loss=99,
    )


def get_thresholds(stage: str) -> LgbAcceptanceThresholds:
    if stage == "v1":
        return thresholds_v1()
    if stage == "v2":
        return thresholds_v2()
    if stage == "v4":
        return thresholds_v4()
    if stage == "v42":
        return thresholds_v42()
    if stage == "v5":
        return thresholds_v5()
    if stage == "v51":
        return thresholds_v51()
    if stage == "v6":
        return thresholds_v6()
    if stage == "v61":
        return thresholds_v61()
    if stage == "v7":
        return thresholds_v7()
    if stage == "v8":
        return thresholds_v8()
    if stage == "v9":
        return thresholds_v9()
    if stage == "v11":
        return thresholds_v11()
    if stage == "v12":
        return thresholds_v12()
    if stage == "v13":
        return thresholds_v13()
    if stage == "v13_triple":
        return thresholds_v13_triple()
    if stage == "v15":
        return thresholds_v15()
    if stage == "oil_v1":
        return thresholds_oil_v1()
    return thresholds_final()


@dataclass
class LgbAcceptanceReport:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    acceptance_stage: str = "v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "acceptance_stage": self.acceptance_stage,
            "metrics": self.metrics,
            "failures": self.failures,
        }


def evaluate_lgb_acceptance(
    val_metrics: dict[str, float],
    test1_metrics: dict[str, float],
    stress_metrics: dict[str, float],
    thresholds: LgbAcceptanceThresholds | None = None,
    stage: str = "v2",
) -> LgbAcceptanceReport:
    th = thresholds or get_thresholds(stage)
    failures: list[str] = []
    metrics = {"validation": val_metrics, "test1": test1_metrics, "stress": stress_metrics}

    if stage not in ("v4", "v42"):
        if val_metrics.get("precision", 0) < th.precision:
            failures.append(f"val_precision={val_metrics.get('precision', 0):.3f}<{th.precision}")
        if val_metrics.get("recall", 0) < th.recall:
            failures.append(f"val_recall={val_metrics.get('recall', 0):.3f}<{th.recall}")
        if val_metrics.get("f1", 0) < th.f1:
            failures.append(f"val_f1={val_metrics.get('f1', 0):.3f}<{th.f1}")

    if test1_metrics.get("win_rate", 0) < th.oos_win_rate:
        failures.append(f"test1_win_rate={test1_metrics.get('win_rate', 0):.3f}<{th.oos_win_rate}")
    if test1_metrics.get("avg_rr", 0) < th.oos_avg_rr:
        failures.append(f"test1_avg_rr={test1_metrics.get('avg_rr', 0):.3f}<{th.oos_avg_rr}")
    if test1_metrics.get("max_daily_signals", 99) > th.max_daily_signals:
        failures.append(
            f"test1_max_daily_signals={test1_metrics.get('max_daily_signals')}>{th.max_daily_signals}"
        )
    if test1_metrics.get("max_drawdown", 1) > th.stress_max_drawdown:
        failures.append(
            f"test1_max_drawdown={test1_metrics.get('max_drawdown', 1):.3f}>{th.stress_max_drawdown}"
        )

    if stage not in ("v4", "v42", "v5", "v6", "v61", "v7", "v8", "v9", "v11", "v12", "v13", "v13_triple", "v15", "oil_v1"):
        if stress_metrics.get("max_drawdown", 1) > th.stress_max_drawdown:
            failures.append(f"stress_dd={stress_metrics.get('max_drawdown', 1):.3f}>{th.stress_max_drawdown}")
        if stress_metrics.get("max_consec_losses", 99) > th.stress_max_consec_loss:
            failures.append(
                f"stress_consec_loss={stress_metrics.get('max_consec_losses')}>{th.stress_max_consec_loss}"
            )

    return LgbAcceptanceReport(
        passed=len(failures) == 0,
        metrics=metrics,
        failures=failures,
        acceptance_stage=stage,
    )
