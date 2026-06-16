# 烛龙三代 · 计划缺口清单

> 对照 `IMPLEMENTATION_PLAN.md` / `LEAP_ROADMAP.md`。勾选表示已完成。

## Phase 0 — 工程基建

- [x] `config.json` 初版
- [x] JSON Schema 文件 `config/config.schema.json`
- [x] 启动时 Schema 校验（`ConfigSchemaValidator`）
- [x] Git 仓库 + `.gitignore`
- [x] pytest 基架扩充
- [ ] 全量 JSON Schema 覆盖所有 config 字段

## Phase 1 — MT5 双通道

- [x] `ZhuLongIndicator.mq5`
- [x] C# `PipeServer`（替代 Python `data_receiver.py`）
- [x] 绘图协议 `draw_signal` / `clear_signal`
- [x] `scripts/smoke_pipe.py`
- [x] `scripts/deploy-mt5-indicator.ps1`（全终端部署）

## Phase 2 — 特征工程

- [x] M1→M5 重采样（`FeatureCacheService`）
- [x] `feature_engine.py` 30 维逻辑
- [x] C# `FeatureCalculator` 22 维对齐 Python
- [x] C# `FeatureMtfCalculator` 6 维 MTF
- [x] `zhulong/feature_schema.json`
- [x] 特征 golden 回归测试
- [x] `scripts/download_m1.py`

## Phase 3 — 训练与模型（**依赖正式模型**）

- [x] `train.py` + `zhulong/training/pipeline.py`
- [x] LightGBM 训练子管线 `zhulong/training/lgb/`
- [x] `zhulong/training/transformer_pretrain.py` 骨架
- [ ] Transformer 15% mask 预训练跑通
- [ ] `reg_expected_return` 正式版
- [ ] **正式生产模型** `acceptance_passed: true`

## Phase 4 — 信号生成

- [x] `inference_engine.py` + C# `SignalGeneratorService`
- [x] `ProductionModelGate`（无正式模型时暂停推理）
- [ ] 实机 L2-3 信号出图（需正式模型）
- [x] `scripts/inject_test_signal.py`（无模型验 DB/匹配）

## Phase 5 — 持仓管理

- [x] `PositionManagerService`（G5/G8）
- [x] 移动止损 / 分批止盈 / 时间止损
- [ ] 实机 L2-4/L2-5 签字（可用注入信号手动验）

## Phase 6 — GUI 与归因

- [x] WinUI 面板 + SQLite + ScottPlot
- [x] `scripts/seed_attribution_data.py`
- [ ] 实机完整闭环一笔交易

## Phase 7 — 回测与交付

- [x] `backtest/engine.py` 对接 `lgb/backtest`
- [x] `scripts/walk_forward.py` 骨架
- [x] Inno Setup + `build_release.ps1`
- [x] `docs/安装说明.md`
- [x] `scripts/verify_clean_vm.ps1`
- [ ] 干净 VM 实机签字
- [ ] 模拟盘 ≥3 交易日签字

## §4.1 增补

- [x] manifest / magic+comment / symbol_mapping
- [x] `RiskGuardService`
- [x] `AlertService` Webhook 结构化事件
- [x] `scripts/backup_trading_db.ps1` + 计划任务注册
- [ ] Telegram/邮件原生告警
- [ ] 多账户

## Leap 2 门禁

- [x] L2-1 脚本检查
- [x] L2-2 管道
- [ ] L2-3 信号（正式模型）
- [ ] L2-4 匹配（可注入信号验）
- [ ] L2-5 改单（可注入信号验）
- [x] L2-6 版本检查脚本
