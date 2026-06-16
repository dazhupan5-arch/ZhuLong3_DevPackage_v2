# 烛龙 ZhuLong_3 基线备份与回滚

## 当前基线（跃迁升级前）

| 项 | 值 |
|----|-----|
| 版本 | **v1.0.22** |
| 基线 ID | `v1.0.22-20260609` |
| Git 标签 | `baseline/v1.0.22-20260609` |
| 冻结分支 | `backup/v1.0.22-stable` |
| 完整目录备份 | `d:\trae_projects\_backups\ZhuLong_3_v1.0.22_baseline_20260609`（约 5.2 GB，含 models） |

## 跃迁升级前检查

1. 托盘退出烛龙，MT5 可保持打开  
2. 确认备份目录存在且大小正常  
3. 在 `ZhuLong_3` 目录执行：`git tag -l "baseline/*"`

## 回滚方式（三选一）

### 方式 A：一键脚本（推荐，含 models）

```powershell
cd D:\trae_projects\ZhuLong_3
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restore-baseline.ps1
```

将当前 `ZhuLong_3` 恢复为 v1.0.22 完整快照（含模型与配置）。

### 方式 B：Git 标签（仅源码，不含被 .gitignore 忽略的 models）

```powershell
cd D:\trae_projects\ZhuLong_3
git fetch --tags
git checkout baseline/v1.0.22-20260609
```

若需继续开发：`git checkout -B upgrade/retry baseline/v1.0.22-20260609`

**注意：** `models/` 大文件在 .gitignore 中，Git 回滚后需从备份目录复制 `models\` 或重新运行 `scripts\pack-installer.ps1` 前的 deploy 脚本。

### 方式 C：手动复制备份目录

```powershell
# 先备份当前失败版本（可选）
Rename-Item D:\trae_projects\ZhuLong_3 D:\trae_projects\ZhuLong_3_failed_upgrade

# 恢复基线
Copy-Item -Recurse D:\trae_projects\_backups\ZhuLong_3_v1.0.22_baseline_20260609 D:\trae_projects\ZhuLong_3
```

## 实机回退（已安装新版本时）

1. 卸载或覆盖安装基线安装包（若备份中含 `output\ZhuLong_Setup_v1.0.22.exe`）  
2. 用户数据 `%APPDATA%\ZhuLong\` 一般可保留；若 config 被改坏，从备份中复制 `config.json` 参考项  
3. 重启 ZhuLong → 设置页确认主品种与多策略开关  

## 跃迁升级工作流建议

```powershell
cd D:\trae_projects\ZhuLong_3
git checkout -B upgrade/next baseline/v1.0.22-20260609   # 在新分支上大改
# … 开发 …
# 失败时：
git checkout backup/v1.0.22-stable
# 或运行 restore-baseline.ps1
```

## 基线功能清单

- 多策略 + SchedulerEngine（动态权重 / 状态机 / 回撤保护）
- WinUI 传递 `primary_symbol`（USOIL/XAUUSD 切换正确）
- 信号展示 strategy、设置页模型回测摘要
- XAUUSD v12 + USOIL v1 验收模型
