# 烛龙 Logo

## 设计

**形**：水平透镜，上金下墨 + 中心竖缝  
**色**：`#0A0A0A` + `#D4A820` → `#141414`

设计源：`vector/zhulong-logo.svg`（真矢量）· Skia 渲染导出 PNG/ICO

## 导出（全量品牌资源）

```bash
py -3 -m pip install skia-python pillow
py -3 scripts/generate_app_icons.py
```

构建 `ZhuLong.App` 时也会自动运行上述脚本（`PrepareForBuild`）。

## 资源落点

| 场景 | 文件 |
|------|------|
| 标题栏 / 侧栏 / 界面内 Logo | `TitleLogo.png`, `StoreLogo.png` |
| 任务栏 / 窗口角标 | `zhulong.ico` + `WM_SETICON` |
| 系统托盘 | `zhulong.ico`（32×32 层） |
| exe 内嵌图标 | `ApplicationIcon` → `zhulong.ico` |
| 桌面 / 开始菜单快捷方式 | 安装包 `{app}\Assets\zhulong.ico` |
| 安装程序 / 卸载图标 | `assets/logo/zhulong.ico` |
| 磁贴 / 启动画面 | `Square*.png`, `SplashScreen*.png`, `Wide310x150Logo*.png` |

代码统一入口：`AppBrandAssets.cs`
