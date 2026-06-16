# ZhuLong.Package（安装打包）

DeepSeek 清单中的 `ZhuLong.Package` 在本工程对应 **Inno Setup 安装包**（见 `installer/build_installer.iss`），
而非 MSIX。原因：WinUI 3 非 MSIX 分发需捆绑 Windows App Runtime，与烛龙 Pro 一致。

构建：

```powershell
.\scripts\pack-installer.ps1
# 输出 output\ZhuLong_Setup_v1.0.1.exe
```

干净 VM 验收：Windows 10/11 x64，无 Python，双击 Setup 安装后启动 ZhuLong.exe。
