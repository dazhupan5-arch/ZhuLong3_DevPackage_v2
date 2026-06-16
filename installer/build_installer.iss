; 烛龙 ZhuLong · Inno Setup（WinUI 3）
; 编译前: scripts\pack-installer.ps1

#define MyAppName "烛龙量化交易系统"
#define MyAppVersion "3.1.11"
#define MyAppPublisher "Stephen.Pan"
#define MyAppExeName "ZhuLong.exe"

[Setup]
AppId={{B2C3D4E5-F6A7-8901-BCDE-F12345678901}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\ZhuLong
DefaultGroupName=烛龙
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=..\output
OutputBaseFilename=ZhuLong_Setup_v3.1.11
SetupIconFile=..\assets\logo\zhulong.ico
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
DiskSpanning=no
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
UsePreviousAppDir=no
; 禁止安装程序自动关闭其他进程（含 MT5 terminal64）；仅 [Code] 中 taskkill ZhuLong.exe
CloseApplications=no
AppMutex=ZhuLong.Trading.App.v3,Global\ZhuLong.Trading.App.v3
SetupLogging=yes

[InstallDelete]
; 覆盖安装时删除旧版自包含 host，避免与 framework-dependent runtimeconfig 混用
Type: files; Name: "{app}\coreclr.dll"
Type: files; Name: "{app}\hostfxr.dll"
Type: files; Name: "{app}\hostpolicy.dll"
Type: files; Name: "{app}\clrjit.dll"
Type: files; Name: "{app}\mscordaccore.dll"
Type: files; Name: "{app}\mscordbi.dll"
Type: files; Name: "{app}\createdump.exe"

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "redist\VC_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "redist\WindowsAppRuntimeInstall-x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "redist\windowsdesktop-runtime-8.0-win-x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "redist\VC_redist.x64.exe"; DestDir: "{app}\redist"; Flags: ignoreversion
Source: "redist\WindowsAppRuntimeInstall-x64.exe"; DestDir: "{app}\redist"; Flags: ignoreversion
Source: "redist\windowsdesktop-runtime-8.0-win-x64.exe"; DestDir: "{app}\redist"; Flags: ignoreversion
Source: "..\publish\win-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\烛龙系统"; Filename: "{app}\LaunchZhuLong.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; IconIndex: 0
Name: "{group}\MT5 手动复制说明"; Filename: "notepad.exe"; Parameters: """{app}\mql5\Libraries\ZhuLong_部署说明.txt"""; Comment: "查看 MT5 桥接文件手动复制步骤"
Name: "{group}\MT5 指标文件目录"; Filename: "{cmd}"; Parameters: "/c explorer ""{app}\indicators"""; Comment: "打开 ZhuLongIndicator 与 DLL 源文件"
Name: "{group}\卸载烛龙"; Filename: "{uninstallexe}"
Name: "{commondesktop}\烛龙系统"; Filename: "{app}\LaunchZhuLong.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; IconIndex: 0

[Run]
; .NET 8 Desktop → WinUI 3 → VC++（框架依赖发布必需 Desktop Runtime，非 SDK）
Filename: "{tmp}\windowsdesktop-runtime-8.0-win-x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "正在安装 .NET 8 Desktop 运行库..."; Flags: runhidden waituntilterminated; Check: NeedInstallDotNet8Desktop
Filename: "{tmp}\WindowsAppRuntimeInstall-x64.exe"; Parameters: "--quiet"; StatusMsg: "正在安装 Windows App Runtime (WinUI 3)，约需 5-10 分钟..."; Flags: runhidden waituntilterminated; Check: NeedInstallWinAppRuntime
Filename: "{tmp}\VC_redist.x64.exe"; Parameters: "/install /passive /norestart"; StatusMsg: "正在安装 Visual C++ 运行库..."; Flags: runhidden waituntilterminated; Check: NeedInstallVcRedist
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install_post_setup.ps1"" -InstallDir ""{app}"""; StatusMsg: "正在配置 Python 智能体依赖..."; Flags: runhidden waituntilterminated
Filename: "{app}\LaunchZhuLong.cmd"; Description: "启动烛龙系统"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=将安装 [name/ver]。%n%n升级前请先退出烛龙。安装程序会自动安装缺失的 .NET 8 Desktop、WinUI 3、VC++ 运行库（已安装则跳过）。%n%nMT5 桥接文件需手动复制（见 mql5\Libraries\ZhuLong_部署说明.txt）。安装程序不会关闭 MT5。

[Code]
function DotNet8DesktopInstalled(): Boolean;
var
  Path: String;
  FindRec: TFindRec;
begin
  Path := ExpandConstant('{commonpf64}\dotnet\shared\Microsoft.WindowsDesktop.App');
  if not DirExists(Path) then
  begin
    Result := False;
    Exit;
  end;
  Result := FindFirst(Path + '\8.0.*', FindRec);
  if Result then
    FindClose(FindRec);
end;

function NeedInstallDotNet8Desktop(): Boolean;
begin
  Result := (not DotNet8DesktopInstalled()) and
    FileExists(ExpandConstant('{tmp}\windowsdesktop-runtime-8.0-win-x64.exe'));
end;

function WinAppRuntimeInstalled(): Boolean;
var
  Version: String;
  FindRec: TFindRec;
begin
  if RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\WindowsAppRuntime\Installed', 'Version', Version) then
  begin
    Result := True;
    Exit;
  end;
  if RegKeyExists(HKLM, 'SOFTWARE\Microsoft\WindowsAppRuntime\Installed') then
  begin
    Result := True;
    Exit;
  end;
  if RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\WindowsAppRuntime\Installed') then
  begin
    Result := True;
    Exit;
  end;
  { AppX packages often omit Installed registry key }
  if FindFirst(ExpandConstant('{commonpf64}\WindowsApps\Microsoft.WindowsAppRuntime.*'), FindRec) then
  begin
    FindClose(FindRec);
    Result := True;
    Exit;
  end;
  if FindFirst(ExpandConstant('{commonpf64}\WindowsApps\MicrosoftCorporationII.WinAppRuntime.Main.*'), FindRec) then
  begin
    FindClose(FindRec);
    Result := True;
    Exit;
  end;
  Result := False;
end;

function NeedInstallWinAppRuntime(): Boolean;
begin
  Result := (not WinAppRuntimeInstalled()) and
    FileExists(ExpandConstant('{tmp}\WindowsAppRuntimeInstall-x64.exe'));
end;

function VcRedistInstalled(): Boolean;
begin
  Result := RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64') or
    RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
end;

function NeedInstallVcRedist(): Boolean;
begin
  Result := (not VcRedistInstalled()) and
    FileExists(ExpandConstant('{tmp}\VC_redist.x64.exe'));
end;

procedure KillZhuLongProcesses();
var
  ResultCode: Integer;
begin
  { 仅结束烛龙本体，不用 /T，避免误伤其它进程 }
  Exec('taskkill.exe', '/F /IM ZhuLong.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(1200);
end;

function InitializeSetup(): Boolean;
begin
  KillZhuLongProcesses();
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  KillZhuLongProcesses();
  Result := '';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { 修复 runtimeconfig（D 盘安装 / 旧版覆盖时可能缺少 WindowsDesktop.App） }
    if FileExists(ExpandConstant('{app}\scripts\fix_runtimeconfig.ps1')) then
      Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\scripts\fix_runtimeconfig.ps1') + '" -StageDir "' + ExpandConstant('{app}') + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    { 通知 Shell 刷新图标缓存，避免快捷方式仍显示旧版淮河 logo }
    Exec('cmd.exe', '/c ie4uinit.exe -show', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
