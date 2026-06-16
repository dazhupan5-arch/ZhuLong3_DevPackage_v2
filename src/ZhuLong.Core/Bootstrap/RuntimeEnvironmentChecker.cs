using Microsoft.Win32;



namespace ZhuLong.Core.Bootstrap;



/// <summary>检测 .NET 8 Desktop / WinUI 运行库是否就绪（与 check_runtime.ps1 对齐）。</summary>

public static class RuntimeEnvironmentChecker

{

    public const string RequiredDotNetMajor = "8.0";



    public sealed record CheckItem(string Name, bool Ok, string Detail);



    public sealed record Report(bool Ready, IReadOnlyList<CheckItem> Items)

    {

        public string Summary => Ready

            ? "运行环境就绪"

            : "运行环境不完整：" + string.Join("；", Items.Where(i => !i.Ok).Select(i => i.Name));

    }



    public static Report Evaluate()

    {

        var items = new List<CheckItem>

        {

            CheckDotNetDesktop(),

            CheckWinAppRuntime(),

            CheckVcRedist(),

        };

        return new Report(items.All(i => i.Ok), items);

    }



    public static string BuildReport()

    {

        var r = Evaluate();

        var lines = new List<string> { "[RuntimeEnvironment] " + r.Summary };

        foreach (var i in r.Items)

            lines.Add($"  {(i.Ok ? "OK" : "MISS")} {i.Name}: {i.Detail}");

        return string.Join(Environment.NewLine, lines);

    }



    public static bool IsDotNet8DesktopInstalled() => CheckDotNetDesktop().Ok;



    public static bool IsWinAppRuntimeInstalled() => CheckWinAppRuntime().Ok;



    private static CheckItem CheckDotNetDesktop()

    {

        var info = DotNetRuntimeDiscovery.ProbeDesktopRuntime();

        if (info.IsReady)

        {

            return new CheckItem(".NET 8 Desktop Runtime", true,

                $"根目录 {info.InstallRoot}；版本 {string.Join(", ", info.DesktopVersions)}");

        }



        foreach (var root in EnumerateSharedRoots())

        {

            var core = FindSharedFrameworkVersions(root, "Microsoft.NETCore.App", RequiredDotNetMajor);

            if (core.Count > 0)

            {

                return new CheckItem(".NET 8 Desktop Runtime", false,

                    $"已检测到 .NET Core ({string.Join(", ", core)}) @ {root}，但缺少 WindowsDesktop 桌面框架。"

                    + " 请安装「.NET 8 Desktop Runtime」。");

            }

        }



        return new CheckItem(".NET 8 Desktop Runtime", false,

            "未检测到 .NET 8 Desktop。请运行安装目录 redist\\windowsdesktop-runtime-8.0-win-x64.exe");

    }



    private static CheckItem CheckWinAppRuntime()

    {

        if (RegKeyExists(Registry.LocalMachine, @"SOFTWARE\Microsoft\WindowsAppRuntime\Installed")

            || RegKeyExists(Registry.LocalMachine, @"SOFTWARE\WOW6432Node\Microsoft\WindowsAppRuntime\Installed"))

        {

            return new CheckItem("Windows App Runtime (WinUI 3)", true, "注册表已注册");

        }



        var windowsApps = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "WindowsApps");

        if (Directory.Exists(windowsApps))

        {

            foreach (var pattern in new[] { "Microsoft.WindowsAppRuntime.*", "MicrosoftCorporationII.WinAppRuntime.Main.*" })

            {

                try

                {

                    var hit = Directory.GetDirectories(windowsApps, pattern).FirstOrDefault();

                    if (hit is not null)

                        return new CheckItem("Windows App Runtime (WinUI 3)", true, Path.GetFileName(hit));

                }

                catch

                {

                    /* ignore */

                }

            }

        }



        return new CheckItem("Windows App Runtime (WinUI 3)", false,

            "未检测到 WinUI 3 运行库。请运行 redist\\WindowsAppRuntimeInstall-x64.exe");

    }



    private static CheckItem CheckVcRedist()

    {

        var ok = RegKeyExists(Registry.LocalMachine, @"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64")

                 || RegKeyExists(Registry.LocalMachine, @"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64");

        return new CheckItem("Visual C++ 2015-2022 x64", ok,

            ok ? "已安装" : "建议安装 redist\\VC_redist.x64.exe");

    }



    private static List<string> FindSharedFrameworkVersions(string sharedRoot, string framework, string majorPrefix)

    {

        var list = new List<string>();

        var dir = Path.Combine(sharedRoot, framework);

        if (!Directory.Exists(dir)) return list;

        foreach (var sub in Directory.GetDirectories(dir))

        {

            var name = Path.GetFileName(sub);

            if (name.StartsWith(majorPrefix + ".", StringComparison.Ordinal))

                list.Add(name);

        }

        return list;

    }



    private static IEnumerable<string> EnumerateSharedRoots()

    {

        var pf = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);

        yield return Path.Combine(pf, "dotnet", "shared");

        var pf86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

        var alt = Path.Combine(pf86, "dotnet", "shared");

        if (!string.Equals(alt, Path.Combine(pf, "dotnet", "shared"), StringComparison.OrdinalIgnoreCase))

            yield return alt;

    }



    private static bool RegKeyExists(RegistryKey hive, string subKey)

    {

        try

        {

            using var k = hive.OpenSubKey(subKey);

            return k is not null;

        }

        catch

        {

            return false;

        }

    }

}


