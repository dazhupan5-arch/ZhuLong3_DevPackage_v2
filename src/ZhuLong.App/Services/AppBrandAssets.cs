using System.Drawing;
using System.Runtime.InteropServices;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Imaging;

namespace ZhuLong.App.Services;

/// <summary>烛龙品牌资源：矢量 SVG 导出 PNG/ICO，全应用统一加载。</summary>
internal static class AppBrandAssets
{
    private const int WmSetIcon = 0x0080;
    private const int IconSmall = 0;
    private const int IconBig = 1;

    internal const string TitleLogoRelative = "Assets/TitleLogo.png";
    internal const string StoreLogoRelative = "Assets/StoreLogo.png";
    internal const string WindowIconRelative = "Assets/zhulong.ico";
    internal const string AppIconRelative = "Assets/app.ico";

    private static Icon? _trayIcon;
    private static Icon? _windowIconSmall;
    private static Icon? _windowIconLarge;

    internal static string? ResolvePath(string relativePath)
    {
        foreach (var dir in EnumerateBaseDirectories())
        {
            var path = Path.GetFullPath(Path.Combine(dir, relativePath));
            if (File.Exists(path))
                return path;
        }

        return null;
    }

    internal static BitmapImage? LoadBitmap(string relativePath, int? decodePixelWidth = null)
    {
        var path = ResolvePath(relativePath);
        if (path is null)
            return null;

        var image = new BitmapImage { DecodePixelType = DecodePixelType.Logical };
        if (decodePixelWidth is > 0)
            image.DecodePixelWidth = decodePixelWidth.Value;
        image.UriSource = new Uri(path);
        return image;
    }

    internal static void ApplyTitleImages(params Microsoft.UI.Xaml.Controls.Image[] images)
    {
        foreach (var image in images)
        {
            var bitmap = LoadBitmap(TitleLogoRelative, 80) ?? LoadBitmap(StoreLogoRelative, 80);
            if (bitmap is not null)
                image.Source = bitmap;
        }
    }

    internal static void ApplyWindowBranding(Microsoft.UI.Windowing.AppWindow appWindow, nint hwnd)
    {
        var iconPath = ResolvePath(WindowIconRelative) ?? ResolvePath(AppIconRelative);
        if (string.IsNullOrEmpty(iconPath))
            return;

        try
        {
            appWindow.SetIcon(iconPath);
        }
        catch
        {
            /* ignore */
        }

        if (hwnd == IntPtr.Zero)
            return;

        try
        {
            ReleaseWindowIcons();
            _windowIconSmall = LoadIconLayer(iconPath, 16) ?? LoadIconLayer(iconPath, 20);
            _windowIconLarge = LoadIconLayer(iconPath, 32) ?? LoadIconLayer(iconPath, 48);
            if (_windowIconSmall is not null)
                _ = SendMessage(hwnd, WmSetIcon, (IntPtr)IconSmall, _windowIconSmall.Handle);
            if (_windowIconLarge is not null)
                _ = SendMessage(hwnd, WmSetIcon, (IntPtr)IconBig, _windowIconLarge.Handle);
        }
        catch
        {
            /* ignore */
        }
    }

    internal static Icon? LoadTrayIcon()
    {
        if (_trayIcon is not null)
            return (Icon)_trayIcon.Clone();

        var iconPath = ResolvePath(WindowIconRelative) ?? ResolvePath(AppIconRelative);
        if (string.IsNullOrEmpty(iconPath))
            return null;

        try
        {
            _trayIcon = LoadIconLayer(iconPath, 32)
                ?? LoadIconLayer(iconPath, 24)
                ?? LoadIconLayer(iconPath, 16);
            return _trayIcon is null ? null : (Icon)_trayIcon.Clone();
        }
        catch
        {
            return null;
        }
    }

    internal static IEnumerable<string> EnumerateBaseDirectories()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var results = new List<string>();
        foreach (var candidate in new[]
                 {
                     AppContext.BaseDirectory,
                     AppDomain.CurrentDomain.BaseDirectory,
                     Environment.ProcessPath is { Length: > 0 } ep ? Path.GetDirectoryName(ep) : null,
                     Directory.GetCurrentDirectory(),
                 })
        {
            if (string.IsNullOrWhiteSpace(candidate))
                continue;
            try
            {
                var full = Path.GetFullPath(candidate);
                if (Directory.Exists(full) && seen.Add(full))
                    results.Add(full);
            }
            catch
            {
                /* skip */
            }
        }

        return results;
    }

    private static Icon? LoadIconLayer(string iconPath, int size)
    {
        try
        {
            return new Icon(iconPath, size, size);
        }
        catch
        {
            return null;
        }
    }

    private static void ReleaseWindowIcons()
    {
        _windowIconSmall?.Dispose();
        _windowIconLarge?.Dispose();
        _windowIconSmall = null;
        _windowIconLarge = null;
    }

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    private static extern IntPtr SendMessage(nint hWnd, int msg, IntPtr wParam, IntPtr lParam);
}
