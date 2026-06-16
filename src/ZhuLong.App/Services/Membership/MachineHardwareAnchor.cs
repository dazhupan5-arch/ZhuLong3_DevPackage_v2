using System.Globalization;
using System.Management;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Win32;

namespace ZhuLong.App.Services.Membership;

internal static class MachineHardwareAnchor
{
    private static readonly object Gate = new();
    private static string? _cached;

    public static string ComputeFingerprintHex64()
    {
        lock (Gate)
        {
            if (!string.IsNullOrEmpty(_cached)) return _cached;

            List<string>? best = null;
            var bestCount = -1;
            for (var i = 0; i < 3; i++)
            {
                var parts = CollectOnce();
                if (parts.Count > bestCount)
                {
                    bestCount = parts.Count;
                    best = parts;
                }
            }

            best ??= ["fallback:" + Environment.MachineName];
            best.Sort(StringComparer.Ordinal);
            _cached = Sha256Hex(string.Join("|", best));
            return _cached;
        }
    }

    internal static string? TryGetCpuModelName()
    {
        try
        {
            using var searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_Processor");
            using var coll = searcher.Get();
            foreach (ManagementObject o in coll)
            {
                try
                {
                    var n = o["Name"]?.ToString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(n)) return n;
                }
                finally { o.Dispose(); }
            }
        }
        catch { /* ignore */ }
        return null;
    }

    private static List<string> CollectOnce()
    {
        var parts = new List<string>();
        var mg = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Cryptography")?.GetValue("MachineGuid") as string;
        if (!string.IsNullOrWhiteSpace(mg)) parts.Add("mg:" + mg.Trim().ToLowerInvariant());
        AppendWmi(parts, "Win32_BIOS", "SerialNumber", "bios");
        AppendWmi(parts, "Win32_BaseBoard", "SerialNumber", "board");
        AppendWmi(parts, "Win32_DiskDrive", "SerialNumber", "disk0", firstOnly: true);
        if (parts.Count == 0) parts.Add("fallback:" + Environment.MachineName);
        return parts;
    }

    private static void AppendWmi(List<string> parts, string cls, string prop, string tag, bool firstOnly = false)
    {
        try
        {
            using var searcher = new ManagementObjectSearcher($"SELECT {prop} FROM {cls}");
            using var coll = searcher.Get();
            var n = 0;
            foreach (ManagementObject o in coll)
            {
                try
                {
                    var v = o[prop]?.ToString()?.Trim();
                    if (string.IsNullOrWhiteSpace(v) ||
                        v.Equals("To be filled by O.E.M.", StringComparison.OrdinalIgnoreCase) ||
                        v.Equals("Default string", StringComparison.OrdinalIgnoreCase))
                        continue;
                    parts.Add($"{tag}:{v.ToLowerInvariant()}");
                    if (++n >= 1 && firstOnly) break;
                }
                finally { o.Dispose(); }
            }
        }
        catch { /* ignore */ }
    }

    private static string Sha256Hex(string s)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(s));
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash) sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}
