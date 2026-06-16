using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ZhuLong.Core.Licensing;

namespace ZhuLong.App.Services.Membership;

/// <summary>与 V13/V14 共用 membership_state.json 路径，授权互通。</summary>
internal static class LicenseStateStore
{
    public static string PrimaryPath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
            "烛龙智核AI",
            "FileBridge",
            "membership_state.json");

    public static string MirrorPath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "烛龙智核AI",
            "FileBridge",
            "membership_state.json");

    public sealed class StateDto
    {
        [JsonPropertyName("schema")]
        public int Schema { get; set; } = 1;

        [JsonPropertyName("hardware_fingerprint_hex")]
        public string HardwareFingerprintHex { get; set; } = "";

        [JsonPropertyName("trial_started_utc")]
        public string TrialStartedUtc { get; set; } = "";

        [JsonPropertyName("license_token")]
        public string? LicenseToken { get; set; }
    }

    public static StateDto LoadForCurrentFingerprint(string currentFingerprintHex64Lower)
    {
        var cur = NormFp(currentFingerprintHex64Lower);
        var p = TryReadFile(PrimaryPath);
        var m = TryReadFile(MirrorPath);
        var trialIso = MergeTrialStartedIso(p, m);
        var (token, storageFp) = ResolveBestTokenAndStorageFingerprint(cur, p, m);
        var merged = new StateDto
        {
            Schema = 1,
            HardwareFingerprintHex = storageFp,
            TrialStartedUtc = trialIso,
            LicenseToken = token,
        };

        if (!StateMatchesPrimaryOnDisk(p, merged))
            Save(merged);

        return merged;
    }

    public static void Save(StateDto state)
    {
        try { WriteOne(PrimaryPath, state); } catch { /* ignore */ }
        try { WriteOne(MirrorPath, state); } catch { /* ignore */ }
    }

    private static string MergeTrialStartedIso(StateDto? p, StateDto? m)
    {
        if (p is null && m is null)
            return DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        if (p is null) return PickNonEmptyTrialOrNow(m!);
        if (m is null) return PickNonEmptyTrialOrNow(p);
        var ta = ParseIso(p.TrialStartedUtc);
        var tb = ParseIso(m.TrialStartedUtc);
        return ta <= tb ? p.TrialStartedUtc : m.TrialStartedUtc;
    }

    private static string PickNonEmptyTrialOrNow(StateDto d) =>
        string.IsNullOrWhiteSpace(d.TrialStartedUtc)
            ? DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)
            : d.TrialStartedUtc;

    private static (string? Token, string StorageFp) ResolveBestTokenAndStorageFingerprint(
        string currentFp, StateDto? p, StateDto? m)
    {
        var key = LicenseClientSecrets.OfflineHmacKeyUtf8;
        var fps = new List<string> { currentFp };
        AddDistinctFp(fps, p?.HardwareFingerprintHex);
        AddDistinctFp(fps, m?.HardwareFingerprintHex);

        var tokens = new List<string>();
        TryAddToken(tokens, p?.LicenseToken);
        TryAddToken(tokens, m?.LicenseToken);
        if (tokens.Count == 0) return (null, currentFp);

        string? bestTok = null;
        var bestExp = DateTimeOffset.MinValue;
        var storageFp = currentFp;

        foreach (var tok in tokens)
        {
            foreach (var fp in fps)
            {
                if (OfflineLicenseCodec.TryValidate(tok, key, fp, out var exp, out _))
                {
                    if (exp > bestExp)
                    {
                        bestExp = exp;
                        bestTok = tok;
                        storageFp = fp;
                    }
                }
            }
        }

        if (bestTok is null && tokens.Count > 0)
        {
            bestTok = tokens[0];
            storageFp = NormFp(p?.HardwareFingerprintHex) is { Length: > 0 } pf ? pf
                : NormFp(m?.HardwareFingerprintHex) is { Length: > 0 } mf ? mf
                : currentFp;
        }

        return (bestTok, storageFp);
    }

    private static void AddDistinctFp(List<string> list, string? fp)
    {
        var n = NormFp(fp);
        if (string.IsNullOrEmpty(n)) return;
        if (list.Any(e => string.Equals(e, n, StringComparison.Ordinal))) return;
        list.Add(n);
    }

    private static void TryAddToken(List<string> list, string? tok)
    {
        var t = string.IsNullOrWhiteSpace(tok) ? null : tok.Trim();
        if (t is null) return;
        if (list.Any(e => string.Equals(e, t, StringComparison.Ordinal))) return;
        list.Add(t);
    }

    private static bool StateMatchesPrimaryOnDisk(StateDto? primary, StateDto merged) =>
        primary is not null &&
        string.Equals(NormFp(primary.HardwareFingerprintHex), NormFp(merged.HardwareFingerprintHex), StringComparison.Ordinal) &&
        string.Equals(primary.LicenseToken ?? "", merged.LicenseToken ?? "", StringComparison.Ordinal) &&
        string.Equals(primary.TrialStartedUtc ?? "", merged.TrialStartedUtc ?? "", StringComparison.Ordinal);

    private static DateTime ParseIso(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return DateTime.MaxValue;
        return DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var d)
            ? d.ToUniversalTime()
            : DateTime.MaxValue;
    }

    private static string NormFp(string? s) => (s ?? "").Trim().ToLowerInvariant();

    private static StateDto? TryReadFile(string path)
    {
        try
        {
            if (!File.Exists(path)) return null;
            return JsonSerializer.Deserialize(
                File.ReadAllText(path),
                LicenseStateJsonSerializationContext.Default.StateDto);
        }
        catch { return null; }
    }

    private static void WriteOne(string path, StateDto state)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(state, LicenseStateJsonSerializationContext.Default.StateDto));
    }
}
