using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using ZhuLong.Core;
using ZhuLong.Core.Licensing;

namespace ZhuLong.App.Services.Membership;

public sealed class MembershipService : IMembershipService
{
    public const int TrialDays = 7;

    private string _fingerprintHex64 = "";
    private LicenseStateStore.StateDto _state = new();
    private DateTimeOffset? _licenseValidUntil;
    private string? _licenseError;

    public bool CanUseApp => IsLicensed || IsTrialActive;

    public bool IsLicensed =>
        _licenseValidUntil is { } u && u.UtcDateTime > DateTime.UtcNow.AddMinutes(-5);

    public bool IsTrialActive
    {
        get
        {
            if (IsLicensed) return false;
            if (!TryParseTrialStart(out var start)) return false;
            return DateTime.UtcNow < start.AddDays(TrialDays);
        }
    }

    public TimeSpan? TrialRemaining
    {
        get
        {
            if (!IsTrialActive || !TryParseTrialStart(out var start)) return null;
            var left = start.AddDays(TrialDays) - DateTime.UtcNow;
            return left < TimeSpan.Zero ? TimeSpan.Zero : left;
        }
    }

    public string TierDisplayName
    {
        get
        {
            if (IsLicensed && _licenseValidUntil is { } u)
                return "已授权（至 " + ChinaTime.Format(u, "yyyy-MM-dd") + " 北京时间）";
            if (IsTrialActive && TrialRemaining is { } tr)
                return $"试用中（剩余 {Math.Max(0, tr.Days)} 天 {tr.Hours} 小时）";
            return "试用已结束";
        }
    }

    public string StatusSummary => BuildStatusSummary();

    public string HardwareIdShort =>
        string.IsNullOrEmpty(HardwareFingerprintHex) || HardwareFingerprintHex == "—"
            ? "—"
            : OfflineLicenseCodec.FingerprintPrefix8(HardwareFingerprintHex);

    private string HardwareFingerprintHex =>
        string.IsNullOrWhiteSpace(_state.HardwareFingerprintHex)
            ? (string.IsNullOrEmpty(_fingerprintHex64) ? "—" : _fingerprintHex64)
            : _state.HardwareFingerprintHex.Trim().ToLowerInvariant();

    public string DiagnosticClipboardText
    {
        get
        {
            var fp = string.IsNullOrEmpty(_fingerprintHex64)
                ? HardwareFingerprintHex
                : _fingerprintHex64;
            if (string.IsNullOrWhiteSpace(fp) || fp == "—")
                fp = MachineHardwareAnchor.ComputeFingerprintHex64();

            var sb = new StringBuilder();
            sb.AppendLine(AppMetadata.FormatVersionLine());
            sb.AppendLine("操作系统：" + RuntimeInformation.OSDescription);
            sb.AppendLine("CPU 型号：" + (MachineHardwareAnchor.TryGetCpuModelName() ?? "—"));
            sb.AppendLine("设备标识（前 8 位）：" + HardwareIdShort);
            sb.AppendLine("完整指纹：" + fp);
            return sb.ToString().TrimEnd();
        }
    }

    public void Refresh()
    {
        _fingerprintHex64 = MachineHardwareAnchor.ComputeFingerprintHex64();
        _state = LicenseStateStore.LoadForCurrentFingerprint(_fingerprintHex64);
        _licenseValidUntil = null;
        _licenseError = null;

        var tok = _state.LicenseToken?.Trim();
        if (string.IsNullOrEmpty(tok)) return;

        var fpForLicense = string.IsNullOrWhiteSpace(_state.HardwareFingerprintHex)
            ? _fingerprintHex64
            : _state.HardwareFingerprintHex.Trim().ToLowerInvariant();

        if (OfflineLicenseCodec.TryValidate(tok, LicenseClientSecrets.OfflineHmacKeyUtf8, fpForLicense, out var until, out var err))
            _licenseValidUntil = until;
        else
            _licenseError = err;
    }

    public bool TryApplyActivationCode(string code, out string? error)
    {
        error = null;
        _fingerprintHex64 = MachineHardwareAnchor.ComputeFingerprintHex64();
        var trimmed = (code ?? "").Trim().Replace("\r", "").Replace("\n", "").Replace(" ", "");
        if (string.IsNullOrEmpty(trimmed))
        {
            error = "请输入授权码。";
            return false;
        }

        if (!OfflineLicenseCodec.TryValidate(trimmed, LicenseClientSecrets.OfflineHmacKeyUtf8, _fingerprintHex64, out var until, out var err))
        {
            error = err;
            return false;
        }

        _state.LicenseToken = trimmed;
        _state.HardwareFingerprintHex = _fingerprintHex64.Trim().ToLowerInvariant();
        LicenseStateStore.Save(_state);
        _licenseValidUntil = until;
        _licenseError = null;
        return true;
    }

    private bool TryParseTrialStart(out DateTime startUtc)
    {
        startUtc = default;
        if (string.IsNullOrWhiteSpace(_state.TrialStartedUtc)) return false;
        if (!DateTime.TryParse(_state.TrialStartedUtc, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var d))
            return false;
        startUtc = d.Kind == DateTimeKind.Utc ? d : d.ToUniversalTime();
        return true;
    }

    private string BuildStatusSummary()
    {
        var sb = new StringBuilder();
        if (IsLicensed && _licenseValidUntil is { } u)
            sb.AppendLine("当前：已授权，到期（北京时间）：" + ChinaTime.Format(u, "yyyy-MM-dd HH:mm"));
        else if (IsTrialActive && TrialRemaining is { } tr)
            sb.AppendLine($"当前：试用中，剩余 {tr.Days} 天 {tr.Hours} 小时。");
        else
            sb.AppendLine("当前：试用已结束。请粘贴授权码后点击「激活」。");

        if (!string.IsNullOrEmpty(_licenseError) && !string.IsNullOrEmpty(_state.LicenseToken))
            sb.AppendLine("授权码校验：" + _licenseError);

        sb.AppendLine("状态：" + TierDisplayName);
        return sb.ToString().TrimEnd();
    }
}
