namespace ZhuLong.App.Services.Membership;

public interface IMembershipService
{
    bool CanUseApp { get; }
    bool IsLicensed { get; }
    bool IsTrialActive { get; }
    TimeSpan? TrialRemaining { get; }
    string TierDisplayName { get; }
    string StatusSummary { get; }
    string HardwareIdShort { get; }
    string DiagnosticClipboardText { get; }
    bool TryApplyActivationCode(string code, out string? error);
    void Refresh();
}
