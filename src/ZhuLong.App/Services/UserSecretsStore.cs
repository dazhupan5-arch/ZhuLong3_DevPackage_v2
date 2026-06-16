using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Windows.Storage;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;

namespace ZhuLong.App.Services;

/// <summary>
/// 用户 API 密钥：环境变量 &gt; WinUI LocalSettings &gt; %APPDATA%\ZhuLong\secrets\*.txt（与烛龙 V14 一致）。
/// </summary>
public sealed class UserSecretsStore : IApiSecrets
{
    private const string KeyFred = "FredApiKey_v3";
    private const string KeyFinnhub = "FinnhubApiKey_v3";
    private const string KeyFmp = "FmpApiKey_v3";
    private const string KeyLlm = "LlmApiKey_v3";

    public string? ResolveFinnhubApiKey() =>
        Env("FINNHUB_API_KEY") ?? Env("FINNHUB_TOKEN") ?? Get(KeyFinnhub, "finnhub_api_key.txt");

    public string? ResolveFmpApiKey() =>
        Env("FMP_API_KEY") ?? Get(KeyFmp, "fmp_api_key.txt");

    public string? ResolveFredApiKey() =>
        Env("FRED_API_KEY") ?? Get(KeyFred, "fred_api_key.txt");

    public string? ResolveLlmApiKey() =>
        Env("LLM_API_KEY") ?? Env("GEMINI_API_KEY") ?? Env("API2D_API_KEY") ?? Get(KeyLlm, "llm_api_key.txt");

    public void SetFinnhubApiKey(string? value) => Set(KeyFinnhub, "finnhub_api_key.txt", value);
    public void SetFmpApiKey(string? value) => Set(KeyFmp, "fmp_api_key.txt", value);
    public void SetFredApiKey(string? value) => Set(KeyFred, "fred_api_key.txt", value);
    public void SetLlmApiKey(string? value) => Set(KeyLlm, "llm_api_key.txt", value);

    public bool HasAnyMacroKey =>
        !string.IsNullOrEmpty(ResolveFredApiKey()) ||
        !string.IsNullOrEmpty(ResolveFinnhubApiKey()) ||
        !string.IsNullOrEmpty(ResolveLlmApiKey());

    public string? ResolveAlertWebhookUrl() =>
        Env("ALERT_WEBHOOK_URL") ?? ReadDisk("alert_webhook.txt");

    private static string? Env(string name)
    {
        var v = Environment.GetEnvironmentVariable(name)?.Trim();
        return string.IsNullOrEmpty(v) ? null : v;
    }

    private static string? Get(string localKey, string fileName)
    {
        var ls = TryLocal(localKey);
        if (!string.IsNullOrEmpty(ls)) return ls;
        return ReadDisk(fileName);
    }

    private static void Set(string localKey, string fileName, string? value)
    {
        var t = string.IsNullOrWhiteSpace(value) ? null : value.Trim();
        WriteLocal(localKey, t);
        WriteDisk(fileName, t);
    }

    private static string? TryLocal(string key)
    {
        try
        {
            if (ApplicationData.Current.LocalSettings.Values.TryGetValue(key, out var o) &&
                o is string s && !string.IsNullOrWhiteSpace(s))
                return s.Trim();
        }
        catch { /* ignore */ }
        return null;
    }

    private static void WriteLocal(string key, string? trimmed)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(trimmed))
                ApplicationData.Current.LocalSettings.Values.Remove(key);
            else
                ApplicationData.Current.LocalSettings.Values[key] = trimmed;
        }
        catch { /* ignore */ }
    }

    private static string? ReadDisk(string fileName)
    {
        try
        {
            var path = Path.Combine(AppPaths.SecretsDir, fileName);
            if (!File.Exists(path)) return null;
            var line = File.ReadAllText(path).Trim();
            return string.IsNullOrEmpty(line) ? null : line;
        }
        catch { return null; }
    }

    private static void WriteDisk(string fileName, string? trimmed)
    {
        try
        {
            var path = Path.Combine(AppPaths.SecretsDir, fileName);
            if (string.IsNullOrWhiteSpace(trimmed))
            {
                if (File.Exists(path)) File.Delete(path);
            }
            else
            {
                File.WriteAllText(path, trimmed);
            }
        }
        catch { /* ignore */ }
    }
}
