namespace ZhuLong.Core.Configuration;

/// <summary>宏观/LLM API 密钥解析（环境变量 &gt; LocalSettings &gt; 磁盘）。</summary>
public interface IApiSecrets
{
    string? ResolveFinnhubApiKey();
    string? ResolveFmpApiKey();
    string? ResolveFredApiKey();
    string? ResolveLlmApiKey();
}
