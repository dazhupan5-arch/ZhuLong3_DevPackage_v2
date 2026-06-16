using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ZhuLong.Core;

namespace ZhuLong.Core.Licensing;

/// <summary>离线授权码（与烛龙 V13/V14 发码工具互通）。</summary>
public static class OfflineLicenseCodec
{
    public sealed class PayloadDto
    {
        [JsonPropertyName("v")]
        public int V { get; set; } = 1;

        [JsonPropertyName("exp")]
        public long Exp { get; set; }

        [JsonPropertyName("fph")]
        public string Fph { get; set; } = "*";
    }

    /// <summary>生成授权码（与 V14 发码工具互通）。</summary>
    public static string Issue(string hmacKeyUtf8, DateTimeOffset validUntilUtc, string fingerprintPrefix8OrStar)
    {
        if (string.IsNullOrEmpty(hmacKeyUtf8))
            throw new ArgumentException("HMAC 密钥不能为空。", nameof(hmacKeyUtf8));

        var dto = new PayloadDto
        {
            V = 1,
            Exp = validUntilUtc.ToUnixTimeSeconds(),
            Fph = NormalizeFph(fingerprintPrefix8OrStar),
        };
        var json = JsonSerializer.Serialize(dto, OfflineLicensePayloadJsonContext.Default.PayloadDto);
        var body = Encoding.UTF8.GetBytes(json);
        var sig = Sign(hmacKeyUtf8, body);
        return $"{Base64UrlEncode(body)}.{Base64UrlEncode(sig)}";
    }

    public static bool TryValidate(
        string token,
        string hmacKeyUtf8,
        string hardwareFingerprintHex64Lower,
        out DateTimeOffset validUntilUtc,
        out string? error)
    {
        validUntilUtc = default;
        error = null;
        if (string.IsNullOrWhiteSpace(token))
        {
            error = "授权码为空。";
            return false;
        }

        if (string.IsNullOrEmpty(hmacKeyUtf8))
        {
            error = "未配置授权密钥。";
            return false;
        }

        var dot = token.IndexOf('.', StringComparison.Ordinal);
        if (dot <= 0 || dot >= token.Length - 1)
        {
            error = "授权码格式无效。";
            return false;
        }

        byte[] body;
        byte[] sig;
        try
        {
            body = Base64UrlDecode(token[..dot].Trim());
            sig = Base64UrlDecode(token[(dot + 1)..].Trim());
        }
        catch
        {
            error = "授权码 Base64 解析失败。";
            return false;
        }

        var expected = Sign(hmacKeyUtf8, body);
        if (expected.Length != sig.Length || !CryptographicOperations.FixedTimeEquals(expected, sig))
        {
            error = "授权码签名无效或密钥不匹配。";
            return false;
        }

        PayloadDto? dto;
        try
        {
            dto = JsonSerializer.Deserialize(
                Encoding.UTF8.GetString(body),
                OfflineLicensePayloadJsonContext.Default.PayloadDto);
        }
        catch
        {
            error = "授权码载荷解析失败。";
            return false;
        }

        if (dto is null || dto.V != 1 || dto.Exp <= 0)
        {
            error = "授权码载荷版本或字段无效。";
            return false;
        }

        var fph = NormalizeFph(dto.Fph);
        if (fph != "*")
        {
            var prefix = FingerprintPrefix8(hardwareFingerprintHex64Lower);
            if (!string.Equals(fph, prefix, StringComparison.OrdinalIgnoreCase))
            {
                error = "授权码与当前设备不匹配（请使用设置页「设备标识」申请授权）。";
                return false;
            }
        }

        validUntilUtc = DateTimeOffset.FromUnixTimeSeconds(dto.Exp);
        if (validUntilUtc < DateTimeOffset.UtcNow.AddMinutes(-5))
        {
            error = "授权码已过期（到期北京时间：" + ChinaTime.Format(validUntilUtc, "yyyy-MM-dd HH:mm") + "）。";
            return false;
        }

        return true;
    }

    public static string FingerprintPrefix8(string hardwareFingerprintHex64Lower)
    {
        var s = (hardwareFingerprintHex64Lower ?? "").Trim().ToLowerInvariant();
        return s.Length < 8 ? s.PadRight(8, '0') : s[..8];
    }

    private static string NormalizeFph(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return "*";
        var t = raw.Trim();
        if (t == "*") return "*";
        t = t.ToLowerInvariant();
        return t.Length > 8 ? t[..8] : t;
    }

    private static byte[] Sign(string keyUtf8, byte[] body)
    {
        using var h = new HMACSHA256(Encoding.UTF8.GetBytes(keyUtf8));
        return h.ComputeHash(body);
    }

    private static string Base64UrlEncode(byte[] data) =>
        Convert.ToBase64String(data).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    private static byte[] Base64UrlDecode(string s)
    {
        var t = s.Replace('-', '+').Replace('_', '/');
        switch (t.Length % 4)
        {
            case 2: t += "=="; break;
            case 3: t += "="; break;
        }
        return Convert.FromBase64String(t);
    }
}
