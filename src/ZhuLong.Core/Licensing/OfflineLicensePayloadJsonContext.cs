using System.Text.Json.Serialization;

namespace ZhuLong.Core.Licensing;

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(OfflineLicenseCodec.PayloadDto))]
internal partial class OfflineLicensePayloadJsonContext : JsonSerializerContext;
