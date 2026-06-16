using System.Text.Json.Serialization;

namespace ZhuLong.App.Services.Membership;

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(LicenseStateStore.StateDto))]
internal partial class LicenseStateJsonSerializationContext : JsonSerializerContext;
