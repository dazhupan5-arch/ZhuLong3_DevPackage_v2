using System.Text.Json;
using Json.Schema;
using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>启动时 JSON Schema 校验 config.json（Phase 0）。</summary>
public static class ConfigSchemaValidator
{
    public static string? ResolveSchemaPath()
    {
        var candidates = new[]
        {
            Path.Combine(AppPaths.InstallDir, "config", "config.schema.json"),
            Path.Combine(AppPaths.FindDevRoot() ?? "", "config", "config.schema.json"),
            Path.Combine(AppPaths.InstallDir, "..", "..", "..", "..", "..", "config", "config.schema.json"),
        };

        foreach (var p in candidates)
        {
            var full = Path.GetFullPath(p);
            if (File.Exists(full)) return full;
        }

        return null;
    }

    public static IReadOnlyList<string> ValidateFile(string configPath, string? schemaPath = null)
    {
        if (!File.Exists(configPath))
            return [$"配置文件不存在: {configPath}"];

        schemaPath ??= ResolveSchemaPath();
        if (schemaPath is null || !File.Exists(schemaPath))
            return ConfigValidator.Validate(AppSettings.Load(configPath));

        var text = File.ReadAllText(configPath);
        if (text.Length > 0 && text[0] == '\uFEFF')
            text = text[1..];

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(text);
        }
        catch (JsonException ex)
        {
            return [$"config.json 非合法 JSON: {ex.Message}"];
        }

        var schema = JsonSchema.FromFile(schemaPath);
        var result = schema.Evaluate(doc.RootElement, new EvaluationOptions
        {
            OutputFormat = OutputFormat.List,
        });

        if (result.IsValid)
            return ConfigValidator.Validate(AppSettings.Load(configPath));

        var errors = new List<string>();
        CollectErrors(result, errors);
        return errors;
    }

    private static void CollectErrors(EvaluationResults result, List<string> errors)
    {
        if (result.Errors is not null)
        {
            foreach (var kv in result.Errors)
                errors.Add($"{result.InstanceLocation}: {kv.Value}");
        }

        if (result.Details is null) return;
        foreach (var child in result.Details)
            CollectErrors(child, errors);
    }
}
