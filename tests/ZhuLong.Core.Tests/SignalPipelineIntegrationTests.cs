using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Data;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

/// <summary>L1-4：Mock 推理 → 信号生成 → SQLite 持久化。</summary>
public sealed class SignalPipelineIntegrationTests : IDisposable
{
    private readonly string _dbPath;
    private readonly DatabaseService _db;

    public SignalPipelineIntegrationTests()
    {
        _dbPath = Path.Combine(Path.GetTempPath(), $"zhulong_l14_{Guid.NewGuid():N}.db");
        var factory = new TestDbContextFactory(_dbPath);
        _db = new DatabaseService(factory, NullLogger<DatabaseService>.Instance);
    }

    [Fact]
    public async Task MockInference_GeneratesSignal_AndPersistsToSqlite()
    {
        await _db.EnsureCreatedAsync();

        var settings = new AppSettings
        {
            SignalFilters = new AppSettings.SignalFilterSettings(),
            SignalGeometry = new AppSettings.SignalGeometrySettings(),
            Mt5 = new AppSettings.Mt5Settings { CommentPrefix = "ZhuLong" },
        };
        var gen = new SignalGeneratorService();
        var inference = new InferenceResult
        {
            Direction = 1,
            Confidence = 0.88,
            EntryOffset = -0.0015,
            ExpectedReturn = 1.2,
        };

        var signal = gen.TryGenerate(settings, "XAUUSD", inference, atrPct: 0.5, closePrice: 2350.0);
        Assert.NotNull(signal);

        await _db.SaveSignalAsync(signal!);
        var rows = await _db.GetRecentSignalsAsync(10);

        Assert.Single(rows);
        Assert.Equal(signal!.SignalId, rows[0].SignalId);
        Assert.Equal("XAUUSD", rows[0].Symbol);
        Assert.Equal("buy", rows[0].Direction);
        Assert.StartsWith("ZhuLong_", rows[0].CommentHint);
    }

    public void Dispose()
    {
        try { File.Delete(_dbPath); } catch { /* ignore */ }
    }

    private sealed class TestDbContextFactory : IDbContextFactory<ZhuLongDbContext>
    {
        private readonly DbContextOptions<ZhuLongDbContext> _options;

        public TestDbContextFactory(string dbPath)
        {
            _options = new DbContextOptionsBuilder<ZhuLongDbContext>()
                .UseSqlite($"Data Source={dbPath}")
                .Options;
        }

        public ZhuLongDbContext CreateDbContext() => new(_options);

        public Task<ZhuLongDbContext> CreateDbContextAsync(CancellationToken ct = default) =>
            Task.FromResult(CreateDbContext());
    }
}
