using Microsoft.EntityFrameworkCore;

namespace ZhuLong.Core.Data;

public sealed class SignalEntity
{
    public string SignalId { get; set; } = "";
    public long Timestamp { get; set; }
    public string Symbol { get; set; } = "";
    public string Direction { get; set; } = "";
    public double EntryPrice { get; set; }
    public double StopLoss { get; set; }
    public double TakeProfit { get; set; }
    public double Confidence { get; set; }
    public double ExpectedReturn { get; set; }
    public int MagicNumber { get; set; }
    public string CommentHint { get; set; } = "";
    public string Strategy { get; set; } = "";
    public string Status { get; set; } = "pending";
    public string? ParamsSnapshot { get; set; }
    public string? AttributionJson { get; set; }
    public long CreatedAt { get; set; }
}

public sealed class TradeEntity
{
    public long TradeId { get; set; }
    public string SignalId { get; set; } = "";
    public long OpenTime { get; set; }
    public double OpenPrice { get; set; }
    public long? CloseTime { get; set; }
    public double? ClosePrice { get; set; }
    public double? PnlPoints { get; set; }
    public double? PnlPercent { get; set; }
    public int? IsWin { get; set; }
    public string? CloseReason { get; set; }
}

public sealed class PositionEventEntity
{
    public long EventId { get; set; }
    public string SignalId { get; set; } = "";
    public long EventTime { get; set; }
    public string EventType { get; set; } = "";
    public double? Price { get; set; }
    public double? Volume { get; set; }
    public double? OldSl { get; set; }
    public double? NewSl { get; set; }
}

public sealed class ZhuLongDbContext : DbContext
{
    public ZhuLongDbContext(DbContextOptions<ZhuLongDbContext> options) : base(options) { }

    public DbSet<SignalEntity> Signals => Set<SignalEntity>();
    public DbSet<TradeEntity> Trades => Set<TradeEntity>();
    public DbSet<PositionEventEntity> PositionEvents => Set<PositionEventEntity>();
    public DbSet<MacroEventEntity> MacroEvents => Set<MacroEventEntity>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<SignalEntity>(e =>
        {
            e.ToTable("signals");
            e.HasKey(x => x.SignalId);
            e.Property(x => x.SignalId).HasColumnName("signal_id");
            e.Property(x => x.Timestamp).HasColumnName("timestamp");
            e.Property(x => x.Symbol).HasColumnName("symbol");
            e.Property(x => x.Direction).HasColumnName("direction");
            e.Property(x => x.EntryPrice).HasColumnName("entry_price");
            e.Property(x => x.StopLoss).HasColumnName("stop_loss");
            e.Property(x => x.TakeProfit).HasColumnName("take_profit");
            e.Property(x => x.Confidence).HasColumnName("confidence");
            e.Property(x => x.ExpectedReturn).HasColumnName("expected_return");
            e.Property(x => x.MagicNumber).HasColumnName("magic_number");
            e.Property(x => x.CommentHint).HasColumnName("comment_hint");
            e.Property(x => x.Strategy).HasColumnName("strategy");
            e.Property(x => x.Status).HasColumnName("status");
            e.Property(x => x.ParamsSnapshot).HasColumnName("params_snapshot");
            e.Property(x => x.AttributionJson).HasColumnName("attribution_json");
            e.Property(x => x.CreatedAt).HasColumnName("created_at");
        });

        modelBuilder.Entity<TradeEntity>(e =>
        {
            e.ToTable("trades");
            e.HasKey(x => x.TradeId);
            e.Property(x => x.TradeId).HasColumnName("trade_id").ValueGeneratedOnAdd();
            e.Property(x => x.SignalId).HasColumnName("signal_id");
            e.Property(x => x.OpenTime).HasColumnName("open_time");
            e.Property(x => x.OpenPrice).HasColumnName("open_price");
            e.Property(x => x.CloseTime).HasColumnName("close_time");
            e.Property(x => x.ClosePrice).HasColumnName("close_price");
            e.Property(x => x.PnlPoints).HasColumnName("pnl_points");
            e.Property(x => x.PnlPercent).HasColumnName("pnl_percent");
            e.Property(x => x.IsWin).HasColumnName("is_win");
            e.Property(x => x.CloseReason).HasColumnName("close_reason");
        });

        modelBuilder.Entity<PositionEventEntity>(e =>
        {
            e.ToTable("position_events");
            e.HasKey(x => x.EventId);
            e.Property(x => x.EventId).HasColumnName("event_id").ValueGeneratedOnAdd();
            e.Property(x => x.SignalId).HasColumnName("signal_id");
            e.Property(x => x.EventTime).HasColumnName("event_time");
            e.Property(x => x.EventType).HasColumnName("event_type");
            e.Property(x => x.Price).HasColumnName("price");
            e.Property(x => x.Volume).HasColumnName("volume");
            e.Property(x => x.OldSl).HasColumnName("old_sl");
            e.Property(x => x.NewSl).HasColumnName("new_sl");
        });

        modelBuilder.Entity<MacroEventEntity>(e =>
        {
            e.ToTable("macro_events");
            e.HasKey(x => x.Id);
            e.Property(x => x.Id).HasColumnName("id").ValueGeneratedOnAdd();
            e.Property(x => x.EventTimeUnix).HasColumnName("event_time_unix");
            e.Property(x => x.EventName).HasColumnName("event_name");
            e.Property(x => x.Impact).HasColumnName("impact");
            e.Property(x => x.Currency).HasColumnName("currency");
            e.Property(x => x.Source).HasColumnName("source");
            e.Property(x => x.FetchedAtUnix).HasColumnName("fetched_at_unix");
            e.Property(x => x.ExternalId).HasColumnName("external_id");
        });
    }
}
