namespace ZhuLong.App.Services;

internal static class Mt5OrderHelper
{
    public static bool ModifySlTpWithRetry(Mt5ApiWrapper mt5, long ticket, double sl, double tp, int maxRetries, Action<string>? log)
    {
        maxRetries = Math.Max(1, maxRetries);
        for (var i = 0; i < maxRetries; i++)
        {
            if (mt5.ModifySlTp(ticket, sl, tp)) return true;
            Thread.Sleep(150 * (i + 1));
        }
        log?.Invoke($"改单失败 ticket={ticket}（已重试 {maxRetries} 次）");
        return false;
    }

    public static bool ClosePartialWithRetry(Mt5ApiWrapper mt5, long ticket, double volume, int maxRetries, Action<string>? log)
    {
        maxRetries = Math.Max(1, maxRetries);
        for (var i = 0; i < maxRetries; i++)
        {
            if (mt5.ClosePartial(ticket, volume)) return true;
            Thread.Sleep(150 * (i + 1));
        }
        log?.Invoke($"部分平仓失败 ticket={ticket}");
        return false;
    }

    public static bool CloseFullWithRetry(Mt5ApiWrapper mt5, long ticket, int maxRetries, Action<string>? log)
    {
        maxRetries = Math.Max(1, maxRetries);
        for (var i = 0; i < maxRetries; i++)
        {
            if (mt5.CloseFull(ticket)) return true;
            Thread.Sleep(150 * (i + 1));
        }
        log?.Invoke($"平仓失败 ticket={ticket}");
        return false;
    }
}
