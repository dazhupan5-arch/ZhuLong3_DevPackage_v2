using Python.Runtime;

namespace ZhuLong.App.Services;

/// <summary>串行化所有 Python.NET GIL 调用，避免信号调度与持仓扫描争用导致卡死。</summary>
public sealed class PythonGilExecutor
{
    private readonly SemaphoreSlim _gate = new(1, 1);

    private static T RunWithGil<T>(Func<T> action)
    {
        if (!PythonEngine.IsInitialized)
            throw new InvalidOperationException("Python 未初始化，请先连接 MT5 或运行一键修复");
        using (Py.GIL())
            return action();
    }

    public T Run<T>(Func<T> action, CancellationToken ct = default)
    {
        _gate.Wait(ct);
        try
        {
            return RunWithGil(action);
        }
        finally
        {
            _gate.Release();
        }
    }

    public bool TryRun<T>(Func<T> action, TimeSpan timeout, out T result, CancellationToken ct = default)
    {
        if (!_gate.Wait(timeout, ct))
        {
            result = default!;
            return false;
        }

        try
        {
            result = RunWithGil(action);
            return true;
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Run(Action action, CancellationToken ct = default) =>
        Run(() => { action(); return 0; }, ct);

    public async Task<T> RunAsync<T>(Func<T> action, CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => RunWithGil(action), ct).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }
}
