// ZhuLongMt5Pipe.dll v5 — MT5 named pipe client
// mode 1 = write (ZhuLong_Data)
// mode 2 = read (ZhuLong_Drawing, server Out)

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace {

struct PipeConn
{
    HANDLE handle = INVALID_HANDLE_VALUE;
    int mode = 0;             // 1=data write, 2=draw read
    std::string readBuf;
};

std::mutex g_mutex;
std::map<int, PipeConn> g_conns;
int g_nextId = 1;

bool BuildPipePath(wchar_t* out, int outChars, const wchar_t* logicalName)
{
    if (!out || outChars < 32 || !logicalName || !logicalName[0])
        return false;
    if (wcsncmp(logicalName, L"\\\\.\\pipe\\", 9) == 0)
        return swprintf_s(out, (size_t)outChars, L"%s", logicalName) > 0;
    return swprintf_s(out, (size_t)outChars, L"\\\\.\\pipe\\%s", logicalName) > 0;
}

bool WaitPipeReady(const wchar_t* path, unsigned int timeoutMs)
{
    const DWORD budget = timeoutMs == 0 ? 3000U : timeoutMs;
    const DWORD deadline = GetTickCount() + budget;
    while ((int)(GetTickCount() - deadline) < 0)
    {
        DWORD remain = deadline - GetTickCount();
        DWORD waitMs = remain > 320U ? 320U : remain;
        if (waitMs < 24U)
            break;
        if (WaitNamedPipeW(path, waitMs))
            return true;
        const DWORD err = GetLastError();
        if (err != ERROR_SEM_TIMEOUT && err != ERROR_FILE_NOT_FOUND && err != ERROR_PIPE_BUSY)
            break;
    }
    return false;
}

HANDLE OpenPipe(const wchar_t* path, int mode)
{
    if (mode == 1)
        return CreateFileW(path, GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return CreateFileW(path, GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}

bool WideToUtf8(const wchar_t* wide, std::string& out)
{
    if (!wide)
        return false;
    const int need = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (need <= 1)
        return false;
    out.resize((size_t)need - 1);
    const int wrote = WideCharToMultiByte(CP_UTF8, 0, wide, -1, &out[0], need, nullptr, nullptr);
    return wrote > 0;
}

bool IsPipeBrokenError(DWORD err)
{
    return err == ERROR_BROKEN_PIPE
        || err == ERROR_PIPE_NOT_CONNECTED
        || err == ERROR_INVALID_HANDLE;
}

PipeConn* FindConn(int handleId, int expectedMode)
{
    auto it = g_conns.find(handleId);
    if (it == g_conns.end() || it->second.mode != expectedMode || it->second.handle == INVALID_HANDLE_VALUE)
        return nullptr;
    return &it->second;
}

} // namespace

extern "C" __declspec(dllexport) int __stdcall ZhuLongPipeConnectV1(
    const wchar_t* pipeLogicalName,
    int mode,
    unsigned int connectTimeoutMs)
{
    if (!pipeLogicalName || (mode != 1 && mode != 2))
        return 0;

    wchar_t path[512];
    if (!BuildPipePath(path, 512, pipeLogicalName))
        return 0;

    if (!WaitPipeReady(path, connectTimeoutMs))
        return 0;

    HANDLE h = OpenPipe(path, mode);
    if (h == INVALID_HANDLE_VALUE)
        return 0;

    if (mode == 1)
    {
        DWORD pipeMode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
        if (!SetNamedPipeHandleState(h, &pipeMode, nullptr, nullptr))
        {
            CloseHandle(h);
            return 0;
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const int id = g_nextId++;
    g_conns[id] = PipeConn{ h, mode };
    return id;
}

// 1=data available (read pipe only), 0=alive/no data, -1=broken or unsupported handle
extern "C" __declspec(dllexport) int __stdcall ZhuLongPipePollV1(int handleId)
{
    HANDLE h = INVALID_HANDLE_VALUE;
  {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto* c = FindConn(handleId, 2);
        if (!c)
            return -1;
        h = c->handle;
    }

    DWORD avail = 0;
    if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr))
    {
        if (IsPipeBrokenError(GetLastError()))
            return -1;
        return 0;
    }
    return avail > 0 ? 1 : 0;
}

extern "C" __declspec(dllexport) int __stdcall ZhuLongPipeWriteLineV1(int handleId, const wchar_t* lineUtf16)
{
    if (!lineUtf16)
        return 0;

    HANDLE h = INVALID_HANDLE_VALUE;
    int mode = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_conns.find(handleId);
        if (it == g_conns.end() || it->second.handle == INVALID_HANDLE_VALUE)
            return 0;
        mode = it->second.mode;
        if (mode != 1)
            return 0;
        h = it->second.handle;
    }

    std::string payload;
    if (!WideToUtf8(lineUtf16, payload))
        return 0;
    if (payload.empty() || payload.back() != '\n')
        payload += '\n';

    const DWORD payloadSize = (DWORD)payload.size();
    const int maxAttempts = 150;
    for (int attempt = 0; attempt < maxAttempts; ++attempt)
    {
        DWORD written = 0;
        if (WriteFile(h, payload.data(), payloadSize, &written, nullptr))
            return written > 0 ? 1 : 0;

        const DWORD err = GetLastError();
        if (err == ERROR_NO_DATA || err == ERROR_PIPE_BUSY || err == ERROR_PIPE_NOT_CONNECTED)
        {
            Sleep(20);
            continue;
        }
        return 0;
    }
    return 0;
}

extern "C" __declspec(dllexport) int __stdcall ZhuLongPipeReadLineV1(
    int handleId,
    char* utf8Out,
    int utf8OutCapacity)
{
    if (!utf8Out || utf8OutCapacity < 8)
        return -1;
    utf8Out[0] = 0;

    HANDLE h = INVALID_HANDLE_VALUE;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto* c = FindConn(handleId, 2);
        if (!c)
            return -1;
        h = c->handle;
    }

    DWORD avail = 0;
    if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr))
    {
        if (IsPipeBrokenError(GetLastError()))
            return -1;
        avail = 0;
    }

    if (avail > 0)
    {
        char tmp[16384];
        if (avail > (DWORD)(sizeof(tmp) - 1))
            avail = (DWORD)(sizeof(tmp) - 1);
        DWORD n = 0;
        if (!ReadFile(h, tmp, avail, &n, nullptr))
        {
            if (IsPipeBrokenError(GetLastError()))
                return -1;
        }
        else if (n > 0)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto* c = FindConn(handleId, 2);
            if (!c)
                return -1;
            c->readBuf.append(tmp, n);
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    auto* c = FindConn(handleId, 2);
    if (!c)
        return -1;

    std::string& buf = c->readBuf;
    size_t pos = buf.find('\n');
    if (pos == std::string::npos)
        return 0;

    size_t len = pos;
    if (len > 0 && buf[len - 1] == '\r')
        len--;
    if (len > (size_t)(utf8OutCapacity - 1))
        len = (size_t)(utf8OutCapacity - 1);

    if (len > 0)
    {
        memcpy(utf8Out, buf.data(), len);
        utf8Out[len] = 0;
    }
    buf.erase(0, pos + 1);
    return len > 0 ? 1 : 0;
}

extern "C" __declspec(dllexport) void __stdcall ZhuLongPipeDisconnectV1(int handleId)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_conns.find(handleId);
    if (it == g_conns.end())
        return;
    if (it->second.handle != INVALID_HANDLE_VALUE)
        CloseHandle(it->second.handle);
    g_conns.erase(it);
}
