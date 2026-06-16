using ZhuLong.Core.Licensing;

namespace ZhuLong.Core.Tests;

public sealed class OfflineLicenseCodecTests
{
    private const string TestKey = "test-hmac-key-for-unit-tests-only!!";

    [Fact]
    public void IssueAndValidate_WildcardFingerprint_Succeeds()
    {
        var until = DateTimeOffset.UtcNow.AddDays(30);
        var fp = new string('a', 64);
        var token = OfflineLicenseCodec.Issue(TestKey, until, "*");

        Assert.True(OfflineLicenseCodec.TryValidate(token, TestKey, fp, out var got, out var err), err);
        Assert.Equal(until.ToUnixTimeSeconds(), got.ToUnixTimeSeconds());
    }

    [Fact]
    public void TryValidate_RejectsWrongKey()
    {
        var token = OfflineLicenseCodec.Issue(TestKey, DateTimeOffset.UtcNow.AddDays(1), "*");
        Assert.False(OfflineLicenseCodec.TryValidate(
            token, "wrong-key------------------------", new string('b', 64), out _, out _));
    }

    [Fact]
    public void TryValidate_BindsDevicePrefix()
    {
        var fpFull = "abcdef0123456789" + new string('0', 48);
        var until = DateTimeOffset.UtcNow.AddDays(7);
        var token = OfflineLicenseCodec.Issue(TestKey, until, "abcdef01");

        Assert.True(OfflineLicenseCodec.TryValidate(token, TestKey, fpFull, out _, out var err), err);
        Assert.False(OfflineLicenseCodec.TryValidate(token, TestKey, fpFull.Replace('a', 'z'), out _, out _));
    }
}
