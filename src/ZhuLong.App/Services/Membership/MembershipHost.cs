namespace ZhuLong.App.Services.Membership;

public static class MembershipHost
{
    public static IMembershipService Instance { get; } = new MembershipService();
}
