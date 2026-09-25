namespace cAlgo.Robots;

public partial class SwapProbe : Robot
{
    private static readonly string[] Names =
    {
        "US500", "NAS100", "US30", "US2000", "GER40", "UK100", "JPN225", "FRA40", "AUS200", "HK50", "EUSTX50",
        "SPA35", "XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD", "EURUSD", "USDJPY",
    };

    protected override void OnStart()
    {
        foreach (var name in Names)
        {
            var symbol = Symbols.GetSymbol(name);
            if (symbol == null)
            {
                Print($"SWAP {name} missing");
                continue;
            }
            Print($"SWAP {name} long={symbol.SwapLong} short={symbol.SwapShort} type={symbol.SwapCalculationType} pip={symbol.PipSize} bid={symbol.Bid}");
        }
        Stop();
    }
}
