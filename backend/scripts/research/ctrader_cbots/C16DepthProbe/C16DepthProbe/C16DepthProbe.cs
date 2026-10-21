using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace cAlgo.Robots;

[Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
public partial class C16DepthProbe : Robot
{
    [Parameter("Pairs", DefaultValue = "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD")]
    public string Pairs { get; set; }

    [Parameter("Maximum levels", DefaultValue = 20, MinValue = 1, MaxValue = 100)]
    public int MaxLevels { get; set; }

    private readonly List<(MarketDepth Depth, Action Handler)> _subscriptions = new();
    private readonly List<(Symbol Symbol, Action<SymbolTickEventArgs> Handler)> _tickSubscriptions = new();
    private long _sequence;

    protected override void OnStart()
    {
        if (!Server.IsConnected)
            throw new InvalidOperationException("cTrader server is not connected; refusing to capture partial data.");
        Server.Disconnected += OnServerDisconnected;

        var pairNames = Pairs.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(name => name.ToUpperInvariant())
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        if (pairNames.Length == 0)
            throw new ArgumentException("Pairs must contain at least one cTrader symbol.");

        foreach (var pairName in pairNames)
        {
            var symbol = Symbols.GetSymbol(pairName)
                ?? throw new ArgumentException($"The cTrader account does not provide {pairName}.");
            Action<SymbolTickEventArgs> tickHandler = _ => EmitSpot(symbol);
            symbol.Tick += tickHandler;
            _tickSubscriptions.Add((symbol, tickHandler));
            var depth = MarketData.GetMarketDepth(symbol.Name);
            Action handler = () => EmitSnapshot(symbol, depth);
            depth.Updated += handler;
            _subscriptions.Add((depth, handler));
            EmitSpot(symbol);
            EmitSnapshot(symbol, depth);
        }

        Print($"C16DOM_STATUS|started|pairs={string.Join(",", pairNames)}|orders=disabled");
    }

    protected override void OnStop()
    {
        Server.Disconnected -= OnServerDisconnected;
        foreach (var (depth, handler) in _subscriptions)
            depth.Updated -= handler;
        _subscriptions.Clear();
        foreach (var (symbol, handler) in _tickSubscriptions)
            symbol.Tick -= handler;
        _tickSubscriptions.Clear();
        Print("C16DOM_STATUS|stopped|orders=disabled");
    }

    private void OnServerDisconnected()
    {
        Print("C16DOM_STATUS|disconnected|capture_stopping=true");
        Stop();
    }

    private void EmitSpot(Symbol symbol)
    {
        var payload = new
        {
            schema_version = 1,
            event_type = "spot",
            symbol = symbol.Name,
            sequence = Interlocked.Increment(ref _sequence),
            source_timestamp_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            bid = symbol.Bid,
            ask = symbol.Ask,
        };
        Print($"C16SPOT|{JsonSerializer.Serialize(payload)}");
    }

    private void EmitSnapshot(Symbol symbol, MarketDepth depth)
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var limit = Math.Clamp(MaxLevels, 1, 100);
        var payload = new
        {
            schema_version = 1,
            event_type = "depth_snapshot",
            symbol = symbol.Name,
            sequence = Interlocked.Increment(ref _sequence),
            source_timestamp_ms = timestamp,
            spot_bid = symbol.Bid,
            spot_ask = symbol.Ask,
            bids = SnapshotLevels(depth.BidEntries, limit),
            asks = SnapshotLevels(depth.AskEntries, limit),
        };
        Print($"C16DOM|{JsonSerializer.Serialize(payload)}");
    }

    private static object[] SnapshotLevels(IReadonlyList<MarketDepthEntry> entries, int limit)
    {
        var count = Math.Min(entries.Count, limit);
        var levels = new object[count];
        for (var index = 0; index < count; index++)
        {
            var entry = entries[index];
            levels[index] = new { price = entry.Price, volume_in_units = entry.VolumeInUnits };
        }
        return levels;
    }
}
