// Indicators calculation web worker
// Runs technical indicator calculations off the main thread

interface CandleData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

interface WorkerMessage {
  type: "calculate";
  candles: CandleData[];
  indicators: string[];
}

interface IndicatorResult {
  name: string;
  values: (number | null)[];
}

function calculateSMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const sum = closes.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }
  return result;
}

function calculateEMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  const multiplier = 2 / (period + 1);
  let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;

  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else if (i === period - 1) {
      result.push(ema);
    } else {
      ema = (closes[i] - ema) * multiplier + ema;
      result.push(ema);
    }
  }
  return result;
}

function calculateRSI(closes: number[], period: number = 14): (number | null)[] {
  const result: (number | null)[] = [];
  const gains: number[] = [];
  const losses: number[] = [];

  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }

  result.push(null); // First candle has no RSI

  for (let i = 0; i < gains.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else if (i === period - 1) {
      const avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      result.push(100 - 100 / (1 + rs));
    } else {
      const prevRSI = result[result.length - 1]!;
      const avgGain = (gains[i] + (period - 1) * (100 / (100 - prevRSI) - 1) * gains[i]) / period;
      const avgLoss = (losses[i] + (period - 1) * (1 / (1 - prevRSI / 100) - 1) * losses[i]) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      result.push(100 - 100 / (1 + rs));
    }
  }
  return result;
}

function calculateBollingerBands(closes: number[], period: number = 20, stdDev: number = 2): { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] } {
  const middle = calculateSMA(closes, period);
  const upper: (number | null)[] = [];
  const lower: (number | null)[] = [];

  for (let i = 0; i < closes.length; i++) {
    if (middle[i] === null) {
      upper.push(null);
      lower.push(null);
    } else {
      const slice = closes.slice(i - period + 1, i + 1);
      const mean = middle[i]!;
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const sd = Math.sqrt(variance) * stdDev;
      upper.push(mean + sd);
      lower.push(mean - sd);
    }
  }

  return { upper, middle, lower };
}

function calculateMACD(closes: number[]): { macd: (number | null)[]; signal: (number | null)[]; histogram: (number | null)[] } {
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);
  const macdLine: (number | null)[] = [];

  for (let i = 0; i < closes.length; i++) {
    if (ema12[i] === null || ema26[i] === null) {
      macdLine.push(null);
    } else {
      macdLine.push(ema12[i]! - ema26[i]!);
    }
  }

  const validMACD = macdLine.filter((v): v is number => v !== null);
  const signalLine = calculateEMA(validMACD, 9);
  
  const signal: (number | null)[] = [];
  const histogram: (number | null)[] = [];
  let validIdx = 0;

  for (let i = 0; i < closes.length; i++) {
    if (macdLine[i] === null) {
      signal.push(null);
      histogram.push(null);
    } else {
      const sig = signalLine[validIdx] ?? null;
      signal.push(sig);
      histogram.push(sig !== null ? macdLine[i]! - sig : null);
      validIdx++;
    }
  }

  return { macd: macdLine, signal, histogram };
}

self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  const { candles, indicators } = e.data;
  const closes = candles.map((c) => c.close);
  const results: IndicatorResult[] = [];

  for (const indicator of indicators) {
    switch (indicator) {
      case "sma_20":
        results.push({ name: "SMA 20", values: calculateSMA(closes, 20) });
        break;
      case "sma_50":
        results.push({ name: "SMA 50", values: calculateSMA(closes, 50) });
        break;
      case "ema_12":
        results.push({ name: "EMA 12", values: calculateEMA(closes, 12) });
        break;
      case "ema_26":
        results.push({ name: "EMA 26", values: calculateEMA(closes, 26) });
        break;
      case "rsi":
        results.push({ name: "RSI", values: calculateRSI(closes) });
        break;
      case "bollinger": {
        const bb = calculateBollingerBands(closes);
        results.push({ name: "BB Upper", values: bb.upper });
        results.push({ name: "BB Middle", values: bb.middle });
        results.push({ name: "BB Lower", values: bb.lower });
        break;
      }
      case "macd": {
        const macd = calculateMACD(closes);
        results.push({ name: "MACD", values: macd.macd });
        results.push({ name: "MACD Signal", values: macd.signal });
        results.push({ name: "MACD Histogram", values: macd.histogram });
        break;
      }
    }
  }

  self.postMessage({ type: "result", results });
};
