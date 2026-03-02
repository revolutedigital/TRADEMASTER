/**
 * WebAssembly loader for performance-critical calculations.
 * Provides fallback to JavaScript implementations when WASM is unavailable.
 */

export interface WASMIndicators {
  calculateSMA(prices: Float64Array, period: number): Float64Array;
  calculateEMA(prices: Float64Array, period: number): Float64Array;
  calculateRSI(prices: Float64Array, period: number): Float64Array;
  calculateBollingerBands(prices: Float64Array, period: number, stdDev: number): { upper: Float64Array; middle: Float64Array; lower: Float64Array };
}

let wasmModule: WASMIndicators | null = null;
let wasmLoadAttempted = false;

/**
 * Attempt to load WASM module for indicator calculations.
 * Falls back to JS implementation if WASM is unavailable.
 */
export async function loadWASM(): Promise<WASMIndicators | null> {
  if (wasmLoadAttempted) return wasmModule;
  wasmLoadAttempted = true;

  try {
    if (typeof WebAssembly === "undefined") {
      console.warn("WebAssembly not supported, using JS fallback");
      return null;
    }

    const response = await fetch("/wasm/indicators.wasm");
    if (!response.ok) {
      console.warn("WASM module not found, using JS fallback");
      return null;
    }

    const buffer = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(buffer);
    wasmModule = module.instance.exports as unknown as WASMIndicators;
    console.info("WASM indicators module loaded successfully");
    return wasmModule;
  } catch (error) {
    console.warn("Failed to load WASM module, using JS fallback:", error);
    return null;
  }
}

/**
 * JavaScript fallback implementations for when WASM is unavailable.
 * These are 10-100x slower but produce identical results.
 */
export const jsIndicators: WASMIndicators = {
  calculateSMA(prices: Float64Array, period: number): Float64Array {
    const result = new Float64Array(prices.length);
    for (let i = 0; i < prices.length; i++) {
      if (i < period - 1) {
        result[i] = NaN;
      } else {
        let sum = 0;
        for (let j = 0; j < period; j++) sum += prices[i - j];
        result[i] = sum / period;
      }
    }
    return result;
  },

  calculateEMA(prices: Float64Array, period: number): Float64Array {
    const result = new Float64Array(prices.length);
    const multiplier = 2 / (period + 1);
    
    // First value is SMA
    let sum = 0;
    for (let i = 0; i < period; i++) {
      result[i] = NaN;
      sum += prices[i];
    }
    result[period - 1] = sum / period;

    for (let i = period; i < prices.length; i++) {
      result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
    }
    return result;
  },

  calculateRSI(prices: Float64Array, period: number): Float64Array {
    const result = new Float64Array(prices.length);
    result[0] = NaN;

    let gainSum = 0, lossSum = 0;
    for (let i = 1; i <= period; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gainSum += change;
      else lossSum -= change;
      result[i] = NaN;
    }

    let avgGain = gainSum / period;
    let avgLoss = lossSum / period;
    result[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

    for (let i = period + 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      avgGain = ((avgGain * (period - 1)) + (change > 0 ? change : 0)) / period;
      avgLoss = ((avgLoss * (period - 1)) + (change < 0 ? -change : 0)) / period;
      result[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    }
    return result;
  },

  calculateBollingerBands(prices: Float64Array, period: number, stdDev: number) {
    const middle = jsIndicators.calculateSMA(prices, period);
    const upper = new Float64Array(prices.length);
    const lower = new Float64Array(prices.length);

    for (let i = 0; i < prices.length; i++) {
      if (isNaN(middle[i])) {
        upper[i] = NaN;
        lower[i] = NaN;
      } else {
        let variance = 0;
        for (let j = 0; j < period; j++) {
          const diff = prices[i - j] - middle[i];
          variance += diff * diff;
        }
        const sd = Math.sqrt(variance / period) * stdDev;
        upper[i] = middle[i] + sd;
        lower[i] = middle[i] - sd;
      }
    }
    return { upper, middle, lower };
  },
};

/**
 * Get the best available indicator calculator.
 * Prefers WASM, falls back to JS.
 */
export async function getIndicatorCalculator(): Promise<WASMIndicators> {
  const wasm = await loadWASM();
  return wasm || jsIndicators;
}
