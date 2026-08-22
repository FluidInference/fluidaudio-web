/**
 * Canonical seed syntax used in requests, manifests, fixtures, and reports.
 *
 * A seed is exactly 64 unsigned bits written as 16 lowercase hexadecimal
 * digits without a prefix. Strings avoid JavaScript's 53-bit integer limit and
 * remain lossless in JSON as well as structured clone.
 */
export type AceSeed = string & { readonly __aceSeed: unique symbol };

export type AceRandomStream = "diffusion-noise" | "planner-sampling";

export const ACE_SEED_CONTRACT = Object.freeze({
  version: "ace-seed-v1",
  authority: "browser-defined",
  syntax: "uint64-lowercase-hex-16",
  generator: "philox4x32-10",
  counterOrder: "little-endian-word-index",
  streams: ["diffusion-noise", "planner-sampling"] as const,
  gaussianMapping: "probit-acklam-binary64-f32-v1",
  categoricalMapping: "u32-midpoint-binary64-cdf-v1",
  transformLocation: "dedicated-worker-cpu",
  referenceInterop: "capture-or-inject-browser-random-inputs",
});

/**
 * A numeric ACE seed is not presumed to match PyTorch or MLX RNG output.
 * Upstream comparisons capture or inject the same browser-defined initial
 * noise and planner words. The transforms below are intentionally executed on
 * the dedicated worker CPU: WebGPU never evaluates a transcendental function
 * to create random inputs.
 */

const MAX_UINT64 = 0xffff_ffff_ffff_ffffn;
const MAX_PHILOX_WORD_INDEX = (MAX_UINT64 << 2n) | 3n;
const CANONICAL_SEED = /^[0-9a-f]{16}$/;
const DECIMAL_SEED = /^(?:0|[1-9][0-9]*)$/;
const HEX_SEED = /^(?:0x)?([0-9a-fA-F]{1,16})$/;

/** Convert accepted user input into the sole serialized seed representation. */
export function canonicalizeSeed(value: string | number | bigint): AceSeed {
  let parsed: bigint;
  if (typeof value === "bigint") {
    parsed = value;
  } else if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) {
      throw new RangeError("A numeric ACE seed must be a safe integer");
    }
    parsed = BigInt(value);
  } else {
    if (CANONICAL_SEED.test(value)) return value as AceSeed;
    const hexadecimal = HEX_SEED.exec(value);
    if (hexadecimal !== null && (value.startsWith("0x") || /[a-fA-F]/.test(value))) {
      parsed = BigInt(`0x${hexadecimal[1]}`);
    } else if (DECIMAL_SEED.test(value)) {
      parsed = BigInt(value);
    } else {
      throw new TypeError(
        "An ACE seed must be an unsigned decimal integer or at most 16 hexadecimal digits",
      );
    }
  }

  if (parsed < 0n || parsed > MAX_UINT64) {
    throw new RangeError("An ACE seed must fit in an unsigned 64-bit integer");
  }
  return parsed.toString(16).padStart(16, "0") as AceSeed;
}

export function isAceSeed(value: unknown): value is AceSeed {
  return typeof value === "string" && CANONICAL_SEED.test(value);
}

export interface PhiloxWords {
  readonly word0: number;
  readonly word1: number;
  readonly word2: number;
  readonly word3: number;
}

const PHILOX_MULTIPLIER_0 = 0xd251_1f53;
const PHILOX_MULTIPLIER_1 = 0xcd9e_8d57;
const PHILOX_WEYL_0 = 0x9e37_79b9;
const PHILOX_WEYL_1 = 0xbb67_ae85;
const PHILOX_ROUNDS = 10;

const RANDOM_STREAM_DOMAINS: Readonly<
  Record<AceRandomStream, readonly [number, number]>
> = Object.freeze({
  "diffusion-noise": [0x4449_4646, 0x4e4f_4953],
  "planner-sampling": [0x504c_414e, 0x5341_4d50],
});

/**
 * Exact Philox 4x32-10 primitive. Every operation is defined modulo 2^32, so
 * TypeScript, WASM, and WGSL implementations can share bit-for-bit fixtures.
 */
export function philox4x32_10(
  counter: PhiloxWords,
  key: readonly [number, number],
): PhiloxWords {
  let word0 = asUint32(counter.word0, "counter.word0");
  let word1 = asUint32(counter.word1, "counter.word1");
  let word2 = asUint32(counter.word2, "counter.word2");
  let word3 = asUint32(counter.word3, "counter.word3");
  let key0 = asUint32(key[0], "key[0]");
  let key1 = asUint32(key[1], "key[1]");

  for (let round = 0; round < PHILOX_ROUNDS; round += 1) {
    const product0 = multiplyHighLow(PHILOX_MULTIPLIER_0, word0);
    const product1 = multiplyHighLow(PHILOX_MULTIPLIER_1, word2);
    const next0 = (product1.high ^ word1 ^ key0) >>> 0;
    const next1 = product1.low;
    const next2 = (product0.high ^ word3 ^ key1) >>> 0;
    const next3 = product0.low;
    word0 = next0;
    word1 = next1;
    word2 = next2;
    word3 = next3;
    key0 = (key0 + PHILOX_WEYL_0) >>> 0;
    key1 = (key1 + PHILOX_WEYL_1) >>> 0;
  }

  return { word0, word1, word2, word3 };
}

/**
 * Return the four random words beginning at a 128-bit counter block. Counter
 * blocks are independently addressable; adding or skipping work in one stream
 * never shifts the other stream.
 */
export function aceRandomWords(
  seed: AceSeed,
  stream: AceRandomStream,
  blockIndex: number | bigint,
): PhiloxWords {
  if (!isAceSeed(seed)) {
    throw new TypeError("ACE random words require a canonical seed");
  }
  const counter = typeof blockIndex === "bigint" ? blockIndex : numberToBigInt(blockIndex);
  if (counter < 0n || counter > MAX_UINT64) {
    throw new RangeError("The Philox block index must fit in an unsigned 64-bit integer");
  }
  const seedValue = BigInt(`0x${seed}`);
  const seedLow = Number(seedValue & 0xffff_ffffn);
  const seedHigh = Number(seedValue >> 32n);
  const [domain0, domain1] = RANDOM_STREAM_DOMAINS[stream];
  return philox4x32_10(
    {
      word0: Number(counter & 0xffff_ffffn),
      word1: Number(counter >> 32n),
      word2: 0,
      word3: 0,
    },
    [(seedLow ^ domain0) >>> 0, (seedHigh ^ domain1) >>> 0],
  );
}

/**
 * Address one word in a Philox stream. Word indices increase through word0,
 * word1, word2, and word3 before advancing the 64-bit block counter.
 */
export function aceRandomWord(
  seed: AceSeed,
  stream: AceRandomStream,
  wordIndex: number | bigint,
): number {
  const index =
    typeof wordIndex === "bigint" ? wordIndex : numberToBigInt(wordIndex);
  if (index < 0n || index > MAX_PHILOX_WORD_INDEX) {
    throw new RangeError("The Philox word index must fit in 66 unsigned bits");
  }
  const words = aceRandomWords(seed, stream, index >> 2n);
  switch (Number(index & 3n)) {
    case 0:
      return words.word0;
    case 1:
      return words.word1;
    case 2:
      return words.word2;
    case 3:
      return words.word3;
    default:
      throw new Error("Unreachable Philox word lane");
  }
}

const UINT32_MIDPOINT_SCALE = 1 / 0x1_0000_0000;
const ACKLAM_LOW_TAIL = 0.02425;
const ACKLAM_HIGH_TAIL = 1 - ACKLAM_LOW_TAIL;
const NATURAL_LOG_OF_TWO = 0.6931471805599453;

// Coefficients from Peter J. Acklam's inverse-normal rational approximation.
// Their binary64 values and the evaluation order below are part of ace-seed-v1.
const ACKLAM_A = Object.freeze([
  -3.969683028665376e1,
  2.209460984245205e2,
  -2.759285104469687e2,
  1.38357751867269e2,
  -3.066479806614716e1,
  2.506628277459239,
] as const);
const ACKLAM_B = Object.freeze([
  -5.447609879822406e1,
  1.615858368580409e2,
  -1.556989798598866e2,
  6.680131188771972e1,
  -1.328068155288572e1,
] as const);
const ACKLAM_C = Object.freeze([
  -7.784894002430293e-3,
  -3.223964580411365e-1,
  -2.400758277161838,
  -2.549732539343734,
  4.374664141464968,
  2.938163982698783,
] as const);
const ACKLAM_D = Object.freeze([
  7.784695709041462e-3,
  3.224671290700398e-1,
  2.445134137142996,
  3.754408661907416,
] as const);

/**
 * Map a uint32 to the midpoint of its equal-width interval in (0, 1).
 * The expression is exactly representable in binary64 before multiplication.
 */
export function aceOpenUnitFloat64FromWord(word: number): number {
  return (asUint32(word, "word") + 0.5) * UINT32_MIDPOINT_SCALE;
}

/**
 * Deterministic inverse-standard-normal mapping used for diffusion noise.
 *
 * The rational probit is evaluated in binary64 with a fixed operation order,
 * then rounded once to binary32. Its tail log and square root use the pinned
 * arithmetic routines below, not host Math transcendental functions. One
 * Philox word always produces exactly one output value.
 */
export function aceGaussianF32FromWord(word: number): number {
  const probability = aceOpenUnitFloat64FromWord(word);
  let normal: number;

  if (probability < ACKLAM_LOW_TAIL) {
    const q = deterministicSquareRoot(-2 * deterministicNaturalLog(probability));
    normal = evaluateAcklamTail(q);
  } else if (probability > ACKLAM_HIGH_TAIL) {
    const q = deterministicSquareRoot(
      -2 * deterministicNaturalLog(1 - probability),
    );
    normal = -evaluateAcklamTail(q);
  } else {
    const q = probability - 0.5;
    const r = q * q;
    const numerator =
      (((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r +
        ACKLAM_A[3]) *
        r +
        ACKLAM_A[4]) *
        r +
        ACKLAM_A[5]) *
      q;
    const denominator =
      ((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r +
        ACKLAM_B[3]) *
        r +
        ACKLAM_B[4]) *
        r +
      1;
    normal = numerator / denominator;
  }

  return Math.fround(normal);
}

/**
 * Fill a contiguous slice of the logical [batch, latent-frame, channel]
 * diffusion tensor. Callers own the shape; this function fixes its flattened
 * row-major word consumption.
 */
export function fillAceDiffusionNoise(
  output: Float32Array,
  seed: AceSeed,
  startWordIndex: number | bigint = 0,
): void {
  if (!isAceSeed(seed)) {
    throw new TypeError("ACE diffusion noise requires a canonical seed");
  }
  const start =
    typeof startWordIndex === "bigint"
      ? startWordIndex
      : numberToBigInt(startWordIndex);
  if (start < 0n) {
    throw new RangeError("The diffusion word index must not be negative");
  }
  const finalIndex =
    start + BigInt(output.length) - (output.length === 0 ? 0n : 1n);
  if (finalIndex > MAX_PHILOX_WORD_INDEX) {
    throw new RangeError("The diffusion word range exceeds the Philox stream");
  }

  let outputIndex = 0;
  let wordIndex = start;
  while (outputIndex < output.length) {
    const words = aceRandomWords(seed, "diffusion-noise", wordIndex >> 2n);
    const lanes = [words.word0, words.word1, words.word2, words.word3] as const;
    let lane = Number(wordIndex & 3n);
    while (lane < lanes.length && outputIndex < output.length) {
      const word = lanes[lane];
      if (word === undefined) throw new Error("Unreachable Philox word lane");
      output[outputIndex] = aceGaussianF32FromWord(word);
      outputIndex += 1;
      wordIndex += 1n;
      lane += 1;
    }
  }
}

/**
 * Select one token from nonnegative categorical weights in ascending token-ID
 * order. Both summation passes are binary64 and have the exact same order.
 * Zero-weight tokens are never selected. The final positive token is a guard
 * against one-ulp accumulation drift at the upper boundary.
 */
export function aceCategoricalTokenFromWord(
  weights: ArrayLike<number>,
  word: number,
): number {
  if (!Number.isSafeInteger(weights.length) || weights.length <= 0) {
    throw new RangeError("Categorical weights must have a non-empty safe length");
  }

  let total = 0;
  let finalPositiveToken = -1;
  for (let token = 0; token < weights.length; token += 1) {
    const sourceWeight = weights[token];
    if (sourceWeight === undefined || !Number.isFinite(sourceWeight)) {
      throw new RangeError("Categorical weights must be finite and nonnegative");
    }
    const weight = Math.fround(sourceWeight);
    if (!Number.isFinite(weight) || weight < 0) {
      throw new RangeError("Categorical weights must be finite and nonnegative");
    }
    if (weight > 0) finalPositiveToken = token;
    total += weight;
  }
  if (!Number.isFinite(total) || total <= 0 || finalPositiveToken < 0) {
    throw new RangeError("Categorical weights must have a finite positive sum");
  }

  const threshold = aceOpenUnitFloat64FromWord(word) * total;
  let cumulative = 0;
  for (let token = 0; token < weights.length; token += 1) {
    const sourceWeight = weights[token];
    if (sourceWeight === undefined) {
      throw new Error("Unreachable categorical weight");
    }
    const weight = Math.fround(sourceWeight);
    cumulative += weight;
    if (weight > 0 && cumulative > threshold) return token;
  }
  return finalPositiveToken;
}

/** One categorical draw at the declared global planner draw ordinal. */
export function acePlannerCategoricalToken(
  seed: AceSeed,
  drawIndex: number | bigint,
  weights: ArrayLike<number>,
): number {
  return aceCategoricalTokenFromWord(
    weights,
    aceRandomWord(seed, "planner-sampling", drawIndex),
  );
}

function evaluateAcklamTail(q: number): number {
  const numerator =
    ((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q +
      ACKLAM_C[3]) *
      q +
      ACKLAM_C[4]) *
      q +
    ACKLAM_C[5];
  const denominator =
    (((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q +
      ACKLAM_D[3]) *
      q +
    1;
  return numerator / denominator;
}

/**
 * Natural log for positive finite binary64 values. Range reduction followed
 * by a 25-term atanh series avoids host transcendental implementations.
 */
function deterministicNaturalLog(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError("Deterministic log requires a positive finite value");
  }
  let mantissa = value;
  let exponent = 0;
  while (mantissa < 1) {
    mantissa *= 2;
    exponent -= 1;
  }
  while (mantissa >= 2) {
    mantissa *= 0.5;
    exponent += 1;
  }

  const y = (mantissa - 1) / (mantissa + 1);
  const ySquared = y * y;
  let term = y;
  let sum = y;
  for (let denominator = 3; denominator <= 49; denominator += 2) {
    term *= ySquared;
    sum += term / denominator;
  }
  return 2 * sum + exponent * NATURAL_LOG_OF_TWO;
}

/** Fixed-count Newton square root using binary64 arithmetic only. */
function deterministicSquareRoot(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      "Deterministic square root requires a finite nonnegative value",
    );
  }
  if (value === 0) return 0;
  let estimate = value >= 1 ? value : 1;
  for (let iteration = 0; iteration < 10; iteration += 1) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}

function numberToBigInt(value: number): bigint {
  if (!Number.isSafeInteger(value)) {
    throw new RangeError("The Philox block index must be a safe integer");
  }
  return BigInt(value);
}

function asUint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer`);
  }
  return value >>> 0;
}

function multiplyHighLow(
  left: number,
  right: number,
): { readonly high: number; readonly low: number } {
  const leftLow = left & 0xffff;
  const leftHigh = left >>> 16;
  const rightLow = right & 0xffff;
  const rightHigh = right >>> 16;
  const lowLow = Math.imul(leftLow, rightLow) >>> 0;
  const cross0 = Math.imul(leftHigh, rightLow) >>> 0;
  const cross1 = Math.imul(leftLow, rightHigh) >>> 0;
  const highHigh = Math.imul(leftHigh, rightHigh) >>> 0;
  const middle =
    (lowLow >>> 16) + (cross0 & 0xffff) + (cross1 & 0xffff);
  return {
    high:
      (highHigh + (cross0 >>> 16) + (cross1 >>> 16) + (middle >>> 16)) >>> 0,
    low: (((middle & 0xffff) << 16) | (lowLow & 0xffff)) >>> 0,
  };
}
