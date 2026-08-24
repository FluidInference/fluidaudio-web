import {
  DICOSE_SAMPLE_RATE,
  DICOSE_STFT_HOP_LENGTH,
  type GaussianSeed,
  type StereoPcm,
} from "./audio.js";

const UINT32_UNIT = 1 / 0x1_0000_0000;

/** Released DiCoSe train/eval item geometry: exactly 11 seconds at 44.1 kHz. */
export const DICOSE_CHUNK_SAMPLES = 485_100;
/** The upstream overlap-add window linearly fades over 10% of a chunk. */
export const DICOSE_CHUNK_FADE_SAMPLES = 48_510;
/** Released Full whole-track inference retains the upstream 50% overlap. */
export const DICOSE_FULL_CHUNK_STEP_SAMPLES = 242_550;
/** Fast overlaps only the existing 10% fade region. */
export const DICOSE_FAST_CHUNK_STEP_SAMPLES = 436_590;
/** Preserve the included 11.89-second oracle as one graph; chunk above 12 seconds. */
export const DICOSE_SINGLE_PASS_SAMPLES = 12 * DICOSE_SAMPLE_RATE;

export interface DiCoSeChunkGeometry {
  readonly stepSamples: number;
  readonly fadeSamples: number;
}

export const DICOSE_FULL_CHUNK_GEOMETRY: DiCoSeChunkGeometry = Object.freeze({
  stepSamples: DICOSE_FULL_CHUNK_STEP_SAMPLES,
  fadeSamples: DICOSE_CHUNK_FADE_SAMPLES,
});

export const DICOSE_FAST_CHUNK_GEOMETRY: DiCoSeChunkGeometry = Object.freeze({
  stepSamples: DICOSE_FAST_CHUNK_STEP_SAMPLES,
  fadeSamples: DICOSE_CHUNK_FADE_SAMPLES,
});

export interface DiCoSeChunkSpan {
  readonly index: number;
  readonly paddedStart: number;
  readonly validSamples: number;
  readonly tailPadding: "zero" | "reflect";
  readonly chunkReadStart: number;
  readonly outputStart: number;
  readonly outputSamples: number;
}

export interface DiCoSeChunkPlan {
  readonly sourceSamples: number;
  readonly paddedSamples: number;
  readonly borderSamples: number;
  readonly geometry: DiCoSeChunkGeometry;
  readonly spans: readonly DiCoSeChunkSpan[];
}

export interface StereoPcmAccumulator {
  readonly left: Float32Array;
  readonly right: Float32Array;
}

/**
 * Plan the generic MSST whole-track policy without materializing its reflected
 * outer padding. Spans that can only affect padding later cropped away are
 * omitted, unlike the upstream helper's redundant final call.
 */
export function makeDiCoSeChunkPlan(
  sourceSamples: number,
  geometry: DiCoSeChunkGeometry = DICOSE_FULL_CHUNK_GEOMETRY,
): DiCoSeChunkPlan {
  if (!Number.isSafeInteger(sourceSamples) || sourceSamples <= DICOSE_CHUNK_SAMPLES) {
    throw new RangeError(`Chunked DiCoSe inference requires more than ${DICOSE_CHUNK_SAMPLES} samples`);
  }
  validateChunkGeometry(geometry);
  const frozenGeometry = Object.freeze({
    stepSamples: geometry.stepSamples,
    fadeSamples: geometry.fadeSamples,
  });
  const borderSamples = DICOSE_CHUNK_SAMPLES - frozenGeometry.stepSamples;
  const paddedSamples = sourceSamples + borderSamples * 2;
  const cropStart = borderSamples;
  const cropEnd = cropStart + sourceSamples;
  const spans: DiCoSeChunkSpan[] = [];
  let rawIndex = 0;
  for (
    let paddedStart = 0;
    paddedStart < paddedSamples;
    paddedStart += frozenGeometry.stepSamples
  ) {
    const validSamples = Math.min(DICOSE_CHUNK_SAMPLES, paddedSamples - paddedStart);
    const intersectionStart = Math.max(paddedStart, cropStart);
    const intersectionEnd = Math.min(paddedStart + validSamples, cropEnd);
    const outputSamples = Math.max(0, intersectionEnd - intersectionStart);
    if (outputSamples > 0) {
      spans.push(Object.freeze({
        index: rawIndex,
        paddedStart,
        validSamples,
        tailPadding: validSamples > DICOSE_CHUNK_SAMPLES / 2 ? "reflect" : "zero",
        chunkReadStart: intersectionStart - paddedStart,
        outputStart: intersectionStart - cropStart,
        outputSamples,
      }));
    }
    rawIndex += 1;
  }
  return Object.freeze({
    sourceSamples,
    paddedSamples,
    borderSamples,
    geometry: frozenGeometry,
    spans: Object.freeze(spans),
  });
}

export function makeDiCoSeChunkWindow(
  chunkSamples = DICOSE_CHUNK_SAMPLES,
  fadeSamples = DICOSE_CHUNK_FADE_SAMPLES,
): Float32Array {
  if (
    !Number.isSafeInteger(chunkSamples) || chunkSamples <= 0 ||
    !Number.isSafeInteger(fadeSamples) || fadeSamples <= 1 || fadeSamples * 2 > chunkSamples
  ) {
    throw new RangeError("Invalid DiCoSe chunk window geometry");
  }
  const window = new Float32Array(chunkSamples);
  window.fill(1);
  for (let index = 0; index < fadeSamples; index += 1) {
    const value = index / (fadeSamples - 1);
    window[index] = Math.fround(value);
    window[chunkSamples - 1 - index] = Math.fround(value);
  }
  return window;
}

function validateChunkGeometry(geometry: DiCoSeChunkGeometry): void {
  const { stepSamples, fadeSamples } = geometry;
  const borderSamples = DICOSE_CHUNK_SAMPLES - stepSamples;
  if (
    !Number.isSafeInteger(stepSamples) || stepSamples <= 0 ||
    !Number.isSafeInteger(fadeSamples) || fadeSamples <= 1 ||
    borderSamples < fadeSamples ||
    stepSamples % DICOSE_STFT_HOP_LENGTH !== 0 ||
    fadeSamples % DICOSE_STFT_HOP_LENGTH !== 0 ||
    borderSamples % DICOSE_STFT_HOP_LENGTH !== 0
  ) {
    throw new RangeError("Invalid DiCoSe chunk geometry");
  }
}

/** Materialize one fixed-size model item from virtual reflected outer padding. */
export function materializeDiCoSeChunk(
  source: StereoPcm,
  plan: DiCoSeChunkPlan,
  span: DiCoSeChunkSpan,
): StereoPcm {
  if (
    source.sampleRate !== DICOSE_SAMPLE_RATE || source.length !== plan.sourceSamples ||
    source.left.length !== source.length || source.right.length !== source.length
  ) {
    throw new RangeError("DiCoSe chunk source does not match its plan");
  }
  const left = new Float32Array(DICOSE_CHUNK_SAMPLES);
  const right = new Float32Array(DICOSE_CHUNK_SAMPLES);
  for (let local = 0; local < span.validSamples; local += 1) {
    const paddedIndex = span.paddedStart + local;
    const sourceIndex = reflectIndex(paddedIndex - plan.borderSamples, source.length);
    left[local] = source.left[sourceIndex]!;
    right[local] = source.right[sourceIndex]!;
  }
  if (span.tailPadding === "reflect") {
    for (let local = span.validSamples; local < DICOSE_CHUNK_SAMPLES; local += 1) {
      const reflected = reflectIndex(local, span.validSamples);
      left[local] = left[reflected]!;
      right[local] = right[reflected]!;
    }
  }
  return Object.freeze({
    sampleRate: DICOSE_SAMPLE_RATE,
    length: DICOSE_CHUNK_SAMPLES,
    left,
    right,
    channels: [left, right] as const,
  });
}

/** Accumulate every stem from one model item using the shared OLA denominator. */
export function overlapAddDiCoSeChunk(
  accumulators: readonly StereoPcmAccumulator[],
  denominator: Float32Array,
  chunks: readonly StereoPcm[],
  span: DiCoSeChunkSpan,
  window: Float32Array,
): void {
  if (accumulators.length === 0 || accumulators.length !== chunks.length) {
    throw new RangeError("DiCoSe overlap-add requires matching output and chunk stems");
  }
  if (window.length !== DICOSE_CHUNK_SAMPLES) {
    throw new RangeError("DiCoSe overlap-add window has the wrong model length");
  }
  const outputEnd = span.outputStart + span.outputSamples;
  const chunkEnd = span.chunkReadStart + span.outputSamples;
  if (
    span.outputStart < 0 || outputEnd > denominator.length ||
    span.chunkReadStart < 0 || chunkEnd > DICOSE_CHUNK_SAMPLES
  ) {
    throw new RangeError("DiCoSe overlap-add span exceeds its buffers");
  }
  for (const accumulator of accumulators) {
    if (accumulator.left.length !== denominator.length || accumulator.right.length !== denominator.length) {
      throw new RangeError("DiCoSe overlap-add accumulator has the wrong output length");
    }
  }
  for (const chunk of chunks) {
    if (
      chunk.sampleRate !== DICOSE_SAMPLE_RATE || chunk.length !== DICOSE_CHUNK_SAMPLES ||
      chunk.left.length !== chunk.length || chunk.right.length !== chunk.length
    ) {
      throw new RangeError("DiCoSe overlap-add chunk has the wrong model geometry");
    }
  }

  for (let offset = 0; offset < span.outputSamples; offset += 1) {
    const sourceIndex = span.chunkReadStart + offset;
    const outputIndex = span.outputStart + offset;
    const weight = window[sourceIndex]!;
    denominator[outputIndex] = denominator[outputIndex]! + weight;
    for (let stem = 0; stem < chunks.length; stem += 1) {
      const chunk = chunks[stem]!;
      const accumulator = accumulators[stem]!;
      accumulator.left[outputIndex] = accumulator.left[outputIndex]! + chunk.left[sourceIndex]! * weight;
      accumulator.right[outputIndex] = accumulator.right[outputIndex]! + chunk.right[sourceIndex]! * weight;
    }
  }
}

/** Divide a completed overlap-add sum in place, rejecting gaps instead of emitting NaNs/noise. */
export function normalizeDiCoSeOverlapAdd(
  accumulators: readonly StereoPcmAccumulator[],
  denominator: Float32Array,
): void {
  if (accumulators.length === 0) throw new RangeError("DiCoSe overlap-add has no output stems");
  for (const accumulator of accumulators) {
    if (accumulator.left.length !== denominator.length || accumulator.right.length !== denominator.length) {
      throw new RangeError("DiCoSe overlap-add accumulator has the wrong output length");
    }
  }
  for (let index = 0; index < denominator.length; index += 1) {
    const weight = denominator[index]!;
    if (!Number.isFinite(weight) || weight <= 0) {
      throw new Error(`DiCoSe overlap-add left output sample ${index} uncovered`);
    }
    for (const accumulator of accumulators) {
      const left = accumulator.left[index]! / weight;
      const right = accumulator.right[index]! / weight;
      if (!Number.isFinite(left) || !Number.isFinite(right)) {
        throw new Error(`DiCoSe overlap-add produced a non-finite sample at ${index}`);
      }
      accumulator.left[index] = Math.fround(left);
      accumulator.right[index] = Math.fround(right);
    }
  }
}

/**
 * Add CD noise keyed by padded-track coordinate. Overlapping chunks therefore
 * see identical noise at the same sample instead of crossfading independent
 * noise fields and creating a periodic seam in the final consistency affine.
 */
export function addDiCoSeCoordinateNoise(
  source: StereoPcm,
  span: DiCoSeChunkSpan,
  stem: number,
  seed: GaussianSeed,
  sigma: number,
): StereoPcm {
  if (
    source.sampleRate !== DICOSE_SAMPLE_RATE || source.length !== DICOSE_CHUNK_SAMPLES ||
    source.left.length !== source.length || source.right.length !== source.length
  ) {
    throw new RangeError("DiCoSe coordinate noise requires one fixed-size model item");
  }
  if (!Number.isSafeInteger(span.paddedStart) || span.paddedStart < 0) {
    throw new RangeError("DiCoSe coordinate noise requires a non-negative padded offset");
  }
  if (!Number.isSafeInteger(stem) || stem < 0 || !Number.isFinite(sigma) || sigma < 0) {
    throw new RangeError("Invalid DiCoSe coordinate-noise stream");
  }
  const seedWords = gaussianSeedWords(seed);
  const left = new Float32Array(source.length);
  const right = new Float32Array(source.length);
  for (let local = 0; local < source.length; local += 1) {
    const coordinate = span.paddedStart + local;
    left[local] = Math.fround(
      source.left[local]! + sigma * coordinateGaussian(seedWords, stem, 0, coordinate),
    );
    right[local] = Math.fround(
      source.right[local]! + sigma * coordinateGaussian(seedWords, stem, 1, coordinate),
    );
  }
  return Object.freeze({
    sampleRate: DICOSE_SAMPLE_RATE,
    length: source.length,
    left,
    right,
    channels: [left, right] as const,
  });
}

function reflectIndex(index: number, length: number): number {
  if (length <= 1) return 0;
  const period = 2 * length - 2;
  const wrapped = ((index % period) + period) % period;
  return wrapped < length ? wrapped : period - wrapped;
}

function gaussianSeedWords(seed: GaussianSeed): readonly [number, number] {
  if (typeof seed === "bigint") {
    if (seed < 0n || seed > 0xffff_ffff_ffff_ffffn) {
      throw new RangeError("Gaussian bigint seed must fit in unsigned 64 bits");
    }
    return [Number(seed & 0xffff_ffffn), Number(seed >> 32n)];
  }
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new RangeError("Gaussian numeric seed must be a non-negative safe integer");
  }
  return [seed >>> 0, Math.floor(seed / 0x1_0000_0000) >>> 0];
}

function coordinateGaussian(
  seed: readonly [number, number],
  stem: number,
  channel: number,
  coordinate: number,
): number {
  const pair = Math.floor(coordinate / 2);
  const low = pair >>> 0;
  const high = Math.floor(pair / 0x1_0000_0000) >>> 0;
  const stream = mix32(
    seed[0] ^ Math.imul(seed[1], 0x9e37_79b9) ^
    Math.imul(stem + 1, 0x85eb_ca6b) ^ Math.imul(channel + 1, 0xc2b2_ae35),
  );
  const first = mix32(stream ^ low ^ Math.imul(high, 0x27d4_eb2d));
  const second = mix32(first ^ 0xa511_e9b3);
  const radius = Math.sqrt(-2 * Math.log((first + 0.5) * UINT32_UNIT));
  const angle = Math.PI * 2 * ((second + 0.5) * UINT32_UNIT);
  return Math.fround(radius * (coordinate % 2 === 0 ? Math.cos(angle) : Math.sin(angle)));
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x7feb_352d);
  mixed ^= mixed >>> 15;
  mixed = Math.imul(mixed, 0x846c_a68b);
  mixed ^= mixed >>> 16;
  return mixed >>> 0;
}
