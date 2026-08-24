/**
 * Browser-side audio primitives used by the DiCoSe runtime.
 *
 * The model's front end is deliberately kept on the CPU: Web Audio handles
 * container decoding, while the deterministic conversion/STFT code below
 * gives WebGPU a stable, explicit tensor layout. Spectra are frame-major:
 * `value[frame * binCount + frequencyBin]`.
 */

export const DICOSE_SAMPLE_RATE = 44_100 as const;
export const DICOSE_STFT_N_FFT = 2_048 as const;
export const DICOSE_STFT_HOP_LENGTH = 441 as const;

const TWO_PI = Math.PI * 2;
const UINT32_UNIT = 1 / 0x1_0000_0000;
const SINC_LOWPASS_FILTER_WIDTH = 6;
const SINC_ROLLOFF = 0.99;
const MAX_SINC_KERNEL_COEFFICIENTS = 8_000_000;

interface SincResampleKernel {
  readonly sourceStride: number;
  readonly targetPhases: number;
  readonly width: number;
  readonly tapCount: number;
  readonly coefficients: Float32Array;
  readonly firstNonzeroTaps: Uint32Array;
  readonly lastNonzeroTaps: Uint32Array;
}

const sincResampleKernelCache = new Map<string, SincResampleKernel>();

/** Structural subset of AudioBuffer, which also makes this easy to test in Node. */
export interface AudioBufferLike {
  readonly sampleRate: number;
  readonly length: number;
  readonly numberOfChannels: number;
  getChannelData(channel: number): Float32Array;
}

/** Two independent, planar channels at one sample rate. */
export interface StereoPcm {
  readonly sampleRate: number;
  readonly length: number;
  readonly left: Float32Array;
  readonly right: Float32Array;
  readonly channels: readonly [Float32Array, Float32Array];
}

export interface DecodeAudioOptions {
  /** Defaults to the DiCoSe model's 44.1 kHz source timeline. */
  readonly targetSampleRate?: number | "source";
  /** Linear exists only to replay the frozen pre-resampler model oracle. */
  readonly resampler?: "sinc" | "linear";
}

/**
 * Decode any browser-supported container (including WAV) and return a fresh,
 * planar stereo PCM buffer. The temporary AudioContext is never started, so
 * this does not require a user gesture or playback permission.
 */
export async function decodeAudioBlob(
  blob: Blob,
  options: DecodeAudioOptions = {},
): Promise<StereoPcm> {
  const AudioContextConstructor = globalThis.AudioContext;
  if (AudioContextConstructor === undefined) {
    throw new Error("Web Audio AudioContext is unavailable in this environment");
  }

  const encoded = await blob.arrayBuffer();
  // decodeAudioData resamples to its context rate. For WAV we can retain the
  // file's rate (including the bundled 22.05 kHz fixture) by inspecting its
  // container header, then perform the canonical bandlimited conversion
  // ourselves.
  const encodedSampleRate = wavSampleRateFromRiff(encoded);
  const context = encodedSampleRate === undefined
    ? new AudioContextConstructor()
    : new AudioContextConstructor({ sampleRate: encodedSampleRate });
  try {
    const decoded = await context.decodeAudioData(encoded);
    return audioBufferToStereoPcm(decoded, options.targetSampleRate, options.resampler);
  } finally {
    // close() does not require an activated context and releases the decoder's
    // resources promptly in a long-running browser session.
    await context.close();
  }
}

/** Alias that makes the canonical WAV input path self-documenting. */
export const decodeWavBlob = decodeAudioBlob;

/**
 * Convert an AudioBuffer into independent stereo channels, duplicating mono
 * inputs and using the first two channels of multichannel sources. Resampling
 * matches torchaudio 2.0.2's default Hann-windowed sinc path rather than using
 * a device-dependent Web Audio rendering pass.
 */
export function audioBufferToStereoPcm(
  source: AudioBufferLike,
  targetSampleRate: number | "source" = DICOSE_SAMPLE_RATE,
  resampler: "sinc" | "linear" = "sinc",
): StereoPcm {
  validateSampleRate(source.sampleRate, "source.sampleRate");
  const outputSampleRate = targetSampleRate === "source"
    ? source.sampleRate
    : targetSampleRate;
  validateSampleRate(outputSampleRate, "targetSampleRate");
  if (!Number.isSafeInteger(source.length) || source.length < 0) {
    throw new RangeError("AudioBuffer length must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(source.numberOfChannels) || source.numberOfChannels < 1) {
    throw new RangeError("AudioBuffer must contain at least one channel");
  }

  const sourceLeft = source.getChannelData(0);
  if (sourceLeft.length !== source.length) {
    throw new RangeError("AudioBuffer left-channel length does not match length");
  }
  const resample = resampler === "sinc"
    ? resampleSinc
    : resampler === "linear"
      ? resampleLinear
      : undefined;
  if (resample === undefined) throw new RangeError(`Unsupported audio resampler: ${String(resampler)}`);
  const left = resample(
    sourceLeft,
    source.sampleRate,
    outputSampleRate,
  );

  let right: Float32Array;
  if (source.numberOfChannels === 1) {
    // Do not alias the two channels: the model pipeline may process or replace
    // one channel independently of the other.
    right = left.slice();
  } else {
    const sourceRight = source.getChannelData(1);
    if (sourceRight.length !== source.length) {
      throw new RangeError("AudioBuffer right-channel length does not match length");
    }
    right = resample(
      sourceRight,
      source.sampleRate,
      outputSampleRate,
    );
  }

  return makeStereoPcm(outputSampleRate, left, right);
}

/**
 * Resample one planar channel with torchaudio 2.0.2's default sinc kernel:
 * lowpass_filter_width=6, rolloff=0.99, and a Hann window. The convolution is
 * zero-padded and trimmed to `ceil(inputLength * targetRate / sourceRate)`, as
 * in `torchaudio.functional.resample`.
 */
export function resampleSinc(
  source: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number,
): Float32Array {
  validateSampleRate(sourceSampleRate, "sourceSampleRate");
  validateSampleRate(targetSampleRate, "targetSampleRate");
  if (source.length === 0) return new Float32Array(0);
  if (sourceSampleRate === targetSampleRate) return source.slice();

  const commonDivisor = greatestCommonDivisor(sourceSampleRate, targetSampleRate);
  const sourceStride = sourceSampleRate / commonDivisor;
  const targetPhases = targetSampleRate / commonDivisor;
  const targetLengthNumerator = source.length * targetPhases;
  if (!Number.isSafeInteger(targetLengthNumerator)) {
    throw new RangeError("resampled audio length exceeds the safe integer range");
  }
  const targetLength = Math.ceil(targetLengthNumerator / sourceStride);
  const kernel = getSincResampleKernel(sourceStride, targetPhases);
  const output = new Float32Array(targetLength);

  for (let targetIndex = 0; targetIndex < targetLength; targetIndex += 1) {
    const sourceFrame = Math.floor(targetIndex / kernel.targetPhases);
    const phase = targetIndex - sourceFrame * kernel.targetPhases;
    const firstSourceIndex = sourceFrame * kernel.sourceStride - kernel.width;
    const firstTap = Math.max(
      kernel.firstNonzeroTaps[phase]!,
      -firstSourceIndex,
    );
    const lastTap = Math.min(
      kernel.lastNonzeroTaps[phase]!,
      source.length - firstSourceIndex,
    );
    const coefficientOffset = phase * kernel.tapCount;
    let sum = 0;
    for (let tap = firstTap; tap < lastTap; tap += 1) {
      sum += source[firstSourceIndex + tap]!
        * kernel.coefficients[coefficientOffset + tap]!;
    }
    output[targetIndex] = Math.fround(sum);
  }
  return output;
}

/** Resample both planar channels without changing their stereo ordering. */
export function resampleStereoSinc(
  source: StereoPcm,
  targetSampleRate: number = DICOSE_SAMPLE_RATE,
): StereoPcm {
  validateStereoPcm(source);
  validateSampleRate(targetSampleRate, "targetSampleRate");
  return makeStereoPcm(
    targetSampleRate,
    resampleSinc(source.left, source.sampleRate, targetSampleRate),
    resampleSinc(source.right, source.sampleRate, targetSampleRate),
  );
}

/** Resample stereo PCM and then enforce an externally declared frame count. */
export function resampleStereoSincToLength(
  source: StereoPcm,
  targetSampleRate: number,
  targetLength: number,
): StereoPcm {
  if (!Number.isSafeInteger(targetLength) || targetLength < 0) {
    throw new RangeError("targetLength must be a non-negative safe integer");
  }
  const resampled = resampleStereoSinc(source, targetSampleRate);
  if (resampled.length === targetLength) return resampled;
  const left = new Float32Array(targetLength);
  const right = new Float32Array(targetLength);
  left.set(resampled.left.subarray(0, targetLength));
  right.set(resampled.right.subarray(0, targetLength));
  return makeStereoPcm(targetSampleRate, left, right);
}

/**
 * Derive the complementary instrumental waveform from an input mixture and
 * its estimated vocals. Both operands must already share the same timeline.
 * The result is deliberately not clamped: valid subtraction can exceed unit
 * peak and `encodeStereoWav` will preserve it in an IEEE-float WAV.
 */
export function subtractStereoPcm(
  mixture: StereoPcm,
  vocals: StereoPcm,
): StereoPcm {
  validateStereoPcm(mixture);
  validateStereoPcm(vocals);
  if (mixture.sampleRate !== vocals.sampleRate || mixture.length !== vocals.length) {
    throw new RangeError("Stereo PCM subtraction requires matching timelines");
  }
  const left = new Float32Array(mixture.length);
  const right = new Float32Array(mixture.length);
  for (let index = 0; index < mixture.length; index += 1) {
    left[index] = Math.fround(mixture.left[index]! - vocals.left[index]!);
    right[index] = Math.fround(mixture.right[index]! - vocals.right[index]!);
  }
  return makeStereoPcm(mixture.sampleRate, left, right);
}

/**
 * Legacy deterministic-reference helper. Production audio paths use
 * `resampleSinc`. Positions beyond the final sample hold that final value.
 */
export function resampleLinear(
  source: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number,
): Float32Array {
  validateSampleRate(sourceSampleRate, "sourceSampleRate");
  validateSampleRate(targetSampleRate, "targetSampleRate");
  if (source.length === 0) return new Float32Array(0);
  if (sourceSampleRate === targetSampleRate) return source.slice();

  const targetLength = Math.max(
    1,
    Math.round((source.length * targetSampleRate) / sourceSampleRate),
  );
  if (!Number.isSafeInteger(targetLength)) {
    throw new RangeError("resampled audio length exceeds the safe integer range");
  }

  const output = new Float32Array(targetLength);
  const sourceLastIndex = source.length - 1;
  const sourceStep = sourceSampleRate / targetSampleRate;
  for (let targetIndex = 0; targetIndex < targetLength; targetIndex += 1) {
    const sourcePosition = targetIndex * sourceStep;
    const beforeIndex = Math.min(
      sourceLastIndex,
      Math.floor(sourcePosition),
    );
    const afterIndex = Math.min(sourceLastIndex, beforeIndex + 1);
    const fraction = sourcePosition - beforeIndex;
    const before = source[beforeIndex]!;
    const after = source[afterIndex]!;
    output[targetIndex] = Math.fround(before + (after - before) * fraction);
  }
  return output;
}

function getSincResampleKernel(
  sourceStride: number,
  targetPhases: number,
): SincResampleKernel {
  const cacheKey = `${sourceStride}:${targetPhases}`;
  const cached = sincResampleKernelCache.get(cacheKey);
  if (cached !== undefined) return cached;

  const baseFrequency = Math.min(sourceStride, targetPhases) * SINC_ROLLOFF;
  const width = Math.ceil(
    SINC_LOWPASS_FILTER_WIDTH * sourceStride / baseFrequency,
  );
  const tapCount = sourceStride + width * 2;
  const coefficientCount = targetPhases * tapCount;
  if (
    !Number.isSafeInteger(coefficientCount) ||
    coefficientCount > MAX_SINC_KERNEL_COEFFICIENTS
  ) {
    throw new RangeError(
      `resampling ratio ${sourceStride}:${targetPhases} requires an impractically large sinc kernel`,
    );
  }

  // This is the phase-major kernel constructed by torchaudio 2.0.2's
  // _get_sinc_resample_kernel. Build in float64 and cast each coefficient to
  // float32, matching its default dtype path.
  const coefficients = new Float32Array(coefficientCount);
  const firstNonzeroTaps = new Uint32Array(targetPhases);
  const lastNonzeroTaps = new Uint32Array(targetPhases);
  const scale = baseFrequency / sourceStride;
  for (let phase = 0; phase < targetPhases; phase += 1) {
    const phaseOffset = -phase / targetPhases;
    const coefficientOffset = phase * tapCount;
    let firstNonzeroTap = tapCount;
    let lastNonzeroTap = 0;
    for (let tap = 0; tap < tapCount; tap += 1) {
      const kernelIndex = tap - width;
      const unclamped = (
        phaseOffset + kernelIndex / sourceStride
      ) * baseFrequency;
      const time = Math.max(
        -SINC_LOWPASS_FILTER_WIDTH,
        Math.min(SINC_LOWPASS_FILTER_WIDTH, unclamped),
      );
      const window = Math.cos(
        time * Math.PI / SINC_LOWPASS_FILTER_WIDTH / 2,
      ) ** 2;
      const angle = time * Math.PI;
      const sinc = angle === 0 ? 1 : Math.sin(angle) / angle;
      const coefficient = Math.fround(
        sinc * window * scale,
      );
      coefficients[coefficientOffset + tap] = coefficient;
      if (coefficient !== 0) {
        firstNonzeroTap = Math.min(firstNonzeroTap, tap);
        lastNonzeroTap = tap + 1;
      }
    }
    firstNonzeroTaps[phase] = firstNonzeroTap;
    lastNonzeroTaps[phase] = lastNonzeroTap;
  }

  const kernel: SincResampleKernel = {
    sourceStride,
    targetPhases,
    width,
    tapCount,
    coefficients,
    firstNonzeroTaps,
    lastNonzeroTaps,
  };
  sincResampleKernelCache.set(cacheKey, kernel);
  return kernel;
}

function greatestCommonDivisor(left: number, right: number): number {
  while (right !== 0) {
    const remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

/** Interleave planar stereo as `[left0, right0, left1, right1, ...]`. */
export function interleaveStereo(source: StereoPcm): Float32Array {
  validateStereoPcm(source);
  const output = new Float32Array(source.length * 2);
  for (let sample = 0; sample < source.length; sample += 1) {
    const offset = sample * 2;
    output[offset] = source.left[sample]!;
    output[offset + 1] = source.right[sample]!;
  }
  return output;
}

export interface StftOptions {
  readonly nFft?: number;
  readonly hopLength?: number;
}

/**
 * One-sided complex spectrum produced by `torch.stft(..., return_complex=True,
 * center=True, pad_mode="reflect", window=torch.hann_window(nFft))`.
 */
export interface CenteredHannStft {
  readonly layout: "frame-frequency";
  readonly window: "hann-periodic";
  readonly center: true;
  readonly nFft: number;
  readonly hopLength: number;
  readonly binCount: number;
  readonly frameCount: number;
  /** Original, unpadded source length. Used as ISTFT's exact default length. */
  readonly sourceLength: number;
  readonly real: Float32Array;
  readonly imag: Float32Array;
}

export interface IstftOptions {
  /** Mirrors torch.istft's `length`; defaults to the STFT source length. */
  readonly length?: number;
}

interface FftPlan {
  readonly size: number;
  readonly bitReversed: Uint32Array;
  readonly twiddleReal: Float32Array;
  readonly twiddleForwardImaginary: Float32Array;
}

const FFT_PLANS = new Map<number, FftPlan>();
const HANN_WINDOWS = new Map<number, Float32Array>();

/** Return a caller-owned periodic Hann window matching torch.hann_window's default. */
export function createPeriodicHannWindow(
  nFft: number = DICOSE_STFT_N_FFT,
): Float32Array {
  validateFftSize(nFft);
  return new Float32Array(periodicHannWindow(nFft));
}

/**
 * Compute a centered, reflection-padded, one-sided Hann STFT. The output is
 * contiguous frame-major complex data, which is convenient for one upload per
 * channel before the WebGPU model begins.
 */
export function centeredHannStft(
  samples: Float32Array,
  options: StftOptions = {},
): CenteredHannStft {
  const { nFft, hopLength } = resolveStftOptions(options);
  const padding = nFft / 2;
  if (samples.length <= padding) {
    throw new RangeError(
      `torch-compatible reflection padding requires more than ${padding} samples`,
    );
  }

  const frameCount = Math.floor(samples.length / hopLength) + 1;
  const binCount = padding + 1;
  const valueCount = frameCount * binCount;
  if (!Number.isSafeInteger(valueCount)) {
    throw new RangeError("STFT output length exceeds the safe integer range");
  }

  const real = new Float32Array(valueCount);
  const imag = new Float32Array(valueCount);
  const fftReal = new Float32Array(nFft);
  const fftImag = new Float32Array(nFft);
  const window = periodicHannWindow(nFft);
  const plan = fftPlan(nFft);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const frameStart = frame * hopLength - padding;
    for (let sample = 0; sample < nFft; sample += 1) {
      const sourceIndex = reflectIndex(frameStart + sample, samples.length);
      fftReal[sample] = Math.fround(
        samples[sourceIndex]! * window[sample]!,
      );
      fftImag[sample] = 0;
    }
    fftInPlace(fftReal, fftImag, plan, false);

    const outputOffset = frame * binCount;
    for (let bin = 0; bin < binCount; bin += 1) {
      real[outputOffset + bin] = fftReal[bin]!;
      imag[outputOffset + bin] = fftImag[bin]!;
    }
  }

  return {
    layout: "frame-frequency",
    window: "hann-periodic",
    center: true,
    nFft,
    hopLength,
    binCount,
    frameCount,
    sourceLength: samples.length,
    real,
    imag,
  };
}

/**
 * Invert a centered Hann STFT with window-squared overlap-add normalization.
 * Passing an explicit length exactly follows `torch.istft(..., length=...)`:
 * it removes the left center padding and then takes that many output samples.
 */
export function centeredHannIstft(
  spectrum: CenteredHannStft,
  options: IstftOptions = {},
): Float32Array {
  validateSpectrum(spectrum);
  const requestedLength = options.length ?? spectrum.sourceLength;
  if (!Number.isSafeInteger(requestedLength) || requestedLength < 0) {
    throw new RangeError("ISTFT length must be a non-negative safe integer");
  }

  const { nFft, hopLength, frameCount, binCount } = spectrum;
  const padding = nFft / 2;
  const overlapLength = nFft + (frameCount - 1) * hopLength;
  if (!Number.isSafeInteger(overlapLength)) {
    throw new RangeError("ISTFT overlap-add length exceeds the safe integer range");
  }

  // Float64 accumulation keeps normalization stable even for long source audio;
  // values are rounded once at the public Float32 output boundary.
  const overlapAdd = new Float64Array(overlapLength);
  const envelope = new Float64Array(overlapLength);
  const fftReal = new Float32Array(nFft);
  const fftImag = new Float32Array(nFft);
  const window = periodicHannWindow(nFft);
  const plan = fftPlan(nFft);
  const nyquist = nFft / 2;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const inputOffset = frame * binCount;
    for (let bin = 0; bin < binCount; bin += 1) {
      fftReal[bin] = spectrum.real[inputOffset + bin]!;
      // rFFT endpoint imaginary components do not represent a real signal and
      // are ignored by torch.fft.irfft as well.
      fftImag[bin] = bin === 0 || bin === nyquist
        ? 0
        : spectrum.imag[inputOffset + bin]!;
    }
    for (let bin = binCount; bin < nFft; bin += 1) {
      const mirroredBin = nFft - bin;
      fftReal[bin] = fftReal[mirroredBin]!;
      fftImag[bin] = -fftImag[mirroredBin]!;
    }

    fftInPlace(fftReal, fftImag, plan, true);
    const outputOffset = frame * hopLength;
    for (let sample = 0; sample < nFft; sample += 1) {
      const weighted = fftReal[sample]! * window[sample]!;
      const targetIndex = outputOffset + sample;
      overlapAdd[targetIndex] = overlapAdd[targetIndex]! + weighted;
      const windowValue = window[sample]!;
      envelope[targetIndex] = envelope[targetIndex]! + windowValue * windowValue;
    }
  }

  const output = new Float32Array(requestedLength);
  for (let sample = 0; sample < requestedLength; sample += 1) {
    const overlapIndex = padding + sample;
    if (overlapIndex >= overlapLength) continue;
    const normalization = envelope[overlapIndex]!;
    if (normalization > 1e-11) {
      output[sample] = Math.fround(overlapAdd[overlapIndex]! / normalization);
    }
  }
  return output;
}

/** Convert one IEEE-754 binary32 number to round-to-nearest-even binary16 bits. */
export function float32ToFloat16Bits(value: number): number {
  FLOAT32_CONVERSION_VIEW[0] = value;
  const bits = UINT32_CONVERSION_VIEW[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7f_ffff;

  if (exponent === 0xff) {
    if (fraction === 0) return sign | 0x7c00;
    const payload = fraction >>> 13;
    return sign | 0x7c00 | (payload === 0 ? 1 : payload);
  }

  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    // All binary32 subnormals are below binary16's representable subnormal
    // range. The leading bit is present for every normal binary32 value here.
    if (halfExponent < -10) return sign;
    const normalizedFraction = fraction | 0x80_0000;
    const shift = 14 - halfExponent;
    let halfFraction = normalizedFraction >>> shift;
    const remainder = normalizedFraction & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (
      remainder > halfway ||
      (remainder === halfway && (halfFraction & 1) !== 0)
    ) {
      halfFraction += 1;
    }
    return sign | halfFraction;
  }

  let halfFraction = fraction >>> 13;
  const remainder = fraction & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfFraction & 1) !== 0)) {
    halfFraction += 1;
    if (halfFraction === 0x400) {
      halfFraction = 0;
      halfExponent += 1;
      if (halfExponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (halfExponent << 10) | halfFraction;
}

/** Decode IEEE-754 binary16 storage bits into the exactly representable binary32 value. */
export function float16BitsToFloat32(bits: number): number {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) {
    throw new RangeError("float16 bits must be an unsigned 16-bit integer");
  }
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    if (fraction === 0) return sign < 0 ? -0 : 0;
    return Math.fround(sign * fraction * 2 ** -24);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return Math.fround(sign * (1 + fraction / 1024) * 2 ** (exponent - 15));
}

/** Pack binary32 tensor values for `array<f16>` / `u32` WebGPU uploads. */
export function packFloat16(values: Float32Array): Uint16Array {
  const packed = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    packed[index] = float32ToFloat16Bits(values[index]!);
  }
  return packed;
}

/** Expand binary16 storage into a binary32 tensor. */
export function unpackFloat16(values: Uint16Array): Float32Array {
  const unpacked = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    unpacked[index] = float16BitsToFloat32(values[index]!);
  }
  return unpacked;
}

export type GaussianSeed = number | bigint;

/**
 * Tiny deterministic random stream for initial CD noise. It intentionally has
 * no dependency on browser crypto or Math.random, so worker, main-thread, and
 * Node test results are reproducible for a given seed.
 */
export class SeededGaussian {
  private state: number;
  private spare: number | undefined;

  constructor(seed: GaussianSeed) {
    this.state = normalizeGaussianSeed(seed);
  }

  next(): number {
    if (this.spare !== undefined) {
      const output = this.spare;
      this.spare = undefined;
      return output;
    }

    const radius = Math.sqrt(-2 * Math.log(this.nextOpenUnit()));
    const angle = TWO_PI * this.nextOpenUnit();
    this.spare = Math.fround(radius * Math.sin(angle));
    return Math.fround(radius * Math.cos(angle));
  }

  private nextOpenUnit(): number {
    return (this.nextUint32() + 0.5) * UINT32_UNIT;
  }

  private nextUint32(): number {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return this.state;
  }
}

/** Fill a caller-owned Float32 tensor in contiguous row-major order. */
export function fillSeededGaussian(
  output: Float32Array,
  seed: GaussianSeed,
): void {
  const random = new SeededGaussian(seed);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = random.next();
  }
}

/** Allocate deterministic standard-normal noise for a model tensor. */
export function seededGaussianNoise(
  length: number,
  seed: GaussianSeed,
): Float32Array {
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError("Gaussian noise length must be a non-negative safe integer");
  }
  const output = new Float32Array(length);
  fillSeededGaussian(output, seed);
  return output;
}

/** Encode one or more equal-length planar channels as little-endian PCM16 WAV. */
export function encodePcm16Wav(
  channels: readonly Float32Array[],
  sampleRate: number = DICOSE_SAMPLE_RATE,
): ArrayBuffer {
  return encodeWav(channels, sampleRate, "pcm16");
}

/** Encode one or more equal-length planar channels as IEEE-float WAV. */
export function encodeFloat32Wav(
  channels: readonly Float32Array[],
  sampleRate: number = DICOSE_SAMPLE_RATE,
): ArrayBuffer {
  return encodeWav(channels, sampleRate, "float32");
}

function encodeWav(
  channels: readonly Float32Array[],
  sampleRate: number,
  encoding: "pcm16" | "float32",
): ArrayBuffer {
  validateSampleRate(sampleRate, "sampleRate");
  if (channels.length < 1 || channels.length > 0xffff) {
    throw new RangeError("WAV must contain between one and 65,535 channels");
  }
  const frameCount = channels[0]?.length;
  if (frameCount === undefined) throw new Error("WAV channels are unexpectedly empty");
  for (let channel = 0; channel < channels.length; channel += 1) {
    const values = channels[channel]!;
    if (values.length !== frameCount) {
      throw new RangeError("all WAV channels must have the same frame count");
    }
  }

  const bytesPerSample = encoding === "pcm16" ? 2 : 4;
  const blockAlign = channels.length * bytesPerSample;
  const dataBytes = frameCount * blockAlign;
  const byteRate = sampleRate * blockAlign;
  if (
    !Number.isSafeInteger(dataBytes) ||
    dataBytes > 0xffff_ffff ||
    !Number.isSafeInteger(byteRate) ||
    byteRate > 0xffff_ffff
  ) {
    throw new RangeError("WAV is too large for the RIFF container");
  }

  const bytes = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(bytes);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, encoding === "pcm16" ? 1 : 3, true);
  view.setUint16(22, channels.length, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let byteOffset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channels.length; channel += 1) {
      const value = channels[channel]![frame]!;
      if (!Number.isFinite(value)) {
        throw new RangeError("WAV samples must be finite");
      }
      if (encoding === "float32") {
        view.setFloat32(byteOffset, value, true);
      } else {
        const clamped = Math.min(1, Math.max(-1, value));
        const integer = clamped < 0
          ? Math.round(clamped * 32_768)
          : Math.round(clamped * 32_767);
        view.setInt16(byteOffset, integer, true);
      }
      byteOffset += bytesPerSample;
    }
  }
  return bytes;
}

/** Encode a model-ready stereo PCM object as an audio/wav Blob. */
export function encodeStereoPcm16Wav(source: StereoPcm): Blob {
  validateStereoPcm(source);
  return new Blob([encodePcm16Wav(source.channels, source.sampleRate)], {
    type: "audio/wav",
  });
}

/** Preserve over-range model output using the same peak rule as upstream. */
export function encodeStereoWav(source: StereoPcm): Blob {
  validateStereoPcm(source);
  let requiresFloat = false;
  for (const channel of source.channels) {
    for (let index = 0; index < channel.length; index += 1) {
      const value = channel[index]!;
      if (!Number.isFinite(value)) throw new RangeError("WAV samples must be finite");
      requiresFloat ||= Math.abs(value) > 1;
    }
  }
  const encoded = requiresFloat
    ? encodeFloat32Wav(source.channels, source.sampleRate)
    : encodePcm16Wav(source.channels, source.sampleRate);
  return new Blob([encoded], { type: "audio/wav" });
}

const FLOAT32_CONVERSION_BUFFER = new ArrayBuffer(4);
const FLOAT32_CONVERSION_VIEW = new Float32Array(FLOAT32_CONVERSION_BUFFER);
const UINT32_CONVERSION_VIEW = new Uint32Array(FLOAT32_CONVERSION_BUFFER);

function makeStereoPcm(
  sampleRate: number,
  left: Float32Array,
  right: Float32Array,
): StereoPcm {
  if (left.length !== right.length) {
    throw new RangeError("stereo channels must have equal lengths");
  }
  return {
    sampleRate,
    length: left.length,
    left,
    right,
    channels: [left, right],
  };
}

function validateStereoPcm(source: StereoPcm): void {
  validateSampleRate(source.sampleRate, "source.sampleRate");
  if (!Number.isSafeInteger(source.length) || source.length < 0) {
    throw new RangeError("stereo PCM length must be a non-negative safe integer");
  }
  if (
    source.left.length !== source.length ||
    source.right.length !== source.length ||
    source.channels[0] !== source.left ||
    source.channels[1] !== source.right
  ) {
    throw new RangeError("stereo PCM channel metadata is inconsistent");
  }
}

function validateSampleRate(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffff_ffff) {
    throw new RangeError(`${name} must be a positive integer sample rate`);
  }
}

function resolveStftOptions(options: StftOptions): {
  readonly nFft: number;
  readonly hopLength: number;
} {
  const nFft = options.nFft ?? DICOSE_STFT_N_FFT;
  const hopLength = options.hopLength ?? DICOSE_STFT_HOP_LENGTH;
  validateFftSize(nFft);
  if (!Number.isSafeInteger(hopLength) || hopLength < 1) {
    throw new RangeError("hopLength must be a positive safe integer");
  }
  return { nFft, hopLength };
}

function validateFftSize(nFft: number): void {
  if (
    !Number.isSafeInteger(nFft) ||
    nFft < 2 ||
    nFft > 1 << 20 ||
    nFft % 2 !== 0 ||
    (nFft & (nFft - 1)) !== 0
  ) {
    throw new RangeError("nFft must be an even power of two no greater than 1,048,576");
  }
}

function validateSpectrum(spectrum: CenteredHannStft): void {
  if (
    spectrum.layout !== "frame-frequency" ||
    spectrum.window !== "hann-periodic" ||
    spectrum.center !== true
  ) {
    throw new TypeError("ISTFT requires a centered periodic-Hann STFT");
  }
  validateFftSize(spectrum.nFft);
  if (!Number.isSafeInteger(spectrum.hopLength) || spectrum.hopLength < 1) {
    throw new RangeError("STFT hopLength must be a positive safe integer");
  }
  if (!Number.isSafeInteger(spectrum.frameCount) || spectrum.frameCount < 1) {
    throw new RangeError("STFT frameCount must be a positive safe integer");
  }
  if (!Number.isSafeInteger(spectrum.sourceLength) || spectrum.sourceLength < 0) {
    throw new RangeError("STFT sourceLength must be a non-negative safe integer");
  }
  const expectedBinCount = spectrum.nFft / 2 + 1;
  const expectedValueCount = spectrum.frameCount * expectedBinCount;
  if (
    spectrum.binCount !== expectedBinCount ||
    !Number.isSafeInteger(expectedValueCount) ||
    spectrum.real.length !== expectedValueCount ||
    spectrum.imag.length !== expectedValueCount
  ) {
    throw new RangeError("STFT complex storage shape is inconsistent");
  }
}

function periodicHannWindow(nFft: number): Float32Array {
  let cached = HANN_WINDOWS.get(nFft);
  if (cached !== undefined) return cached;
  cached = new Float32Array(nFft);
  for (let index = 0; index < nFft; index += 1) {
    cached[index] = Math.fround(
      0.5 - 0.5 * Math.cos((TWO_PI * index) / nFft),
    );
  }
  HANN_WINDOWS.set(nFft, cached);
  return cached;
}

function reflectIndex(index: number, length: number): number {
  if (index >= 0 && index < length) return index;
  const period = length * 2 - 2;
  let reflected = index % period;
  if (reflected < 0) reflected += period;
  return reflected < length ? reflected : period - reflected;
}

function fftPlan(size: number): FftPlan {
  let plan = FFT_PLANS.get(size);
  if (plan !== undefined) return plan;

  const bitCount = Math.round(Math.log2(size));
  const bitReversed = new Uint32Array(size);
  for (let value = 0; value < size; value += 1) {
    let remainder = value;
    let reversed = 0;
    for (let bit = 0; bit < bitCount; bit += 1) {
      reversed = (reversed << 1) | (remainder & 1);
      remainder >>>= 1;
    }
    bitReversed[value] = reversed;
  }

  const twiddleReal = new Float32Array(size / 2);
  const twiddleForwardImaginary = new Float32Array(size / 2);
  for (let index = 0; index < size / 2; index += 1) {
    const phase = (TWO_PI * index) / size;
    twiddleReal[index] = Math.fround(Math.cos(phase));
    twiddleForwardImaginary[index] = Math.fround(-Math.sin(phase));
  }
  plan = { size, bitReversed, twiddleReal, twiddleForwardImaginary };
  FFT_PLANS.set(size, plan);
  return plan;
}

function fftInPlace(
  real: Float32Array,
  imag: Float32Array,
  plan: FftPlan,
  inverse: boolean,
): void {
  const { size, bitReversed, twiddleReal, twiddleForwardImaginary } = plan;
  for (let index = 0; index < size; index += 1) {
    const mirrored = bitReversed[index]!;
    if (index >= mirrored) continue;
    const realValue = real[index]!;
    real[index] = real[mirrored]!;
    real[mirrored] = realValue;
    const imagValue = imag[index]!;
    imag[index] = imag[mirrored]!;
    imag[mirrored] = imagValue;
  }

  for (let butterflySize = 2; butterflySize <= size; butterflySize *= 2) {
    const halfSize = butterflySize / 2;
    const twiddleStride = size / butterflySize;
    for (let start = 0; start < size; start += butterflySize) {
      for (let offset = 0; offset < halfSize; offset += 1) {
        const twiddleIndex = offset * twiddleStride;
        const twiddleR = twiddleReal[twiddleIndex]!;
        const forwardImaginary = twiddleForwardImaginary[twiddleIndex]!;
        const twiddleI = inverse ? -forwardImaginary : forwardImaginary;
        const upperIndex = start + offset;
        const lowerIndex = upperIndex + halfSize;
        const lowerR = real[lowerIndex]!;
        const lowerI = imag[lowerIndex]!;
        const transformedR = twiddleR * lowerR - twiddleI * lowerI;
        const transformedI = twiddleR * lowerI + twiddleI * lowerR;
        const upperR = real[upperIndex]!;
        const upperI = imag[upperIndex]!;
        real[upperIndex] = upperR + transformedR;
        imag[upperIndex] = upperI + transformedI;
        real[lowerIndex] = upperR - transformedR;
        imag[lowerIndex] = upperI - transformedI;
      }
    }
  }

  if (inverse) {
    for (let index = 0; index < size; index += 1) {
      real[index] = real[index]! / size;
      imag[index] = imag[index]! / size;
    }
  }
}

function normalizeGaussianSeed(seed: GaussianSeed): number {
  let low: number;
  let high: number;
  if (typeof seed === "bigint") {
    if (seed < 0n || seed > 0xffff_ffff_ffff_ffffn) {
      throw new RangeError("Gaussian bigint seed must fit in unsigned 64 bits");
    }
    low = Number(seed & 0xffff_ffffn);
    high = Number(seed >> 32n);
  } else {
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new RangeError("Gaussian numeric seed must be a non-negative safe integer");
    }
    low = seed >>> 0;
    high = Math.floor(seed / 0x1_0000_0000) >>> 0;
  }

  let state = (low ^ Math.imul(high, 0x9e37_79b9)) >>> 0;
  // xorshift32 has a forbidden all-zero state.
  if (state === 0) state = 0x6d2b_79f5;
  return state;
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

/** Return a native RIFF/WAVE sample rate without decoding PCM payload bytes. */
function wavSampleRateFromRiff(encoded: ArrayBuffer): number | undefined {
  if (encoded.byteLength < 12) return undefined;
  const view = new DataView(encoded);
  if (
    readAscii(view, 0, 4) !== "RIFF" ||
    readAscii(view, 8, 4) !== "WAVE"
  ) {
    return undefined;
  }

  let chunkOffset = 12;
  while (chunkOffset + 8 <= view.byteLength) {
    const chunkId = readAscii(view, chunkOffset, 4);
    const chunkLength = view.getUint32(chunkOffset + 4, true);
    const dataOffset = chunkOffset + 8;
    if (dataOffset + chunkLength > view.byteLength) return undefined;
    if (chunkId === "fmt " && chunkLength >= 16) {
      const sampleRate = view.getUint32(dataOffset + 4, true);
      return Number.isSafeInteger(sampleRate) && sampleRate > 0
        ? sampleRate
        : undefined;
    }
    chunkOffset = dataOffset + chunkLength + (chunkLength & 1);
  }
  return undefined;
}

function readAscii(
  view: DataView,
  offset: number,
  length: number,
): string {
  let value = "";
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(view.getUint8(offset + index));
  }
  return value;
}
