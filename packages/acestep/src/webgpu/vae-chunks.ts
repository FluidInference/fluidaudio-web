import {
  checkedAceProduct,
  requirePositiveSafeInteger,
} from "./kernels/correctness-utils.js";
import {
  ACE_OOBLECK_DECODER_CONFIG,
  type AceVaeDecoderConfig,
  type AceVaeDecoderGraphPlan,
  planAceVaeDecoder,
} from "./vae-decoder.js";

export const ACE_VAE_DEFAULT_CHUNK_FRAMES = 256;
export const ACE_VAE_DEFAULT_OVERLAP_FRAMES = 64;
export const ACE_VAE_NEAR_SILENCE_PEAK = 1e-6;
export const ACE_DEFAULT_NORMALIZATION_DB = -1;

export interface AceVaeChunkOptions {
  readonly chunkFrames?: number;
  readonly overlapFrames?: number;
  readonly config?: AceVaeDecoderConfig;
}

export interface AceVaeDecodeWindow {
  readonly index: number;
  readonly coreStartLatentFrame: number;
  readonly coreEndLatentFrame: number;
  readonly windowStartLatentFrame: number;
  readonly windowEndLatentFrame: number;
  readonly latentWindowFrames: number;
  readonly discardPrefixLatentFrames: number;
  readonly discardSuffixLatentFrames: number;
  readonly discardPrefixAudioFrames: number;
  readonly discardSuffixAudioFrames: number;
  readonly outputStartAudioFrame: number;
  readonly outputAudioFrames: number;
  readonly decodedAudioFrames: number;
}

export interface AceVaeChunkedDecodePlan {
  readonly batch: 1;
  readonly latentFrames: number;
  readonly chunkFrames: number;
  readonly overlapFrames: number;
  readonly strideFrames: number;
  readonly hopLength: number;
  readonly sampleRateHz: number;
  readonly audioChannels: number;
  readonly outputAudioFrames: number;
  readonly outputInterleavedElements: number;
  readonly outputFloat32Bytes: number;
  readonly maximumWindowFrames: number;
  readonly maximumDecodedInterleavedElements: number;
  readonly maximumDecodedFloat32Bytes: number;
  readonly direct: boolean;
  readonly windows: readonly AceVaeDecodeWindow[];
  readonly decoderWorkspacePlan: AceVaeDecoderGraphPlan;
}

export interface AceVaeChunkBackend {
  /**
   * Return one raw FP32 NLC/interleaved stereo window. The caller releases it
   * after `sink.writeCore` resolves.
   */
  decodeWindow(window: AceVaeDecodeWindow): Promise<Float32Array>;
}

export interface AceVaeRawChunkSink {
  /** Consume the view before the promise resolves; it aliases the window. */
  writeCore(
    window: AceVaeDecodeWindow,
    interleavedCore: Float32Array,
  ): Promise<void>;
}

export interface AceVaeRawStreamStats {
  readonly peak: number;
  readonly finiteSamples: number;
  readonly outputInterleavedElements: number;
  readonly windowsDecoded: number;
  readonly cooperativeIdleMs: number;
}

export interface AceVaeRawStreamOptions {
  readonly signal?: AbortSignal;
  /** @internal Test seam; production uses a real one-millisecond timer. */
  readonly yieldQueueIdle?: () => Promise<void>;
  /** @internal Capture-only decoder/scan/raw-write attribution. */
  readonly onTrace?: (trace: AceVaeRawStreamTrace) => void;
  /** @internal Deterministic capture clock. */
  readonly now?: () => number;
}

export interface AceVaeRawStreamTrace {
  readonly schema: "ace-vae-raw-stream-capture-v1";
  readonly startedAtMs: number;
  readonly completedAtMs: number;
  readonly wallMs: number;
  readonly windowCount: number;
  readonly outputElements: number;
  readonly outputBytes: number;
  readonly maximumDecodedWindowBytes: number;
  readonly timing: Readonly<{
    readonly decodeAndReadbackMs: number;
    readonly finitePeakScanMs: number;
    readonly rawOpfsWriteMs: number;
    readonly queueEmptyGapMs: number;
  }>;
}

export interface AceVaePostprocessPlan {
  readonly rawPeak: number;
  /** FP32 divisor used by upstream `pred_wavs / peak.clamp(min=1.0)`. */
  readonly safetyDivisor: number;
  readonly peakAfterSafetyScale: number;
  readonly targetDb: number;
  readonly targetAmplitude: number;
  /** Upstream `normalize_audio`; one for peaks below the silence threshold. */
  readonly normalizationScale: number;
  readonly totalScale: number;
  readonly finalPeak: number;
  readonly nearSilence: boolean;
}

/**
 * Mirror the pinned MLX 256/64 overlap-discard decoder geometry exactly.
 *
 * There is no crossfade. Each decoded window drops the audio corresponding to
 * its contextual latent overlap, then writes its core at `coreStart * 1920`.
 */
export function planAceVaeChunkedDecode(
  latentFrames: number,
  options: AceVaeChunkOptions = {},
): AceVaeChunkedDecodePlan {
  requirePositiveSafeInteger(latentFrames, "ACE VAE chunk latent frames");
  const chunkFrames = options.chunkFrames ?? ACE_VAE_DEFAULT_CHUNK_FRAMES;
  const overlapFrames =
    options.overlapFrames ?? ACE_VAE_DEFAULT_OVERLAP_FRAMES;
  const config = options.config ?? ACE_OOBLECK_DECODER_CONFIG;
  requirePositiveSafeInteger(chunkFrames, "ACE VAE chunk size");
  if (!Number.isSafeInteger(overlapFrames) || overlapFrames < 0) {
    throw new RangeError(
      "ACE VAE overlap must be a non-negative safe integer",
    );
  }
  const strideFrames = chunkFrames - 2 * overlapFrames;
  if (strideFrames <= 0) {
    throw new RangeError(
      "ACE VAE chunk size must exceed twice the overlap",
    );
  }
  const hopLength = config.downsamplingRatios.reduce(
    (product, ratio) => product * ratio,
    1,
  );
  requirePositiveSafeInteger(hopLength, "ACE VAE hop length");
  const direct = latentFrames <= chunkFrames;
  const steps = direct ? 1 : Math.ceil(latentFrames / strideFrames);
  const windows: AceVaeDecodeWindow[] = [];
  for (let index = 0; index < steps; index += 1) {
    const coreStartLatentFrame = direct ? 0 : index * strideFrames;
    const coreEndLatentFrame = direct
      ? latentFrames
      : Math.min(coreStartLatentFrame + strideFrames, latentFrames);
    if (coreStartLatentFrame >= coreEndLatentFrame) {
      throw new Error("ACE VAE chunk planner produced an empty core");
    }
    const windowStartLatentFrame = direct
      ? 0
      : Math.max(0, coreStartLatentFrame - overlapFrames);
    const windowEndLatentFrame = direct
      ? latentFrames
      : Math.min(latentFrames, coreEndLatentFrame + overlapFrames);
    const latentWindowFrames =
      windowEndLatentFrame - windowStartLatentFrame;
    const discardPrefixLatentFrames =
      coreStartLatentFrame - windowStartLatentFrame;
    const discardSuffixLatentFrames =
      windowEndLatentFrame - coreEndLatentFrame;
    windows.push(Object.freeze({
      index,
      coreStartLatentFrame,
      coreEndLatentFrame,
      windowStartLatentFrame,
      windowEndLatentFrame,
      latentWindowFrames,
      discardPrefixLatentFrames,
      discardSuffixLatentFrames,
      discardPrefixAudioFrames: discardPrefixLatentFrames * hopLength,
      discardSuffixAudioFrames: discardSuffixLatentFrames * hopLength,
      outputStartAudioFrame: coreStartLatentFrame * hopLength,
      outputAudioFrames:
        (coreEndLatentFrame - coreStartLatentFrame) * hopLength,
      decodedAudioFrames: latentWindowFrames * hopLength,
    }));
  }
  const maximumWindowFrames = Math.max(
    ...windows.map((window) => window.latentWindowFrames),
  );
  const outputAudioFrames = latentFrames * hopLength;
  const outputInterleavedElements = checkedAceProduct(
    [outputAudioFrames, config.audioChannels],
    "ACE VAE output samples",
  );
  const maximumDecodedInterleavedElements = checkedAceProduct(
    [maximumWindowFrames, hopLength, config.audioChannels],
    "ACE VAE maximum decoded window",
  );
  const plannedOutputFrames = windows.reduce(
    (total, window) => total + window.outputAudioFrames,
    0,
  );
  if (plannedOutputFrames !== outputAudioFrames) {
    throw new Error("ACE VAE chunk cores do not cover the output exactly once");
  }
  return Object.freeze({
    batch: 1,
    latentFrames,
    chunkFrames,
    overlapFrames,
    strideFrames,
    hopLength,
    sampleRateHz: config.sampleRateHz,
    audioChannels: config.audioChannels,
    outputAudioFrames,
    outputInterleavedElements,
    outputFloat32Bytes:
      outputInterleavedElements * Float32Array.BYTES_PER_ELEMENT,
    maximumWindowFrames,
    maximumDecodedInterleavedElements,
    maximumDecodedFloat32Bytes:
      maximumDecodedInterleavedElements * Float32Array.BYTES_PER_ELEMENT,
    direct,
    windows: Object.freeze(windows),
    decoderWorkspacePlan: planAceVaeDecoder(maximumWindowFrames, config),
  });
}

/**
 * Sequentially decode and offload raw chunk cores with bounded JS storage.
 * Discarded overlap never contributes to peak measurement or the output.
 */
export async function streamAceVaeRawChunks(
  plan: AceVaeChunkedDecodePlan,
  backend: AceVaeChunkBackend,
  sink: AceVaeRawChunkSink,
  options: AceVaeRawStreamOptions = {},
): Promise<AceVaeRawStreamStats> {
  const capture = options.onTrace !== undefined;
  const now = options.now ?? defaultNow;
  const startedAtMs = capture ? now() : 0;
  let decodeAndReadbackMs = 0;
  let finitePeakScanMs = 0;
  let rawOpfsWriteMs = 0;
  let queueEmptyGapMs = 0;
  let maximumDecodedWindowBytes = 0;
  let peak = 0;
  let finiteSamples = 0;
  let writtenElements = 0;
  let cooperativeIdleMs = 0;
  for (const window of plan.windows) {
    options.signal?.throwIfAborted();
    const decodeStartedAt = capture ? now() : 0;
    const decoded = await backend.decodeWindow(window);
    if (capture) {
      decodeAndReadbackMs += nonnegativeElapsed(now(), decodeStartedAt);
      maximumDecodedWindowBytes = Math.max(
        maximumDecodedWindowBytes,
        decoded.byteLength,
      );
    }
    const expectedElements =
      window.decodedAudioFrames * plan.audioChannels;
    if (decoded.length !== expectedElements) {
      throw new RangeError(
        `ACE VAE window ${window.index} returned ${decoded.length} samples; expected ${expectedElements}`,
      );
    }
    const start = window.discardPrefixAudioFrames * plan.audioChannels;
    const end =
      decoded.length - window.discardSuffixAudioFrames * plan.audioChannels;
    const core = decoded.subarray(start, end);
    const expectedCoreElements =
      window.outputAudioFrames * plan.audioChannels;
    if (core.length !== expectedCoreElements) {
      throw new Error(
        `ACE VAE window ${window.index} trim produced ${core.length} samples; expected ${expectedCoreElements}`,
      );
    }
    const scanStartedAt = capture ? now() : 0;
    for (const value of core) {
      if (!Number.isFinite(value)) {
        throw new Error(
          `ACE VAE window ${window.index} produced a non-finite raw sample`,
        );
      }
      peak = Math.max(peak, Math.abs(value));
      finiteSamples += 1;
    }
    if (capture) finitePeakScanMs += nonnegativeElapsed(now(), scanStartedAt);
    const writeStartedAt = capture ? now() : 0;
    await sink.writeCore(window, core);
    if (capture) rawOpfsWriteMs += nonnegativeElapsed(now(), writeStartedAt);
    writtenElements += core.length;
    if (window.index + 1 < plan.windows.length) {
      // The preceding window's readback has drained. Make the queue-empty
      // interval explicit instead of assuming CPU scanning/OPFS happened to
      // take long enough before the next window uploads and submits work.
      const idleStartedAt = capture ? now() : 0;
      await (options.yieldQueueIdle ?? yieldAceVaeQueueIdle)();
      if (capture) queueEmptyGapMs += nonnegativeElapsed(now(), idleStartedAt);
      cooperativeIdleMs += 1;
      options.signal?.throwIfAborted();
    }
  }
  if (writtenElements !== plan.outputInterleavedElements) {
    throw new Error(
      `ACE VAE stream wrote ${writtenElements} samples; expected ${plan.outputInterleavedElements}`,
    );
  }
  if (capture) {
    const completedAtMs = now();
    try {
      options.onTrace?.(Object.freeze({
        schema: "ace-vae-raw-stream-capture-v1",
        startedAtMs,
        completedAtMs,
        wallMs: nonnegativeElapsed(completedAtMs, startedAtMs),
        windowCount: plan.windows.length,
        outputElements: writtenElements,
        outputBytes: writtenElements * Float32Array.BYTES_PER_ELEMENT,
        maximumDecodedWindowBytes,
        timing: Object.freeze({
          decodeAndReadbackMs,
          finitePeakScanMs,
          rawOpfsWriteMs,
          queueEmptyGapMs,
        }),
      }));
    } catch {
      // Capture is observational and cannot change raw stream publication.
    }
  }
  return Object.freeze({
    peak,
    finiteSamples,
    outputInterleavedElements: writtenElements,
    windowsDecoded: plan.windows.length,
    cooperativeIdleMs,
  });
}

async function yieldAceVaeQueueIdle(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 1));
}

function defaultNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function nonnegativeElapsed(completedAtMs: number, startedAtMs: number): number {
  return Math.max(0, completedAtMs - startedAtMs);
}

/**
 * Reproduce the two upstream whole-waveform operations without clipping.
 * A bounded implementation can capture raw FP32 samples in OPFS, measure `rawPeak`, and
 * apply the safety division and normalization multiplication as two FP32
 * operations during a second streaming pass. `totalScale` is diagnostic only;
 * reciprocal-multiply is not equivalent to the upstream elementwise division.
 */
export function deriveAceVaePostprocessPlan(
  rawPeak: number,
  targetDb = ACE_DEFAULT_NORMALIZATION_DB,
): AceVaePostprocessPlan {
  if (!Number.isFinite(rawPeak) || rawPeak < 0) {
    throw new RangeError("ACE VAE raw peak must be finite and non-negative");
  }
  const rawPeakF32 = Math.fround(rawPeak);
  if (!Number.isFinite(targetDb) || targetDb > 0) {
    throw new RangeError("ACE VAE normalization target must be finite and <= 0 dB");
  }
  const safetyDivisor = rawPeakF32 > 1 ? rawPeakF32 : Math.fround(1);
  // The sample that produced the finite FP32 peak divides by itself exactly.
  const peakAfterSafetyScale = rawPeakF32 > 1 ? 1 : rawPeakF32;
  const nearSilence = peakAfterSafetyScale < ACE_VAE_NEAR_SILENCE_PEAK;
  const targetAmplitude = Math.fround(10 ** (targetDb / 20));
  const normalizationScale = nearSilence
    ? Math.fround(1)
    : Math.fround(targetAmplitude / peakAfterSafetyScale);
  const totalScale = Math.fround(
    Math.fround(1 / safetyDivisor) * normalizationScale,
  );
  return Object.freeze({
    rawPeak: rawPeakF32,
    safetyDivisor,
    peakAfterSafetyScale,
    targetDb,
    targetAmplitude,
    normalizationScale,
    totalScale,
    finalPeak: Math.fround(peakAfterSafetyScale * normalizationScale),
    nearSilence,
  });
}
