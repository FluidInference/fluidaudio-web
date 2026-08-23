import {
  ACE_OOBLECK_DECODER_CONFIG,
  planAceVaeDecoder,
} from "../src/webgpu/vae-decoder.js";
import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";

export type AceOpt0011ArmId = "fp32-256" | "fp16-256" | "fp16-512";

export interface AceOpt0011ArmPlan {
  readonly id: AceOpt0011ArmId;
  readonly windowFrames: 256 | 512;
  readonly overlapFrames: 64;
  readonly storageElementBytes: 2 | 4;
  readonly parameterElements: number;
  readonly parameterBytes: number;
  readonly maximumActivationElements: number;
  readonly workspaceBytes: number;
  readonly allWorkspaceBytes: number;
  readonly inputBytes: number;
  readonly outputFloat32Bytes: number;
  readonly readbackFloat32Bytes: number;
  readonly namedBufferSubtotalBytes: number;
}

export interface AceOpt0011ChunkGeometry {
  readonly logicalLatentFrames: number;
  readonly windowFrames: 256 | 512;
  readonly overlapFrames: 64;
  readonly strideFrames: 128 | 384;
  readonly windowCount: number;
  readonly decodedLatentFrames: number;
  readonly duplicatedLatentFrames: number;
}

export interface AceOpt0011TemporalSupportClass {
  readonly firstRelativeLatentFrame: number;
  readonly lastRelativeLatentFrame: number;
  readonly outputPhaseCount: number;
}

export interface AceOpt0011TemporalSupportPlan {
  readonly hopLength: 1920;
  readonly maximumPastLatentFrames: number;
  readonly maximumFutureLatentFrames: number;
  readonly maximumRadiusLatentFrames: number;
  readonly classes: readonly AceOpt0011TemporalSupportClass[];
}

const FP32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const OUTPUT_CHANNELS = ACE_OOBLECK_DECODER_CONFIG.audioChannels;
const INPUT_CHANNELS = ACE_OOBLECK_DECODER_CONFIG.decoderInputChannels;

export function planAceOpt0011Arm(id: AceOpt0011ArmId): AceOpt0011ArmPlan {
  if (id !== "fp32-256" && id !== "fp16-256" && id !== "fp16-512") {
    throw new RangeError(`Unknown OPT-0011 arm ${String(id)}`);
  }
  const windowFrames = id === "fp16-512" ? 512 : 256;
  const storageElementBytes: 2 | 4 =
    id === "fp32-256" ? 4 : 2;
  const decoder = planAceVaeDecoder(windowFrames);
  const parameterBytes = decoder.parameterElements * storageElementBytes;
  const workspaceBytes =
    decoder.maximumActivationElements * storageElementBytes;
  const allWorkspaceBytes = workspaceBytes * 3;
  const inputBytes = windowFrames * INPUT_CHANNELS * storageElementBytes;
  const outputFloat32Bytes =
    windowFrames * decoder.hopLength * OUTPUT_CHANNELS * FP32_BYTES;
  const readbackFloat32Bytes = outputFloat32Bytes;
  return Object.freeze({
    id,
    windowFrames,
    overlapFrames: 64,
    storageElementBytes,
    parameterElements: decoder.parameterElements,
    parameterBytes,
    maximumActivationElements: decoder.maximumActivationElements,
    workspaceBytes,
    allWorkspaceBytes,
    inputBytes,
    outputFloat32Bytes,
    readbackFloat32Bytes,
    namedBufferSubtotalBytes:
      parameterBytes +
      allWorkspaceBytes +
      inputBytes +
      outputFloat32Bytes +
      readbackFloat32Bytes,
  });
}

export function planAceOpt0011ChunkGeometry(
  logicalLatentFrames: number,
  windowFrames: 256 | 512,
): AceOpt0011ChunkGeometry {
  if (windowFrames !== 256 && windowFrames !== 512) {
    throw new RangeError(
      `OPT-0011 window frames must be exactly 256 or 512; received ${String(windowFrames)}`,
    );
  }
  const plan = planAceVaeChunkedDecode(logicalLatentFrames, {
    chunkFrames: windowFrames,
    overlapFrames: 64,
  });
  const decodedLatentFrames = plan.windows.reduce(
    (sum, window) => sum + window.latentWindowFrames,
    0,
  );
  return Object.freeze({
    logicalLatentFrames,
    windowFrames,
    overlapFrames: 64,
    strideFrames: plan.strideFrames as 128 | 384,
    windowCount: plan.windows.length,
    decodedLatentFrames,
    duplicatedLatentFrames: decodedLatentFrames - logicalLatentFrames,
  });
}

/**
 * Generate the frozen little-endian xorshift32 latent fixture registered by
 * OPT-0011. The returned bytes are directly hashable and uploadable.
 */
export function createAceOpt0011LatentFixture(frames: number): Uint8Array {
  if (!Number.isSafeInteger(frames) || frames <= 0) {
    throw new RangeError("OPT-0011 fixture frames must be a positive safe integer");
  }
  const scalarCount = frames * INPUT_CHANNELS;
  if (!Number.isSafeInteger(scalarCount)) {
    throw new RangeError("OPT-0011 fixture scalar count exceeds safe integer range");
  }
  const bytes = new Uint8Array(scalarCount * FP32_BYTES);
  const view = new DataView(bytes.buffer);
  let state = 0x0011_0512;
  for (let index = 0; index < scalarCount; index += 1) {
    state = (state ^ (state << 13)) >>> 0;
    state = (state ^ (state >>> 17)) >>> 0;
    state = (state ^ (state << 5)) >>> 0;
    const value = Math.fround((state >>> 8) / 8_388_608 - 1);
    view.setFloat32(index * FP32_BYTES, value, true);
  }
  return bytes;
}

/**
 * Enumerate the exact infinite-domain temporal dependencies of all 1,920
 * decoder output phases. Residual skip paths are included by set union.
 */
export function planAceOpt0011TemporalSupport(): AceOpt0011TemporalSupportPlan {
  const hopLength = ACE_OOBLECK_DECODER_CONFIG.downsamplingRatios.reduce(
    (product, ratio) => product * ratio,
    1,
  );
  if (hopLength !== 1920) {
    throw new Error(`OPT-0011 expected hop length 1920, received ${hopLength}`);
  }
  const counts = new Map<string, AceOpt0011TemporalSupportClass>();
  let maximumPastLatentFrames = 0;
  let maximumFutureLatentFrames = 0;
  const reverseStrides = [
    ...ACE_OOBLECK_DECODER_CONFIG.downsamplingRatios,
  ];

  for (let phase = 0; phase < hopLength; phase += 1) {
    let support = expandConv1d(new Set([phase]), 1);
    for (const stride of reverseStrides) {
      support = expandConv1d(support, 9);
      support = expandConv1d(support, 3);
      support = expandConv1d(support, 1);
      support = reverseConvTranspose1d(support, stride);
    }
    support = expandConv1d(support, 1);
    const relativeBase = Math.floor(phase / hopLength);
    const first = Math.min(...support) - relativeBase;
    const last = Math.max(...support) - relativeBase;
    maximumPastLatentFrames = Math.max(maximumPastLatentFrames, -first);
    maximumFutureLatentFrames = Math.max(maximumFutureLatentFrames, last);
    const key = `${first}:${last}`;
    const previous = counts.get(key);
    counts.set(key, Object.freeze({
      firstRelativeLatentFrame: first,
      lastRelativeLatentFrame: last,
      outputPhaseCount: (previous?.outputPhaseCount ?? 0) + 1,
    }));
  }

  const classes = [...counts.values()].sort(
    (left, right) =>
      left.firstRelativeLatentFrame - right.firstRelativeLatentFrame ||
      left.lastRelativeLatentFrame - right.lastRelativeLatentFrame,
  );
  return Object.freeze({
    hopLength: 1920,
    maximumPastLatentFrames,
    maximumFutureLatentFrames,
    maximumRadiusLatentFrames: Math.max(
      maximumPastLatentFrames,
      maximumFutureLatentFrames,
    ),
    classes: Object.freeze(classes),
  });
}

function expandConv1d(input: ReadonlySet<number>, dilation: number): Set<number> {
  const output = new Set<number>();
  for (const time of input) {
    for (let kernel = 0; kernel < 7; kernel += 1) {
      output.add(time + (kernel - 3) * dilation);
    }
  }
  return output;
}

function reverseConvTranspose1d(
  outputTimes: ReadonlySet<number>,
  stride: number,
): Set<number> {
  const inputTimes = new Set<number>();
  const padding = Math.ceil(stride / 2);
  for (const outputTime of outputTimes) {
    for (let kernel = 0; kernel < 2 * stride; kernel += 1) {
      const numerator = outputTime + padding - kernel;
      if (numerator % stride === 0) inputTimes.add(numerator / stride);
    }
  }
  if (inputTimes.size === 0) {
    throw new Error("OPT-0011 transpose support unexpectedly became empty");
  }
  return inputTimes;
}
