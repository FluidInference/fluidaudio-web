import { describe, expect, it, vi } from "vitest";

import {
  ACE_DEFAULT_NORMALIZATION_DB,
  ACE_VAE_NEAR_SILENCE_PEAK,
  deriveAceVaePostprocessPlan,
  planAceVaeChunkedDecode,
  streamAceVaeRawChunks,
} from "../src/webgpu/vae-chunks.js";
import type { AceVaeDecoderConfig } from "../src/webgpu/vae-decoder.js";
import {
  AceVaeRawF32FileSink,
  createAceFloat32WavHeader,
  type AceSeekableFile,
  writeNormalizedAceVaeFloat32Wav,
  writeNormalizedAceVaeFloat32WavCooperatively,
} from "../src/webgpu/vae-wav.js";

const TOY_CONFIG: AceVaeDecoderConfig = {
  id: "chunk-test",
  decoderInputChannels: 1,
  decoderChannels: 1,
  audioChannels: 2,
  channelMultiples: [1],
  downsamplingRatios: [2],
  sampleRateHz: 48_000,
};

describe("ACE production VAE overlap-discard plan", () => {
  it("pins 256/64 windows and exact 1,920x CPU-offload coordinates", () => {
    const plan = planAceVaeChunkedDecode(750);
    expect(plan).toMatchObject({
      latentFrames: 750,
      chunkFrames: 256,
      overlapFrames: 64,
      strideFrames: 128,
      hopLength: 1_920,
      sampleRateHz: 48_000,
      audioChannels: 2,
      outputAudioFrames: 1_440_000,
      outputInterleavedElements: 2_880_000,
      outputFloat32Bytes: 11_520_000,
      maximumWindowFrames: 256,
      maximumDecodedInterleavedElements: 983_040,
      maximumDecodedFloat32Bytes: 3_932_160,
      direct: false,
    });
    expect(plan.windows).toHaveLength(6);
    expect(plan.windows[0]).toMatchObject({
      coreStartLatentFrame: 0,
      coreEndLatentFrame: 128,
      windowStartLatentFrame: 0,
      windowEndLatentFrame: 192,
      discardPrefixAudioFrames: 0,
      discardSuffixAudioFrames: 122_880,
      outputStartAudioFrame: 0,
      outputAudioFrames: 245_760,
    });
    expect(plan.windows[1]).toMatchObject({
      coreStartLatentFrame: 128,
      coreEndLatentFrame: 256,
      windowStartLatentFrame: 64,
      windowEndLatentFrame: 320,
      discardPrefixAudioFrames: 122_880,
      discardSuffixAudioFrames: 122_880,
      outputStartAudioFrame: 245_760,
    });
    expect(plan.windows.at(-1)).toMatchObject({
      coreStartLatentFrame: 640,
      coreEndLatentFrame: 750,
      windowStartLatentFrame: 576,
      windowEndLatentFrame: 750,
      discardPrefixAudioFrames: 122_880,
      discardSuffixAudioFrames: 0,
      outputStartAudioFrame: 1_228_800,
      outputAudioFrames: 211_200,
    });
    expect(plan.decoderWorkspacePlan.workspaceBytes).toBe(251_658_240);
  });

  it("mirrors the upstream ceil-step tail and direct-decode boundary", () => {
    const tail = planAceVaeChunkedDecode(257);
    expect(tail.windows).toHaveLength(3);
    expect(tail.windows.at(-1)).toMatchObject({
      coreStartLatentFrame: 256,
      coreEndLatentFrame: 257,
      windowStartLatentFrame: 192,
      windowEndLatentFrame: 257,
      outputAudioFrames: 1_920,
    });

    const direct = planAceVaeChunkedDecode(256);
    expect(direct.direct).toBe(true);
    expect(direct.windows).toEqual([
      expect.objectContaining({
        coreStartLatentFrame: 0,
        coreEndLatentFrame: 256,
        windowStartLatentFrame: 0,
        windowEndLatentFrame: 256,
        discardPrefixLatentFrames: 0,
        discardSuffixLatentFrames: 0,
      }),
    ]);
  });

  it("rejects a non-positive overlap-discard stride", () => {
    expect(() => planAceVaeChunkedDecode(100, {
      chunkFrames: 128,
      overlapFrames: 64,
    })).toThrow(/exceed twice/);
  });
});

describe("ACE bounded raw/offload and float WAV passes", () => {
  it("discards seams, measures the assembled raw peak, and emits canonical f32 WAV", async () => {
    const plan = planAceVaeChunkedDecode(7, {
      chunkFrames: 4,
      overlapFrames: 1,
      config: TOY_CONFIG,
    });
    const rawFile = new MemorySeekableFile();
    const rawSink = new AceVaeRawF32FileSink(rawFile, plan);
    const idle = vi.fn(async () => undefined);
    const traces: unknown[] = [];
    let captureClock = 0;
    const stats = await streamAceVaeRawChunks(
      plan,
      {
        async decodeWindow(window) {
          const values = new Float32Array(
            window.decodedAudioFrames * plan.audioChannels,
          );
          for (let frame = 0; frame < window.decodedAudioFrames; frame += 1) {
            const globalFrame =
              window.windowStartLatentFrame * plan.hopLength + frame;
            values[frame * 2] = globalFrame;
            values[frame * 2 + 1] = -globalFrame;
          }
          return values;
        },
      },
      rawSink,
      {
        yieldQueueIdle: idle,
        now: () => captureClock++,
        onTrace: (trace) => traces.push(trace),
      },
    );
    rawSink.finish();
    expect(stats).toEqual({
      peak: 13,
      finiteSamples: 28,
      outputInterleavedElements: 28,
      windowsDecoded: 4,
      cooperativeIdleMs: 3,
    });
    expect(idle).toHaveBeenCalledTimes(3);
    expect(traces).toEqual([
      expect.objectContaining({
        schema: "ace-vae-raw-stream-capture-v1",
        windowCount: 4,
        outputElements: 28,
        outputBytes: 112,
        maximumDecodedWindowBytes: 64,
      }),
    ]);
    expect(rawFile.bytes.byteLength).toBe(28 * 4);
    const rawValues = readFloat32Le(rawFile.bytes);
    expect(Array.from(rawValues)).toEqual(
      Array.from({ length: 14 }, (_, frame) => [frame, -frame]).flat(),
    );

    const postprocess = deriveAceVaePostprocessPlan(stats.peak);
    const wavFile = new MemorySeekableFile();
    const result = writeNormalizedAceVaeFloat32Wav(
      rawFile,
      wavFile,
      plan,
      postprocess,
      3,
    );
    expect(result).toMatchObject({
      headerBytes: 44,
      dataBytes: 112,
      wavBytes: 156,
    });
    expect(result.outputPeak).toBeCloseTo(10 ** (-1 / 20), 6);
    expect(new TextDecoder().decode(wavFile.bytes.subarray(0, 4))).toBe("RIFF");
    expect(new TextDecoder().decode(wavFile.bytes.subarray(8, 12))).toBe("WAVE");
    const header = new DataView(
      wavFile.bytes.buffer,
      wavFile.bytes.byteOffset,
      44,
    );
    expect(header.getUint16(20, true)).toBe(3);
    expect(header.getUint16(22, true)).toBe(2);
    expect(header.getUint32(24, true)).toBe(48_000);
    expect(header.getUint16(34, true)).toBe(32);
    expect(header.getUint32(40, true)).toBe(112);
    const finalValues = readFloat32Le(wavFile.bytes.subarray(44));
    expect(finalValues.at(-2)).toBeCloseTo(10 ** (-1 / 20), 6);
    expect(finalValues.at(-1)).toBeCloseTo(-(10 ** (-1 / 20)), 6);
  });

  it("keeps safety scaling and -1 dBFS normalization distinct and never clamps", () => {
    const loud = deriveAceVaePostprocessPlan(2);
    expect(loud.safetyDivisor).toBe(2);
    expect(loud.peakAfterSafetyScale).toBe(1);
    expect(loud.targetDb).toBe(ACE_DEFAULT_NORMALIZATION_DB);
    expect(loud.normalizationScale).toBeCloseTo(10 ** (-1 / 20));
    expect(loud.totalScale).toBeCloseTo((10 ** (-1 / 20)) / 2);
    expect(loud.finalPeak).toBeCloseTo(10 ** (-1 / 20));

    const quiet = deriveAceVaePostprocessPlan(
      ACE_VAE_NEAR_SILENCE_PEAK / 2,
    );
    expect(quiet.nearSilence).toBe(true);
    expect(quiet.totalScale).toBe(1);

  });

  it("matches independently captured wrapped-FP32 scalar gain bits", () => {
    const plan = deriveAceVaePostprocessPlan(1.0000001192092896);
    expect(f32Bits(plan.rawPeak)).toBe(0x3f800001);
    expect(f32Bits(plan.safetyDivisor)).toBe(0x3f800001);
    expect(f32Bits(plan.peakAfterSafetyScale)).toBe(0x3f800000);
    expect(f32Bits(plan.targetAmplitude)).toBe(0x3f642905);
    expect(f32Bits(plan.normalizationScale)).toBe(0x3f642905);
    expect(f32Bits(plan.totalScale)).toBe(0x3f642903);
  });

  it("rounds upstream safety and normalization gains as two FP32 operations", () => {
    const plan = planAceVaeChunkedDecode(1, { config: TOY_CONFIG });
    const rawFile = new MemorySeekableFile();
    const raw = new Float32Array([
      1.0000001192092896,
      -0.9999998211860657,
      0.5,
      -1,
    ]);
    rawFile.write(new Uint8Array(raw.buffer), { at: 0 });
    const postprocess = deriveAceVaePostprocessPlan(raw[0]!);
    const sequential = Math.fround(
      Math.fround(raw[1]! / postprocess.safetyDivisor) *
        postprocess.normalizationScale,
    );
    const combined = Math.fround(raw[1]! * postprocess.totalScale);
    expect(sequential).not.toBe(combined);
    const wavFile = new MemorySeekableFile();
    writeNormalizedAceVaeFloat32Wav(
      rawFile,
      wavFile,
      plan,
      postprocess,
      1,
    );
    expect(readFloat32Le(wavFile.bytes.subarray(44))[1]).toBe(sequential);
  });

  it("matches independently captured Torch 2.10 FP32 divide-then-normalize bits", () => {
    const plan = planAceVaeChunkedDecode(1, { config: TOY_CONFIG });
    const rawFile = new MemorySeekableFile();
    const raw = new Float32Array([
      1.0416107177734375,
      -0.9999998211860657,
      0.5,
      -1,
    ]);
    rawFile.write(new Uint8Array(raw.buffer), { at: 0 });
    const postprocess = deriveAceVaePostprocessPlan(raw[0]!);
    const wavFile = new MemorySeekableFile();
    writeNormalizedAceVaeFloat32Wav(rawFile, wavFile, plan, postprocess, 1);
    expect([...readFloat32Le(wavFile.bytes.subarray(44))].map(f32Bits)).toEqual([
      0x3f642905,
      0xbf5b0ba9,
      0x3edb0bab,
      0xbf5b0bab,
    ]);
  });

  it("keeps cooperative block output byte-exact and observes aborts after a task yield", async () => {
    const plan = planAceVaeChunkedDecode(3, { config: TOY_CONFIG });
    const rawFile = new MemorySeekableFile();
    const raw = new Float32Array(plan.outputInterleavedElements);
    for (let index = 0; index < raw.length; index += 1) {
      raw[index] = Math.fround((index - 5) / 7);
    }
    rawFile.write(new Uint8Array(raw.buffer), { at: 0 });
    const postprocess = deriveAceVaePostprocessPlan(2);
    const synchronous = new MemorySeekableFile();
    writeNormalizedAceVaeFloat32Wav(
      rawFile,
      synchronous,
      plan,
      postprocess,
      1,
    );

    const cooperative = new MemorySeekableFile();
    const yields = vi.fn(async () => undefined);
    const traces: unknown[] = [];
    let captureClock = 0;
    const result = await writeNormalizedAceVaeFloat32WavCooperatively(
      rawFile,
      cooperative,
      plan,
      postprocess,
      {
        blockAudioFrames: 1,
        yieldEveryBlocks: 1,
        yieldToEventLoop: yields,
        now: () => captureClock++,
        onTrace: (trace) => traces.push(trace),
      },
    );
    expect(cooperative.bytes).toEqual(synchronous.bytes);
    expect(result.wavBytes).toBe(synchronous.bytes.byteLength);
    expect(yields).toHaveBeenCalledTimes(plan.outputAudioFrames - 1);
    expect(traces).toEqual([
      expect.objectContaining({
        schema: "ace-vae-wav-write-capture-v1",
        blockAudioFrames: 1,
        blockCount: plan.outputAudioFrames,
        yieldCount: plan.outputAudioFrames - 1,
        rawReadBytes: plan.outputFloat32Bytes,
        wavDataBytes: plan.outputFloat32Bytes,
        maximumReadBufferBytes: 8,
        maximumScaledBufferBytes: 8,
      }),
    ]);

    const controller = new AbortController();
    const partial = new MemorySeekableFile();
    await expect(writeNormalizedAceVaeFloat32WavCooperatively(
      rawFile,
      partial,
      plan,
      postprocess,
      {
        blockAudioFrames: 1,
        signal: controller.signal,
        yieldEveryBlocks: 1,
        yieldToEventLoop: async () => controller.abort(),
      },
    )).rejects.toMatchObject({ name: "AbortError" });
    expect(partial.bytes.byteLength).toBeLessThan(synchronous.bytes.byteLength);
  });

  it("matches the committed reference's 44-byte IEEE-float header", () => {
    const header = createAceFloat32WavHeader(1, 2, 48_000);
    expect(Array.from(header)).toEqual([
      82, 73, 70, 70, 44, 0, 0, 0,
      87, 65, 86, 69, 102, 109, 116, 32,
      16, 0, 0, 0, 3, 0, 2, 0,
      128, 187, 0, 0, 0, 220, 5, 0,
      8, 0, 32, 0, 100, 97, 116, 97,
      8, 0, 0, 0,
    ]);
  });

  it("rejects non-finite raw decoder samples before persistence", async () => {
    const plan = planAceVaeChunkedDecode(1, { config: TOY_CONFIG });
    await expect(streamAceVaeRawChunks(
      plan,
      { async decodeWindow() { return new Float32Array([NaN, 0, 0, 0]); } },
      { async writeCore() {} },
    )).rejects.toThrow(/non-finite raw sample/);
  });
});

class MemorySeekableFile implements AceSeekableFile {
  bytes = new Uint8Array();

  write(buffer: ArrayBufferView, options: { readonly at: number }): number {
    const source = new Uint8Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength,
    );
    this.ensureSize(options.at + source.byteLength);
    this.bytes.set(source, options.at);
    return source.byteLength;
  }

  read(buffer: ArrayBufferView, options: { readonly at: number }): number {
    const destination = new Uint8Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength,
    );
    const available = Math.max(
      0,
      Math.min(destination.byteLength, this.bytes.byteLength - options.at),
    );
    destination.set(this.bytes.subarray(options.at, options.at + available));
    return available;
  }

  truncate(newSize: number): void {
    const next = new Uint8Array(newSize);
    next.set(this.bytes.subarray(0, newSize));
    this.bytes = next;
  }

  flush(): void {}

  private ensureSize(size: number): void {
    if (this.bytes.byteLength >= size) return;
    const next = new Uint8Array(size);
    next.set(this.bytes);
    this.bytes = next;
  }
}

function readFloat32Le(bytes: Uint8Array): Float32Array {
  const values = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(index * 4, true);
  }
  return values;
}

function f32Bits(value: number): number {
  const f32 = new Float32Array([value]);
  return new Uint32Array(f32.buffer)[0]!;
}
