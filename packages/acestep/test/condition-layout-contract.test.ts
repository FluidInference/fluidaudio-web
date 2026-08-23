import { describe, expect, it } from "vitest";

import {
  ACE_CONTEXT_CHANNELS,
  ACE_SILENCE_SOURCE_CHANNELS,
  ACE_SILENCE_SOURCE_FRAMES,
  aceCorrectnessDirectContextWgsl,
  aceCorrectnessSilenceExpandWgsl,
  createAceDirectContextCpu,
  expandAceSilenceLatentCpu,
  planAceDirectContext,
  planAceSilenceExpand,
} from "../src/webgpu/kernels/condition-layout.js";

describe("ACE condition layout contract", () => {
  it("pins and plans the authenticated NCT silence source", () => {
    expect(ACE_SILENCE_SOURCE_CHANNELS).toBe(64);
    expect(ACE_SILENCE_SOURCE_FRAMES).toBe(15_000);
    expect(ACE_CONTEXT_CHANNELS).toBe(128);
    expect(planAceSilenceExpand({ batch: 2, frames: 750 })).toMatchObject({
      batch: 2,
      frames: 750,
      channels: 64,
      sourceFrames: 15_000,
      outputElements: 96_000,
    });
    expect(planAceDirectContext({ batch: 2, frames: 3 })).toMatchObject({
      sourceElements: 384,
      maskElements: 6,
      outputElements: 768,
    });
  });

  it("slices, transposes, and batch-expands NCT silence exactly", () => {
    const source = new Float32Array(
      ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES,
    );
    for (let channel = 0; channel < ACE_SILENCE_SOURCE_CHANNELS; channel += 1) {
      source[channel * ACE_SILENCE_SOURCE_FRAMES] = channel + 0.25;
      source[channel * ACE_SILENCE_SOURCE_FRAMES + 1] = channel + 0.5;
    }
    const output = expandAceSilenceLatentCpu(source, 2, 2);
    expect(output[0]).toBe(0.25);
    expect(output[63]).toBe(63.25);
    expect(output[64]).toBe(0.5);
    expect(output[127]).toBe(63.5);
    expect([...output.slice(128, 256)]).toEqual([...output.slice(0, 128)]);
  });

  it("tiles the authenticated source when the requested duration is longer", () => {
    const source = new Float32Array(
      ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES,
    );
    source[0] = 1.25;
    source[ACE_SILENCE_SOURCE_FRAMES - 1] = 9.5;
    const output = expandAceSilenceLatentCpu(
      source,
      1,
      ACE_SILENCE_SOURCE_FRAMES + 1,
    );
    expect(output[(ACE_SILENCE_SOURCE_FRAMES - 1) * 64]).toBe(9.5);
    expect(output[ACE_SILENCE_SOURCE_FRAMES * 64]).toBe(1.25);
  });

  it("concatenates source channels with the channel-repeated chunk mask", () => {
    const source = new Float32Array(4 * 64);
    for (let index = 0; index < source.length; index += 1) source[index] = index;
    const output = createAceDirectContextCpu(source, [1, 0, 1, 1], 1, 4);
    expect([...output.slice(0, 64)]).toEqual([...source.slice(0, 64)]);
    expect([...output.slice(64, 128)]).toEqual(Array(64).fill(1));
    expect([...output.slice(128, 192)]).toEqual([...source.slice(64, 128)]);
    expect([...output.slice(192, 256)]).toEqual(Array(64).fill(0));
  });

  it("emits profile-specific direct GPU conversions", () => {
    const reference = aceCorrectnessSilenceExpandWgsl("reference-bf16", {
      batch: 1,
      frames: 2,
    });
    expect(reference).toContain("frame % SOURCE_FRAMES");
    expect(reference).toContain("array<f32>");
    expect(reference).not.toContain("enable f16;");

    const fp16 = aceCorrectnessDirectContextWgsl("raw-fp16", {
      batch: 1,
      frames: 2,
    });
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("f16(f32(chunk_mask[row]))");
  });

  it("accepts tiled source slices and rejects non-binary masks", () => {
    expect(planAceSilenceExpand({ batch: 1, frames: 15_001 }).frames).toBe(
      15_001,
    );
    expect(() => createAceDirectContextCpu(
      new Float32Array(64),
      [2],
      1,
      1,
    )).toThrow(/zero or one/);
  });
});
