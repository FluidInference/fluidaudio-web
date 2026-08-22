import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import { ACE_TURBO_V1_CORRECTNESS_PROFILE } from "../src/api.js";
import {
  ACE_DIT_TIMESTEP_FREQUENCY_BITS_FP32,
  ACE_DIT_TIMESTEP_FREQUENCY_SHA256,
  aceCorrectnessDitConcatenateWgsl,
  aceCorrectnessDitLinearUpdateWgsl,
  aceCorrectnessDitModulationWgsl,
  aceCorrectnessDitPatchProjectionWgsl,
  aceCorrectnessDitTimestepEmbeddingWgsl,
  aceCorrectnessDitUnpatchProjectionWgsl,
  planAceDitConcatenate,
  planAceDitLinearUpdate,
  planAceDitModulation,
  planAceDitPatchProjection,
  planAceDitTimestepEmbedding,
  planAceDitUnpatchProjection,
} from "../src/webgpu/kernels/dit-plumbing.js";
import {
  ACE_TORCH_210_TIMESTEP_VECTORS,
  ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS,
} from "./dit-timestep-torch-vectors.js";

describe("ACE DiT-only plumbing contracts", () => {
  it("plans context/audio concatenation without materializing padding", () => {
    expect(planAceDitConcatenate({
      batch: 2,
      time: 325,
      leftWidth: 128,
      rightWidth: 64,
    })).toEqual({
      elements: 124_800,
      workgroupsX: 488,
      workgroupsY: 1,
    });
    const source = aceCorrectnessDitConcatenateWgsl("reference-bf16", {
      batch: 2,
      time: 325,
      leftWidth: 128,
      rightWidth: 64,
    });
    expect(source).toContain("if (column < LEFT_WIDTH)");
    expect(source).toContain("right[row * RIGHT_WIDTH + (column - LEFT_WIDTH)]");
  });

  it("uses the exact source Conv1d [out,in,kernel] patch layout", () => {
    expect(planAceDitPatchProjection({
      batch: 1,
      time: 325,
      inputChannels: 192,
      hiddenSize: 2_048,
      patchSize: 2,
    })).toEqual({
      batch: 1,
      time: 325,
      inputChannels: 192,
      hiddenSize: 2_048,
      patchSize: 2,
      paddedTime: 326,
      tokens: 163,
      elements: 333_824,
      workgroupsX: 1_304,
      workgroupsY: 1,
    });
    const source = aceCorrectnessDitPatchProjectionWgsl("reference-bf16", {
      batch: 1,
      time: 325,
      inputChannels: 192,
      hiddenSize: 2_048,
      patchSize: 2,
    });
    expect(source).toContain("if (time < TIME)");
    expect(source).toContain(
      "(hidden * INPUT_CHANNELS + channel) * PATCH_SIZE + kernel_index",
    );
    expect(source).toContain(
      "(batch * TIME + time) * INPUT_CHANNELS + channel",
    );
    expect(source).toContain("var sum: f32 = load_bias(hidden)");
  });

  it("uses the exact source ConvTranspose1d [in,out,kernel] layout and crops", () => {
    expect(planAceDitUnpatchProjection({
      batch: 1,
      time: 325,
      outputChannels: 64,
      hiddenSize: 2_048,
      patchSize: 2,
    })).toEqual({
      batch: 1,
      time: 325,
      outputChannels: 64,
      hiddenSize: 2_048,
      patchSize: 2,
      tokens: 163,
      elements: 20_800,
      workgroupsX: 82,
      workgroupsY: 1,
    });
    const source = aceCorrectnessDitUnpatchProjectionWgsl("reference-bf16", {
      batch: 1,
      time: 325,
      outputChannels: 64,
      hiddenSize: 2_048,
      patchSize: 2,
    });
    expect(source).toContain(
      "(hidden * OUTPUT_CHANNELS + channel) * PATCH_SIZE + kernel_index",
    );
    expect(source).toContain("let token = time / PATCH_SIZE");
    expect(source).toContain("const TIME: u32 = 325u");
  });

  it("transposes layer modulation to contiguous group-major bindings", () => {
    expect(planAceDitModulation({
      batch: 2,
      groups: 6,
      width: 2_048,
      projectionLayout: "per-group",
    })).toEqual({
      batch: 2,
      groups: 6,
      width: 2_048,
      projectionLayout: "per-group",
      groupElements: 4_096,
      elements: 24_576,
      workgroupsX: 96,
      workgroupsY: 1,
    });
    const source = aceCorrectnessDitModulationWgsl("reference-bf16", {
      batch: 2,
      groups: 6,
      width: 2_048,
      projectionLayout: "per-group",
    });
    expect(source).toContain("let group = output_index / GROUP_ELEMENTS");
    expect(source).toContain(
      "let projection_index = (batch * GROUPS + group) * WIDTH + column",
    );
    expect(source).toContain("projection[projection_index] + load_table(table_index)");

    const outputSource = aceCorrectnessDitModulationWgsl("reference-bf16", {
      batch: 2,
      groups: 2,
      width: 2_048,
      projectionLayout: "per-batch",
    });
    expect(outputSource).toContain("let projection_index = batch * WIDTH + column");
  });

  it("uses the authenticated Torch frequency vector and BF16 scale semantics", () => {
    expect(planAceDitTimestepEmbedding({
      batch: 2,
      dimension: 256,
      scale: 1_000,
      maximumPeriod: 10_000,
    })).toEqual({ elements: 512, workgroupsX: 2, workgroupsY: 1 });
    const timestep = aceCorrectnessDitTimestepEmbeddingWgsl("reference-bf16", {
      batch: 2,
      dimension: 256,
      scale: 1_000,
      maximumPeriod: 10_000,
    });
    expect(timestep).not.toContain("exp(");
    expect(timestep).toContain("TIMESTEP_FREQUENCY_BITS");
    expect(timestep).toContain("round_bfloat16(timestep[batch] * SCALE)");
    expect(timestep).toContain("bitcast<f32>(TIMESTEP_FREQUENCY_BITS[frequency_index])");
    expect(timestep).toContain("select(sin(phase), cos(phase), dimension < HALF)");

    expect(ACE_DIT_TIMESTEP_FREQUENCY_BITS_FP32).toHaveLength(128);
    const bytes = new ArrayBuffer(128 * 4);
    const view = new DataView(bytes);
    ACE_DIT_TIMESTEP_FREQUENCY_BITS_FP32.forEach((word, index) =>
      view.setUint32(index * 4, word, true)
    );
    expect(createHash("sha256").update(new Uint8Array(bytes)).digest("hex"))
      .toBe(ACE_DIT_TIMESTEP_FREQUENCY_SHA256);
    expect(ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS).toHaveLength(16);
    expect(ACE_TORCH_210_TIMESTEP_VECTORS.map((vector) => vector.timestep))
      .toEqual(ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16);
    expect(ACE_TORCH_210_TIMESTEP_VECTORS.map((vector) => vector.scaledBfloat16))
      .toEqual([1_000, 952, 900, 832, 752, 644, 500, 300]);
    for (const vector of ACE_TORCH_210_TIMESTEP_VECTORS) {
      expect(vector.selectedFloat32Bits).toHaveLength(16);
      expect(vector.selectedFloat16Bits).toHaveLength(16);
    }

    expect(() => planAceDitTimestepEmbedding({
      batch: 1,
      dimension: 128,
      scale: 1_000,
      maximumPeriod: 10_000,
    })).toThrow(/no authenticated Torch 2\.10 frequency vector/);
    expect(() => planAceDitTimestepEmbedding({
      batch: 1,
      dimension: 256,
      scale: 1,
      maximumPeriod: 10_000,
    })).toThrow(/no authenticated Torch 2\.10 frequency vector/);

    expect(planAceDitLinearUpdate({
      batch: 1,
      time: 400,
      channels: 64,
      coefficient: 0.046875,
    })).toEqual({ elements: 25_600, workgroupsX: 100, workgroupsY: 1 });
    const update = aceCorrectnessDitLinearUpdateWgsl("reference-bf16", {
      batch: 1,
      time: 400,
      channels: 64,
      coefficient: 0.046875,
    });
    expect(update).toContain("const COEFFICIENT: f32 = 0.046875");
    expect(update).toContain("latent[index] - velocity[index] * COEFFICIENT");
  });

  it("retains FP16 storage arithmetic only in the raw profile", () => {
    const patch = aceCorrectnessDitPatchProjectionWgsl("raw-fp16", {
      batch: 1,
      time: 4,
      inputChannels: 3,
      hiddenSize: 2,
      patchSize: 2,
    });
    expect(patch).toContain("enable f16");
    expect(patch).toContain("var sum: f16 = bias[hidden]");
    expect(patch).toContain("input[input_index] * weight[weight_index]");
    expect(patch).not.toContain("load_bf16");
  });

  it("rejects malformed geometry, coefficients, and profiles", () => {
    expect(() => planAceDitPatchProjection({
      batch: 1,
      time: 0,
      inputChannels: 192,
      hiddenSize: 2_048,
      patchSize: 2,
    })).toThrow();
    expect(() => planAceDitModulation({
      batch: 1,
      groups: -1,
      width: 2,
      projectionLayout: "per-group",
    })).toThrow();
    expect(() => planAceDitLinearUpdate({
      batch: 1,
      time: 1,
      channels: 1,
      coefficient: 1.01,
    })).toThrow();
    expect(() => aceCorrectnessDitConcatenateWgsl("future" as never, {
      batch: 1,
      time: 1,
      leftWidth: 1,
      rightWidth: 1,
    })).toThrow(/Unknown ACE DiT/);
  });
});
