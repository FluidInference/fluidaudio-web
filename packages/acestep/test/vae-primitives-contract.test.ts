import { describe, expect, it } from "vitest";

import {
  aceCorrectnessVaeAddWgsl,
  aceCorrectnessVaeConv1dWgsl,
  aceCorrectnessVaeConvTranspose1dWgsl,
  aceCorrectnessVaeConvTranspose1dPartWgsl,
  aceCorrectnessVaeSnakeWgsl,
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
  planAceVaePointwise,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("ACE FP32 Oobleck primitive contracts", () => {
  it("plans symmetric dilated Conv1d in NLC with [out,kernel,in] weights", () => {
    const shape = {
      batch: 1,
      inputFrames: 5,
      inputChannels: 2,
      outputChannels: 3,
      kernelSize: 7,
      stride: 1,
      dilation: 3,
      padding: 9,
    } as const;
    expect(planAceVaeConv1d(shape)).toMatchObject({
      outputFrames: 5,
      inputElements: 10,
      weightElements: 42,
      outputElements: 15,
      workgroupsX: 1,
      workgroupsY: 1,
    });
    const source = aceCorrectnessVaeConv1dWgsl(shape, true);
    expect(source).toContain("output_time * STRIDE + kernel * DILATION");
    expect(source).toContain(
      "(output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS",
    );
    expect(source).toContain("input[input_base + input_channel]");
    expect(source).toContain("array<f32>");
    expect(source).not.toContain("enable f16");

    const noBiasSource = aceCorrectnessVaeConv1dWgsl(shape, false);
    expect(noBiasSource).not.toContain("var<storage, read> bias");
    expect(noBiasSource).toContain("var sum = 0.0;");
    expect(noBiasSource).toContain("@binding(2) var<storage, read_write> output");
  });

  it("matches PyTorch ConvTranspose1d output length and inverse stride test", () => {
    const shape = {
      batch: 1,
      inputFrames: 11,
      inputChannels: 4,
      outputChannels: 2,
      kernelSize: 12,
      stride: 6,
      dilation: 1,
      padding: 3,
      outputPadding: 0,
    } as const;
    expect(planAceVaeConvTranspose1d(shape)).toMatchObject({
      outputFrames: 66,
      inputElements: 44,
      weightElements: 96,
      outputElements: 132,
    });
    const source = aceCorrectnessVaeConvTranspose1dWgsl(shape, true);
    expect(source).toContain("input_numerator % STRIDE");
    expect(source).toContain("input_numerator / STRIDE");
    expect(source).toContain(
      "(output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS",
    );
  });

  it("writes output-axis package parts into full interleaved channel lanes", () => {
    const source = aceCorrectnessVaeConvTranspose1dPartWgsl({
      batch: 1,
      inputFrames: 3,
      inputChannels: 2,
      outputChannels: 5,
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      padding: 1,
      outputPadding: 0,
      firstOutputChannel: 2,
      partOutputChannels: 3,
    }, true);
    expect(source).toContain("const TOTAL_OUTPUT_CHANNELS: u32 = 5u");
    expect(source).toContain("const FIRST_OUTPUT_CHANNEL: u32 = 2u");
    expect(source).toContain("const PART_OUTPUT_CHANNELS: u32 = 3u");
    expect(source).toContain("bias[output_channel]");
    expect(source).toContain("local_output_channel * KERNEL_SIZE");
    expect(source).toContain("TOTAL_OUTPUT_CHANNELS + output_channel");
  });

  it("keeps Snake log scales, epsilon, and residual add as separate FP32 kernels", () => {
    const shape = { batch: 1, frames: 7, channels: 3 } as const;
    expect(planAceVaePointwise(shape)).toEqual({
      ...shape,
      elements: 21,
      workgroupsX: 1,
      workgroupsY: 1,
    });
    const snake = aceCorrectnessVaeSnakeWgsl(shape);
    expect(snake).toContain("let alpha_value = exp(alpha[channel])");
    expect(snake).toContain("let beta_value = exp(beta[channel])");
    expect(snake).toContain("let reciprocal_beta = 1.0 / (beta_value + 1e-9)");
    expect(snake).toContain("reciprocal_beta * periodic * periodic");
    const add = aceCorrectnessVaeAddWgsl(shape);
    expect(add).toContain("output[index] = left[index] + right[index]");
  });

  it("uses one dynamic uniform range without changing source-order primitive math", () => {
    const convShape = {
      batch: 1,
      inputFrames: 5,
      inputChannels: 2,
      outputChannels: 3,
      kernelSize: 3,
      stride: 1,
      dilation: 1,
      padding: 1,
    } as const;
    const conv = aceCorrectnessVaeConv1dWgsl(convShape, true, true);
    expect(conv).toContain("struct AceVaeOutputRangeControl");
    expect(conv).toContain("@binding(4)");
    expect(conv).not.toContain("hasDynamicOffset");
    expect(conv).toContain("@builtin(num_workgroups) num_workgroups");
    expect(conv).toContain("local_output_index >= output_range.count");
    expect(conv).toContain(
      "let output_index = output_range.base + local_output_index",
    );
    expect(conv.indexOf("for (var kernel = 0u")).toBeLessThan(
      conv.indexOf("for (var input_channel = 0u"),
    );

    const transposePart = aceCorrectnessVaeConvTranspose1dPartWgsl({
      ...convShape,
      outputPadding: 0,
      firstOutputChannel: 1,
      partOutputChannels: 2,
    }, true, true);
    expect(transposePart).toContain(
      "let part_index = output_range.base + local_output_index",
    );
    expect(transposePart).toContain("local_output_channel * KERNEL_SIZE");

    const pointwiseShape = { batch: 1, frames: 7, channels: 3 } as const;
    expect(aceCorrectnessVaeSnakeWgsl(pointwiseShape, true)).toContain(
      "let index = output_range.base + local_output_index",
    );
    expect(aceCorrectnessVaeAddWgsl(pointwiseShape, true)).toContain(
      "let index = output_range.base + local_output_index",
    );
  });

  it("rejects malformed or empty convolution geometry", () => {
    expect(() => planAceVaeConv1d({
      batch: 1,
      inputFrames: 2,
      inputChannels: 1,
      outputChannels: 1,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 0,
    })).toThrow(/exceeds/);
    expect(() => planAceVaeConvTranspose1d({
      batch: 1,
      inputFrames: 2,
      inputChannels: 1,
      outputChannels: 1,
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      padding: 1,
      outputPadding: 2,
    })).toThrow(/output padding/);
  });
});
