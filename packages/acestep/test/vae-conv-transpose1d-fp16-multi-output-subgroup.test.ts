import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
  aceOpt0026NativeWeightIndex,
  aceOpt0026PolyphaseWeightIndex,
  aceOpt0026VaeConvTranspose1dWgsl,
  packAceOpt0026VaeConvTranspose1dWeights,
  planAceOpt0026VaeConvTranspose1d,
  planAceOpt0026VaeConvTranspose1dRange,
  unpackAceOpt0026VaeConvTranspose1dWeights,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";

const PRODUCTION_SHAPES = Object.freeze([
  shape(300, 2_048, 1_024, 10),
  shape(3_000, 1_024, 512, 6),
  shape(18_000, 512, 256, 4),
  shape(72_000, 256, 128, 4),
  shape(288_000, 128, 128, 2),
]);

describe("OPT-0026 multi-output subgroup ConvTranspose1D", () => {
  it("gives every subgroup four rows and 128 adjacent output channels", () => {
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP).toBe(4);
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP).toBe(16);
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE).toBe(4);
    expect(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP).toBe(128);
    expect(
      ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE *
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE,
    ).toBe(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP);
    expect(
      ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP *
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
    ).toBe(ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP);

    for (const productionShape of PRODUCTION_SHAPES) {
      const plan = planAceOpt0026VaeConvTranspose1d(productionShape);
      const range = planAceOpt0026VaeConvTranspose1dRange(plan, {
        base: 0,
        count: plan.outputElements,
      });
      expect(range.workgroupsY).toBe(productionShape.outputChannels / 128);
      expect(range.workgroupsZ).toBe(productionShape.stride);
      expect(plan.workgroupStorageBytes).toBe(0);
      expect(plan.workgroupBarrierCount).toBe(0);
    }
  });

  it("owns channel, row, phase, and tail coordinates exactly once", () => {
    const candidate = shape(17, 19, 137, 6);
    const plan = planAceOpt0026VaeConvTranspose1d(candidate);
    const firstOutputTime = 5;
    const outputRows = 53;
    const range = planAceOpt0026VaeConvTranspose1dRange(plan, {
      base: firstOutputTime * candidate.outputChannels,
      count: outputRows * candidate.outputChannels,
    });
    const owners = new Uint8Array(range.count);

    for (let x = 0; x < range.workgroupsX; x += 1) {
      for (let y = 0; y < range.workgroupsY; y += 1) {
        for (let phase = 0; phase < range.workgroupsZ; phase += 1) {
          for (let subgroup = 0;
            subgroup < ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP;
            subgroup += 1) {
            for (let lane = 0;
              lane < ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE;
              lane += 1) {
              for (let row = 0;
                row < ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP;
                row += 1) {
                const phaseRow =
                  x * ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP +
                  subgroup *
                    ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP + row;
                const rowOffset = phase + phaseRow * candidate.stride;
                if (rowOffset >= outputRows) continue;
                for (let member = 0;
                  member < ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE;
                  member += 1) {
                  const channel =
                    y * ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP +
                    lane *
                      ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE +
                    member;
                  if (channel >= candidate.outputChannels) continue;
                  const index = rowOffset * candidate.outputChannels + channel;
                  owners[index] = owners[index]! + 1;
                }
              }
            }
          }
        }
      }
    }
    expect([...owners].every((count) => count === 1)).toBe(true);
  });

  it("round-trips every raw weight bit through the polyphase layout", () => {
    for (const stride of [2, 4, 6, 10]) {
      const candidate = shape(3, 5, 7, stride);
      const words = candidate.outputChannels * candidate.kernelSize *
        candidate.inputChannels;
      const native = Uint16Array.from(
        { length: words },
        (_, index) => [
          0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7e01,
          index & 0xffff,
        ][index % 8]!,
      );
      const packed = packAceOpt0026VaeConvTranspose1dWeights(native, candidate);
      const inverse = unpackAceOpt0026VaeConvTranspose1dWeights(
        packed,
        candidate,
      );
      expect(inverse).toEqual(native);

      const visited = new Uint8Array(words);
      for (let phase = 0; phase < stride; phase += 1) {
        for (let tap = 0; tap < 2; tap += 1) {
          const kernel = phase + tap * stride;
          for (let inputChannel = 0;
            inputChannel < candidate.inputChannels; inputChannel += 1) {
            for (let outputChannel = 0;
              outputChannel < candidate.outputChannels; outputChannel += 1) {
              const source = aceOpt0026NativeWeightIndex(
                candidate,
                outputChannel,
                kernel,
                inputChannel,
              );
              const destination = aceOpt0026PolyphaseWeightIndex(
                candidate,
                phase,
                tap,
                inputChannel,
                outputChannel,
              );
              expect(packed[destination]).toBe(native[source]);
              visited[source] = visited[source]! + 1;
            }
          }
        }
      }
      expect([...visited].every((count) => count === 1)).toBe(true);
    }
  });

  it("matches native ConvTranspose output and FP32 term order exactly", () => {
    for (const stride of [2, 4, 6, 10]) {
      const candidate = shape(4, 5, 7, stride);
      const plan = planAceOpt0026VaeConvTranspose1d(candidate);
      const input = Float32Array.from(
        { length: plan.inputElements },
        (_, index) => Math.fround(((index * 7) % 17 - 8) / 16),
      );
      const native = Float32Array.from(
        { length: plan.weightElements },
        (_, index) => Math.fround(((index * 11) % 19 - 9) / 32),
      );
      const bias = Float32Array.from(
        { length: candidate.outputChannels },
        (_, index) => Math.fround((index - 3) / 64),
      );
      const packed = packFloatWeights(native, candidate);
      const expected = nativeOutput(candidate, input, native, bias);
      const actual = polyphaseOutput(candidate, input, packed, bias);
      expect(actual).toEqual(expected);
    }
  });

  it("emits vectorized four-channel FP32 accumulators without barriers", () => {
    const source = aceOpt0026VaeConvTranspose1dWgsl(
      shape(17, 19, 137, 6),
    );
    expect(source).toContain("lane * 4u");
    expect(source).toContain("var sum3 = initial_sum;");
    expect(source).toContain("var weight7 = vec4<f32>(0.0);");
    expect(source).toContain("subgroupBroadcast(lane_input3_7, 7u)");
    expect(source).toContain("sum3 = sum3 +");
    expect(source).toContain("output_channel + 3u");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("var<workgroup>");
  });

  it("rejects malformed weights and output ranges", () => {
    const candidate = shape(4, 5, 7, 2);
    const plan = planAceOpt0026VaeConvTranspose1d(candidate);
    expect(() => packAceOpt0026VaeConvTranspose1dWeights(
      new Uint16Array(plan.weightElements - 1),
      candidate,
    )).toThrow(/require/);
    expect(() => planAceOpt0026VaeConvTranspose1dRange(plan, {
      base: 1,
      count: plan.outputElements,
    })).toThrow(/complete in-bounds NLC rows/);
  });
});

function shape(
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  stride: number,
): AceVaeConvTranspose1dShape {
  return Object.freeze({
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 2 * stride,
    stride,
    dilation: 1,
    padding: stride / 2,
    outputPadding: 0,
  });
}

function packFloatWeights(
  native: Float32Array,
  candidate: AceVaeConvTranspose1dShape,
): Float32Array {
  const packed = new Float32Array(native.length);
  for (let phase = 0; phase < candidate.stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      const kernel = phase + tap * candidate.stride;
      for (let inputChannel = 0;
        inputChannel < candidate.inputChannels; inputChannel += 1) {
        for (let outputChannel = 0;
          outputChannel < candidate.outputChannels; outputChannel += 1) {
          packed[aceOpt0026PolyphaseWeightIndex(
            candidate,
            phase,
            tap,
            inputChannel,
            outputChannel,
          )] = native[aceOpt0026NativeWeightIndex(
            candidate,
            outputChannel,
            kernel,
            inputChannel,
          )]!;
        }
      }
    }
  }
  return packed;
}

function nativeOutput(
  candidate: AceVaeConvTranspose1dShape,
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
): Float32Array {
  const outputFrames = candidate.inputFrames * candidate.stride;
  const output = new Float32Array(outputFrames * candidate.outputChannels);
  for (let outputTime = 0; outputTime < outputFrames; outputTime += 1) {
    const padded = outputTime + candidate.padding;
    for (let outputChannel = 0;
      outputChannel < candidate.outputChannels; outputChannel += 1) {
      let sum = bias[outputChannel]!;
      for (let kernel = 0; kernel < candidate.kernelSize; kernel += 1) {
        if (padded < kernel || (padded - kernel) % candidate.stride !== 0) {
          continue;
        }
        const inputTime = (padded - kernel) / candidate.stride;
        if (inputTime >= candidate.inputFrames) continue;
        for (let inputChannel = 0;
          inputChannel < candidate.inputChannels; inputChannel += 1) {
          sum = Math.fround(sum + Math.fround(
            input[inputTime * candidate.inputChannels + inputChannel]! *
              weight[aceOpt0026NativeWeightIndex(
                candidate,
                outputChannel,
                kernel,
                inputChannel,
              )]!,
          ));
        }
      }
      output[outputTime * candidate.outputChannels + outputChannel] = sum;
    }
  }
  return output;
}

function polyphaseOutput(
  candidate: AceVaeConvTranspose1dShape,
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
): Float32Array {
  const outputFrames = candidate.inputFrames * candidate.stride;
  const output = new Float32Array(outputFrames * candidate.outputChannels);
  for (let outputTime = 0; outputTime < outputFrames; outputTime += 1) {
    const padded = outputTime + candidate.padding;
    const phase = padded % candidate.stride;
    const firstInputTime = Math.floor(padded / candidate.stride);
    for (let outputChannel = 0;
      outputChannel < candidate.outputChannels; outputChannel += 1) {
      let sum = bias[outputChannel]!;
      for (let tap = 0; tap < 2; tap += 1) {
        const inputTime = firstInputTime - tap;
        if (inputTime < 0 || inputTime >= candidate.inputFrames) continue;
        for (let chunk = 0;
          chunk < Math.ceil(
            candidate.inputChannels /
              ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
          ); chunk += 1) {
          for (let member = 0;
            member < ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK;
            member += 1) {
            const inputChannel =
              chunk * ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK +
              member;
            if (inputChannel >= candidate.inputChannels) continue;
            sum = Math.fround(sum + Math.fround(
              input[inputTime * candidate.inputChannels + inputChannel]! *
                weight[aceOpt0026PolyphaseWeightIndex(
                  candidate,
                  phase,
                  tap,
                  inputChannel,
                  outputChannel,
                )]!,
            ));
          }
        }
      }
      output[outputTime * candidate.outputChannels + outputChannel] = sum;
    }
  }
  return output;
}
