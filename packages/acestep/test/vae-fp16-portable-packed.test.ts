import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
  aceOpt0028VaeK1PortablePackedWeightIndex,
  aceOpt0028VaeK1PortablePackedWgsl,
  planAceOpt0028VaeK1PortablePacked,
} from "../src/webgpu/kernels/vae-k1-fp16-portable-packed.js";
import {
  packAceOpt0025VaeK1WeightU16,
} from "../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
  aceOpt0028VaeConvTranspose1dPortablePackedWeightIndex,
  aceOpt0028VaeConvTranspose1dPortablePackedWgsl,
  planAceOpt0028VaeConvTranspose1dPortablePacked,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-portable-packed.js";
import {
  aceOpt0026PolyphaseWeightIndex,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  planAceFp16VaeConv1dRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeConvTranspose1dShape,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("OPT-0028 portable exact-packed kernel counterparts", () => {
  it("indexes revision-6 K1 tiles without a native-layout copy", () => {
    const shape = k1Shape(37, 128);
    const plan = planAceOpt0028VaeK1PortablePacked(shape);
    expect(plan).toMatchObject({
      kernelId: ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
      family: "k1",
      outputStorage: "float16",
      weightLayout: "conv1d-k1-cout128-cin32-tile-major-f16-v1",
      packedWeightStorageShape: [1, 4, 32, 128],
      workgroupSize: 128,
    });
    expect(plan.workgroupStorageBytes).toBeGreaterThan(0);

    const native = Uint16Array.from(
      { length: plan.weightElements },
      (_, index) => index,
    );
    const packed = packAceOpt0025VaeK1WeightU16(native, 128, 128);
    const visited = new Uint8Array(plan.weightElements);
    for (let inputChannel = 0; inputChannel < 128; inputChannel += 1) {
      for (let outputChannel = 0; outputChannel < 128; outputChannel += 1) {
        const packedIndex = aceOpt0028VaeK1PortablePackedWeightIndex(
          shape,
          inputChannel,
          outputChannel,
        );
        expect(packed[packedIndex]).toBe(
          native[outputChannel * 128 + inputChannel],
        );
        visited[packedIndex] = visited[packedIndex]! + 1;
      }
    }
    expect([...visited].every((count) => count === 1)).toBe(true);

    const range = planAceFp16VaeConv1dRange(plan, {
      base: 3 * 128,
      count: 19 * 128,
    });
    expect(range).toMatchObject({
      firstOutputTime: 3,
      outputRowCount: 19,
      workgroupsX: 2,
      workgroupsY: 16,
    });
  });

  it("stages packed K1 operands in workgroup memory and retains Cin order", () => {
    const source = aceOpt0028VaeK1PortablePackedWgsl(k1Shape(37, 128));
    expect(source).toContain("var<workgroup> input_tile");
    expect(source).toContain("var<workgroup> weight_tile");
    expect(source).toMatch(/inner_tile = input_channel \/\s*32u/);
    expect(source).toContain("column_tile * PACKED_INNER_TILES");
    expect(source).toContain("input_channel_chunk += 1u");
    expect(source).toContain("chunk_channel += 1u");
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expect(source).toContain("= f16(sum);");
    expect(source.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    expect(source).not.toContain("enable subgroups");
    expect(source).not.toContain("subgroupBroadcast");
  });

  it("indexes every polyphase coordinate exactly as revision 6", () => {
    for (const stride of [2, 4, 6, 10]) {
      const shape = transposeShape(17, 32, 24, stride);
      const plan = planAceOpt0028VaeConvTranspose1dPortablePacked(shape);
      expect(plan).toMatchObject({
        kernelId:
          ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
        weightLayout:
          "conv-transpose1d-phase-tap-input-output-f16-v1",
        packedWeightStorageShape: [stride, 2, 32, 24],
        workgroupSize: 128,
      });
      expect(plan.workgroupStorageBytes).toBeGreaterThan(0);
      const visited = new Uint8Array(plan.weightElements);
      for (let phase = 0; phase < stride; phase += 1) {
        for (let tap = 0; tap < 2; tap += 1) {
          for (let inputChannel = 0; inputChannel < 32; inputChannel += 1) {
            for (let outputChannel = 0; outputChannel < 24;
              outputChannel += 1) {
              const index =
                aceOpt0028VaeConvTranspose1dPortablePackedWeightIndex(
                  shape,
                  phase,
                  tap,
                  inputChannel,
                  outputChannel,
                );
              expect(index).toBe(aceOpt0026PolyphaseWeightIndex(
                shape,
                phase,
                tap,
                inputChannel,
                outputChannel,
              ));
              visited[index] = visited[index]! + 1;
            }
          }
        }
      }
      expect([...visited].every((count) => count === 1)).toBe(true);

      const range = planAceFp16VaeConvTranspose1dCongruentRange(plan, {
        base: 5 * shape.outputChannels,
        count: 20 * shape.outputChannels,
      });
      expect(range.workgroupsZ).toBe(stride);
      expect(range.workgroupsY).toBe(3);
    }
  });

  it("stages polyphase operands and retains tap-then-Cin FP32 order", () => {
    const source = aceOpt0028VaeConvTranspose1dPortablePackedWgsl(
      transposeShape(17, 32, 24, 6),
    );
    expect(source).toContain("var<workgroup> input_tile");
    expect(source).toContain("var<workgroup> weight_tile");
    expect(source).toContain("for (var tap = 0u; tap < 2u; tap += 1u)");
    expect(source).toContain("input_channel_chunk += 1u");
    expect(source).toContain("chunk_channel += 1u");
    expect(source).toContain(
      "((congruent_phase * 2u + tap) * INPUT_CHANNELS + input_channel)",
    );
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expect(source).toContain("= f16(sum);");
    expect(source.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    expect(source).not.toContain("enable subgroups");
    expect(source).not.toContain("subgroupBroadcast");
  });
});

function k1Shape(
  frames: number,
  channels: number,
): AceVaeConv1dShape {
  return Object.freeze({
    batch: 1,
    inputFrames: frames,
    inputChannels: channels,
    outputChannels: channels,
    kernelSize: 1,
    stride: 1,
    dilation: 1,
    padding: 0,
  });
}

function transposeShape(
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
