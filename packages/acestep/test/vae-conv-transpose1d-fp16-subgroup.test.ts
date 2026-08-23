import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  AceOpt0022VaeConvTranspose1dSubgroupKernel,
  aceOpt0022VaeConvTranspose1dSubgroupWgsl,
  planAceOpt0022VaeConvTranspose1d,
  planAceOpt0022VaeConvTranspose1dRange,
  type AceOpt0022VaeConvTranspose1dBindings,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.js";
import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";

describe("OPT-0022 fixed32 subgroup polyphase ConvTranspose1D", () => {
  it("pins one fixed WG128/four-subgroup 16x128 geometry", () => {
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE).toBe(128);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP).toBe(16);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP).toBe(32);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP).toBe(128);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS).toBe(2);
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK).toBe(8);
    expect(
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE *
        ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
    ).toBe(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE);
    expect(
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP *
        ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
    ).toBe(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP);

    const plan = planAceOpt0022VaeConvTranspose1d(
      tailShape(65, 137, 6),
    );
    expect(plan).toMatchObject({
      kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
      weightLayout: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
      outputFrames: 102,
      inputElements: 1_105,
      weightElements: 106_860,
      outputElements: 13_974,
      inputStorageBytes: 2_210,
      inputBindingBytes: 2_212,
      polyphaseWeightStorageBytes: 213_720,
      polyphaseWeightBindingBytes: 213_720,
      biasStorageBytes: 274,
      biasBindingBytes: 276,
      outputStorageBytes: 27_948,
      outputBindingBytes: 27_948,
      inputChannelChunkCount: 9,
      workgroupStorageBytes: 0,
      workgroupBarrierCount: 0,
    });
  });

  it("owns OC, Cin, and range tails exactly once", () => {
    const shape = tailShape(65, 137, 6);
    const plan = planAceOpt0022VaeConvTranspose1d(shape);
    const base = 5 * shape.outputChannels;
    const count = 17 * shape.outputChannels;
    const range = planAceOpt0022VaeConvTranspose1dRange(plan, {
      base,
      count,
    });
    expect(range).toEqual({
      base,
      count,
      batch: 0,
      firstOutputTime: 5,
      firstOutputRow: 5,
      outputRowCount: 17,
      workgroupsX: 1,
      workgroupsY: 2,
      workgroupsZ: 6,
    });

    const owners = new Uint8Array(count);
    for (let x = 0; x < range.workgroupsX; x += 1) {
      for (let y = 0; y < range.workgroupsY; y += 1) {
        for (let phase = 0; phase < range.workgroupsZ; phase += 1) {
          for (
            let subgroup = 0;
            subgroup < ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP;
            subgroup += 1
          ) {
            for (
              let lane = 0;
              lane < ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE;
              lane += 1
            ) {
              const outputChannel =
                y * ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP +
                subgroup *
                  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP +
                lane;
              if (outputChannel >= shape.outputChannels) continue;
              for (
                let row = 0;
                row < ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP;
                row += 1
              ) {
                const outputOffset = phase +
                  (x * ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP +
                    row) * shape.stride;
                if (outputOffset >= range.outputRowCount) continue;
                const index = outputOffset * shape.outputChannels +
                  outputChannel;
                owners[index] = owners[index]! + 1;
              }
            }
          }
        }
      }
    }
    expect([...owners].every((ownerCount) => ownerCount === 1)).toBe(true);
    expect(plan.inputChannels % plan.inputChannelChunk).toBe(1);
    expect(plan.outputChannels % plan.channelsPerWorkgroup).toBe(9);

    expect(() => planAceOpt0022VaeConvTranspose1dRange(plan, {
      base: 1,
      count,
    })).toThrow(/complete in-bounds NLC rows/);
    expect(() => planAceOpt0022VaeConvTranspose1dRange(plan, {
      base,
      count: count + plan.outputElements,
    })).toThrow(/complete in-bounds NLC rows/);
  });

  it("maps native O-K-I bits bijectively to phase-tap-input-output", () => {
    for (const stride of [2, 4, 6, 10] as const) {
      const shape = {
        ...tailShape(5, 7, stride),
        inputFrames: 3,
      };
      const native = Uint16Array.from(
        { length: shape.outputChannels * shape.kernelSize * shape.inputChannels },
        (_, index) => [
          0x0000,
          0x8000,
          0x0001,
          0x8001,
          0x7e01,
          0x7c00,
          0xfc00,
          index & 0xffff,
        ][index % 8]!,
      );
      const polyphase = packPolyphaseReference(native, shape);
      expect(polyphase).toHaveLength(native.length);
      const visited = new Uint8Array(native.length);
      for (let phase = 0; phase < stride; phase += 1) {
        for (let tap = 0; tap < 2; tap += 1) {
          const kernel = phase + tap * stride;
          for (let inputChannel = 0; inputChannel < 5; inputChannel += 1) {
            for (let outputChannel = 0; outputChannel < 7; outputChannel += 1) {
              const nativeIndex =
                (outputChannel * shape.kernelSize + kernel) * 5 +
                inputChannel;
              const packedIndex =
                ((phase * 2 + tap) * 5 + inputChannel) * 7 + outputChannel;
              expect(polyphase[packedIndex]).toBe(native[nativeIndex]);
              visited[nativeIndex] = visited[nativeIndex]! + 1;
            }
          }
        }
      }
      expect([...visited].every((count) => count === 1)).toBe(true);
    }
  });

  it("matches the exact kernel-then-Cin term sequence at every boundary", () => {
    for (const stride of [2, 4, 6, 10] as const) {
      const shape = {
        ...tailShape(5, 7, stride),
        inputFrames: 4,
      };
      for (let outputTime = 0;
        outputTime < shape.inputFrames * stride;
        outputTime += 1) {
        const reference: string[] = [];
        const paddedTime = outputTime + shape.padding;
        for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
          if (paddedTime < kernel) continue;
          const numerator = paddedTime - kernel;
          if (numerator % stride !== 0) continue;
          const inputTime = numerator / stride;
          if (inputTime >= shape.inputFrames) continue;
          for (let inputChannel = 0; inputChannel < 5; inputChannel += 1) {
            reference.push(`${kernel}:${inputTime}:${inputChannel}`);
          }
        }

        const candidate: string[] = [];
        const phase = paddedTime % stride;
        const firstInputTime = Math.floor(paddedTime / stride);
        for (let tap = 0; tap < 2; tap += 1) {
          const inputTime = firstInputTime - tap;
          if (inputTime < 0 || inputTime >= shape.inputFrames) continue;
          const kernel = phase + tap * stride;
          for (let chunk = 0; chunk < Math.ceil(5 / 8); chunk += 1) {
            for (let member = 0; member < 8; member += 1) {
              const inputChannel = chunk * 8 + member;
              if (inputChannel >= 5) continue;
              candidate.push(`${kernel}:${inputTime}:${inputChannel}`);
            }
          }
        }
        expect(candidate, `stride=${stride} output=${outputTime}`)
          .toEqual(reference);
      }
    }
  });

  it("emits uniform chunk8 broadcasts and exact scalar FP32 adds", () => {
    const source = aceOpt0022VaeConvTranspose1dSubgroupWgsl(
      tailShape(65, 137, 6),
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID}`,
    );
    expect(source).toContain(
      `// weight-layout: ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain("subgroup_size != 32u");
    expect(source).toContain("subgroup >= 4u");
    expect(source).toContain("group.x * 16u");
    expect(source).toContain("group.y * 128u");
    expect(source).toContain("subgroup * 32u");
    expect(source).toContain("let phase = group.z;");
    expect(source).toContain(
      "let congruent_kernel = phase_first_padded_time % STRIDE;",
    );
    expect(source).toContain("for (var tap = 0u; tap < 2u; tap += 1u)");
    expect(source).toContain(
      "input_channel_chunk < INPUT_CHANNEL_CHUNKS;",
    );
    expect(source).toContain(
      "((congruent_kernel * 2u + tap) * INPUT_CHANNELS +",
    );
    expect(source).toContain(
      "input_channel0) * OUTPUT_CHANNELS + output_channel",
    );
    expect(source.match(/subgroupBroadcast\(/g)).toHaveLength(16 * 8);
    expect(source.match(/ = sum\d+ \+/g)).toHaveLength(16 * 8);
    expect(source.match(/\] = f16\(sum\d+\);/g)).toHaveLength(16);
    for (let row = 0; row < 16; row += 1) {
      for (let member = 0; member < 8; member += 1) {
        const broadcast = source.indexOf(
          `subgroupBroadcast(lane_input${row}, ${member}u);`,
        );
        const guardedAdd = source.indexOf(
          `sum${row} = sum${row} +`,
          broadcast,
        );
        expect(broadcast).toBeGreaterThan(0);
        expect(guardedAdd).toBeGreaterThan(broadcast);
      }
    }
    const tap = source.indexOf("var tap = 0u;");
    const chunk = source.indexOf("var input_channel_chunk = 0u;", tap);
    const weight = source.indexOf("var weight_operand0: f16", chunk);
    const rowInput = source.indexOf("var lane_input0: f32", weight);
    const add = source.indexOf("sum0 = sum0 +", rowInput);
    expect(tap).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(weight);
    expect(weight).toBeLessThan(rowInput);
    expect(rowInput).toBeLessThan(add);
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bsubgroup(Add|Mul|Min|Max)\s*\(/);
    expect(source).not.toMatch(/\batomic\w*\s*\(/);
  });

  it("loads all eight broadcast sources independently of an OC=129 tail", () => {
    const shape = tailShape(65, 129, 6);
    const plan = planAceOpt0022VaeConvTranspose1d(shape);
    expect(plan.outputChannels % plan.channelsPerWorkgroup).toBe(1);

    // In the last channel workgroup, subgroup zero has only lane zero's output
    // active. Lanes 0..7 must nevertheless load all eight source Cin members
    // because their broadcasts feed that one active output lane.
    const lastGroupFirstOutputChannel =
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP;
    const activeOutputLanes = Array.from(
      { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE },
      (_, lane) => lastGroupFirstOutputChannel + lane < shape.outputChannels,
    );
    expect(activeOutputLanes.filter(Boolean)).toHaveLength(1);
    const sourceLoadLanes = Array.from(
      { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK },
      (_, lane) => lane,
    );
    expect(sourceLoadLanes).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    const nonzeroInputs = sourceLoadLanes.map((lane) => lane + 1);
    const nonzeroWeights = sourceLoadLanes.map((lane) => 9 - lane);
    const completeDot = sourceLoadLanes.reduce(
      (sum, lane) => sum + nonzeroInputs[lane]! * nonzeroWeights[lane]!,
      0,
    );
    const incorrectlyOutputGatedDot = sourceLoadLanes.reduce(
      (sum, lane) => activeOutputLanes[lane]
        ? sum + nonzeroInputs[lane]! * nonzeroWeights[lane]!
        : sum,
      0,
    );
    expect(completeDot).toBe(156);
    expect(incorrectlyOutputGatedDot).toBe(9);

    const source = aceOpt0022VaeConvTranspose1dSubgroupWgsl(shape);
    expect(source).toContain(
      "let output_row_active0 =\n" +
        "    output_range_offset0 < output_row_count;",
    );
    expect(source).toContain(
      "let output_active0 =\n" +
        "    output_channel_active && output_row_active0;",
    );
    const laneLoadStart = source.indexOf("var lane_input0: f32 = 0.0;");
    const laneLoadEnd = source.indexOf(
      "subgroupBroadcast(lane_input0, 0u);",
      laneLoadStart,
    );
    const laneLoadBlock = source.slice(laneLoadStart, laneLoadEnd);
    expect(laneLoadBlock).toContain(
      "output_row_active0 && input_valid0",
    );
    expect(laneLoadBlock).not.toContain("output_active0");
    const addBlock = source.slice(
      laneLoadEnd,
      source.indexOf("var lane_input1: f32 = 0.0;", laneLoadEnd),
    );
    expect(addBlock).toContain("output_active0 && input_valid0");
  });

  it("pins all five C300 ranges and exact aggregate accounting", () => {
    const graph = planAceVaeDecoder(300);
    const quanta = planAceVaeDecoderQuanta(graph);
    const candidateWorkgroups: number[] = [];
    let exactRangeCount = 0;
    let baselineWorkgroupsTotal = 0;
    let candidateWorkgroupsTotal = 0;
    let candidateScheduledMacs = 0;
    let validMacs = 0;
    let baselineBarrierEvents = 0;
    let baselineInputBytes = 0;
    let candidateInputBytes = 0;
    let candidateWeightBytes = 0;

    for (const [operationIndex, operation] of graph.operations.entries()) {
      if (operation.kind !== "conv-transpose1d") continue;
      const plan = planAceOpt0022VaeConvTranspose1d(operation.shape);
      const basePlan = planAceFp16VaeConvTranspose1d(operation.shape);
      const ranges = quanta.quanta.filter(
        (quantum) => quantum.operationIndex === operationIndex,
      );
      exactRangeCount += ranges.length;
      let operationCandidateWorkgroups = 0;
      for (const quantum of ranges) {
        const range = {
          base: quantum.logicalOutputBase,
          count: quantum.logicalOutputCount,
        };
        const candidate = planAceOpt0022VaeConvTranspose1dRange(plan, range);
        const baseline = planAceFp16VaeConvTranspose1dCongruentRange(
          basePlan,
          range,
        );
        const candidateCount = candidate.workgroupsX *
          candidate.workgroupsY * candidate.workgroupsZ;
        const baselineCount = baseline.workgroupsX *
          baseline.workgroupsY * baseline.workgroupsZ;
        operationCandidateWorkgroups += candidateCount;
        candidateWorkgroupsTotal += candidateCount;
        baselineWorkgroupsTotal += baselineCount;
        candidateScheduledMacs += candidateCount *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS *
          plan.inputChannelChunkCount *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK;
        candidateWeightBytes += candidateCount *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS *
          plan.inputChannelChunkCount *
          ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK * 2;
        baselineBarrierEvents += baselineCount * 2 *
          basePlan.inputChannelChunkCount * 2;
      }
      candidateWorkgroups.push(operationCandidateWorkgroups);

      const validTapRows = countValidTapRows(operation.shape);
      validMacs += validTapRows * operation.shape.inputChannels *
        operation.shape.outputChannels;
      baselineInputBytes += validTapRows * operation.shape.inputChannels *
        Math.ceil(operation.shape.outputChannels / 8) * 2;
      candidateInputBytes += validTapRows * operation.shape.inputChannels *
        Math.ceil(
          operation.shape.outputChannels /
            ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
        ) * ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP * 2;
    }

    expect(candidateWorkgroups).toEqual([
      4_320,
      5_784,
      9_000,
      18_000,
      36_000,
    ]);
    expect(exactRangeCount).toBe(378);
    expect(baselineWorkgroupsTotal).toBe(1_169_664);
    expect(candidateWorkgroupsTotal).toBe(73_104);
    expect(baselineWorkgroupsTotal / candidateWorkgroupsTotal).toBe(16);
    expect(candidateWorkgroupsTotal * 4).toBe(292_416);
    expect(baselineBarrierEvents).toBe(28_594_176);
    expect(candidateScheduledMacs).toBe(117_121_744_896);
    expect(validMacs).toBe(88_055_578_624);
    expect(baselineInputBytes).toBe(22_013_894_656);
    expect(candidateInputBytes).toBe(5_503_473_664);
    expect(candidateWeightBytes).toBe(14_640_218_112);
    expect(candidateInputBytes + candidateWeightBytes)
      .toBe(20_143_691_776);
  });

  it("compiles once, caches bindings, and encodes immutable range controls", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      device,
      fixed32Capability,
    );
    const shape = tailShape(65, 137, 6);
    const plan = planAceOpt0022VaeConvTranspose1d(shape);
    const bindings = bindingsFor(shape);
    const control = fakeBuffer(1_024);
    const first = await kernel.createDispatch(
      "first",
      shape,
      bindings,
      rangeBinding(control, 256, 5 * 137, 17 * 137),
    );
    const second = await kernel.createDispatch(
      "second",
      shape,
      bindings,
      rangeBinding(control, 512, 22 * 137, 13 * 137),
    );
    expect(first.outputRange).toMatchObject({
      firstOutputTime: 5,
      outputRowCount: 17,
      workgroupsX: 1,
      workgroupsY: 2,
      workgroupsZ: 6,
    });
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).map(({ buffer }) => buffer?.minBindingSize))
      .toEqual([
        plan.inputBindingBytes,
        plan.polyphaseWeightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
        16,
      ]);
    expect(Array.from(layout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const group = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect(Array.from(group.entries).at(1)?.resource)
      .toEqual(bindings.polyphaseWeight);
    expect(Array.from(group.entries).at(-1)?.resource)
      .toEqual({ buffer: control, offset: 0, size: 16 });

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [1, 2, 6],
      [1, 2, 6],
    ]);

    const otherBindings = bindingsFor(shape);
    await kernel.createDispatch(
      "other-bindings",
      shape,
      otherBindings,
      rangeBinding(control, 768, 0, 5 * 137),
    );
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);

    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      bindings,
      rangeBinding(control, 256, 5 * 137, 17 * 137),
    )).rejects.toThrow(/was destroyed/);
  });

  it("fails closed before compilation on capability, limits, and aliases", async () => {
    expect(() => AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      fakeDevice({ shaderF16: false }),
      fixed32Capability,
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      fakeDevice({ subgroups: false }),
      fixed32Capability,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      fixed32Capability,
    )).toThrow(/requires WG128 in X/);
    expect(() => AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 127 }),
      fixed32Capability,
    )).toThrow(/requires WG128 in X/);

    const shape = tailShape(65, 137, 10);
    const plan = planAceOpt0022VaeConvTranspose1d(shape);
    const bindings = bindingsFor(shape);
    const dispatchDevice = fakeDevice({ maximumDispatch: 9 });
    await expect(kernelFor(dispatchDevice).createDispatch(
      "dispatch-limit",
      shape,
      bindings,
      fullRange(fakeBuffer(256), shape),
    )).rejects.toThrow(/dispatch dimension/);
    expect(dispatchDevice.createShaderModule).not.toHaveBeenCalled();

    const uniformDevice = fakeDevice({ maximumUniformBinding: 15 });
    await expect(kernelFor(uniformDevice).createDispatch(
      "uniform-limit",
      shape,
      bindings,
      fullRange(fakeBuffer(256), shape),
    )).rejects.toThrow(/invalid buffer limits/);
    expect(uniformDevice.createShaderModule).not.toHaveBeenCalled();

    const storageDevice = fakeDevice({
      maximumStorageBinding: plan.polyphaseWeightBindingBytes - 4,
    });
    await expect(kernelFor(storageDevice).createDispatch(
      "storage-limit",
      shape,
      bindings,
      fullRange(fakeBuffer(256), shape),
    )).rejects.toThrow(/polyphase weight exceeds the device buffer limits/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const shortDevice = fakeDevice();
    await expect(kernelFor(shortDevice).createDispatch(
      "short-weight",
      shape,
      {
        ...bindings,
        polyphaseWeight: fakeBinding(plan.polyphaseWeightBindingBytes - 4),
      },
      fullRange(fakeBuffer(256), shape),
    )).rejects.toThrow(/polyphase weight does not expose/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const alignmentDevice = fakeDevice({ uniformAlignment: 512 });
    await expect(kernelFor(alignmentDevice).createDispatch(
      "control-alignment",
      shape,
      bindings,
      fullRange(fakeBuffer(1_024), shape, 256),
    )).rejects.toThrow(/aligned 16-byte immutable record/);
    expect(alignmentDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const shared = fakeBuffer(
      plan.inputBindingBytes + plan.outputBindingBytes,
    );
    await expect(kernelFor(aliasDevice).createDispatch(
      "alias",
      shape,
      {
        ...bindings,
        input: {
          buffer: shared,
          offset: 0,
          size: plan.inputBindingBytes,
        },
        output: {
          buffer: shared,
          offset: 256,
          size: plan.outputBindingBytes,
        },
      },
      fullRange(fakeBuffer(256), shape),
    )).rejects.toThrow(/input and output bindings must not overlap/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("evicts compile failures and rejects an in-flight destruction race", async () => {
    const diagnosticDevice = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic subgroup diagnostic",
        type: "error",
        lineNum: 91,
        linePos: 7,
      }], []],
    });
    const shape = tailShape(65, 137, 6);
    const bindings = bindingsFor(shape);
    const range = fullRange(fakeBuffer(256), shape);
    const diagnosticKernel = kernelFor(diagnosticDevice);
    await expect(diagnosticKernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      range,
    )).rejects.toThrow(/91:7 synthetic subgroup diagnostic/);
    expect(diagnosticDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    await expect(diagnosticKernel.createDispatch(
      "retry",
      shape,
      bindings,
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
    });
    expect(diagnosticDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(diagnosticDevice.createComputePipelineAsync).toHaveBeenCalledOnce();

    const gate = deferred<void>();
    const raceDevice = fakeDevice({ pipelineGate: gate.promise });
    const raceKernel = kernelFor(raceDevice);
    const pending = raceKernel.createDispatch(
      "race",
      shape,
      bindingsFor(shape),
      fullRange(fakeBuffer(256), shape),
    );
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    expect(raceDevice.createComputePipelineAsync).toHaveBeenCalledOnce();
    raceKernel.destroy();
    gate.resolve();
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(raceDevice.createBindGroup).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const fixed32Capability = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

function tailShape(
  inputChannels: number,
  outputChannels: number,
  stride: 2 | 4 | 6 | 10,
): AceVaeConvTranspose1dShape {
  return {
    batch: 1,
    inputFrames: 17,
    inputChannels,
    outputChannels,
    kernelSize: 2 * stride,
    stride,
    dilation: 1,
    padding: Math.ceil(stride / 2),
    outputPadding: 0,
  };
}

function packPolyphaseReference(
  native: Uint16Array,
  shape: AceVaeConvTranspose1dShape,
): Uint16Array {
  const packed = new Uint16Array(native.length);
  for (let phase = 0; phase < shape.stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      const kernel = phase + tap * shape.stride;
      for (
        let inputChannel = 0;
        inputChannel < shape.inputChannels;
        inputChannel += 1
      ) {
        for (
          let outputChannel = 0;
          outputChannel < shape.outputChannels;
          outputChannel += 1
        ) {
          packed[
            ((phase * 2 + tap) * shape.inputChannels + inputChannel) *
              shape.outputChannels + outputChannel
          ] = native[
            (outputChannel * shape.kernelSize + kernel) *
              shape.inputChannels + inputChannel
          ]!;
        }
      }
    }
  }
  return packed;
}

function countValidTapRows(shape: AceVaeConvTranspose1dShape): number {
  let count = 0;
  const outputFrames = planAceOpt0022VaeConvTranspose1d(shape).outputFrames;
  for (let outputTime = 0; outputTime < outputFrames; outputTime += 1) {
    const paddedTime = outputTime + shape.padding;
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      if (paddedTime < kernel) continue;
      const numerator = paddedTime - kernel;
      if (numerator % shape.stride !== 0) continue;
      if (numerator / shape.stride >= shape.inputFrames) continue;
      count += 1;
    }
  }
  return count;
}

function bindingsFor(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0022VaeConvTranspose1dBindings {
  const plan = planAceOpt0022VaeConvTranspose1d(shape);
  return {
    input: fakeBinding(plan.inputBindingBytes),
    polyphaseWeight: fakeBinding(plan.polyphaseWeightBindingBytes),
    bias: fakeBinding(plan.biasBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
}

function fullRange(
  control: GPUBuffer,
  shape: AceVaeConvTranspose1dShape,
  offset = 0,
): AceVaeOutputRangeBinding {
  const plan = planAceOpt0022VaeConvTranspose1d(shape);
  return rangeBinding(control, offset, 0, plan.outputElements);
}

function rangeBinding(
  buffer: GPUBuffer,
  offset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return {
    base,
    count,
    control: { buffer, offset, size: 16 },
  };
}

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroupLayout: ReturnType<typeof vi.fn>;
  readonly createPipelineLayout: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly shaderF16?: boolean;
  readonly subgroups?: boolean;
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumUniformBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
  readonly pipelineGate?: Promise<void>;
} = {}): FakeDevice {
  const features: string[] = [];
  if (options.shaderF16 !== false) features.push("shader-f16");
  if (options.subgroups !== false) features.push("subgroups");
  const compilationMessageBatches = [
    ...(options.compilationMessageBatches ?? [[]]),
  ];
  return {
    features: new Set(features),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxUniformBufferBindingSize:
        options.maximumUniformBinding ?? 65_536,
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      minUniformBufferOffsetAlignment: options.uniformAlignment ?? 256,
      minStorageBufferOffsetAlignment: options.storageAlignment ?? 256,
    },
    createShaderModule: vi.fn(() => {
      const messages = compilationMessageBatches.shift() ?? [];
      return {
        label: "module",
        getCompilationInfo: vi.fn(async () => ({ messages })),
      };
    }),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => {
      await options.pipelineGate;
      return fakePipeline();
    }),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
  } as unknown as FakeDevice;
}

function kernelFor(
  device: GPUDevice,
): AceOpt0022VaeConvTranspose1dSubgroupKernel {
  return AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
    device,
    fixed32Capability,
  );
}

function fakePipeline(): GPUComputePipeline {
  return { label: "pipeline" } as GPUComputePipeline;
}

function fakePass(): GPUComputePassEncoder & {
  readonly setBindGroup: ReturnType<typeof vi.fn>;
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly setBindGroup: ReturnType<typeof vi.fn>;
    readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
  };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => {
    resolve = accept;
  });
  return { promise, resolve };
}
