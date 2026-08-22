import { describe, expect, it, vi } from "vitest";

import {
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_K1_FP16_TILE_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  type AcePackageTensorRecord,
} from "../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
} from "../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
} from "../src/webgpu/kernels/vae-k1-fp16-portable-packed.js";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-portable-packed.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-shape-selector.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
} from "../src/webgpu/kernels/vae-pointwise-fp16.js";
import {
  ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
} from "../src/webgpu/kernels/vae-snake-fp16.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  type AceOpt0011VaeOperationBindings,
  type AceOpt0011VaePackageBindings,
  type AceOpt0011VaeResolvedTensor,
} from "../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
  ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
  ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES,
  ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8,
  ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES,
  ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT,
  ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT,
  AceOpt0011Fp16VaeDecoderRuntime,
  planAceOpt0011Fp16VaeChunkDispatches,
  planAceOpt0011Fp16VaeDecoderDynamicControls,
  planAceOpt0011Fp16VaeWindowDynamicControls,
  type AceOpt0011Fp16VaeDecoderBindings,
  type AceOpt0011Fp16VaeWindowBindings,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  planAceVaeDecoder,
} from "../src/webgpu/vae-decoder.js";

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6, COPY_DST: 1 << 3 });
vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const PLAN = planAceVaeDecoder(256);
const REVISION7_ROW_REUSE_LABELS: ReadonlySet<string> = new Set(
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(({ operationLabel }) =>
    operationLabel
  ),
);
const REVISION7_TRANSPOSE_BY_LABEL: ReadonlyMap<
  string,
  "channel" | "row"
> = new Map(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(
  ({ operationLabel, reuseAxis }) => [operationLabel, reuseAxis] as const,
));
const PACKAGE = createPackageBindings();
const PACKED_PACKAGE = createPackageBindings(true);
const REVISION7_PACKAGE = createPackageBindings("revision7");

describe("OPT-0011 package-native FP16 VAE decoder runtime", () => {
  it("freezes one dense ingress-plus-graph dynamic-control plan", () => {
    expect(ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT).toBe(88);
    expect(ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT).toBe(3_942);
    expect(ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT).toBe(3_943);
    expect(ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES).toBe(16);
    const controls = planAceOpt0011Fp16VaeDecoderDynamicControls(256);
    expect(controls).toMatchObject({
      recordBytes: 16,
      recordAlignment: 256,
      recordCount: 3_943,
      byteLength: 1_009_168,
    });
    expect(controls.records).toHaveLength(
      ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT,
    );
    expect(controls.records[0]).toEqual({
      recordIndex: 0,
      sequenceIndex: 0,
      graphQuantumIndex: null,
      operationIndex: null,
      operationLabel: "f32-latent-to-f16-decoder-input",
      operationKind: "ingress-cast",
      outputBase: 0,
      outputCount: 16_384,
      byteOffset: 0,
    });
    expect(controls.records.at(-1)).toMatchObject({
      recordIndex: 3_942,
      sequenceIndex: 3_942,
      graphQuantumIndex: 3_941,
      operationIndex: 87,
      operationLabel: "conv2",
      operationKind: "conv1d",
      byteOffset: 3_942 * 256,
    });
    expect(controls.records.every((record, index) =>
      record.recordIndex === index &&
      record.sequenceIndex === index &&
      record.byteOffset === index * 256 &&
      record.outputCount > 0
    )).toBe(true);
    expect(countBy(controls.records.slice(1), (record) =>
      record.operationKind
    )).toEqual({
      conv1d: 2_459,
      "conv-transpose1d": 322,
      snake: 813,
      add: 348,
    });
    expect(Object.isFrozen(controls)).toBe(true);
    expect(Object.isFrozen(controls.records)).toBe(true);
    expect(controls.records.every(Object.isFrozen)).toBe(true);
    expect(() => planAceOpt0011Fp16VaeDecoderDynamicControls(3)).toThrow(
      /valid uniform alignment/,
    );
    expect(() => planAceOpt0011Fp16VaeDecoderDynamicControls(2 ** 21))
      .toThrow(/exceeds WGSL's u32 domain/);
  });

  it("composes all committed kernels in exact B-256 FIFO order", async () => {
    const device = fakeDevice();
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
    const bindings = createRuntimeBindings(PACKAGE);
    const dispatch = await runtime.createDecoderDispatch("complete-b256", bindings);

    expect(dispatch).toMatchObject({
      label: "complete-b256",
      runtimeProfileId: "opt-0011-mixed-fp16-portable-v1",
      kernelSetId: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY.id,
      kernelTopology: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY,
      operationCount: 88,
      graphQuantumCount: 3_942,
      primitiveCount: 3_943,
    });
    expect(dispatch.plan).toMatchObject({
      inputFrames: 256,
      outputFrames: 491_520,
      primitiveCount: 88,
    });
    expect(dispatch.precisionMap).toBe(ACE_OPT_0011_VAE_FP16_PRECISION_MAP);
    expect(dispatch.precisionMap.entries).toHaveLength(89);
    expect(dispatch.graphQuanta).toHaveLength(3_942);
    expect(dispatch.quanta).toHaveLength(3_943);
    expect(dispatch.quanta[0]).toBe(dispatch.ingressQuantum);
    expect(dispatch.quanta.slice(1)).toEqual(dispatch.graphQuanta);
    expect(dispatch.quanta.map((quantum) => quantum.sequenceIndex)).toEqual(
      Array.from({ length: 3_943 }, (_, index) => index),
    );
    expect(countBy(dispatch.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 2_459,
      [ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID]: 322,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 813,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 348,
    });
    expect(dispatch.graphQuanta.every((quantum) =>
      quantum.graphQuantumIndex !== null &&
      quantum.operationIndex !== null &&
      quantum.control.sequenceIndex === quantum.sequenceIndex &&
      quantum.precision.graphOperationIndex === quantum.operationIndex &&
      quantum.precision.output.storage ===
        (quantum.operationIndex === 87 ? "float32" : "float16")
    )).toBe(true);
    const finalQuanta = dispatch.graphQuanta.filter((quantum) =>
      quantum.operationIndex === 87
    );
    expect(finalQuanta).toHaveLength(4);
    expect(finalQuanta.every((quantum) =>
      quantum.operationLabel === "conv2" &&
      quantum.kernelId === ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID &&
      quantum.precision.kernelFamily === "conv1d-k7" &&
      quantum.precision.parameters.length === 1 &&
      quantum.precision.output.storage === "float32"
    )).toBe(true);

    expect(device.createBuffer).toHaveBeenCalledOnce();
    const descriptor = device.createBuffer.mock.calls[0]?.[0] as
      GPUBufferDescriptor;
    expect(descriptor).toEqual({
      label: "complete-b256-fp16-vae-range-controls",
      size: 1_009_168,
      usage: (1 << 6) | (1 << 3),
    });
    expect(device.queue.writeBuffer).toHaveBeenCalledOnce();
    const payload = device.queue.writeBuffer.mock.calls[0]?.[2] as Uint32Array;
    expect(payload).toBeInstanceOf(Uint32Array);
    for (const index of [0, 1, 2, 1_337, 3_942]) {
      const record = dispatch.dynamicControls.records[index]!;
      const wordOffset = record.byteOffset / 4;
      expect(payload[wordOffset]).toBe(record.outputBase);
      expect(payload[wordOffset + 1]).toBe(record.outputCount);
      expect(payload[wordOffset + 2]).toBe(0);
      expect(payload[wordOffset + 3]).toBe(0);
    }

    const pass = fakePass();
    for (const quantum of dispatch.quanta) quantum.encode(pass);
    expect(pass.setBindGroup).toHaveBeenCalledTimes(3_943);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(3_943);
    expect(pass.setBindGroup.mock.calls.every((call, index) =>
      (call[2] as readonly number[])[0] === index * 256
    )).toBe(true);
    expect(pass.setBindGroup.mock.calls[0]?.[2]).toEqual([0]);
    expect(pass.setBindGroup.mock.calls.at(-1)?.[2]).toEqual([3_942 * 256]);
    expect(Object.isFrozen(dispatch)).toBe(true);
    expect(Object.isFrozen(dispatch.quanta)).toBe(true);
    expect(dispatch.quanta.every(Object.isFrozen)).toBe(true);

    const controlBuffer = device.createdBuffers[0]!;
    runtime.destroy();
    runtime.destroy();
    expect(controlBuffer.destroyMock).toHaveBeenCalledOnce();
    expect(() => dispatch.ingressQuantum.encode(pass)).toThrow(/was destroyed/);
    await expect(runtime.createDecoderDispatch("after-destroy", bindings))
      .rejects.toThrow(/was destroyed/);
  }, 30_000);

  it("routes every B/C K7 quantum through the explicit fixed32 hybrid", async () => {
    const device = fakeDevice({
      subgroups: true,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    expect(runtime).toMatchObject({
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      kernelSetId:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
      precisionMap:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
    });

    const b = await runtime.createDecoderDispatch(
      "hybrid-b256",
      createRuntimeBindings(PACKAGE),
    );
    expect(b.precisionMap).toBe(
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
    );
    expect(countBy(b.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 2_045,
      [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 414,
      [ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID]: 322,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 813,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 348,
    });
    expect(b.graphQuanta.every((quantum) =>
      quantum.precision.kernelFamily === "conv1d-k7"
        ? quantum.kernelId === ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
        : quantum.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
    )).toBe(true);

    const c = await runtime.createChunkDispatchSet(
      "hybrid-c512",
      512,
      512,
      createWindowBindings(PACKAGE, 512),
    );
    expect(c).toMatchObject({
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      kernelSetId:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
    });
    expect(countBy(c.dispatches[0]!.quanta, (quantum) => quantum.kernelId))
      .toEqual({
        [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
        [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 4_090,
        [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 819,
        [ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID]: 644,
        [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
        [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
      });
    expect(c.dispatches[0]!.graphQuanta.reduce(
      (total, quantum) =>
        total + (quantum.kernelId === ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
          ? quantum.estimatedMaximumMultiplyAccumulates
          : 0),
      0,
    )).toBe(960_545_947_648);

    runtime.destroy();
  }, 30_000);

  it("routes every B/C transpose quantum through the explicit congruent hybrid", async () => {
    const device = fakeDevice({
      subgroups: true,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    expect(runtime).toMatchObject({
      runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      kernelSetId:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
      precisionMap:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
    });

    const b = await runtime.createDecoderDispatch(
      "congruent-b256",
      createRuntimeBindings(PACKAGE),
    );
    expect(countBy(b.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 2_045,
      [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 414,
      [ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID]: 322,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 813,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 348,
    });
    expect(b.graphQuanta.every((quantum) =>
      quantum.precision.kernelFamily === "conv-transpose1d"
        ? quantum.kernelId ===
          ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
        : quantum.kernelId !==
          ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
    )).toBe(true);
    const transposeQuantum = b.graphQuanta.find((quantum) =>
      quantum.kernelId ===
        ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
    )!;
    const transposeOperation = b.plan.operations[
      transposeQuantum.operationIndex!
    ]!;
    expect(transposeOperation.kind).toBe("conv-transpose1d");
    const pass = fakePass();
    transposeQuantum.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledOnce();
    expect(pass.dispatchWorkgroups.mock.calls[0]?.[2]).toBe(
      transposeOperation.kind === "conv-transpose1d"
        ? transposeOperation.shape.stride
        : undefined,
    );

    const c = await runtime.createChunkDispatchSet(
      "congruent-c512",
      512,
      512,
      createWindowBindings(PACKAGE, 512),
    );
    expect(c).toMatchObject({
      runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      kernelSetId:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
    });
    expect(countBy(c.dispatches[0]!.quanta, (quantum) => quantum.kernelId))
      .toEqual({
        [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
        [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 4_090,
        [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 819,
        [ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID]: 644,
        [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
        [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
      });

    runtime.destroy();
  }, 30_000);

  it("routes packed K1 and transpose weights through the exact OPT-0028 kernels", async () => {
    const device = fakeDevice({
      subgroups: true,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    expect(runtime).toMatchObject({
      runtimeProfileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
    });

    const dispatch = await runtime.createDecoderDispatch(
      "exact-packed-b256",
      createRuntimeBindings(PACKED_PACKAGE),
    );
    expect(countBy(dispatch.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 2_045,
      [ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID]: 414,
      [ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID]: 322,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 813,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 348,
    });
    expect(dispatch.graphQuanta.every((quantum) => {
      if (quantum.precision.kernelFamily === "conv1d-k1") {
        return quantum.kernelId === ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID;
      }
      if (quantum.precision.kernelFamily === "conv-transpose1d") {
        return quantum.kernelId === ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID;
      }
      return true;
    })).toBe(true);
    runtime.destroy();
  }, 30_000);

  it("routes the complete C512 transpose topology through the OPT-0040 static selector", async () => {
    const device = fakeDevice({
      subgroups: true,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    expect(runtime).toMatchObject({
      runtimeProfileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
      precisionMap:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
    });
    expect(runtime.kernelTopology.convTranspose1d).toBe(
      ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
    );

    const set = await runtime.createChunkDispatchSet(
      "shape-selected-c512",
      512,
      512,
      createWindowBindings(PACKED_PACKAGE, 512),
    );
    expect(set.topology).toMatchObject({
      uniqueWindowFrames: [512],
      aggregateGraphQuantumCount:
        ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
      aggregateSequenceQuantumCount:
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
    });
    const dispatch = set.dispatches[0]!;
    expect(dispatch).toMatchObject({
      runtimeProfileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY.id,
      graphQuantumCount: ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
      primitiveCount: ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
    });
    expect(countBy(dispatch.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 4_090,
      [ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID]: 819,
      [ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID]: 368,
      [ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID]: 276,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
    });

    const transposeQuanta = dispatch.graphQuanta.filter((quantum) =>
      quantum.operationKind === "conv-transpose1d"
    );
    expect(transposeQuanta).toHaveLength(644);
    expect(countBy(transposeQuanta, (quantum) => quantum.operationLabel))
      .toEqual({
        "block-0-conv-t1": 92,
        "block-1-conv-t1": 138,
        "block-2-conv-t1": 138,
        "block-3-conv-t1": 138,
        "block-4-conv-t1": 138,
      });
    expect(transposeQuanta.every((quantum) =>
      quantum.operationIndex !== null &&
      quantum.precision.kernelFamily === "conv-transpose1d" &&
      (quantum.operationLabel === "block-0-conv-t1" ||
          quantum.operationLabel === "block-1-conv-t1" ||
          quantum.operationLabel === "block-2-conv-t1"
        ? quantum.kernelId ===
          ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID
        : quantum.kernelId ===
          ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID)
    )).toBe(true);

    const selectorModules = device.createShaderModule.mock.calls.filter(
      (call) => String((call[0] as GPUShaderModuleDescriptor).label).includes(
        ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
      ),
    );
    expect(selectorModules).toHaveLength(5);
    expect(selectorModules.filter((call) =>
      String((call[0] as GPUShaderModuleDescriptor).code).includes(
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
      )
    )).toHaveLength(3);
    expect(selectorModules.filter((call) =>
      String((call[0] as GPUShaderModuleDescriptor).code).includes(
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
      )
    )).toHaveLength(2);

    const retainedTranspose = transposeQuanta[0]!;
    runtime.destroy();
    expect(device.createdBuffers[0]!.destroyMock).toHaveBeenCalledOnce();
    expect(() => retainedTranspose.encode(fakePass())).toThrow(/destroyed/);
  }, 30_000);

  it("routes the complete revision-7 C512 graph through OPT-0057 and OPT-0052", async () => {
    const device = fakeDevice({
      subgroups: true,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId: "opt-0054-mixed-fp16-fixed32-revision7-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    expect(runtime).toMatchObject({
      runtimeProfileId: "opt-0054-mixed-fp16-fixed32-revision7-v1",
      kernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
    });
    expect(runtime.kernelTopology).toMatchObject({
      conv1dK7: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      convTranspose1d:
        ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    });

    const set = await runtime.createChunkDispatchSet(
      "revision7-c512",
      512,
      512,
      createWindowBindings(REVISION7_PACKAGE, 512),
    );
    const dispatch = set.dispatches[0]!;
    expect(dispatch).toMatchObject({
      runtimeProfileId: "opt-0054-mixed-fp16-fixed32-revision7-v1",
      kernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY.id,
      graphQuantumCount: ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
      primitiveCount: ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
    });

    const k7 = dispatch.graphQuanta.filter(({ precision }) =>
      precision.kernelFamily === "conv1d-k7"
    );
    expect(k7).toHaveLength(4_090);
    expect(k7.every((quantum) =>
      REVISION7_ROW_REUSE_LABELS.has(quantum.operationLabel)
        ? quantum.kernelId ===
          ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
        : quantum.kernelId === ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
    )).toBe(true);
    expect(new Set(k7.filter(({ kernelId }) =>
      kernelId === ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
    ).map(({ operationLabel }) => operationLabel))).toEqual(
      new Set(REVISION7_ROW_REUSE_LABELS),
    );

    const transpose = dispatch.graphQuanta.filter(({ operationKind }) =>
      operationKind === "conv-transpose1d"
    );
    expect(transpose).toHaveLength(644);
    expect(transpose.every((quantum) => {
      if (quantum.operationLabel === "block-0-conv-t1") {
        return quantum.kernelId ===
          ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID;
      }
      const reuse = REVISION7_TRANSPOSE_BY_LABEL.get(quantum.operationLabel);
      return reuse === "channel"
        ? quantum.kernelId ===
          ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
        : reuse === "row" && quantum.kernelId ===
          ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID;
    })).toBe(true);

    const retainedRowK7 = k7.find(({ kernelId }) =>
      kernelId === ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
    )!;
    const retainedTransposeK4 = transpose.find(({ kernelId }) =>
      kernelId === ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID
    )!;
    runtime.destroy();
    expect(device.createdBuffers[0]!.destroyMock).toHaveBeenCalledOnce();
    expect(() => retainedRowK7.encode(fakePass())).toThrow(/destroyed/);
    expect(() => retainedTransposeK4.encode(fakePass())).toThrow(/destroyed/);
  }, 30_000);

  it("routes revision-6 packed weights through portable workgroup owners", async () => {
    const device = fakeDevice({
      subgroups: false,
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId: "opt-0028-mixed-fp16-portable-exact-packed-v1",
    });
    expect(runtime).toMatchObject({
      runtimeProfileId: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
    });

    const b = await runtime.createDecoderDispatch(
      "portable-exact-packed-b256",
      createRuntimeBindings(PACKED_PACKAGE),
    );
    expect(countBy(b.quanta, (quantum) => quantum.kernelId)).toEqual({
      [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
      [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 2_045,
      [ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID]: 414,
      [ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID]: 322,
      [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 813,
      [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 348,
    });

    const c = await runtime.createChunkDispatchSet(
      "portable-exact-packed-c512",
      512,
      512,
      createWindowBindings(PACKED_PACKAGE, 512),
    );
    expect(c).toMatchObject({
      runtimeProfileId: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
    });
    expect(countBy(c.dispatches[0]!.quanta, (quantum) => quantum.kernelId))
      .toEqual({
        [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
        [ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID]: 4_090,
        [ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID]: 819,
        [ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID]: 644,
        [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
        [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
      });
    const shaderSources = device.createShaderModule.mock.calls.map((call) =>
      String((call[0] as GPUShaderModuleDescriptor).code)
    );
    expect(shaderSources.some((source) =>
      source.includes(ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID)
    )).toBe(true);
    expect(shaderSources.some((source) =>
      source.includes(
        ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
      )
    )).toBe(true);
    expect(shaderSources.every((source) => !source.includes("enable subgroups")))
      .toBe(true);

    runtime.destroy();
  }, 30_000);

  it("rejects unauthenticated subgroup bounds before hybrid allocation", () => {
    const missingFeature = fakeDevice({ subgroups: false });
    expect(() => AceOpt0011Fp16VaeDecoderRuntime.create(missingFeature, {
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    })).toThrow(/authenticated 32\/32 subgroups/);
    expect(missingFeature.createBuffer).not.toHaveBeenCalled();
    expect(missingFeature.createShaderModule).not.toHaveBeenCalled();

    const variableWidth = fakeDevice({ subgroups: true });
    expect(() => AceOpt0011Fp16VaeDecoderRuntime.create(
      variableWidth,
      {
        runtimeProfileId:
          "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
        subgroupMinSize: 16,
        subgroupMaxSize: 32,
      } as never,
    )).toThrow(/authenticated 32\/32 subgroups/);
    expect(variableWidth.createBuffer).not.toHaveBeenCalled();
    expect(variableWidth.createShaderModule).not.toHaveBeenCalled();
  });

  it("authenticates every package operation index, label, kind, and tensor role", async () => {
    const baseBindings = createRuntimeBindings(PACKAGE);
    const forgeries: readonly [string, AceOpt0011VaePackageBindings][] = [
      ["identity", freezePackage({
        ...PACKAGE,
        manifestSha256: "0".repeat(64),
      })],
      ["index", replaceOperation(PACKAGE, 0, (operation) => ({
        ...operation,
        operationIndex: 1,
      }))],
      ["label", replaceOperation(PACKAGE, 12, (operation) => ({
        ...operation,
        label: "forged-label",
      }))],
      ["kind", replaceOperation(PACKAGE, 1, (operation) => ({
        ...operation,
        kind: "add",
      }))],
      ["role", swapConv1dWeightRole(PACKAGE)],
      ["transpose", forgeFirstTransposePartEnd(PACKAGE)],
    ];
    for (const [name, forged] of forgeries) {
      const device = fakeDevice();
      const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
      await expect(runtime.createDecoderDispatch(name, {
        ...baseBindings,
        package: forged,
      }), name).rejects.toThrow(/OPT-0011/);
      expect(device.createBuffer, name).not.toHaveBeenCalled();
      expect(device.createShaderModule, name).not.toHaveBeenCalled();
      runtime.destroy();
    }
  });

  it("fails before allocation for unsupported limits, short spans, and aliases", async () => {
    for (const options of [
      { shaderF16: false },
      { maximumBuffer: 125_829_119 },
      { maximumStorageBinding: 125_829_119 },
      { maximumWorkgroupStorage: 16_383 },
      { maximumInvocations: 255 },
      { maximumWorkgroupSizeX: 255 },
      { maximumWorkgroupSizeY: 7 },
      { maximumStorageBuffers: 3 },
      { maximumUniformBinding: 15 },
      { storageAlignment: 3 },
      { uniformAlignment: 6 },
    ]) {
      expect(() => AceOpt0011Fp16VaeDecoderRuntime.create(
        fakeDevice(options),
      ), JSON.stringify(options)).toThrow(/OPT-0011/);
    }

    const valid = createRuntimeBindings(PACKAGE);
    const workspaceBytes = PLAN.maximumActivationElements * 2;
    const malformed: readonly [string, AceOpt0011Fp16VaeDecoderBindings][] = [
      ["short", {
        ...valid,
        workspaces: [
          fakeBinding(workspaceBytes - 4),
          valid.workspaces[1],
          valid.workspaces[2],
        ],
      }],
      ["misaligned", {
        ...valid,
        output: {
          buffer: fakeBuffer(PLAN.outputElements * 4 + 256),
          offset: 4,
          size: PLAN.outputElements * 4,
        },
      }],
      ["alias", {
        ...valid,
        workspaces: [
          valid.workspaces[0],
          valid.workspaces[0],
          valid.workspaces[2],
        ],
      }],
      ["weight-alias", {
        ...valid,
        decoderInput: Object.values(PACKAGE.tensors)[0]!.binding,
      }],
    ];
    for (const [name, bindings] of malformed) {
      const device = fakeDevice();
      const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
      await expect(runtime.createDecoderDispatch(name, bindings), name)
        .rejects.toThrow();
      expect(device.createBuffer, name).not.toHaveBeenCalled();
      expect(device.createShaderModule, name).not.toHaveBeenCalled();
      runtime.destroy();
    }
  });

  it("destroys the one control buffer on allocation-scope or compile failure", async () => {
    const scopedDevice = fakeDevice({
      scopeResults: [
        { message: "invalid control allocation" } as GPUValidationError,
        null,
        null,
      ],
    });
    const scopedRuntime = AceOpt0011Fp16VaeDecoderRuntime.create(scopedDevice);
    await expect(scopedRuntime.createDecoderDispatch(
      "scope-failure",
      createRuntimeBindings(PACKAGE),
    )).rejects.toThrow(/invalid control allocation/);
    expect(scopedDevice.createdBuffers[0]!.destroyMock).toHaveBeenCalledOnce();
    expect(scopedDevice.queue.writeBuffer).not.toHaveBeenCalled();

    const compileDevice = fakeDevice({
      pipelineFailure: new Error("compile failure"),
    });
    const compileRuntime = AceOpt0011Fp16VaeDecoderRuntime.create(compileDevice);
    await expect(compileRuntime.createDecoderDispatch(
      "compile-failure",
      createRuntimeBindings(PACKAGE),
    )).rejects.toThrow(/compile failure/);
    expect(compileDevice.createdBuffers[0]!.destroyMock).toHaveBeenCalledOnce();
    expect(compileDevice.queue.writeBuffer).toHaveBeenCalledOnce();
    compileRuntime.destroy();
  });

  it("pins the complete C-512 topology and exact active binding prefixes", () => {
    const controls = planAceOpt0011Fp16VaeWindowDynamicControls(512, 256);
    expect(controls).toMatchObject({
      recordBytes: 16,
      recordAlignment: 256,
      recordCount: ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
      byteLength: ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES,
    });
    expect(countBy(controls.records.slice(1), (record) =>
      record.operationKind
    )).toEqual({
      conv1d: 4_909,
      "conv-transpose1d": 644,
      snake: 1_611,
      add: 690,
    });

    const planned = planAceOpt0011Fp16VaeChunkDispatches(512, 512, 256);
    const [topology] = planned.topologies;
    expect(planned.uniqueWindowFrames).toEqual([512]);
    expect(planned.windowTopologyIndices).toEqual([0]);
    expect(topology).toMatchObject({
      inputFrames: 512,
      operationCount: 88,
      graphQuantumCount: ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
      sequenceQuantumCount: ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
      fp16WorkspaceBytes: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      activeStagingInputBytes: 131_072,
      activeDecoderInputBytes: 65_536,
      activeOutputBytes: 7_864_320,
      decoderCommandBufferCountAtBatch8: 982,
      commandBufferCountAtBatch8:
        ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8,
    });
    expect(topology?.quantumFamilyCounts).toEqual({
      conv1d: 4_909,
      "conv-transpose1d": 644,
      snake: 1_611,
      add: 690,
    });
    expect(topology?.plan.requiredTensorNames).toHaveLength(145);
    expect(topology?.plan.requiredTensorNames).toEqual(PLAN.requiredTensorNames);
    expect(Object.isFrozen(planned)).toBe(true);
    expect(Object.isFrozen(planned.topologies)).toBe(true);
    expect(Object.isFrozen(topology)).toBe(true);
    expect(Object.isFrozen(topology?.dynamicControls)).toBe(true);
    expect(Object.isFrozen(topology?.dynamicControls.records)).toBe(true);
    expect(topology?.dynamicControls.records.every(Object.isFrozen)).toBe(true);
  });

  it("derives exact unpadded B/C 1024 edge shapes and aggregate coverage", () => {
    const b = planAceOpt0011Fp16VaeChunkDispatches(1_024, 256, 256);
    expect(b.uniqueWindowFrames).toEqual([192, 256]);
    expect(b.chunkPlan.windows.map((window) => window.latentWindowFrames))
      .toEqual([192, 256, 256, 256, 256, 256, 256, 192]);
    expect(b.windowTopologyIndices).toEqual([0, 1, 1, 1, 1, 1, 1, 0]);
    expect(b).toMatchObject({
      maximumWindowFramesProfile: 256,
      maximumFp16WorkspaceBytes: 125_829_120,
      aggregateGraphQuantumCount: 29_586,
      aggregateSequenceQuantumCount: 29_594,
      aggregateCommandBufferCountAtBatch8: 3_708,
    });

    const c = planAceOpt0011Fp16VaeChunkDispatches(1_024, 512, 256);
    expect(c.uniqueWindowFrames).toEqual([320, 448, 512]);
    expect(c.chunkPlan.windows.map((window) => window.latentWindowFrames))
      .toEqual([448, 512, 320]);
    expect(c.windowTopologyIndices).toEqual([1, 2, 0]);
    expect(c).toMatchObject({
      maximumWindowFramesProfile: 512,
      maximumFp16WorkspaceBytes: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      aggregateGraphQuantumCount: 19_684,
      aggregateSequenceQuantumCount: 19_687,
      aggregateCommandBufferCountAtBatch8: 2_465,
    });
    for (const planned of [b, c]) {
      expect(planned.chunkPlan.windows.reduce(
        (frames, window) => frames + window.outputAudioFrames,
        0,
      )).toBe(planned.chunkPlan.outputAudioFrames);
      expect(planned.topologies.every((topology, index) =>
        topology.inputFrames === planned.uniqueWindowFrames[index] &&
        topology.operationCount === 88 &&
        topology.plan.requiredTensorNames.length === 145 &&
        topology.plan.requiredTensorNames.every((name, tensorIndex) =>
          name === PLAN.requiredTensorNames[tensorIndex]
        )
      )).toBe(true);
      expect(new Set(planned.windowTopologyIndices).size)
        .toBe(planned.topologies.length);
    }

    // The direct three-minute target is 4,500 latent frames. Keep its edge
    // inventory exact too; none of these shapes may be padded to 256 or 512.
    expect(planAceOpt0011Fp16VaeChunkDispatches(
      4_500,
      256,
      256,
    ).uniqueWindowFrames).toEqual([84, 192, 212, 256]);
    expect(planAceOpt0011Fp16VaeChunkDispatches(
      4_500,
      512,
      256,
    ).uniqueWindowFrames).toEqual([340, 448, 512]);
  });

  it("creates and reuses one immutable dispatch/control set per C-1024 shape", async () => {
    const device = fakeDevice({
      maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
    const bindings = createWindowBindings(PACKAGE, 512);
    const set = await runtime.createChunkDispatchSet(
      "complete-c1024",
      1_024,
      512,
      bindings,
    );

    expect(set.dispatches.map((dispatch) => dispatch.plan.inputFrames))
      .toEqual([320, 448, 512]);
    expect(set.windows.map(({ window }) => window.latentWindowFrames))
      .toEqual([448, 512, 320]);
    expect(set.windows[0]?.dispatch).toBe(set.dispatches[1]);
    expect(set.windows[1]?.dispatch).toBe(set.dispatches[2]);
    expect(set.windows[2]?.dispatch).toBe(set.dispatches[0]);
    expect(set.dispatches.map((dispatch) => dispatch.activeOutputBytes))
      .toEqual([4_915_200, 6_881_280, 7_864_320]);
    expect(set.dispatches.every((dispatch) =>
      dispatch.plan.inputFrames * 64 * 4 ===
        dispatch.activeStagingInputBytes &&
      dispatch.plan.inputFrames * 64 * 2 ===
        dispatch.activeDecoderInputBytes &&
      dispatch.plan.outputElements * 4 === dispatch.activeOutputBytes
    )).toBe(true);
    expect(device.createBuffer).toHaveBeenCalledTimes(3);
    expect(device.queue.writeBuffer).toHaveBeenCalledTimes(3);
    expect(device.createBuffer.mock.calls.map((call) =>
      (call[0] as GPUBufferDescriptor).size
    )).toEqual(set.topology.topologies.map((topology) =>
      topology.dynamicControls.byteLength
    ));
    expect(Object.isFrozen(set)).toBe(true);
    expect(Object.isFrozen(set.dispatches)).toBe(true);
    expect(Object.isFrozen(set.windows)).toBe(true);
    expect(set.dispatches.every(Object.isFrozen)).toBe(true);
    expect(set.windows.every(Object.isFrozen)).toBe(true);

    const controls = [...device.createdBuffers];
    runtime.destroy();
    expect(controls.every((buffer) =>
      buffer.destroyMock.mock.calls.length === 1
    )).toBe(true);
  }, 30_000);

  it("preflights C limits and rolls a multi-shape build back transactionally", async () => {
    for (const options of [
      {
        maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES - 1,
        maximumStorageBinding: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      },
      {
        maximumBuffer: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
        maximumStorageBinding:
          ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES - 1,
      },
    ]) {
      const device = fakeDevice(options);
      const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
      await expect(runtime.createChunkDispatchSet(
        "c-limit",
        1_024,
        512,
        createWindowBindings(PACKAGE, 512),
      )).rejects.toThrow(/251658240/);
      expect(device.createBuffer).not.toHaveBeenCalled();
      expect(device.createShaderModule).not.toHaveBeenCalled();
      runtime.destroy();
    }

    const allocationFailure = {
      message: "reject second exact shape",
    } as GPUValidationError;
    const device = fakeDevice({
      scopeResults: [
        null,
        null,
        null,
        allocationFailure,
        null,
        null,
      ],
    });
    const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device);
    await expect(runtime.createChunkDispatchSet(
      "transaction",
      1_024,
      512,
      createWindowBindings(PACKAGE, 512),
    )).rejects.toThrow(/reject second exact shape/);
    expect(device.createdBuffers).toHaveLength(2);
    expect(device.createdBuffers.every((buffer) =>
      buffer.destroyMock.mock.calls.length === 1
    )).toBe(true);
    runtime.destroy();
    expect(device.createdBuffers.every((buffer) =>
      buffer.destroyMock.mock.calls.length === 1
    )).toBe(true);
  }, 30_000);
});

function createRuntimeBindings(
  packageBindings: AceOpt0011VaePackageBindings,
): AceOpt0011Fp16VaeDecoderBindings {
  return createWindowBindings(packageBindings, 256) as
    AceOpt0011Fp16VaeDecoderBindings;
}

function createWindowBindings(
  packageBindings: AceOpt0011VaePackageBindings,
  inputFrames: number,
): AceOpt0011Fp16VaeWindowBindings {
  const plan = planAceVaeDecoder(inputFrames);
  const workspaceBytes = plan.maximumActivationElements * 2;
  return Object.freeze({
    stagingInput: fakeBinding(plan.inputElements * 4),
    decoderInput: fakeBinding(plan.inputElements * 2),
    workspaces: Object.freeze([
      fakeBinding(workspaceBytes),
      fakeBinding(workspaceBytes),
      fakeBinding(workspaceBytes),
    ] as const),
    output: fakeBinding(plan.outputElements * 4),
    package: packageBindings,
  });
}

function createPackageBindings(
  mode: false | true | "revision7" = false,
): AceOpt0011VaePackageBindings {
  const packed = mode !== false;
  const revision7 = mode === "revision7";
  const tensors: Record<string, AceOpt0011VaeResolvedTensor> = {};
  const tensor = (
    logicalTensor: string,
    shape: readonly number[],
    layout?: AcePackageTensorRecord["layout"],
    storageShapeInput: readonly number[] = shape,
  ): AceOpt0011VaeResolvedTensor => {
    const existing = tensors[logicalTensor];
    if (existing !== undefined) return existing;
    const logicalShape = Object.freeze([...shape]);
    const storageShape = Object.freeze([...storageShapeInput]);
    const byteLength = shape.reduce((product, extent) => product * extent, 1) * 2;
    const record = Object.freeze({
      logicalTensor,
      logicalShape,
      storageShape,
      ...(layout === undefined ? {} : { layout }),
      dtype: "float16",
      phase: "vae",
      lifetime: "vae",
      partAxis: 0,
      partStart: 0,
      partEnd: shape[0]!,
      byteOffset: 0,
      byteLength,
    }) as unknown as AcePackageTensorRecord;
    const resolved = Object.freeze({
      logicalTensor,
      logicalShape,
      physicalTensor: logicalTensor,
      record,
      binding: Object.freeze({
        buffer: fakeBuffer(byteLength),
        offset: 0,
        size: byteLength,
      }),
    });
    tensors[logicalTensor] = resolved;
    return resolved;
  };
  const operations = PLAN.operations.map((operation, operationIndex) => {
    const base = { operationIndex, label: operation.label };
    switch (operation.kind) {
      case "conv1d":
        const packedK1 = packed && operation.shape.kernelSize === 1;
        const rowReuse = revision7 &&
          REVISION7_ROW_REUSE_LABELS.has(operation.label);
        return Object.freeze({
          ...base,
          kind: "conv1d" as const,
          weight: tensor(operation.weight, [
            operation.shape.outputChannels,
            operation.shape.kernelSize,
            operation.shape.inputChannels,
          ], packed
            ? packedK1
              ? ACE_VAE_K1_FP16_TILE_LAYOUT
              : rowReuse
                ? ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT
              : ACE_VAE_CONV1D_FP16_LAYOUT
            : undefined, packedK1
              ? [
                  operation.shape.outputChannels / 128,
                  operation.shape.inputChannels / 32,
                  32,
                  128,
                ]
              : rowReuse
              ? [
                  7,
                  operation.shape.inputChannels / 4,
                  operation.shape.outputChannels / 64,
                  32,
                  2,
                  4,
                ]
              : [
                  operation.shape.outputChannels,
                  operation.shape.kernelSize,
                  operation.shape.inputChannels,
                ]),
          ...(operation.bias === undefined
            ? {}
            : {
                bias: tensor(operation.bias, [operation.shape.outputChannels]),
              }),
        });
      case "conv-transpose1d":
        const reuseAxis = revision7
          ? REVISION7_TRANSPOSE_BY_LABEL.get(operation.label)
          : undefined;
        const outputsPerLane = reuseAxis === "channel" ? 8 : 4;
        return Object.freeze({
          ...base,
          kind: "conv-transpose1d" as const,
          weight: tensor(operation.weight, [
            operation.shape.outputChannels,
            operation.shape.kernelSize,
            operation.shape.inputChannels,
          ], packed
            ? reuseAxis === undefined
              ? ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT
              : ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT
            : undefined, packed
              ? reuseAxis === undefined
                ? [
                  operation.shape.kernelSize / 2,
                  2,
                  operation.shape.inputChannels,
                  operation.shape.outputChannels,
                ]
                : [
                    operation.shape.kernelSize / 2,
                    2,
                    operation.shape.inputChannels / 4,
                    operation.shape.outputChannels / (outputsPerLane * 32),
                    32,
                    outputsPerLane,
                    4,
                  ]
              : [
                  operation.shape.outputChannels,
                  operation.shape.kernelSize,
                  operation.shape.inputChannels,
                ]),
          bias: tensor(operation.bias, [operation.shape.outputChannels]),
        });
      case "snake":
        return Object.freeze({
          ...base,
          kind: "snake" as const,
          alpha: tensor(operation.alpha, [operation.shape.channels]),
          beta: tensor(operation.beta, [operation.shape.channels]),
        });
      case "add":
        return Object.freeze({ ...base, kind: "add" as const });
    }
  });
  return Object.freeze({
    manifestSha256: revision7
      ? ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256
      : packed
      ? ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256
      : ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: revision7
      ? ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES
      : packed
      ? ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES
      : ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
    residentWeightBytes: ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    weightFiles: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
    tensors: Object.freeze(tensors),
    operations: Object.freeze(operations),
  });
}

function freezePackage(
  packageBindings: AceOpt0011VaePackageBindings,
): AceOpt0011VaePackageBindings {
  return Object.freeze({
    ...packageBindings,
    tensors: Object.freeze({ ...packageBindings.tensors }),
    operations: Object.freeze([...packageBindings.operations]),
  });
}

function replaceOperation(
  packageBindings: AceOpt0011VaePackageBindings,
  index: number,
  replace: (
    operation: AceOpt0011VaeOperationBindings,
  ) => object,
): AceOpt0011VaePackageBindings {
  const operations = [...packageBindings.operations];
  operations[index] = Object.freeze(replace(operations[index]!)) as
    AceOpt0011VaeOperationBindings;
  return freezePackage({ ...packageBindings, operations });
}

function swapConv1dWeightRole(
  packageBindings: AceOpt0011VaePackageBindings,
): AceOpt0011VaePackageBindings {
  const conv1d = packageBindings.operations.filter((operation) =>
    operation.kind === "conv1d"
  );
  const left = conv1d.find((operation) =>
    operation.kind === "conv1d" &&
    operation.weight.logicalShape.join(",") === "128,7,128"
  );
  const right = conv1d.find((operation) =>
    operation.kind === "conv1d" &&
    operation !== left &&
    operation.weight.logicalShape.join(",") === "128,7,128"
  );
  if (left?.kind !== "conv1d" || right?.kind !== "conv1d") {
    throw new Error("fixture is missing same-shaped Conv1D roles");
  }
  return replaceOperation(
    packageBindings,
    left.operationIndex,
    (operation) => ({ ...operation, weight: right.weight }),
  );
}

function forgeFirstTransposePartEnd(
  packageBindings: AceOpt0011VaePackageBindings,
): AceOpt0011VaePackageBindings {
  const operation = packageBindings.operations.find((candidate) =>
    candidate.kind === "conv-transpose1d"
  );
  if (operation?.kind !== "conv-transpose1d") {
    throw new Error("fixture is missing ConvTranspose1D");
  }
  const forgedRecord = Object.freeze({
    ...operation.weight.record,
    partEnd: operation.weight.record.partEnd - 1,
  });
  const forgedTensor = Object.freeze({
    ...operation.weight,
    record: forgedRecord,
  });
  const tensors = Object.freeze({
    ...packageBindings.tensors,
    [operation.weight.logicalTensor]: forgedTensor,
  });
  const operations = packageBindings.operations.map((candidate) =>
    candidate.operationIndex === operation.operationIndex
      ? Object.freeze({ ...candidate, weight: forgedTensor })
      : candidate
  );
  return freezePackage({ ...packageBindings, tensors, operations });
}

type FakeBuffer = GPUBuffer & {
  readonly destroyMock: ReturnType<typeof vi.fn>;
};

interface FakeDeviceDiagnostics {
  readonly createBuffer: ReturnType<typeof vi.fn>;
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createdBuffers: readonly FakeBuffer[];
  readonly queue: GPUQueue & {
    readonly writeBuffer: ReturnType<typeof vi.fn>;
  };
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly shaderF16?: boolean;
  readonly subgroups?: boolean;
  readonly maximumBuffer?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumWorkgroupStorage?: number;
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupSizeY?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBuffers?: number;
  readonly maximumUniformBinding?: number;
  readonly storageAlignment?: number;
  readonly uniformAlignment?: number;
  readonly scopeResults?: readonly (GPUError | null)[];
  readonly pipelineFailure?: Error;
} = {}): FakeDevice {
  const createdBuffers: FakeBuffer[] = [];
  const scopeResults = [...(options.scopeResults ?? [null, null, null])];
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
    const buffer = fakeBuffer(Number(descriptor.size));
    createdBuffers.push(buffer);
    return buffer;
  });
  const createShaderModule = vi.fn(() => ({
    getCompilationInfo: vi.fn(async () => ({ messages: [] })),
  }));
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      ...(options.subgroups === true ? ["subgroups"] : []),
    ]),
    limits: {
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxComputeWorkgroupStorageSize:
        options.maximumWorkgroupStorage ?? 32_768,
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupSizeY: options.maximumWorkgroupSizeY ?? 256,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxBindGroups: 4,
      maxBindingsPerBindGroup: 1_000,
      maxStorageBuffersPerShaderStage: options.maximumStorageBuffers ?? 10,
      maxUniformBuffersPerShaderStage: 12,
      maxDynamicUniformBuffersPerPipelineLayout: 10,
      maxUniformBufferBindingSize: options.maximumUniformBinding ?? 65_536,
      minStorageBufferOffsetAlignment: options.storageAlignment ?? 256,
      minUniformBufferOffsetAlignment: options.uniformAlignment ?? 256,
    },
    queue: {
      writeBuffer: vi.fn(),
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => scopeResults.shift() ?? null),
    createBuffer,
    createShaderModule,
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => {
      if (options.pipelineFailure !== undefined) {
        throw options.pipelineFailure;
      }
      return { label: "pipeline" } as GPUComputePipeline;
    }),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createdBuffers,
  } as unknown as FakeDevice;
}

function fakeBuffer(size: number): FakeBuffer {
  const destroyMock = vi.fn(() => undefined);
  return {
    size,
    destroy: destroyMock,
    destroyMock,
  } as unknown as FakeBuffer;
}

function fakeBinding(size: number): GPUBufferBinding {
  return Object.freeze({ buffer: fakeBuffer(size), offset: 0, size });
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

function countBy<T>(
  values: readonly T[],
  key: (value: T) => PropertyKey,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const name = String(key(value));
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}
