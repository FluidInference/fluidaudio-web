import { describe, expect, it, vi } from "vitest";

import { ACE_DIRECT_DCW_CONFIGURATION } from "../src/api.js";
import type {
  AceGpuLogicalTensor,
  AceGpuTensorPhase,
} from "../src/model/gpu-tensors.js";
import {
  ACE_DIT_GRAPH_QUANTUM_COUNT,
  ACE_DIT_PROFILE_FAMILIES,
  ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
  ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT,
  AceDitResidentModel,
  createAceDitGraphQuantumPlan,
  planAceDitPhysicalQuantumBatches,
  type AceDitPhysicalCommandBatchDescriptor,
  type AceDitPhysicalCommandDescriptor,
  type AceDitPhysicalCommandDescriptorTable,
  type AceDitProfileFamily,
  type AceDitGraphRunResult,
  type AceDitGraphRunOptions,
} from "../src/webgpu/dit-graph.js";
import {
  ACE_TURBO_DIT_CONFIG,
  planAceDitEvaluation,
} from "../src/webgpu/ace-dit.js";
import {
  AceDitBackendDeviceLostError,
  AceDitGpuBackend,
  planAceDitBackendArena,
  planAceDitGpuBackendMemory,
  planAceDitPhysicalCommandBufferCount,
  resolveAceDitGemmSelection,
  resolveAceDitMixedGemmSelection,
  type AceDitGpuBackendProgress,
  type AceOpt0018DitCommandProfile,
  type AceOpt0034DitSchedulingProfile,
  type AceOpt0067DitCommandProfile,
  type AceOpt0080DitCommandProfile,
  type AceOpt0080FullDitCommandProfile,
  type AceDitPreparedGpuResources,
} from "../src/webgpu/dit-backend.js";
import {
  aceDitExpectedLogicalShape,
  isAceDitGemmWeightTensorName,
  isAceDitRepeatedDenseWeightTensorName,
} from "../src/webgpu/ace-dit-package.js";
import {
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE } from
  "../src/webgpu/kernels/dit-f16-dense-input-producers.js";
import { ACE_FP16_PORTABLE_PROFILE } from "../src/webgpu/capabilities.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
} from "../src/webgpu/capabilities.js";
import {
  planAceCompositeCooperativeQuantumCount,
  planAceTiledGemm,
} from "../src/webgpu/kernels/gemm.js";
import { planAceSubgroupGemm } from "../src/webgpu/kernels/subgroup-gemm.js";
import { planAceFixed32TiledFullAttention } from "../src/webgpu/kernels/attention.js";
import {
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_TILE_LAYOUT,
  ACE_DIT_DENSE_FP16_TILE_LAYOUT,
  ACE_DIT_DENSE_FP16_TRANSFORMATION,
  ACE_DIT_DENSE_K4_FP16_LAYOUT,
  ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
} from "../src/model/manifest.js";

Object.defineProperty(globalThis, "GPUBufferUsage", {
  configurable: true,
  value: Object.freeze({
    MAP_READ: 1,
    COPY_SRC: 4,
    COPY_DST: 8,
    STORAGE: 128,
  }),
});
Object.defineProperty(globalThis, "GPUMapMode", {
  configurable: true,
  value: Object.freeze({ READ: 1 }),
});

const SHAPE = Object.freeze({ batch: 1, latentFrames: 2, conditionTokens: 1 });

describe("ACE DiT concrete backend memory", () => {
  it("selects only the authenticated fixed-32 reference subgroup backend", () => {
    expect(resolveAceDitGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
    )).toMatchObject({
      modelProfile: "reference-bf16",
      backend: "fixed32-subgroups",
      gemmConfiguration: { backend: "subgroups" },
    });
    expect(resolveAceDitGemmSelection(
      ACE_REFERENCE_PORTABLE_PROFILE,
      16,
      64,
    )).toMatchObject({ backend: "portable" });
    expect(resolveAceDitGemmSelection(
      ACE_FP16_PORTABLE_PROFILE,
      32,
      32,
    )).toMatchObject({
      modelProfile: "raw-fp16",
      backend: "portable",
    });
    expect(() => resolveAceDitGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      16,
      32,
    )).toThrow(/fixed 32-lane/);
    expect(() => resolveAceDitGemmSelection({
      ...ACE_REFERENCE_SUBGROUP_PROFILE,
      modelProfile: "raw-fp16",
    })).toThrow(/authenticated profile/);
    expect(resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
    )).toMatchObject({
      modelProfile: "reference-bf16",
      backend: "mixed-opt-0037-k4",
      denseRuntimeProfile: ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
      denseGemmConfiguration: {
        backend: "opt-0037-k4-fp16-partials",
      },
      attentionConfiguration: {
        backend: "fixed32-subgroup-query8",
      },
    });
    expect(resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    )).toMatchObject({
      modelProfile: "reference-bf16",
      backend: "mixed-opt-0056-selective",
      denseRuntimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      denseGemmConfiguration: {
        backend: "opt-0056-selective-k4-exact-down",
      },
    });
  });

  it("resolves the OPT-0088 portable mixed selection only for the hosted product tuple", () => {
    const selection = resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_PORTABLE_PROFILE,
      undefined,
      undefined,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      2_250,
      98,
    );
    expect(selection).toEqual({
      modelProfile: "reference-bf16",
      backend: "mixed-opt-0088-portable",
      denseRuntimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      attentionRuntimeProfile:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      gemmConfiguration: {
        backend: "portable",
        weightLayout: ACE_DIT_GEMM_TILE_LAYOUT,
      },
      denseGemmConfiguration: { backend: "opt-0088-dense-portable" },
      attentionConfiguration: { backend: "portable" },
    });
    expect(Object.isFrozen(selection)).toBe(true);
    expect(Object.isFrozen(selection.denseGemmConfiguration)).toBe(true);
    expect(Object.isFrozen(selection.attentionConfiguration)).toBe(true);
    // Reported subgroup sizes are capability hints; the portable tuple never
    // consumes them and resolves identically when they are present.
    expect(resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_PORTABLE_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      2_250,
      98,
    )).toEqual(selection);
    // Portable plus any rev8/selective dense profile stays closed.
    for (const denseProfile of [
      ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    ] as const) {
      expect(() => resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_PORTABLE_PROFILE,
        undefined,
        undefined,
        denseProfile,
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        2_250,
        98,
      )).toThrow(/cannot combine|Portable mixed DiT/);
    }
    // Portable plus any non-production attention profile stays closed.
    for (const attentionProfile of [
      ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    ] as const) {
      expect(() => resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_PORTABLE_PROFILE,
        undefined,
        undefined,
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
        attentionProfile,
        2_250,
        98,
      )).toThrow(/Portable mixed DiT/);
    }
    // OPT-0070 token-count authentication still applies to the portable arm.
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_PORTABLE_PROFILE,
      undefined,
      undefined,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toThrow(/token counts/);
    // The raw-FP16 portable profile is not the reference portable profile.
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_FP16_PORTABLE_PROFILE,
      undefined,
      undefined,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      2_250,
      98,
    )).toThrow(/fixed32 reference profile/);
    // The fixed32 subgroup arm is unchanged: 32-lane subgroups stay required.
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      16,
      64,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      2_250,
      98,
    )).toThrow(/fixed 32-lane/);
  });

  it("plans OPT-0088 portable mixed physical commands with the portable geometry", () => {
    const shape = { batch: 1, latentFrames: 4_500, conditionTokens: 98 };
    const portable = planAceDitPhysicalCommandBufferCount(shape, "portable");
    const mixedPortable = planAceDitPhysicalCommandBufferCount(
      shape,
      "mixed-opt-0088-portable",
    );
    expect(mixedPortable).toBe(1_785);
    const layerShapes = [
      [2_048, 2_048], [2_048, 1_024], [2_048, 1_024],
      [2_048, 2_048], [2_048, 2_048], [2_048, 2_048],
      [2_048, 6_144], [2_048, 6_144], [6_144, 2_048],
    ].map(([inner, columns]) =>
      ({ rows: 2_250, inner: inner!, columns: columns! })
    );
    // The dense package geometry replaces the tiled reference geometry in
    // every one of the 24 layers across all 8 evaluations; attention stays a
    // rangeless portable dispatch and contributes no extra physical command.
    const denseLayer = planAceCompositeCooperativeQuantumCount(
      layerShapes.map(planAceOpt0009DenseGemm),
    );
    const tiledLayer = planAceCompositeCooperativeQuantumCount(
      layerShapes.map(planAceTiledGemm),
    );
    expect(denseLayer).toBe(9);
    expect(tiledLayer).toBe(68);
    expect(portable - mixedPortable).toBe(24 * 8 * (tiledLayer - denseLayer));
    // The M150 generalized shape keeps the pinned portable accounting.
    expect(planAceDitPhysicalCommandBufferCount(
      { batch: 1, latentFrames: 300, conditionTokens: 97 },
      "mixed-opt-0088-portable",
    )).toBe(1_017);
    // Memory planning admits the backend and keeps plan-vs-plan parity.
    const memory = planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      3_150_917_888,
      "mixed-opt-0088-portable",
    );
    expect(memory.gemmBackend).toBe("mixed-opt-0088-portable");
    expect(memory.commandBufferCount).toBe(mixedPortable + 1);
    // The OPT-0081 diagnostic dense-input seam stays closed to the portable
    // backend.
    expect(() => planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      3_150_917_888,
      "mixed-opt-0088-portable",
      undefined,
      false,
      false,
      undefined,
      false,
      ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
    )).toThrow(/mixed-opt-0009/);
    expect(() => planAceDitPhysicalCommandBufferCount(
      shape,
      "future-backend" as never,
    )).toThrow(/Unknown ACE DiT GEMM backend/);
  });

  it("plans physical command buffers with the selected GEMM geometry", () => {
    const shape = { batch: 1, latentFrames: 641, conditionTokens: 1 };
    const portable = planAceDitPhysicalCommandBufferCount(shape, "portable");
    const subgroup = planAceDitPhysicalCommandBufferCount(
      shape,
      "fixed32-subgroups",
    );
    const layerShapes = [
      [2_048, 2_048], [2_048, 1_024], [2_048, 1_024],
      [2_048, 2_048], [2_048, 2_048], [2_048, 2_048],
      [2_048, 6_144], [2_048, 6_144], [6_144, 2_048],
    ].map(([inner, columns]) => ({ rows: 321, inner: inner!, columns: columns! }));
    expect(planAceCompositeCooperativeQuantumCount(
      layerShapes.map(planAceTiledGemm),
    )).toBe(10);
    expect(planAceCompositeCooperativeQuantumCount(
      layerShapes.map(planAceSubgroupGemm),
    )).toBe(14);
    const memory = planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      1_024,
      "fixed32-subgroups",
    );
    expect(memory.gemmBackend).toBe("fixed32-subgroups");
    expect(memory.commandBufferCount).toBe(subgroup + 1);
    expect(subgroup - portable).toBe(24 * 8 * (14 - 10));
  });

  it("keeps M150 generalized attention portable and accounts for M2250 ranges", () => {
    expect(planAceDitPhysicalCommandBufferCount(
      { batch: 1, latentFrames: 300, conditionTokens: 97 },
      "mixed-opt-0009",
    )).toBe(1_017);
    const shape = {
      batch: 1,
      latentFrames: 4_500,
      conditionTokens: 98,
    };
    const attention = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "full",
    });
    expect(attention.outputRangeCount).toBe(5);
    const crossAttentionC97 = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 97,
      headDimension: 128,
      mode: "full",
    });
    const crossAttentionC98 = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 98,
      headDimension: 128,
      mode: "full",
    });
    const scheduledMultiplyAdds = (
      plan: ReturnType<typeof planAceFixed32TiledFullAttention>,
    ) => plan.outputRanges.reduce(
      (total, range) => total + range.multiplyAdds,
      0,
    );
    expect(scheduledMultiplyAdds(crossAttentionC98)).toBe(451_584_000);
    expect(
      scheduledMultiplyAdds(crossAttentionC98) -
        scheduledMultiplyAdds(crossAttentionC97),
    ).toBe(4_608_000);
    expect(planAceDitPhysicalCommandBufferCount(
      shape,
      "mixed-opt-0009",
    )).toBe(2_553);
    expect(planAceDitPhysicalCommandBufferCount(
      shape,
      "mixed-opt-0037-k4",
    )).toBe(2_553);
    expect(planAceDitPhysicalCommandBufferCount(
      shape,
      "mixed-opt-0056-selective",
    )).toBe(2_553);
    const ordinary = planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      3_150_917_888,
      "mixed-opt-0009",
    );
    const tapped = planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      3_150_917_888,
      "mixed-opt-0009",
      1,
      true,
    );
    expect(tapped.commandBufferCount).toBe(ordinary.commandBufferCount);
    expect(tapped.evaluationReadbackBytes).toBe(
      8 * ordinary.readbackBufferBytes,
    );
    expect(tapped.detachedEvaluationLatentBytes).toBe(
      8 * ordinary.detachedFinalLatentBytes,
    );
    expect(tapped.accountedGpuBytes - ordinary.accountedGpuBytes).toBe(
      tapped.evaluationReadbackBytes,
    );
  });

  it("colors only disjoint lifetimes into exact aligned arena slots", () => {
    const arena = planAceDitBackendArena("reference-bf16", {
      batch: 1,
      latentFrames: 6_000,
      conditionTokens: 256,
    });
    const graph = planAceDitGpuBackendMemory(
      "reference-bf16",
      { batch: 1, latentFrames: 6_000, conditionTokens: 256 },
      3_000_000_000,
    );
    expect(arena.logicalRoleBytes).toBe(
      graph.graph.unaliasedGraphBytesExcludingWeights,
    );
    expect(arena.allocatedArenaBytes).toBeGreaterThanOrEqual(
      graph.graph.minimumGraphBytesExcludingWeights,
    );
    expect(arena.allocatedArenaBytes).toBeLessThan(arena.logicalRoleBytes);
    expect(arena.slots.length).toBeLessThan(arena.roles.length);
    expect(arena.alignmentOverheadBytes).toBeGreaterThanOrEqual(0);
    expect(arena.lifetimeReuseSavingsBytes).toBeGreaterThan(0);
    for (const slot of arena.slots) {
      expect(slot.byteLength % 256).toBe(0);
      const roles = slot.roles.map((key) =>
        arena.roles.find((role) => role.key === key)!
      );
      for (let left = 0; left < roles.length; left += 1) {
        for (let right = left + 1; right < roles.length; right += 1) {
          expect(
            overlap(roles[left]!.lifetimes, roles[right]!.lifetimes),
            `${roles[left]!.key} aliases ${roles[right]!.key}`,
          ).toBe(false);
        }
      }
    }
    expect(graph.commandBufferCount).toBe(
      planAceDitPhysicalCommandBufferCount({
        batch: 1,
        latentFrames: 6_000,
        conditionTokens: 256,
      }) + 1,
    );
    expect(graph.commandBufferCount).toBeGreaterThan(250);
    expect(graph.accountedGpuBytes).toBe(
      graph.residentWeightBytes + graph.arena.allocatedArenaBytes +
        graph.readbackBufferBytes,
    );
    expect(graph.boundedCpuBytes).toBeGreaterThanOrEqual(
      graph.detachedFinalLatentBytes,
    );
  });

  it("halves activation roles for raw FP16 while retaining FP32 controls", () => {
    const reference = planAceDitGpuBackendMemory(
      "reference-bf16",
      SHAPE,
      1_024,
    );
    const fp16 = planAceDitGpuBackendMemory("raw-fp16", SHAPE, 1_024);
    expect(reference.graph.conditionInputBytes).toBe(
      2 * fp16.graph.conditionInputBytes,
    );
    expect(reference.graph.layerScratchBytes).toBe(
      2 * fp16.graph.layerScratchBytes,
    );
    expect(reference.graph.ropeBytes).toBe(fp16.graph.ropeBytes);
    expect(reference.graph.timestepInputBytes).toBe(
      fp16.graph.timestepInputBytes,
    );
    expect(fp16.maximumEncodingStagingBytes).toBeGreaterThan(0);
    expect(reference.maximumEncodingStagingBytes).toBe(0);
  });
});

describe("ACE DiT resident phase ownership", () => {
  it("destroys a transferred phase that is not exclusively DiT", () => {
    const destroy = vi.fn();
    const phase = {
      phases: ["dit", "constants"],
      destroy,
    } as unknown as AceGpuTensorPhase;
    expect(() => AceDitResidentModel.take(phase, "raw-fp16")).toThrow(
      /exclusively resident dit phase/,
    );
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});

describe("ACE DiT concrete factory", () => {
  it("compiles 249 logical stages without eagerly recording commands", async () => {
    const fixture = fakeFactoryDevice("reference-bf16");
    const phases = fakeMixedDitPhases("opt-0037-k4");
    const progress: AceDitGpuBackendProgress[] = [];
    const condition = Float32Array.from(
      { length: 2_048 },
      (_, index) => Math.fround((index % 7 - 3) / 16),
    );
    const context = Float32Array.from(
      { length: 2 * 128 },
      (_, index) => Math.fround((index % 5 - 2) / 8),
    );
    const initialLatent = Float32Array.from(
      { length: 2 * 64 },
      (_, index) => Math.fround((index % 9 - 4) / 4),
    );
    const expectedInitialLatent = Float32Array.from(initialLatent);
    const backend = await AceDitGpuBackend.create({
      device: fixture.device,
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      ditDenseRuntimeProfile: ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
      shape: SHAPE,
      inputs: { condition, context, initialLatent },
      dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
      ownedDitWeights: phases.reference.value,
      ownedDitDenseWeights: phases.mixed.value,
      onProgress: (event) => progress.push(event),
    });
    condition.fill(99);
    context.fill(99);
    initialLatent.fill(99);

    expect(progress.filter((event) => event.stage === "compile")).toHaveLength(
      ACE_DIT_GRAPH_QUANTUM_COUNT,
    );
    expect(progress.at(-1)).toMatchObject({
      stage: "compile",
      compiledQuanta: 249,
      totalQuanta: 249,
    });
    expect(fixture.writeBufferCalls).toHaveLength(16);
    expect(fixture.createdCommandBuffers).toHaveLength(0);
    expect(fixture.scopeDepth.value).toBe(0);
    expect(phases.reference.destroy).not.toHaveBeenCalled();
    expect(phases.mixed.destroy).not.toHaveBeenCalled();
    const bindGroupLabels = vi.mocked(fixture.device.createBindGroup).mock.calls
      .map(([descriptor]) => descriptor.label ?? "");
    const denseK4Labels = bindGroupLabels.filter((label) =>
      label.endsWith("-opt-0032-bindings")
    );
    expect(denseK4Labels).toHaveLength(24 * 8 * 9);
    expect(new Set(denseK4Labels.map((label) =>
      label.replace(/^ace-dit-eval-\d+-/u, "ace-dit-eval-*/")
    ))).toHaveLength(24 * 9);
    expect(denseK4Labels.some((label) => label.includes("cross-cache")))
      .toBe(false);
    expect(bindGroupLabels.filter((label) =>
      /^ace-dit-cross-cache-\d+-(?:key|value)-projection-opt-0003-range-0-bindings$/u
        .test(label)
    )).toHaveLength(48);
    expect(
      Array.from(new Uint16Array(
        fixture.writeBufferCalls[0]!.bytes.buffer,
        fixture.writeBufferCalls[0]!.bytes.byteOffset,
        4,
      )),
    ).toEqual([0x0000, 0xbe40, 0x0000, 0xbe00]);

    vi.useFakeTimers();
    let result;
    try {
      const running = backend.run();
      await vi.runAllTimersAsync();
      result = await running;
    } finally {
      vi.useRealTimers();
    }
    expect(result.finalLatent).toEqual(expectedInitialLatent);
    expect(fixture.submissions).toHaveLength(250);
    expect(fixture.createdCommandBuffers).toHaveLength(250);
    expect(fixture.maximumOutstanding.value).toBe(1);
    await backend.destroy();
    await backend.destroy();
    expect(phases.reference.destroy).toHaveBeenCalledTimes(1);
    expect(phases.mixed.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.destroyedBuffers.value).toBe(fixture.createdBuffers.length);
  });

  it("fails before allocation and destroys wrong-phase weights", async () => {
    const destroy = vi.fn();
    await expect(AceDitGpuBackend.create({
      device: {} as GPUDevice,
      executionProfile: ACE_FP16_PORTABLE_PROFILE,
      shape: SHAPE,
      inputs: {
        condition: new Float32Array(2_048),
        context: new Float32Array(256),
        initialLatent: new Float32Array(128),
      },
      dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
      ownedDitWeights: {
        phases: ["vae"],
        destroy,
      } as unknown as AceGpuTensorPhase,
      ownedDitDenseWeights: fakeMixedDitPhases().mixed.value,
    })).rejects.toThrow(/exclusive reference and authenticated dense dit phases/);
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it("destroys transferred weights when the execution profile is spoofed", async () => {
    const fixture = fakeFactoryDevice("reference-bf16");
    const phases = fakeMixedDitPhases();
    await expect(AceDitGpuBackend.create({
      device: fixture.device,
      executionProfile: {
        ...ACE_REFERENCE_SUBGROUP_PROFILE,
        matrixArithmetic: "spoofed" as never,
      },
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      shape: SHAPE,
      inputs: {
        condition: new Float32Array(2_048),
        context: new Float32Array(256),
        initialLatent: new Float32Array(128),
      },
      dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
      ownedDitWeights: phases.reference.value,
      ownedDitDenseWeights: phases.mixed.value,
    })).rejects.toThrow(/authenticated profile/);
    expect(phases.reference.destroy).toHaveBeenCalledTimes(1);
    expect(phases.mixed.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.createdBuffers).toHaveLength(0);
  });

  it("cancels compilation transactionally and destroys every allocation", async () => {
    const fixture = fakeFactoryDevice("reference-bf16");
    const phases = fakeMixedDitPhases();
    const controller = new AbortController();
    await expect(AceDitGpuBackend.create({
      device: fixture.device,
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      shape: SHAPE,
      inputs: {
        condition: new Float32Array(2_048),
        context: new Float32Array(256),
        initialLatent: new Float32Array(128),
      },
      dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
      ownedDitWeights: phases.reference.value,
      ownedDitDenseWeights: phases.mixed.value,
      signal: controller.signal,
      onProgress: (progress) => {
        if (progress.stage === "compile" && progress.compiledQuanta === 10) {
          controller.abort();
        }
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(phases.reference.destroy).toHaveBeenCalledTimes(1);
    expect(phases.mixed.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.createdCommandBuffers).toHaveLength(0);
    expect(fixture.destroyedBuffers.value).toBe(fixture.createdBuffers.length);
    expect(fixture.scopeDepth.value).toBe(0);
  });

  it("rejects malformed logical inputs and releases the transferred phase", async () => {
    const fixture = fakeFactoryDevice("reference-bf16");
    const phases = fakeMixedDitPhases();
    const condition = new Float32Array(2_048);
    condition[17] = Number.NaN;
    await expect(AceDitGpuBackend.create({
      device: fixture.device,
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      shape: SHAPE,
      inputs: {
        condition,
        context: new Float32Array(256),
        initialLatent: new Float32Array(128),
      },
      dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
      ownedDitWeights: phases.reference.value,
      ownedDitDenseWeights: phases.mixed.value,
    })).rejects.toThrow(/condition\[17\] is not finite/);
    expect(phases.reference.destroy).toHaveBeenCalledTimes(1);
    expect(phases.mixed.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.createdBuffers).toHaveLength(0);
  });
});

describe("ACE DiT backend execution and readback", () => {
  it("rejects a prepared-resource GEMM backend mismatch and tears down", () => {
    const fixture = preparedFixture(
      "reference-bf16",
      new Float32Array(4).buffer,
    );
    expect(() => AceDitGpuBackend.fromPreparedResources({
      ...fixture.resources,
      gemmBackend: "fixed32-subgroups",
    })).toThrow(/prepared resources violate/);
    expect(fixture.graph.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });

  it.each([
    {
      profile: "reference-bf16" as const,
      expected: paddedFloat32([0.5, -1.25, 3, 0.125]),
      mapped: paddedFloat32([0.5, -1.25, 3, 0.125]).buffer,
    },
    {
      profile: "raw-fp16" as const,
      expected: paddedFloat32([0.5, -1.25, 3, 0.125]),
      mapped: paddedUint16([0x3800, 0xbd00, 0x4200, 0x3000]).buffer,
    },
  ])("drains the final copy and detaches $profile output", async ({
    profile,
    expected,
    mapped,
  }) => {
    const fixture = preparedFixture(profile, mapped);
    const backend = AceDitGpuBackend.fromPreparedResources(fixture.resources);
    const result = await backend.run();
    expect(result.finalLatent).toEqual(expected);
    expect(result.finalLatent.buffer).not.toBe(mapped);
    expect(result).toMatchObject({
      commandBuffersSubmitted: 250,
      queueDrains: 250,
      cooperativeIdleMs: 249,
      completedEvaluations: 8,
    });
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.submissions[0]).toHaveLength(1);
    expect(fixture.idle).toHaveBeenCalledTimes(1);
    expect(fixture.readback.mapAsync).toHaveBeenCalledWith(
      GPUMapMode.READ,
      0,
      mapped.byteLength,
    );
    expect(fixture.readback.unmap).toHaveBeenCalledTimes(1);
    expect(fixture.resources.onCommandProfile).toBeUndefined();
    expect(fixture.resources.now).toBeUndefined();
    expect(fixture.graph.physicalCommandDescriptors).toBeUndefined();
    expect(fixture.graph.run.mock.calls[0]![0]).not.toHaveProperty(
      "onCommandBufferDrained",
    );
    expect(fixture.graph.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    await expect(backend.run()).rejects.toMatchObject({
      name: "InvalidStateError",
    });
  });

  it("forwards the capture-free production submission policy to the graph", async () => {
    const mapped = paddedFloat32([0.5, -1.25, 3, 0.125]).buffer;
    const fixture = preparedFixture("reference-bf16", mapped);
    const memory = planAceDitGpuBackendMemory(
      "reference-bf16",
      fixture.resources.shape,
      fixture.resources.memory.residentWeightBytes,
      "mixed-opt-0009",
    );
    const backend = AceDitGpuBackend.fromPreparedResources({
      ...fixture.resources,
      gemmBackend: "mixed-opt-0009",
      memory,
      ditSubmissionPolicy: "depth2-phase-epoch4",
    });
    await expect(backend.run()).resolves.toMatchObject({
      completedEvaluations: 8,
      commandBuffersSubmitted: 250,
      queueDrains: 250,
      cooperativeIdleMs: 249,
    });
    expect(fixture.graph.run.mock.calls[0]![0]).toMatchObject({
      submissionPolicy: "depth2-phase-epoch4",
      physicalQuantaPerCommandBuffer: 1,
    });
    expect(fixture.graph.run.mock.calls[0]![0]).not.toHaveProperty(
      "opt0080SchedulingProfile",
    );
    expect(fixture.graph.physicalCommandDescriptors).toBeUndefined();
    expect(fixture.graph.destroy).toHaveBeenCalledOnce();
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it("rejects production depth two when prepared resources enable capture", () => {
    const fixture = opt0018PreparedFixture();
    expect(() => AceDitGpuBackend.fromPreparedResources({
      ...fixture.resources,
      ditSubmissionPolicy: "depth2-phase-epoch4",
    })).toThrow(/prepared resources violate/u);
    expect(fixture.graph.destroy).toHaveBeenCalledOnce();
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it("maps eight in-command sampler copies with zero extra submits or drains", async () => {
    const values = paddedFloat32([0.5, -1.25, 3, 0.125]);
    const mapped = values.buffer;
    const fixture = preparedFixture("reference-bf16", mapped);
    const evaluationReadbacks = Array.from({ length: 8 }, () => ({
      size: mapped.byteLength,
      mapAsync: vi.fn(async () => undefined),
      getMappedRange: vi.fn(() => mapped),
      unmap: vi.fn(),
    } as unknown as GPUBuffer));
    const memory = planAceDitGpuBackendMemory(
      "reference-bf16",
      fixture.resources.shape,
      fixture.resources.memory.residentWeightBytes,
      "portable",
      1,
      true,
    );
    const backend = AceDitGpuBackend.fromPreparedResources({
      ...fixture.resources,
      evaluationReadbacks,
      memory,
    });
    const result = await backend.run();
    expect(result.evaluationLatents).toHaveLength(8);
    expect(result.evaluationLatents?.every((latent) =>
      new Uint32Array(latent.buffer).every((word, index) =>
        word === new Uint32Array(values.buffer)[index]
      )
    )).toBe(true);
    expect(result).toMatchObject({
      commandBuffersSubmitted: 250,
      queueDrains: 250,
      cooperativeIdleMs: 249,
      completedEvaluations: 8,
    });
    expect(fixture.graph.run.mock.calls[0]![0].evaluationReadbacks)
      .toBe(evaluationReadbacks);
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.idle).toHaveBeenCalledTimes(1);
    for (const readback of evaluationReadbacks) {
      expect(readback.mapAsync).toHaveBeenCalledOnce();
      expect(readback.unmap).toHaveBeenCalledOnce();
    }
    expect(fixture.graph.destroy).toHaveBeenCalledOnce();
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it("captures and reconciles the exact 2553+readback M2250 topology", async () => {
    const fixture = opt0018PreparedFixture();
    const backend = AceDitGpuBackend.fromPreparedResources(fixture.resources);
    const result = await backend.run();
    expect(result).toMatchObject({
      commandBuffersSubmitted: 2_554,
      queueDrains: 2_554,
      cooperativeIdleMs: 2_553,
      completedEvaluations: 8,
    });
    expect(fixture.onCommandProfile).toHaveBeenCalledOnce();
    const profile = fixture.onCommandProfile.mock.calls[0]![0];
    expect(profile).toMatchObject({
      schema: "ace-dit-m2250-command-profile-v1",
      graphCommandBufferCount: 2_553,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 2_554,
      graphRequestedIdleMs: 2_552,
      graphToReadbackRequestedIdleMs: 1,
      timingStorageBytes: 2_553 * (Float64Array.BYTES_PER_ELEMENT + 1),
    });
    expect(profile.descriptorTable).toMatchObject({
      sha256:
        "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
      serializedBytes: 1_869_566,
      memberCount: 6_833,
    });
    expect(profile.timings).toHaveLength(2_553);
    expect(profile.precompute.commandBufferCount).toBe(25);
    expect(profile.evaluations.map((value) => value.commandBufferCount)).toEqual(
      Array.from({ length: 8 }, () => 316),
    );
    expect(Object.fromEntries(ACE_DIT_PROFILE_FAMILIES.map((family) => [
      family,
      profile.families[family].commandBufferCount,
    ]))).toEqual({
      precompute: 1,
      "cross-cache": 24,
      timestep: 8,
      input: 8,
      "attention-projections": 192,
      "self-full": 384,
      "self-sliding": 0,
      "cross-attention": 0,
      "feed-forward": 576,
      plumbing: 0,
      output: 8,
      "sampler-dcw": 8,
      mixed: 1_344,
    });
    expect(profile.familyByBucket.mixed.map(
      (value) => value.commandBufferCount,
    )).toEqual([0, 168, 168, 168, 168, 168, 168, 168, 168]);
    expect(profile.graphSubmitThroughDrainMs).toBeCloseTo(
      profile.timings.reduce((total, value) => total + value, 0),
      10,
    );
    expect(profile.slowest).toHaveLength(16);
    expect(profile.slowest.map((value) => value.submitThroughDrainMs)).toEqual(
      [...profile.slowest]
        .map((value) => value.submitThroughDrainMs)
        .sort((left, right) => right - left),
    );
    expect(fixture.graph.destroy).toHaveBeenCalledOnce();
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it("captures only the exact OPT-0067 precompute/evaluation-0 prefix", async () => {
    const fixture = opt0018PreparedFixture();
    const originalRun = fixture.graph.run.getMockImplementation()!;
    const descriptors = fixture.graph.physicalCommandDescriptors!.descriptors;
    fixture.graph.run.mockImplementation(async (
      options: AceDitGraphRunOptions,
    ) => {
      const full = await originalRun({
        ...options,
        onCommandBufferDrained: undefined,
        onProgress: undefined,
      });
      for (const descriptor of descriptors.slice(
        0,
        ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT,
      )) {
        options.onCommandBufferDrained?.(
          descriptor,
          (descriptor.physicalIndex % 23 + 1) / 100,
        );
      }
      return Object.freeze({
        ...full,
        commandBuffersSubmitted:
          ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT,
        queueDrains:
          ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT,
        cooperativeIdleMs:
          ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT - 1,
        completedEvaluations: 1 as const,
      });
    });
    const onOpt0067EvaluationProfile = vi.fn(
      (_profile: AceOpt0067DitCommandProfile) => undefined,
    );
    const { onCommandProfile: _omitted, ...baseResources } = fixture.resources;
    const backend = AceDitGpuBackend.fromPreparedResources({
      ...baseResources,
      onOpt0067EvaluationProfile,
      opt0067EvaluationLimit: 1,
    });
    const result = await backend.run();
    expect(result).toMatchObject({
      commandBuffersSubmitted: 342,
      queueDrains: 342,
      cooperativeIdleMs: 341,
      completedEvaluations: 1,
    });
    expect(fixture.graph.run.mock.calls.at(-1)![0]).toMatchObject({
      physicalQuantaPerCommandBuffer: 1,
      opt0067EvaluationLimit: 1,
    });
    expect(fixture.graph.run.mock.calls.at(-1)![0]).not.toHaveProperty(
      "evaluationReadbacks",
    );
    expect(onOpt0067EvaluationProfile).toHaveBeenCalledOnce();
    const profile = onOpt0067EvaluationProfile.mock.calls[0]![0];
    expect(profile).toMatchObject({
      schema: "ace-dit-opt0067-evaluation0-command-profile-v1",
      graphCommandBufferCount: 341,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 342,
      graphQueueDrainCount: 341,
      totalQueueDrainCount: 342,
      graphRequestedIdleMs: 340,
      evaluationCommandBufferCount: 316,
      evaluationRequestedIdleMs: 316,
      timingStorageBytes: 341 * (Float64Array.BYTES_PER_ELEMENT + 1),
    });
    expect(profile.timings).toHaveLength(341);
    expect(profile.precompute.commandBufferCount).toBe(25);
    expect(profile.evaluation.commandBufferCount).toBe(316);
    expect(profile.familyByBucket["self-full"].map(
      ({ commandBufferCount }) => commandBufferCount,
    )).toEqual([0, 48]);
    expect(fixture.graph.destroy).toHaveBeenCalledOnce();
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it.each([
    ["depth1-epoch1", 341, 340, 1, 1_000, false],
    ["opt-0080-depth2-epoch4", 86, 85, 2, 1_000, false],
    ["opt-0080-depth2-epoch4", 86, 85, 2, 1, true],
  ] as const)(
    "captures nested OPT-0080 topology for %s without summing fence latencies",
    async (
      schedulingProfile,
      expectedDrains,
      expectedIdles,
      expectedMaximum,
      epochDurationDivisor,
      rejectsOversizedEpochWalls,
    ) => {
      const fixture = opt0018PreparedFixture();
      const originalRun = fixture.graph.run.getMockImplementation()!;
      const table = fixture.graph.physicalCommandDescriptors!;
      const descriptors = table.descriptors.slice(0, 341);
      expect(descriptors.reduce(
        (total, descriptor) => total + descriptor.members.length,
        0,
      )).not.toBe(table.memberCount);
      const epochs = opt0080EpochPlan(schedulingProfile);
      fixture.graph.run.mockImplementation(async (
        options: AceDitGraphRunOptions,
      ) => {
        const full = await originalRun({
          ...options,
          onCommandBufferDrained: undefined,
          onProgress: undefined,
          onOpt0080CommandBufferCompleted: undefined,
          onOpt0080CompletionEpochDrained: undefined,
          onOpt0080PhaseStarted: undefined,
        });
        options.onOpt0080PhaseStarted?.(Object.freeze({
          phaseIndex: 0,
          firstCommandBufferIndex: 0,
          commandBufferCount: 25,
        }));
        let completedEpochs = 0;
        for (let index = 0; index < descriptors.length; index += 1) {
          if (index === 25) {
            options.onOpt0080PhaseStarted?.(Object.freeze({
              phaseIndex: 1,
              firstCommandBufferIndex: 25,
              commandBufferCount: 316,
            }));
          }
          const descriptor = descriptors[index]!;
          const epoch = epochs.find((value) =>
            index >= value.firstCommandBufferIndex &&
            index <= value.lastCommandBufferIndex
          )!;
          const trueQueueDrain = index === epoch.lastCommandBufferIndex;
          if (trueQueueDrain) completedEpochs += 1;
          const graphProgress = Object.freeze({
            completedQuanta: index + 1,
            totalQuanta: 341,
            completedCommandBuffers: index + 1,
            totalCommandBuffers: 341,
            queueDrains: completedEpochs,
            cooperativeIdleMs: Math.max(0, completedEpochs - 1),
            completedEvaluations: index === 340 ? 1 : 0,
            totalEvaluations: 8 as const,
            quantum: createAceDitGraphQuantumPlan()[
              descriptor.logicalIndex
            ]!,
            subquantumIndex: descriptor.subquantumIndex,
            subquantumCount: descriptor.subquantumCount,
            commandId: descriptor.commandId,
            batchIndex: index,
            batchFirstPhysicalQuantum: index,
            batchPhysicalQuantumCount: 1,
            physicalQuantaPerCommandBuffer: 1 as const,
          });
          const schedulingProgress = Object.freeze({
            completedCommandBuffers: index + 1,
            totalCommandBuffers: 341,
            completionFenceRequestedCount: index + 1,
            completionFenceSettledCount: index + 1,
            completionFenceRejectedCount: 0,
            trueQueueDrainCount: completedEpochs,
            completionEpochCount: completedEpochs,
            requestedCooperativeIdleMs: Math.max(0, completedEpochs - 1),
            cooperativeIdleTurns: Math.max(0, completedEpochs - 1),
            outstandingCommandBuffers: 0,
          });
          options.onOpt0080CommandBufferCompleted?.(Object.freeze({
            commandBufferIndex: index,
            submitThroughCompletionFenceMs: (index % 19 + 1) / 100,
            trueQueueDrain,
            completionEpochIndex: epoch.completionEpochIndex,
            descriptor,
            schedulingProgress,
            graphProgress,
          }));
          if (trueQueueDrain) {
            options.onOpt0080CompletionEpochDrained?.(Object.freeze({
              ...epoch,
              submitThroughTrueDrainMs:
                epoch.commandBufferCount / epochDurationDivisor,
            }));
          }
        }
        return Object.freeze({
          ...full,
          commandBuffersSubmitted: 341,
          queueDrains: expectedDrains,
          cooperativeIdleMs: expectedIdles,
          completedEvaluations: 1 as const,
          opt0080Scheduling: Object.freeze({
            profile: schedulingProfile,
            completionFenceRequestedCount: 341,
            completionFenceSettledCount: 341,
            completionFenceRejectedCount: 0,
            trueQueueDrainCount: expectedDrains,
            completionEpochCount: expectedDrains,
            requestedCooperativeIdleMs: expectedIdles,
            cooperativeIdleTurns: expectedIdles,
            maximumOutstandingCommandBuffers: expectedMaximum,
            maximumPendingDescriptorCount: expectedMaximum,
          }),
        });
      });
      const onOpt0080EvaluationProfile = vi.fn(
        (_profile: AceOpt0080DitCommandProfile) => undefined,
      );
      const onOpt0080CommandBufferCompleted = vi.fn();
      const { onCommandProfile: _omitted, ...baseResources } = fixture.resources;
      const backend = AceDitGpuBackend.fromPreparedResources({
        ...baseResources,
        opt0067EvaluationLimit: 1,
        opt0080SchedulingProfile: schedulingProfile,
        onOpt0080EvaluationProfile,
        onOpt0080CommandBufferCompleted,
      });
      if (rejectsOversizedEpochWalls) {
        await expect(backend.run()).rejects.toThrow(
          /disjoint wall timing did not reconcile/u,
        );
        expect(onOpt0080EvaluationProfile).not.toHaveBeenCalled();
        expect(fixture.graph.destroy).toHaveBeenCalledOnce();
        expect(fixture.destroy).toHaveBeenCalledOnce();
        return;
      }
      const result = await backend.run();
      expect(result).toMatchObject({
        commandBuffersSubmitted: 342,
        queueDrains: expectedDrains + 1,
        cooperativeIdleMs: expectedIdles + 1,
        completedEvaluations: 1,
      });
      const runOptions = fixture.graph.run.mock.calls.at(-1)![0];
      expect(runOptions).toMatchObject({
        opt0067EvaluationLimit: 1,
        opt0080SchedulingProfile: schedulingProfile,
      });
      expect(runOptions).not.toHaveProperty("opt0062AttentionIdentityReadback");
      expect(onOpt0080EvaluationProfile).toHaveBeenCalledOnce();
      expect(onOpt0080CommandBufferCompleted).toHaveBeenCalledTimes(341);
      expect(onOpt0080CommandBufferCompleted.mock.calls[0]![0]).toMatchObject({
        commandBufferIndex: 0,
        descriptor: { physicalIndex: 0 },
      });
      const profile = onOpt0080EvaluationProfile.mock.calls[0]![0];
      expect(profile).toMatchObject({
        schema: "ace-dit-opt0080-evaluation0-command-profile-v1",
        schedulingProfile,
        descriptorTable: { memberCount: 6_833 },
        topology: {
          schedulingProfile,
          descriptorTableMemberCount: 6_833,
          graphCommandBufferCount: 341,
          totalCommandBufferCount: 342,
          commandBuffersSubmitted: 342,
          completionFenceRequestedCount: 342,
          completionFenceSettledCount: 342,
          completionFenceRejectedCount: 0,
          graphTrueQueueDrainCount: expectedDrains,
          totalTrueQueueDrainCount: expectedDrains + 1,
          graphCooperativeIdleTurns: expectedIdles,
          totalCooperativeIdleTurns: expectedIdles + 1,
          maximumOutstandingCommandBuffers: expectedMaximum,
          maximumPendingDescriptorCount: expectedMaximum,
          pendingDescriptorCountAfterCleanup: 0,
          graphToReadbackRequestedIdleMs: 1,
        },
      });
      expect(profile.topology.submitThroughCompletionFenceMs).toHaveLength(341);
      expect(profile.topology.graphCompletionEpochs).toHaveLength(expectedDrains);
      expect(profile.precomputeWallMs + profile.evaluationWallMs).toBeCloseTo(
        profile.graphWallMs,
        12,
      );
      expect(profile.topology.graphCompletionEpochs.reduce(
        (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
        0,
      )).toBeLessThanOrEqual(profile.graphWallMs);
      expect(profile).not.toHaveProperty("graphEpochWallSumMs");
      expect(fixture.graph.destroy).toHaveBeenCalledOnce();
      expect(fixture.destroy).toHaveBeenCalledOnce();
    },
  );

  it.each([
    ["depth1-epoch1", 2_553, 2_552, 1],
    ["opt-0080-depth2-epoch4", 639, 638, 2],
  ] as const)(
    "captures full OPT-0080 walls and topology for %s",
    async (
      schedulingProfile,
      expectedDrains,
      expectedIdles,
      expectedMaximum,
    ) => {
      const fixture = opt0018PreparedFixture();
      const originalRun = fixture.graph.run.getMockImplementation()!;
      const table = fixture.graph.physicalCommandDescriptors!;
      const descriptors = table.descriptors;
      const phaseCounts = [25, 316, 316, 316, 316, 316, 316, 316, 316];
      const epochs = opt0080EpochPlan(schedulingProfile, phaseCounts);
      const phaseStarts = new Map<number, number>();
      let first = 0;
      for (let phaseIndex = 0; phaseIndex < phaseCounts.length; phaseIndex += 1) {
        phaseStarts.set(first, phaseIndex);
        first += phaseCounts[phaseIndex]!;
      }
      fixture.graph.run.mockImplementation(async (
        options: AceDitGraphRunOptions,
      ) => {
        const full = await originalRun({
          ...options,
          onCommandBufferDrained: undefined,
          onProgress: undefined,
          onOpt0080CommandBufferCompleted: undefined,
          onOpt0080CompletionEpochDrained: undefined,
          onOpt0080PhaseStarted: undefined,
        });
        let completedEpochs = 0;
        for (let index = 0; index < descriptors.length; index += 1) {
          const phaseIndex = phaseStarts.get(index);
          if (phaseIndex !== undefined) {
            options.onOpt0080PhaseStarted?.(Object.freeze({
              phaseIndex,
              firstCommandBufferIndex: index,
              commandBufferCount: phaseCounts[phaseIndex]!,
            }));
          }
          const descriptor = descriptors[index]!;
          const epoch = epochs.find((value) =>
            index >= value.firstCommandBufferIndex &&
            index <= value.lastCommandBufferIndex
          )!;
          const trueQueueDrain = index === epoch.lastCommandBufferIndex;
          if (trueQueueDrain) completedEpochs += 1;
          const graphProgress = Object.freeze({
            completedQuanta: index + 1,
            totalQuanta: 2_553,
            completedCommandBuffers: index + 1,
            totalCommandBuffers: 2_553,
            queueDrains: completedEpochs,
            cooperativeIdleMs: Math.max(0, completedEpochs - 1),
            completedEvaluations: index === 2_552 ? 8 : 0,
            totalEvaluations: 8 as const,
            quantum: createAceDitGraphQuantumPlan()[descriptor.logicalIndex]!,
            subquantumIndex: descriptor.subquantumIndex,
            subquantumCount: descriptor.subquantumCount,
            commandId: descriptor.commandId,
            batchIndex: index,
            batchFirstPhysicalQuantum: index,
            batchPhysicalQuantumCount: 1,
            physicalQuantaPerCommandBuffer: 1 as const,
          });
          const schedulingProgress = Object.freeze({
            completedCommandBuffers: index + 1,
            totalCommandBuffers: 2_553,
            completionFenceRequestedCount: index + 1,
            completionFenceSettledCount: index + 1,
            completionFenceRejectedCount: 0,
            trueQueueDrainCount: completedEpochs,
            completionEpochCount: completedEpochs,
            requestedCooperativeIdleMs: Math.max(0, completedEpochs - 1),
            cooperativeIdleTurns: Math.max(0, completedEpochs - 1),
            outstandingCommandBuffers: 0,
          });
          options.onOpt0080CommandBufferCompleted?.(Object.freeze({
            commandBufferIndex: index,
            submitThroughCompletionFenceMs: (index % 19 + 1) / 100,
            trueQueueDrain,
            completionEpochIndex: epoch.completionEpochIndex,
            descriptor,
            schedulingProgress,
            graphProgress,
          }));
          if (trueQueueDrain) {
            options.onOpt0080CompletionEpochDrained?.(Object.freeze({
              ...epoch,
              submitThroughTrueDrainMs: epoch.commandBufferCount / 1_000,
            }));
          }
        }
        return Object.freeze({
          ...full,
          commandBuffersSubmitted: 2_553,
          queueDrains: expectedDrains,
          cooperativeIdleMs: expectedIdles,
          completedEvaluations: 8 as const,
          opt0080Scheduling: Object.freeze({
            profile: schedulingProfile,
            completionFenceRequestedCount: 2_553,
            completionFenceSettledCount: 2_553,
            completionFenceRejectedCount: 0,
            trueQueueDrainCount: expectedDrains,
            completionEpochCount: expectedDrains,
            requestedCooperativeIdleMs: expectedIdles,
            cooperativeIdleTurns: expectedIdles,
            maximumOutstandingCommandBuffers: expectedMaximum,
            maximumPendingDescriptorCount: expectedMaximum,
          }),
        });
      });
      const onOpt0080FullProfile = vi.fn(
        (_profile: AceOpt0080FullDitCommandProfile) => undefined,
      );
      const onOpt0080CommandBufferCompleted = vi.fn();
      const { onCommandProfile: _omitted, ...baseResources } = fixture.resources;
      const backend = AceDitGpuBackend.fromPreparedResources({
        ...baseResources,
        opt0080FullSchedulingProfile: schedulingProfile,
        onOpt0080FullProfile,
        onOpt0080CommandBufferCompleted,
      });
      const result = await backend.run();
      expect(result).toMatchObject({
        commandBuffersSubmitted: 2_554,
        queueDrains: expectedDrains + 1,
        cooperativeIdleMs: expectedIdles + 1,
        completedEvaluations: 8,
      });
      expect(fixture.graph.run.mock.calls.at(-1)![0]).toMatchObject({
        opt0080FullGraph: true,
        opt0080SchedulingProfile: schedulingProfile,
      });
      expect(onOpt0080CommandBufferCompleted).toHaveBeenCalledTimes(2_553);
      expect(onOpt0080FullProfile).toHaveBeenCalledOnce();
      const profile = onOpt0080FullProfile.mock.calls[0]![0];
      expect(profile).toMatchObject({
        schema: "ace-dit-opt0080-full-command-profile-v1",
        schedulingProfile,
        descriptorTable: { memberCount: 6_833 },
        topology: {
          graphCommandBufferCount: 2_553,
          totalCommandBufferCount: 2_554,
          completionFenceRequestedCount: 2_554,
          graphTrueQueueDrainCount: expectedDrains,
          totalTrueQueueDrainCount: expectedDrains + 1,
          graphCooperativeIdleTurns: expectedIdles,
          totalCooperativeIdleTurns: expectedIdles + 1,
          maximumOutstandingCommandBuffers: expectedMaximum,
          maximumPendingDescriptorCount: expectedMaximum,
        },
      });
      expect(profile.evaluationWallMs).toHaveLength(8);
      expect(profile.precomputeWallMs + profile.evaluationWallMs.reduce(
        (sum, wall) => sum + wall,
        0,
      )).toBeCloseTo(profile.graphWallMs, 12);
      expect(profile.topology.graphCompletionEpochs).toHaveLength(
        expectedDrains,
      );
      expect(profile.topology.submitThroughCompletionFenceMs).toHaveLength(
        2_553,
      );
      expect(fixture.graph.destroy).toHaveBeenCalledOnce();
      expect(fixture.destroy).toHaveBeenCalledOnce();
    },
  );

  it.each([
    [1, 2_553, 1],
    [8, 320, 1],
    [16, 160, 9],
  ] as const)(
    "captures exact OPT-0034 batch%i M2250 scheduling telemetry",
    async (
      physicalQuantaPerCommandBuffer,
      expectedGraphCommandBuffers,
      expectedTail,
    ) => {
      const fixture = opt0034PreparedFixture(
        physicalQuantaPerCommandBuffer,
      );
      const backend = AceDitGpuBackend.fromPreparedResources(
        fixture.resources,
      );
      const result = await backend.run();
      expect(result).toMatchObject({
        commandBuffersSubmitted: expectedGraphCommandBuffers + 1,
        queueDrains: expectedGraphCommandBuffers + 1,
        cooperativeIdleMs: expectedGraphCommandBuffers,
        completedEvaluations: 8,
      });
      expect(fixture.onSchedulingProfile).toHaveBeenCalledOnce();
      const profile = fixture.onSchedulingProfile.mock.calls[0]![0];
      expect(profile).toMatchObject({
        schema: "ace-dit-opt0034-command-buffer-coalescing-v1",
        physicalGraphQuantumCount: 2_553,
        physicalQuantaPerCommandBuffer,
        graphCommandBufferCount: expectedGraphCommandBuffers,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: expectedGraphCommandBuffers + 1,
        graphQueueDrainCount: expectedGraphCommandBuffers,
        totalQueueDrainCount: expectedGraphCommandBuffers + 1,
        graphRequestedIdleMs: expectedGraphCommandBuffers - 1,
        graphToReadbackRequestedIdleMs: 1,
        maximumPhysicalQuantaPerBatch:
          physicalQuantaPerCommandBuffer,
        descriptorTableSha256:
          "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
      });
      expect(profile.batches).toHaveLength(expectedGraphCommandBuffers);
      expect(profile.batches[0]).toMatchObject({
        batchIndex: 0,
        firstPhysicalIndex: 0,
        physicalQuantumCount: physicalQuantaPerCommandBuffer,
      });
      expect(profile.batches.at(-1)).toMatchObject({
        batchIndex: expectedGraphCommandBuffers - 1,
        lastPhysicalIndex: 2_552,
        physicalQuantumCount: expectedTail,
      });
      expect(profile.batches.reduce(
        (total, batch) => total + batch.physicalQuantumCount,
        0,
      )).toBe(2_553);
      expect(profile.graphSubmitThroughDrainMs).toBeCloseTo(
        profile.batches.reduce(
          (total, batch) => total + batch.submitThroughDrainMs,
          0,
        ),
        10,
      );
      expect(profile.physicalPrimitiveCount).toBe(
        fixture.descriptors.reduce(
          (total, descriptor) => total + descriptor.primitiveCount,
          0,
        ),
      );
      expect(profile.scheduledMultiplyAdds).toBe(
        fixture.descriptors.reduce(
          (total, descriptor) => total + descriptor.scheduledMultiplyAdds,
          0,
        ),
      );
      const runOptions = fixture.graph.run.mock.calls[0]![0];
      expect(runOptions.physicalQuantaPerCommandBuffer).toBe(
        physicalQuantaPerCommandBuffer,
      );
      expect(runOptions.onPhysicalBatchDrained).toBeTypeOf("function");
      expect(runOptions).not.toHaveProperty("onCommandBufferDrained");
      expect(fixture.graph.destroy).toHaveBeenCalledOnce();
      expect(fixture.destroy).toHaveBeenCalledOnce();
    },
  );

  it("honors cancellation after the graph drain and before readback submit", async () => {
    const idle = Promise.withResolvers<void>();
    const fixture = preparedFixture("reference-bf16", new Float32Array(4).buffer, {
      yieldQueueIdle: vi.fn(() => idle.promise),
    });
    const backend = AceDitGpuBackend.fromPreparedResources(fixture.resources);
    const controller = new AbortController();
    const running = backend.run(controller.signal);
    await waitUntil(() => fixture.idle.mock.calls.length === 1);
    controller.abort();
    idle.resolve();
    await expect(running).rejects.toMatchObject({ name: "AbortError" });
    expect(fixture.submissions).toHaveLength(0);
    expect(fixture.readback.mapAsync).not.toHaveBeenCalled();
    expect(fixture.graph.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });

  it("turns device loss into a typed failure and tears down once", async () => {
    const graphWait = Promise.withResolvers<void>();
    const fixture = preparedFixture("reference-bf16", new Float32Array(4).buffer, {
      graphWait,
    });
    const backend = AceDitGpuBackend.fromPreparedResources(fixture.resources);
    const running = backend.run();
    await waitUntil(() => fixture.graph.run.mock.calls.length === 1);
    fixture.deviceLost.resolve({
      reason: "unknown",
      message: "test adapter vanished",
    } as GPUDeviceLostInfo);
    graphWait.resolve();
    await expect(running).rejects.toBeInstanceOf(AceDitBackendDeviceLostError);
    expect(fixture.graph.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });
});

interface Lifetime {
  readonly firstQuantum: number;
  readonly lastQuantum: number;
}

function overlap(left: readonly Lifetime[], right: readonly Lifetime[]): boolean {
  return left.some((a) => right.some((b) =>
    a.firstQuantum <= b.lastQuantum && b.firstQuantum <= a.lastQuantum
  ));
}

function fakeMixedDitPhases(
  denseProfile: "opt-0009" | "opt-0037-k4" = "opt-0009",
): Readonly<{
  reference: Readonly<{
    value: AceGpuTensorPhase;
    destroy: ReturnType<typeof vi.fn>;
  }>;
  mixed: Readonly<{
    value: AceGpuTensorPhase;
    destroy: ReturnType<typeof vi.fn>;
  }>;
}> {
  const reference = fakeDitPhase("reference-shared");
  const mixed = fakeDitPhase("mixed-layers", denseProfile);
  return Object.freeze({ reference, mixed });
}

function fakeDitPhase(
  profile: "reference-shared" | "mixed-layers",
  denseProfile: "opt-0009" | "opt-0037-k4" = "opt-0009",
): Readonly<{
  value: AceGpuTensorPhase;
  destroy: ReturnType<typeof vi.fn>;
}> {
  const destroy = vi.fn();
  const weightBuffer = { size: 1_000_000_000 } as GPUBuffer;
  const logicalTensor = (name: string): AceGpuLogicalTensor => {
    const logicalShape = aceDitExpectedLogicalShape(name);
    const elements = logicalShape.reduce((product, extent) => product * extent, 1);
    const tiled = isAceDitGemmWeightTensorName(name);
    const dense = profile === "mixed-layers" &&
      isAceDitRepeatedDenseWeightTensorName(name);
    const storageShape = dense
      ? denseProfile === "opt-0037-k4"
        ? [logicalShape[0]! / 128, logicalShape[1]! / 4, 4, 32, 4]
        : [...logicalShape]
      : [Math.ceil(elements / 2)];
    const transformation = dense
      ? denseProfile === "opt-0037-k4"
        ? ACE_DIT_DENSE_K4_FP16_TRANSFORMATION
        : ACE_DIT_DENSE_FP16_TRANSFORMATION
      : tiled
        ? ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
        : "preserve-bf16-bits-pack-u32-pairs";
    return {
      logicalTensor: name,
      logicalShape,
      parts: [{
        tensorName: name,
        tensor: {
          logicalTensor: name,
          logicalShape,
          storageShape,
          byteLength: elements * 2,
          phase: "dit",
          lifetime: "dit",
          source: `ace-turbo-weights:${name.slice("ace.".length)}`,
          dtype: dense ? "float16" : "uint32-bf16-pairs",
          layout: dense
            ? denseProfile === "opt-0037-k4"
              ? ACE_DIT_DENSE_K4_FP16_LAYOUT
              : ACE_DIT_DENSE_FP16_TILE_LAYOUT
            : tiled
              ? ACE_DIT_GEMM_TILE_LAYOUT
              : "source-row-major-bf16-pairs-lsb-u32",
          transformation,
          partAxis: 0,
          partStart: 0,
          partEnd: logicalShape[0]!,
        },
        binding: { buffer: weightBuffer, offset: 0, size: weightBuffer.size },
      }],
    } as unknown as AceGpuLogicalTensor;
  };
  return Object.freeze({
    value: {
      phases: Object.freeze(["dit"]),
      residentBytes: profile === "reference-shared"
        ? ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES
        : ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
      packageManifest: {
        profile: profile === "reference-shared"
          ? "reference"
          : "fp16-dit-dense-experimental",
      },
      logicalTensor,
      binding: (name: string) => logicalTensor(name).parts[0]!.binding,
      destroy,
    } as unknown as AceGpuTensorPhase,
    destroy,
  });
}

interface FakeBuffer extends GPUBuffer {
  readonly backing: ArrayBuffer;
}

function fakeFactoryDevice(profile: "reference-bf16" | "raw-fp16") {
  const createdBuffers: FakeBuffer[] = [];
  const writeBufferCalls: Array<{ readonly bytes: Uint8Array<ArrayBuffer> }> = [];
  const createdCommandBuffers: GPUCommandBuffer[] = [];
  const scopeDepth = { value: 0 };
  const destroyedBuffers = { value: 0 };
  const submissions: GPUCommandBuffer[][] = [];
  const maximumOutstanding = { value: 0 };
  let outstanding = 0;
  const lost = Promise.withResolvers<GPUDeviceLostInfo>();
  const queue = {
    writeBuffer(
      buffer: GPUBuffer,
      bufferOffset: number,
      data: GPUAllowSharedBufferSource,
    ) {
      const bytes = data instanceof ArrayBuffer || data instanceof SharedArrayBuffer
        ? new Uint8Array(data)
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const owned = Uint8Array.from(bytes);
      new Uint8Array((buffer as FakeBuffer).backing, bufferOffset, owned.length)
        .set(owned);
      writeBufferCalls.push({ bytes: owned });
    },
    submit(commandBuffers: Iterable<GPUCommandBuffer>) {
      const values = [...commandBuffers];
      submissions.push(values);
      outstanding += 1;
      maximumOutstanding.value = Math.max(maximumOutstanding.value, outstanding);
      for (const command of values) {
        (command as GPUCommandBuffer & { readonly execute?: () => void }).execute?.();
      }
    },
    onSubmittedWorkDone: vi.fn(async () => {
      outstanding -= 1;
      return undefined;
    }),
  } as unknown as GPUQueue;
  const device = {
    features: new Set(
      profile === "raw-fp16"
        ? ["shader-f16"]
        : ["shader-f16", "subgroups"],
    ),
    limits: {
      maxBufferSize: 1_000_000_000,
      maxStorageBufferBindingSize: 1_000_000_000,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    queue,
    lost: lost.promise,
    pushErrorScope: vi.fn(() => {
      scopeDepth.value += 1;
    }),
    popErrorScope: vi.fn(async () => {
      scopeDepth.value -= 1;
      return null;
    }),
    createBuffer(descriptor: GPUBufferDescriptor) {
      const backing = new ArrayBuffer(descriptor.size);
      let destroyed = false;
      const buffer = {
        size: descriptor.size,
        label: descriptor.label,
        backing,
        destroy() {
          if (destroyed) return;
          destroyed = true;
          destroyedBuffers.value += 1;
        },
        mapAsync: vi.fn(async () => undefined),
        getMappedRange: vi.fn(() => backing),
        unmap: vi.fn(),
      } as unknown as FakeBuffer;
      createdBuffers.push(buffer);
      return buffer;
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    } as unknown as GPUShaderModule)),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: () => ({} as GPUBindGroupLayout),
    } as unknown as GPUComputePipeline)),
    createBindGroup: vi.fn(() => ({} as GPUBindGroup)),
    createCommandEncoder({ label }: GPUCommandEncoderDescriptor = {}) {
      let copy: (() => void) | undefined;
      return {
        beginComputePass() {
          return {
            setPipeline: vi.fn(),
            setBindGroup: vi.fn(),
            dispatchWorkgroups: vi.fn(),
            end: vi.fn(),
          } as unknown as GPUComputePassEncoder;
        },
        copyBufferToBuffer(
          source: GPUBuffer,
          sourceOffset: number,
          destination: GPUBuffer,
          destinationOffset: number,
          size: number,
        ) {
          copy = () => {
            const bytes = new Uint8Array(
              (source as FakeBuffer).backing,
              sourceOffset,
              size,
            );
            new Uint8Array(
              (destination as FakeBuffer).backing,
              destinationOffset,
              size,
            ).set(bytes);
          };
        },
        finish({ label: commandLabel }: GPUCommandBufferDescriptor = {}) {
          const command = {
            label: commandLabel ?? label,
            ...(copy === undefined ? {} : { execute: copy }),
          } as unknown as GPUCommandBuffer;
          createdCommandBuffers.push(command);
          return command;
        },
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  return {
    device,
    createdBuffers,
    writeBufferCalls,
    createdCommandBuffers,
    scopeDepth,
    destroyedBuffers,
    submissions,
    maximumOutstanding,
  };
}

interface PreparedOptions {
  readonly yieldQueueIdle?: ReturnType<typeof vi.fn<() => Promise<void>>>;
  readonly graphWait?: PromiseWithResolvers<void>;
}

function opt0080EpochPlan(
  schedulingProfile: "depth1-epoch1" | "opt-0080-depth2-epoch4",
  phaseCounts: readonly number[] = [25, 316],
) {
  const maximum = schedulingProfile === "depth1-epoch1" ? 1 : 4;
  const epochs: Array<Readonly<{
    completionEpochIndex: number;
    phaseIndex: number;
    firstCommandBufferIndex: number;
    lastCommandBufferIndex: number;
    commandBufferCount: number;
  }>> = [];
  let first = 0;
  for (const [phaseIndex, count] of phaseCounts.entries()) {
    const end = first + count;
    while (first < end) {
      const commandBufferCount = Math.min(maximum, end - first);
      epochs.push(Object.freeze({
        completionEpochIndex: epochs.length,
        phaseIndex,
        firstCommandBufferIndex: first,
        lastCommandBufferIndex: first + commandBufferCount - 1,
        commandBufferCount,
      }));
      first += commandBufferCount;
    }
  }
  return Object.freeze(epochs);
}

function opt0018PreparedFixture() {
  const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
    batch: 1,
    latentFrames: 4_500,
    conditionTokens: 98,
  });
  const descriptors = createOpt0018DescriptorOracle();
  const descriptorTable: AceDitPhysicalCommandDescriptorTable = Object.freeze({
    descriptors,
    sha256:
      "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
    serializedBytes: 1_869_566,
    memberCount: 6_833,
    preparationMs: 12.5,
  });
  const mappedValues = new Float32Array(288_000);
  mappedValues[0] = 1;
  mappedValues[287_999] = -0.5;
  const mapped = mappedValues.buffer;
  const source = { size: mapped.byteLength } as GPUBuffer;
  const graphResult: AceDitGraphRunResult = Object.freeze({
    commandBuffersSubmitted: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
    queueDrains: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
    cooperativeIdleMs: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT - 1,
    completedEvaluations: 8,
    finalLatent: { buffer: source, offset: 0, size: mapped.byteLength },
  });
  const finalQuantum = createAceDitGraphQuantumPlan().at(-1)!;
  const graph = {
    commandBufferCount: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
    physicalCommandDescriptors: descriptorTable,
    run: vi.fn(async (runOptions) => {
      for (const descriptor of descriptors) {
        runOptions.onCommandBufferDrained?.(
          descriptor,
          (descriptor.physicalIndex % 23 + 1) / 100,
        );
      }
      runOptions.onProgress?.({
        completedQuanta: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        totalQuanta: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        queueDrains: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        cooperativeIdleMs: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT - 1,
        completedEvaluations: 8,
        totalEvaluations: 8,
        quantum: finalQuantum,
        subquantumIndex: 0,
        subquantumCount: 1,
        commandId: finalQuantum.label,
      });
      return graphResult;
    }),
    destroy: vi.fn(async () => undefined),
  };
  const submissions: GPUCommandBuffer[][] = [];
  const readback = {
    size: mapped.byteLength,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  const device = {
    lost: deviceLost.promise,
    queue: {
      submit(commandBuffers: Iterable<GPUCommandBuffer>) {
        submissions.push([...commandBuffers]);
      },
      onSubmittedWorkDone: vi.fn(async () => undefined),
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer: vi.fn(),
        finish: () => ({ label: "readback" }) as GPUCommandBuffer,
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  const destroy = vi.fn();
  const idle = vi.fn(async () => undefined);
  const onCommandProfile = vi.fn(
    (_profile: AceOpt0018DitCommandProfile) => undefined,
  );
  let clock = 0;
  const memory = planAceDitGpuBackendMemory(
    "reference-bf16",
    plan,
    3_150_917_888,
    "mixed-opt-0009",
  );
  const resources: AceDitPreparedGpuResources = {
    device,
    modelProfile: "reference-bf16",
    gemmBackend: "mixed-opt-0009",
    shape: plan,
    graph,
    readback,
    memory,
    yieldQueueIdle: idle,
    now: () => ++clock,
    onCommandProfile,
    destroy,
  };
  return {
    resources,
    graph,
    readback,
    submissions,
    idle,
    destroy,
    deviceLost,
    onCommandProfile,
  };
}

function opt0034PreparedFixture(
  physicalQuantaPerCommandBuffer: 1 | 8 | 16,
) {
  const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
    batch: 1,
    latentFrames: 4_500,
    conditionTokens: 98,
  });
  const descriptors = createOpt0018DescriptorOracle();
  const descriptorTable: AceDitPhysicalCommandDescriptorTable = Object.freeze({
    descriptors,
    sha256:
      "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
    serializedBytes: 1_869_566,
    memberCount: 6_833,
    preparationMs: 12.5,
  });
  const batches = planAceDitPhysicalQuantumBatches(
    ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
    physicalQuantaPerCommandBuffer,
  );
  const mappedValues = new Float32Array(288_000);
  mappedValues[0] = 1;
  mappedValues[287_999] = -0.5;
  const mapped = mappedValues.buffer;
  const source = { size: mapped.byteLength } as GPUBuffer;
  const graphResult: AceDitGraphRunResult = Object.freeze({
    commandBuffersSubmitted: batches.length,
    queueDrains: batches.length,
    cooperativeIdleMs: batches.length - 1,
    completedEvaluations: 8,
    finalLatent: { buffer: source, offset: 0, size: mapped.byteLength },
  });
  const finalQuantum = createAceDitGraphQuantumPlan().at(-1)!;
  const graph = {
    commandBufferCount: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
    physicalCommandDescriptors: descriptorTable,
    run: vi.fn(async (runOptions) => {
      for (const batch of batches) {
        const members = descriptors.slice(
          batch.firstPhysicalIndex,
          batch.lastPhysicalIndexExclusive,
        );
        const descriptor: AceDitPhysicalCommandBatchDescriptor = Object.freeze({
          batchIndex: batch.batchIndex,
          batchCount: batches.length,
          firstPhysicalIndex: batch.firstPhysicalIndex,
          lastPhysicalIndex: batch.lastPhysicalIndexExclusive - 1,
          physicalQuantumCount: batch.physicalQuantumCount,
          primitiveCount: members.reduce(
            (total, member) => total + member.primitiveCount,
            0,
          ),
          scheduledMultiplyAdds: members.reduce(
            (total, member) => total + member.scheduledMultiplyAdds,
            0,
          ),
          commandIds: Object.freeze(members.map((member) => member.commandId)),
        });
        runOptions.onPhysicalBatchDrained?.(
          descriptor,
          (batch.batchIndex % 23 + 1) / 10,
        );
      }
      runOptions.onProgress?.({
        completedQuanta: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        totalQuanta: ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        completedCommandBuffers: batches.length,
        totalCommandBuffers: batches.length,
        queueDrains: batches.length,
        cooperativeIdleMs: batches.length - 1,
        completedEvaluations: 8,
        totalEvaluations: 8,
        quantum: finalQuantum,
        subquantumIndex: 0,
        subquantumCount: 1,
        commandId: finalQuantum.label,
        batchIndex: batches.length - 1,
        batchFirstPhysicalQuantum: batches.at(-1)!.firstPhysicalIndex,
        batchPhysicalQuantumCount: batches.at(-1)!.physicalQuantumCount,
        physicalQuantaPerCommandBuffer,
      });
      return graphResult;
    }),
    destroy: vi.fn(async () => undefined),
  };
  const submissions: GPUCommandBuffer[][] = [];
  const readback = {
    size: mapped.byteLength,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  const device = {
    lost: deviceLost.promise,
    queue: {
      submit(commandBuffers: Iterable<GPUCommandBuffer>) {
        submissions.push([...commandBuffers]);
      },
      onSubmittedWorkDone: vi.fn(async () => undefined),
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer: vi.fn(),
        finish: () => ({ label: "readback" }) as GPUCommandBuffer,
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  const destroy = vi.fn();
  const idle = vi.fn(async () => undefined);
  const onSchedulingProfile = vi.fn(
    (_profile: AceOpt0034DitSchedulingProfile) => undefined,
  );
  let clock = 0;
  const memory = planAceDitGpuBackendMemory(
    "reference-bf16",
    plan,
    3_150_917_888,
    "mixed-opt-0009",
    physicalQuantaPerCommandBuffer,
  );
  const resources: AceDitPreparedGpuResources = {
    device,
    modelProfile: "reference-bf16",
    gemmBackend: "mixed-opt-0009",
    shape: plan,
    graph,
    readback,
    memory,
    physicalQuantaPerCommandBuffer,
    yieldQueueIdle: idle,
    now: () => ++clock,
    onSchedulingProfile,
    destroy,
  };
  return {
    resources,
    graph,
    descriptors,
    readback,
    submissions,
    idle,
    destroy,
    deviceLost,
    onSchedulingProfile,
  };
}

function createOpt0018DescriptorOracle(): readonly AceDitPhysicalCommandDescriptor[] {
  const familyCounts = [
    ["precompute", 1],
    ["cross-cache", 24],
    ...Array.from({ length: 8 }, () => [
      ["timestep", 1],
      ["input", 1],
      ["attention-projections", 24],
      ["self-full", 48],
      ["feed-forward", 72],
      ["mixed", 168],
      ["output", 1],
      ["sampler-dcw", 1],
    ] as const).flat(),
  ] as const satisfies readonly (readonly [AceDitProfileFamily, number])[];
  const families = familyCounts.flatMap(([family, count]) =>
    Array.from({ length: count }, () => family)
  );
  expect(families).toHaveLength(ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT);
  const extraMembers = new Uint16Array(families.length);
  let remainingExtras = 2_934;
  while (remainingExtras > 0) {
    for (
      let index = 1;
      index < families.length - 1 && remainingExtras > 0;
      index += 1
    ) {
      extraMembers[index]! += 1;
      remainingExtras -= 1;
    }
  }
  let physicalIndex = 0;
  let evaluation = -1;
  let indexWithinBucket = 0;
  const descriptors = families.map((family) => {
    if (physicalIndex === 25 || indexWithinBucket === 316) {
      evaluation += 1;
      indexWithinBucket = 0;
    }
    const precompute = physicalIndex < 25;
    const effectiveEvaluation = precompute ? null : evaluation;
    const isFirst = physicalIndex === 0;
    const isLast = physicalIndex === families.length - 1;
    const logicalKind = isFirst
      ? "condition-projection"
      : precompute
      ? "cross-cache"
      : family === "timestep"
      ? "timestep"
      : family === "input"
      ? "input-projection"
      : family === "output"
      ? "output-projection"
      : family === "sampler-dcw"
      ? "sampler"
      : "layer";
    const commandId = isFirst
      ? "ace-dit-condition-projection"
      : isLast
      ? "ace-dit-eval-7-sampler"
      : `opt-0018-command-${physicalIndex}`;
    const minimumMembers = family === "mixed" ? 2 : isLast ? 3 : 1;
    const memberCount = minimumMembers + extraMembers[physicalIndex]!;
    const members = Array.from({ length: memberCount }, (_, memberIndex) => {
      const memberFamily = family === "mixed"
        ? (memberIndex % 2 === 0 ? "attention-projections" : "plumbing")
        : family;
      const id = isFirst
        ? "ace-dit-condition-projection-range-0"
        : isLast
        ? [
            "ace-dit-eval-7-sampler-sampler-update",
            "ace-dit-eval-7-sampler-predicted-clean",
            "ace-dit-eval-7-sampler-dcw",
          ][memberIndex] ?? `${commandId}-member-${memberIndex}`
        : `${commandId}-member-${memberIndex}`;
      return Object.freeze({
        id,
        family: memberFamily as Exclude<AceDitProfileFamily, "mixed">,
        backend: "oracle",
        kernel: "oracle",
        scheduledMultiplyAdds: memberIndex + 1,
      });
    });
    const descriptor = Object.freeze({
      physicalIndex,
      logicalIndex: isFirst ? 0 : isLast ? 248 : Math.min(247, physicalIndex),
      logicalKind,
      commandId,
      subquantumIndex: 0,
      subquantumCount: 1,
      evaluation: effectiveEvaluation,
      layer: logicalKind === "layer" || logicalKind === "cross-cache"
        ? physicalIndex % 24
        : null,
      family,
      primitiveCount: members.length,
      scheduledMultiplyAdds: members.reduce(
        (total, member) => total + member.scheduledMultiplyAdds,
        0,
      ),
      members: Object.freeze(members),
    }) as AceDitPhysicalCommandDescriptor;
    physicalIndex += 1;
    indexWithinBucket += precompute ? 0 : 1;
    return descriptor;
  });
  expect(descriptors.reduce(
    (total, descriptor) => total + descriptor.members.length,
    0,
  )).toBe(6_833);
  return Object.freeze(descriptors);
}

function preparedFixture(
  profile: "reference-bf16" | "raw-fp16",
  mapped: ArrayBuffer,
  options: PreparedOptions = {},
) {
  const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
    batch: 1,
    latentFrames: 1,
    conditionTokens: 1,
  });
  const source = { size: mapped.byteLength } as GPUBuffer;
  const graphResult: AceDitGraphRunResult = Object.freeze({
    commandBuffersSubmitted: memoryCommandCount(profile, plan),
    queueDrains: memoryCommandCount(profile, plan),
    cooperativeIdleMs: memoryCommandCount(profile, plan) - 1,
    completedEvaluations: 8,
    finalLatent: { buffer: source, offset: 0, size: mapped.byteLength },
  });
  const finalQuantum = createAceDitGraphQuantumPlan().at(-1)!;
  const graph = {
    commandBufferCount: memoryCommandCount(profile, plan),
    physicalCommandDescriptors: undefined,
    run: vi.fn(async (runOptions) => {
      if (options.graphWait !== undefined) {
        await Promise.race([
          options.graphWait.promise,
          new Promise<never>((_resolve, reject) => {
            runOptions.signal.addEventListener(
              "abort",
              () => reject(runOptions.signal.reason),
              { once: true },
            );
          }),
        ]);
      }
      runOptions.signal.throwIfAborted();
      runOptions.onProgress?.({
        completedQuanta: memoryCommandCount(profile, plan),
        totalQuanta: memoryCommandCount(profile, plan),
        queueDrains: memoryCommandCount(profile, plan),
        cooperativeIdleMs: memoryCommandCount(profile, plan) - 1,
        completedEvaluations: 8,
        totalEvaluations: 8,
        quantum: finalQuantum,
        subquantumIndex: 0,
        subquantumCount: 1,
        commandId: finalQuantum.label,
      });
      return graphResult;
    }),
    destroy: vi.fn(async () => undefined),
  };
  const submissions: GPUCommandBuffer[][] = [];
  const readback = {
    size: Math.max(256, mapped.byteLength),
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  const device = {
    lost: deviceLost.promise,
    queue: {
      submit(commandBuffers: Iterable<GPUCommandBuffer>) {
        submissions.push([...commandBuffers]);
      },
      onSubmittedWorkDone: vi.fn(async () => undefined),
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer: vi.fn(),
        finish: () => ({ label: "readback" }) as GPUCommandBuffer,
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  const destroy = vi.fn();
  const idle = options.yieldQueueIdle ?? vi.fn(async () => undefined);
  const memory = planAceDitGpuBackendMemory(profile, plan, 1_024);
  const resources: AceDitPreparedGpuResources = {
    device,
    modelProfile: profile,
    gemmBackend: "portable",
    shape: plan,
    graph,
    readback,
    memory,
    yieldQueueIdle: idle,
    destroy,
  };
  return {
    resources,
    graph,
    readback,
    submissions,
    idle,
    destroy,
    deviceLost,
  };
}

function memoryCommandCount(
  profile: "reference-bf16" | "raw-fp16",
  plan: ReturnType<typeof planAceDitEvaluation>,
): number {
  return planAceDitGpuBackendMemory(profile, plan, 1_024).commandBufferCount - 1;
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let turn = 0; turn < 100; turn += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error("test condition did not become true");
}

function paddedFloat32(prefix: readonly number[]): Float32Array<ArrayBuffer> {
  const values = new Float32Array(64);
  values.set(prefix);
  return values;
}

function paddedUint16(prefix: readonly number[]): Uint16Array<ArrayBuffer> {
  const values = new Uint16Array(64);
  values.set(prefix);
  return values;
}
