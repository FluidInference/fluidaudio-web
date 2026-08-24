import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  AceOpt0011Fp16VaeChunkGpuBackend,
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
  resolveOpt0080VaeProductionWindowSchedulingProfile,
  type AceOpt0080VaeSchedulingEvidence,
  type AceOpt0011Fp16VaeProfileFamily,
  type AceOpt0011Fp16VaePreparedResources,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
} from "../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
  ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY,
  planAceOpt0011Fp16VaeChunkDispatches,
  type AceOpt0011Fp16VaeChunkDispatchSet,
  type AceOpt0011Fp16VaeWindowDispatch,
} from
  "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
  type AceVaeRuntimeProfile,
} from "../src/webgpu/vae-fp16-profile.js";
import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import type { AceVaeChunkGpuBackendProgress } from
  "../src/webgpu/vae-backend.js";
import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";

Object.defineProperty(globalThis, "GPUMapMode", {
  configurable: true,
  value: Object.freeze({ READ: 1 }),
});

describe("OPT-0011 production FP16 VAE backend planning", () => {
  it("retains every unpadded C1024 exact shape and accounts its controls", () => {
    const plan = planAceVaeChunkedDecode(1_024, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    const topology = planAceOpt0011Fp16VaeChunkDispatches(1_024, 512, 256);
    const memory = planAceOpt0011Fp16VaeChunkGpuBackendMemory(plan, 256);

    expect(plan.windows.map((window) => window.latentWindowFrames)).toEqual([
      448,
      512,
      320,
    ]);
    expect(topology.uniqueWindowFrames).toEqual([320, 448, 512]);
    expect(memory.controlBufferBytes).toBe(5_039_152);
    expect(memory.controlBufferBytes).toBe(topology.uniqueDynamicControlBytes);
    expect(memory.accountedGpuBytes).toBe(944_730_672);
    expect(memory.workspaceBufferBytes).toBe(251_658_240);
    expect(memory.maximumWindowFrames).toBe(512);
    expect(memory.quantaPerCommandBuffer).toBe(8);
  });

  it("plans a 300-frame direct smoke without padding to 512", () => {
    const plan = planAceVaeChunkedDecode(300, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    const memory = planAceOpt0011Fp16VaeChunkGpuBackendMemory(plan, 256);

    expect(plan.windows).toHaveLength(1);
    expect(plan.windows[0]!.latentWindowFrames).toBe(300);
    expect(plan.maximumWindowFrames).toBe(300);
    expect(memory.maximumWindowFrames).toBe(512);
    expect(memory.accountedGpuBytes).toBeGreaterThan(939_691_520);
  });

  it("rejects the old B256 production chunk geometry", () => {
    const plan = planAceVaeChunkedDecode(1_024, {
      chunkFrames: 256,
      overlapFrames: 64,
    });
    expect(() =>
      planAceOpt0011Fp16VaeChunkGpuBackendMemory(plan, 256)
    ).toThrow(/exact C-512\/64/);
  });

  it("retains the explicit hybrid profile and kernel topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      kernelSetId:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
  });

  it("retains the explicit congruent-transpose production topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      kernelSetId:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
  });

  it("retains the production OPT-0028 exact-packed topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
  });

  it("retains the opt-in OPT-0040 shape-selected topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
  });

  it("retains the explicit OPT-0054 revision-7 mixed-layout topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId: "opt-0054-mixed-fp16-fixed32-revision7-v1",
      kernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
    expect(fixture.runtimeDestroy).toHaveBeenCalledOnce();
    expect(fixture.weightDestroy).toHaveBeenCalledOnce();
  });

  it("retains the selectable OPT-0028 portable exact-packed topology", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
  });

  it("retains the portable OPT-0088 dual-K4 topology without subgroups", async () => {
    const fixture = preparedFixture(
      1,
      undefined,
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    expect(backend).toMatchObject({
      runtimeProfileId: "opt-0088-mixed-fp16-portable-dual-k4-v1",
      kernelSetId:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY.id,
      kernelTopology:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY,
    });
    await backend.destroy();
    expect(fixture.runtimeDestroy).toHaveBeenCalledOnce();
    expect(fixture.weightDestroy).toHaveBeenCalledOnce();
  });

  it("rejects an unknown runtime profile before GPU allocation", async () => {
    const plan = planAceVaeChunkedDecode(10, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    const createBuffer = vi.fn();
    const createShaderModule = vi.fn();
    const weightDestroy = vi.fn();
    await expect(AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: { createBuffer, createShaderModule } as unknown as GPUDevice,
      plan,
      finalLatents: new Float32Array(plan.latentFrames * 64),
      authenticatedPackage: {} as never,
      ownedVaeWeights: { destroy: weightDestroy } as unknown as AceGpuTensorPhase,
      maximumWindowFrames: 512,
      runtimeProfileId: "forged-runtime-profile",
    } as never)).rejects.toThrow(/unknown runtime profile forged-runtime-profile/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createShaderModule).not.toHaveBeenCalled();
    expect(weightDestroy).toHaveBeenCalledOnce();
  });

  it("rejects the public OPT-0072 identity at the physical backend boundary", async () => {
    const plan = planAceVaeChunkedDecode(10, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    const createBuffer = vi.fn();
    const createShaderModule = vi.fn();
    const weightDestroy = vi.fn();
    await expect(AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: { createBuffer, createShaderModule } as unknown as GPUDevice,
      plan,
      finalLatents: new Float32Array(plan.latentFrames * 64),
      authenticatedPackage: {} as never,
      ownedVaeWeights: { destroy: weightDestroy } as unknown as AceGpuTensorPhase,
      maximumWindowFrames: 512,
      runtimeProfileId:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    } as never)).rejects.toThrow(/unknown runtime profile opt-0072/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createShaderModule).not.toHaveBeenCalled();
    expect(weightDestroy).toHaveBeenCalledOnce();
  });

  it("rejects unauthenticated congruent-profile subgroup bounds before allocation", async () => {
    const plan = planAceVaeChunkedDecode(10, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    const createBuffer = vi.fn();
    const createShaderModule = vi.fn();
    const weightDestroy = vi.fn();
    await expect(AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: { createBuffer, createShaderModule } as unknown as GPUDevice,
      plan,
      finalLatents: new Float32Array(plan.latentFrames * 64),
      authenticatedPackage: {} as never,
      ownedVaeWeights: { destroy: weightDestroy } as unknown as AceGpuTensorPhase,
      maximumWindowFrames: 512,
      runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 16,
    } as never)).rejects.toThrow(/authenticated 32\/32 subgroups/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createShaderModule).not.toHaveBeenCalled();
    expect(weightDestroy).toHaveBeenCalledOnce();
  });

  it("rejects a nested dispatch that claims another kernel set", () => {
    const fixture = preparedFixture(1);
    const dispatch = fixture.resources.dispatchSet.dispatches[0]!;
    const forgedDispatch = Object.freeze({
      ...dispatch,
      runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" as const,
    });
    const dispatchSet = Object.freeze({
      ...fixture.resources.dispatchSet,
      dispatches: Object.freeze([forgedDispatch]),
      windows: Object.freeze([{
        window: fixture.plan.windows[0]!,
        dispatch: forgedDispatch,
      }]),
    });
    expect(() => AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      dispatchSet,
    })).toThrow(/diverged from the chunk plan/);
    expect(fixture.runtimeDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.weightDestroy).toHaveBeenCalledTimes(1);
  });

  it("submits batch-8 compute passes then one active-prefix readback", async () => {
    const fixture = preparedFixture(9);
    const original = Float32Array.from(fixture.resources.finalLatents);
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    fixture.resources.finalLatents.fill(99);

    const output = await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(output).toHaveLength(
      fixture.plan.windows[0]!.decodedAudioFrames * 2,
    );
    expect(fixture.uploads).toEqual([original]);
    expect(fixture.submissions.map((record) => record.kind)).toEqual([
      "compute",
      "compute",
      "readback",
    ]);
    expect(fixture.submissions.map((record) => record.dispatches)).toEqual([
      8,
      1,
      0,
    ]);
    expect(fixture.submissions.map((record) => record.computePasses)).toEqual([
      1,
      1,
      0,
    ]);
    expect(fixture.submissions[2]!.copyBytes).toBe(output.byteLength);
    expect(fixture.submissions[2]!.copyBytes).toBeLessThan(
      fixture.resources.buffers.output.size,
    );
    expect(fixture.idle).toHaveBeenCalledTimes(2);

    await backend.destroy();
    await backend.destroy();
    expect(fixture.runtimeDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.weightDestroy).toHaveBeenCalledTimes(1);
    for (const destroy of fixture.bufferDestroys) {
      expect(destroy).toHaveBeenCalledTimes(1);
    }
  });

  it("submits the OPT-0027 batch-64 candidate without changing dispatch order", async () => {
    const fixture = preparedFixture(
      65,
      undefined,
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
      [],
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
    );
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );

    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(fixture.submissions.map((record) => record.kind)).toEqual([
      "compute",
      "compute",
      "readback",
    ]);
    expect(fixture.submissions.map((record) => record.dispatches)).toEqual([
      64,
      1,
      0,
    ]);
    expect(fixture.idle).toHaveBeenCalledTimes(2);
    expect(backend.memory.quantaPerCommandBuffer).toBe(64);

    await backend.destroy();
  });

  it("retains singleton commands while screening depth-one and depth-two epochs", async () => {
    const controlFixture = preparedFixture(33);
    const controlEvidence: AceOpt0080VaeSchedulingEvidence[] = [];
    const controlProgress: AceVaeChunkGpuBackendProgress[] = [];
    const control = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...controlFixture.resources,
      onProgress: (progress) => controlProgress.push(progress),
    });
    await control.decodeWindow(
      controlFixture.plan.windows[0]!,
      undefined,
      {
        schedulingProfile: "depth1-epoch1",
        onSchedulingEvidence: (evidence) => controlEvidence.push(evidence),
      },
    );
    expect(controlFixture.submissions.map(({ kind }) => kind)).toEqual([
      "compute", "compute", "compute", "compute", "compute", "readback",
    ]);
    expect(controlFixture.idle).toHaveBeenCalledTimes(5);
    expect(controlEvidence).toHaveLength(1);
    expect(controlEvidence[0]).toMatchObject({
      schema: "ace-opt-0080-vae-window-scheduling-v1",
      windowIndex: 0,
      schedulingProfile: "depth1-epoch1",
      decoderQuantumCount: 33,
      quantaPerCommandBuffer: 8,
      decoderCommandBufferCount: 5,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 6,
      commandBuffersSubmitted: 6,
      completionFenceRequestedCount: 6,
      completionFenceSettledCount: 6,
      completionFenceRejectedCount: 0,
      trueQueueDrainCount: 6,
      completionEpochCount: 6,
      cooperativeIdleTurns: 5,
      requestedCooperativeIdleMs: 5,
      maximumOutstandingCommandBuffers: 1,
    });
    expect(controlEvidence[0]!.schedulingWallMs).toBeGreaterThanOrEqual(0);
    expect(controlEvidence[0]!.commandCompletions).toHaveLength(6);
    expect(controlEvidence[0]!.commandCompletions.every(
      ({ trueQueueDrain }) => trueQueueDrain,
    )).toBe(true);
    expect(controlEvidence[0]!.completionEpochs.map((epoch) => [
      epoch.firstCommandBufferIndex,
      epoch.lastCommandBufferIndex,
    ])).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]);
    expect(controlProgress.at(-1)).toMatchObject({
      completedCommandBuffers: 6,
      queueDrains: 6,
      cooperativeIdleMs: 5,
      stage: "readback",
    });
    await control.destroy();

    const candidateFixture = preparedFixture(33);
    const candidateEvidence: AceOpt0080VaeSchedulingEvidence[] = [];
    const candidateProgress: AceVaeChunkGpuBackendProgress[] = [];
    const candidate = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...candidateFixture.resources,
      onProgress: (progress) => candidateProgress.push(progress),
    });
    await candidate.decodeWindow(
      candidateFixture.plan.windows[0]!,
      undefined,
      {
        schedulingProfile: "depth2-phase-epoch4",
        onSchedulingEvidence: (evidence) => candidateEvidence.push(evidence),
      },
    );
    expect(candidateFixture.submissions.map(({ kind }) => kind)).toEqual(
      controlFixture.submissions.map(({ kind }) => kind),
    );
    expect(candidateFixture.idle).toHaveBeenCalledTimes(1);
    expect(candidateEvidence).toHaveLength(1);
    expect(candidateEvidence[0]).toMatchObject({
      schedulingProfile: "depth2-phase-epoch4",
      decoderCommandBufferCount: 5,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 6,
      commandBuffersSubmitted: 6,
      completionFenceRequestedCount: 6,
      completionFenceSettledCount: 6,
      completionFenceRejectedCount: 0,
      trueQueueDrainCount: 2,
      completionEpochCount: 2,
      cooperativeIdleTurns: 1,
      requestedCooperativeIdleMs: 1,
      maximumOutstandingCommandBuffers: 2,
    });
    expect(candidateEvidence[0]!.schedulingWallMs).toBeGreaterThanOrEqual(0);
    expect(candidateEvidence[0]!.commandCompletions.map((completion) => [
      completion.commandBufferIndex,
      completion.commandKind,
      completion.trueQueueDrain,
      completion.completionEpochIndex,
    ])).toEqual([
      [0, "decoder", false, 0],
      [1, "decoder", false, 0],
      [2, "decoder", false, 0],
      [3, "decoder", true, 0],
      [4, "decoder", false, 1],
      [5, "readback", true, 1],
    ]);
    expect(candidateEvidence[0]!.completionEpochs.map((epoch) => [
      epoch.firstCommandBufferIndex,
      epoch.lastCommandBufferIndex,
      epoch.commandBufferCount,
    ])).toEqual([[0, 3, 4], [4, 5, 2]]);
    expect(candidateProgress.at(-1)).toMatchObject({
      completedCommandBuffers: 6,
      queueDrains: 2,
      cooperativeIdleMs: 1,
      stage: "readback",
    });
    await candidate.destroy();
  });

  it("snapshots a production policy and permits an explicit per-window control", async () => {
    const fixture = preparedFixture(33);
    const progress: AceVaeChunkGpuBackendProgress[] = [];
    const prepared = {
      ...fixture.resources,
      submissionPolicy: "depth2-phase-epoch4" as
        "depth1-epoch1" | "depth2-phase-epoch4",
      onProgress: (event: AceVaeChunkGpuBackendProgress) =>
        progress.push(event),
    };
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      prepared,
    );
    prepared.submissionPolicy = "depth1-epoch1";

    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(fixture.idle).toHaveBeenCalledTimes(1);
    expect(progress.at(-1)).toMatchObject({
      queueDrains: 2,
      cooperativeIdleMs: 1,
    });
    await backend.destroy();

    const overrideFixture = preparedFixture(33);
    const evidence: AceOpt0080VaeSchedulingEvidence[] = [];
    const override = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...overrideFixture.resources,
      submissionPolicy: "depth2-phase-epoch4",
    });
    await override.decodeWindow(
      overrideFixture.plan.windows[0]!,
      undefined,
      {
        schedulingProfile: "depth1-epoch1",
        onSchedulingEvidence: (value) => evidence.push(value),
      },
    );
    expect(overrideFixture.idle).toHaveBeenCalledTimes(5);
    expect(evidence[0]?.schedulingProfile).toBe("depth1-epoch1");
    await override.destroy();
  });

  it("resolves the production tag only for the exact C2314 window", () => {
    expect(resolveOpt0080VaeProductionWindowSchedulingProfile(
      ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
      2_314,
    )).toBe("depth2-phase-epoch4");
    for (const frames of [214, 512, 2_313, 2_315, 2_378]) {
      expect(resolveOpt0080VaeProductionWindowSchedulingProfile(
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
        frames,
      )).toBe("depth1-epoch1");
    }
    expect(resolveOpt0080VaeProductionWindowSchedulingProfile(
      undefined,
      2_314,
    )).toBe("depth1-epoch1");
  });

  it("runs the exact rev7 OPT-0066 C2314 production tag as 555+1 singletons", async () => {
    const fixture = preparedOpt0080ProductionFixture();
    const progress: AceVaeChunkGpuBackendProgress[] = [];
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      productionSchedulingPolicy:
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
      onProgress: (event) => progress.push(event),
    });

    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(fixture.submissions).toHaveLength(556);
    expect(fixture.submissions.slice(0, -1).every(
      ({ kind, dispatches }) => kind === "compute" && dispatches > 0,
    )).toBe(true);
    expect(fixture.submissions.at(-1)).toMatchObject({
      kind: "readback",
      dispatches: 0,
    });
    expect(progress.at(-1)).toMatchObject({
      completedDecoderQuanta: 35_498,
      totalDecoderQuanta: 35_498,
      completedCommandBuffers: 556,
      totalCommandBuffers: 556,
      queueDrains: 139,
      cooperativeIdleMs: 138,
      stage: "readback",
    });
    await backend.destroy();
  });

  it.each([
    ["global submission policy", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({ ...resources, submissionPolicy: "depth1-epoch1" as const })],
    ["physical runtime profile", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({
      ...resources,
      runtimeProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
    })],
    ["converter revision", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({
      ...resources,
      runtimeProfile: {
        ...resources.runtimeProfile,
        packageConverterRevision: 6,
      } as AceVaeRuntimeProfile,
    })],
    ["maximum window", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({ ...resources, maximumWindowFrames: 512 as const })],
    ["quanta per command buffer", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({ ...resources, quantaPerCommandBuffer: 8 })],
    ["family profiler", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({ ...resources, onFamilyProfile: vi.fn() })],
    ["idle test seam", (
      resources: AceOpt0011Fp16VaePreparedResources,
    ) => ({ ...resources, yieldQueueIdle: vi.fn(async () => undefined) })],
  ] as const)("rejects production scheduling with a changed %s", (
    _label,
    mutate,
  ) => {
    const fixture = preparedOpt0080ProductionFixture();
    expect(() => AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...mutate(fixture.resources),
      productionSchedulingPolicy:
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
    })).toThrow(/requires exact revision-7 OPT-0066 C2378\/qpc64/u);
    expect(fixture.runtimeDestroy).toHaveBeenCalledOnce();
    expect(fixture.weightDestroy).toHaveBeenCalledOnce();
  });

  it("rejects a benchmark callback on a production-selected backend", async () => {
    const fixture = preparedOpt0080ProductionFixture();
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      productionSchedulingPolicy:
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
    });

    await expect(backend.decodeWindow(
      fixture.plan.windows[0]!,
      undefined,
      {
        schedulingProfile: "depth1-epoch1",
        onSchedulingEvidence: vi.fn(),
      },
    )).rejects.toThrow(/cannot use a benchmark callback/u);
    expect(fixture.uploads).toHaveLength(0);
    expect(fixture.submissions).toHaveLength(0);
    await backend.destroy();
  });

  it("settles the one prefetched successor before candidate cancellation unwinds", async () => {
    const fixture = preparedFixture(33);
    const controller = new AbortController();
    const reason = Object.freeze({ kind: "opt-0080-vae-test-cancellation" });
    const evidence: AceOpt0080VaeSchedulingEvidence[] = [];
    let progressCount = 0;
    let submissionsAtAbort = 0;
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      onProgress: () => {
        progressCount += 1;
        submissionsAtAbort = fixture.submissions.length;
        controller.abort(reason);
      },
    });

    await expect(backend.decodeWindow(
      fixture.plan.windows[0]!,
      controller.signal,
      {
        schedulingProfile: "depth2-phase-epoch4",
        onSchedulingEvidence: (value) => evidence.push(value),
      },
    )).rejects.toBe(reason);
    expect(submissionsAtAbort).toBe(2);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.submissions.every(({ kind }) => kind === "compute")).toBe(true);
    expect(progressCount).toBe(1);
    expect(fixture.idle).not.toHaveBeenCalled();
    expect(evidence).toHaveLength(0);

    await backend.destroy(reason);
    expect(fixture.runtimeDestroy).toHaveBeenCalledOnce();
    expect(fixture.weightDestroy).toHaveBeenCalledOnce();
  });

  it("refuses overlapping depth-two fence times as family attribution", async () => {
    const fixture = preparedFixture(17);
    const profiles: AceOpt0011Fp16VaeWindowFamilyProfile[] = [];
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      onFamilyProfile: (profile) => profiles.push(profile),
    });
    await expect(backend.decodeWindow(
      fixture.plan.windows[0]!,
      undefined,
      {
        schedulingProfile: "depth2-phase-epoch4",
        onSchedulingEvidence: () => undefined,
      },
    )).rejects.toThrow(/cannot attribute overlapping/);
    expect(fixture.uploads).toHaveLength(0);
    expect(fixture.submissions).toHaveLength(0);
    expect(profiles).toHaveLength(0);

    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(profiles).toHaveLength(1);
    await backend.destroy();
  });

  it("aggregates only homogeneous production batches by kernel family", async () => {
    const quantumFamilies = [
      ...Array<AceOpt0011Fp16VaeProfileFamily>(8).fill("k7-conv1d"),
      ...Array<AceOpt0011Fp16VaeProfileFamily>(8).fill("k1-conv1d"),
      ...Array<AceOpt0011Fp16VaeProfileFamily>(8).fill("conv-transpose1d"),
      ...Array<AceOpt0011Fp16VaeProfileFamily>(8).fill("snake"),
      ...Array<AceOpt0011Fp16VaeProfileFamily>(8).fill("add"),
      ...Array.from(
        { length: 8 },
        (_, index): AceOpt0011Fp16VaeProfileFamily =>
          index % 2 === 0 ? "k7-conv1d" : "snake",
      ),
    ];
    const fixture = preparedFixture(
      quantumFamilies.length,
      undefined,
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE,
      quantumFamilies,
    );
    const profiles: AceOpt0011Fp16VaeWindowFamilyProfile[] = [];
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      ...fixture.resources,
      onFamilyProfile: (profile) => profiles.push(profile),
    });

    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(fixture.submissions).toHaveLength(7);
    expect(profiles).toHaveLength(1);
    const profile = profiles[0]!;
    expect(profile).toMatchObject({
      windowIndex: 0,
      inputFrames: 10,
      quantaPerCommandBuffer: 8,
      decoderBatchCount: 6,
      decoderQuantumCount: 48,
      homogeneousBatchCount: 5,
      homogeneousQuantumCount: 40,
      mixedBatchCount: 1,
      mixedQuantumCount: 8,
    });
    for (const family of [
      "k7-conv1d",
      "k1-conv1d",
      "conv-transpose1d",
      "snake",
      "add",
    ] as const) {
      expect(profile.families[family]).toMatchObject({
        batchCount: 1,
        quantumCount: 8,
      });
      expect(profile.families[family].submitThroughDrainMs)
        .toBeGreaterThanOrEqual(0);
    }
    const attributedMs = Object.values(profile.families).reduce(
      (total, family) => total + family.submitThroughDrainMs,
      profile.mixedSubmitThroughDrainMs,
    );
    expect(profile.decoderSubmitThroughDrainMs).toBeCloseTo(attributedMs, 10);

    await backend.destroy();
  });

  it("drains a submitted batch before cancellation releases ownership", async () => {
    const controller = new AbortController();
    const fixture = preparedFixture(17, controller);
    const backend = AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );

    await expect(
      backend.decodeWindow(fixture.plan.windows[0]!),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.maximumOutstanding()).toBe(1);
    await backend.destroy();
    expect(fixture.runtimeDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.weightDestroy).toHaveBeenCalledTimes(1);
  });
});

interface CommandRecord {
  kind: "compute" | "readback";
  computePasses: number;
  dispatches: number;
  copyBytes: number;
}

function kernelTopologyFor(runtimeProfile: AceVaeRuntimeProfile) {
  if (
    runtimeProfile.id === "opt-0028-mixed-fp16-portable-exact-packed-v1"
  ) {
    return ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY;
  }
  if (runtimeProfile.id === "opt-0028-mixed-fp16-fixed32-exact-packed-v1") {
    return ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
  ) {
    return ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY;
  }
  if (runtimeProfile.id === "opt-0054-mixed-fp16-fixed32-revision7-v1") {
    return ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
  ) {
    return ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY;
  }
  if (runtimeProfile.id === "opt-0088-mixed-fp16-portable-dual-k4-v1") {
    return ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
  ) {
    return ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY;
  }
  return runtimeProfile.id ===
      "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
    ? ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY
    : ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY;
}

function precisionMapFor(runtimeProfile: AceVaeRuntimeProfile) {
  if (
    runtimeProfile.id === "opt-0028-mixed-fp16-portable-exact-packed-v1"
  ) {
    return ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP;
  }
  if (runtimeProfile.id === "opt-0028-mixed-fp16-fixed32-exact-packed-v1") {
    return ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP;
  }
  if (
    runtimeProfile.id ===
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
  ) {
    return ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP;
  }
  if (runtimeProfile.id === "opt-0054-mixed-fp16-fixed32-revision7-v1") {
    return ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP;
  }
  if (
    runtimeProfile.id ===
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
  ) {
    return ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP;
  }
  if (runtimeProfile.id === "opt-0088-mixed-fp16-portable-dual-k4-v1") {
    return ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP;
  }
  if (
    runtimeProfile.id ===
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
  ) {
    return ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP;
  }
  return runtimeProfile.id ===
      "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
    ? ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP
    : ACE_OPT_0011_VAE_FP16_PRECISION_MAP;
}

function preparedFixture(
  quantumCount: number,
  cancelAfterFirstDrain?: AbortController,
  runtimeProfile: AceVaeRuntimeProfile =
    ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
  quantumFamilies: readonly AceOpt0011Fp16VaeProfileFamily[] = [],
  quantaPerCommandBuffer?: number,
  options: Readonly<{
    latentFrames?: number;
    maximumWindowFrames?: 512 | 2_378;
    omitIdleOverride?: true;
  }> = {},
) {
  const maximumWindowFrames = options.maximumWindowFrames ?? 512;
  const plan = planAceVaeChunkedDecode(options.latentFrames ?? 10, {
    chunkFrames: maximumWindowFrames,
    overlapFrames: 64,
  });
  const maximumDecoderPlan = planAceVaeDecoder(maximumWindowFrames);
  const uploads: Float32Array[] = [];
  const submissions: CommandRecord[] = [];
  const bufferDestroys: ReturnType<typeof vi.fn>[] = [];
  let outstanding = 0;
  let maximumOutstanding = 0;
  let drains = 0;
  const queue = {
    writeBuffer(
      _buffer: GPUBuffer,
      _offset: number,
      source: GPUAllowSharedBufferSource,
    ) {
      const view = ArrayBuffer.isView(source)
        ? new Float32Array(
            source.buffer,
            source.byteOffset,
            source.byteLength / 4,
          )
        : new Float32Array(source);
      uploads.push(Float32Array.from(view));
    },
    submit(commandBuffers: Iterable<GPUCommandBuffer>) {
      const batch = [...commandBuffers] as unknown as CommandRecord[];
      expect(batch).toHaveLength(1);
      submissions.push(batch[0]!);
      outstanding += 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
    },
    async onSubmittedWorkDone() {
      drains += 1;
      outstanding -= 1;
      if (drains === 1) cancelAfterFirstDrain?.abort();
    },
  };
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    lost: new Promise<never>(() => undefined),
    queue,
    createCommandEncoder() {
      const record: CommandRecord = {
        kind: "compute",
        computePasses: 0,
        dispatches: 0,
        copyBytes: 0,
      };
      return {
        beginComputePass() {
          record.computePasses += 1;
          return {
            noteDispatch() {
              record.dispatches += 1;
            },
            end() {},
          };
        },
        copyBufferToBuffer(
          _source: GPUBuffer,
          _sourceOffset: number,
          _destination: GPUBuffer,
          _destinationOffset: number,
          bytes: number,
        ) {
          record.kind = "readback";
          record.copyBytes = bytes;
        },
        finish() {
          return record;
        },
      };
    },
  } as unknown as GPUDevice;
  const makeBuffer = (size: number, readable = false): GPUBuffer => {
    const destroy = vi.fn();
    bufferDestroys.push(destroy);
    return {
      size,
      destroy,
      ...(readable
        ? {
            mapAsync: vi.fn(async () => undefined),
            getMappedRange: vi.fn(
              (_offset: number, bytes: number) => new ArrayBuffer(bytes),
            ),
            unmap: vi.fn(),
          }
        : {}),
    } as unknown as GPUBuffer;
  };
  const buffers = {
    stagingInput: makeBuffer(maximumDecoderPlan.inputElements * 4),
    decoderInput: makeBuffer(maximumDecoderPlan.inputElements * 2),
    workspaces: [
      makeBuffer(maximumDecoderPlan.maximumActivationElements * 2),
      makeBuffer(maximumDecoderPlan.maximumActivationElements * 2),
      makeBuffer(maximumDecoderPlan.maximumActivationElements * 2),
    ],
    output: makeBuffer(maximumDecoderPlan.outputElements * 4),
    readback: makeBuffer(maximumDecoderPlan.outputElements * 4, true),
  } as AceOpt0011Fp16VaePreparedResources["buffers"];
  if (quantumFamilies.length !== 0 && quantumFamilies.length !== quantumCount) {
    throw new Error("profile fixture family count does not match its quanta");
  }
  const operationIndices = Object.fromEntries(
    [
      "k7-conv1d",
      "k1-conv1d",
      "conv-transpose1d",
      "snake",
      "add",
    ].map((family) => [
      family,
      operationIndexForFamily(
        plan.decoderWorkspacePlan.operations,
        family as AceOpt0011Fp16VaeProfileFamily,
      ),
    ]),
  ) as Record<AceOpt0011Fp16VaeProfileFamily, number>;
  const quanta = Object.freeze(Array.from(
    { length: quantumCount },
    (_, index) => {
      const family = quantumFamilies[index];
      const operationIndex = family === undefined
        ? undefined
        : operationIndices[family];
      const operation = operationIndex === undefined
        ? undefined
        : plan.decoderWorkspacePlan.operations[operationIndex];
      return {
        ...(operationIndex === undefined
          ? {}
          : {
              operationIndex,
              operationKind: operation!.kind,
            }),
        encode(pass: GPUComputePassEncoder) {
          (pass as unknown as { noteDispatch(): void }).noteDispatch();
        },
      };
    },
  ));
  const dispatch = {
    runtimeProfileId: runtimeProfile.id,
    kernelSetId: runtimeProfile.kernelSetId,
    kernelTopology: kernelTopologyFor(runtimeProfile),
    precisionMap: precisionMapFor(runtimeProfile),
    plan: plan.decoderWorkspacePlan,
    quanta,
    decoderCommandBufferCountAtBatch8: Math.ceil(quantumCount / 8),
    activeStagingInputBytes: plan.latentFrames * 64 * 4,
    activeOutputBytes: plan.maximumDecodedFloat32Bytes,
  } as unknown as AceOpt0011Fp16VaeWindowDispatch;
  const dispatchSet = {
    runtimeProfileId: runtimeProfile.id,
    kernelSetId: runtimeProfile.kernelSetId,
    kernelTopology: kernelTopologyFor(runtimeProfile),
    topology: { chunkPlan: plan },
    dispatches: Object.freeze([dispatch]),
    windows: Object.freeze([{ window: plan.windows[0]!, dispatch }]),
  } as unknown as AceOpt0011Fp16VaeChunkDispatchSet;
  const runtimeDestroy = vi.fn();
  const weightDestroy = vi.fn();
  const idle = vi.fn(async () => undefined);
  const resources: AceOpt0011Fp16VaePreparedResources = {
    device,
    plan,
    runtimeProfile,
    finalLatents: new Float32Array(plan.latentFrames * 64).fill(0.25),
    ownedVaeWeights: { destroy: weightDestroy } as unknown as AceGpuTensorPhase,
    buffers,
    runtime: { destroy: runtimeDestroy },
    dispatchSet,
    ...(maximumWindowFrames === 512
      ? {}
      : { maximumWindowFrames }),
    ...(cancelAfterFirstDrain === undefined
      ? {}
      : { signal: cancelAfterFirstDrain.signal }),
    ...(options.omitIdleOverride === true ? {} : { yieldQueueIdle: idle }),
    ...(quantaPerCommandBuffer === undefined
      ? {}
      : { quantaPerCommandBuffer }),
  };
  return {
    plan,
    resources,
    uploads,
    submissions,
    bufferDestroys,
    runtimeDestroy,
    weightDestroy,
    idle,
    maximumOutstanding: () => maximumOutstanding,
  };
}

function preparedOpt0080ProductionFixture() {
  return preparedFixture(
    35_498,
    undefined,
    ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
    [],
    64,
    {
      latentFrames: 2_314,
      maximumWindowFrames: 2_378,
      omitIdleOverride: true,
    },
  );
}

function operationIndexForFamily(
  operations: AceOpt0011Fp16VaeWindowDispatch["plan"]["operations"],
  family: AceOpt0011Fp16VaeProfileFamily,
): number {
  const index = operations.findIndex((operation) => {
    switch (family) {
      case "k7-conv1d":
        return operation.kind === "conv1d" && operation.shape.kernelSize === 7;
      case "k1-conv1d":
        return operation.kind === "conv1d" && operation.shape.kernelSize === 1;
      case "conv-transpose1d":
        return operation.kind === "conv-transpose1d";
      case "snake":
        return operation.kind === "snake";
      case "add":
        return operation.kind === "add";
    }
  });
  if (index < 0) throw new Error(`missing ${family} fixture operation`);
  return index;
}
