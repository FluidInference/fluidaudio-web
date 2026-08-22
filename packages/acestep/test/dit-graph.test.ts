import { describe, expect, it } from "vitest";

import {
  ACE_DIRECT_DCW_CONFIGURATION,
  ACE_TURBO_V1_CORRECTNESS_PROFILE,
} from "../src/api.js";
import type {
  AceDitCrossCacheScratch,
  AceDitLayerScratch,
  AceDitOutputScratch,
  AceDitTimestepScratch,
} from "../src/webgpu/ace-dit.js";
import type { AceDitResolvedPackageWeights } from "../src/webgpu/ace-dit-package.js";
import {
  ACE_DIT_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
  AceDitGraphOwner,
  classifyAceOpt0018DitCommandMember,
  createAceDitGraphControlData,
  createAceDitGraphQuantumPlan,
  planAceDitPhysicalQuantumBatches,
  planAceDitGraphMemory,
  type AceDitGraphBindings,
  type AceDitGraphModel,
  type AceDitGraphQuantum,
  type AceDitPhysicalCommandDescriptorTable,
} from "../src/webgpu/dit-graph.js";

const SHAPE = Object.freeze({
  batch: 1,
  latentFrames: 2,
  conditionTokens: 1,
});

const LAYER_SCRATCH_KEYS = Object.freeze([
  "modulation",
  "selfNormalized",
  "selfModulated",
  "selfQueryFlat",
  "selfKeyFlat",
  "selfValueFlat",
  "selfQueryHeads",
  "selfKeyHeads",
  "selfValueHeads",
  "selfNormalizedQueryHeads",
  "selfNormalizedKeyHeads",
  "selfRotatedQueryHeads",
  "selfRotatedKeyHeads",
  "selfAttentionHeads",
  "selfMergedAttention",
  "selfProjectedAttention",
  "afterSelfAttention",
  "crossNormalized",
  "crossQueryFlat",
  "crossQueryHeads",
  "crossNormalizedQueryHeads",
  "crossAttentionHeads",
  "crossMergedAttention",
  "crossProjectedAttention",
  "afterCrossAttention",
  "mlpNormalized",
  "mlpModulated",
  "gate",
  "up",
  "gatedActivation",
  "projectedMlp",
] as const satisfies readonly (keyof AceDitLayerScratch)[]);

const TIMESTEP_SCRATCH_KEYS = Object.freeze([
  "timestepFrequency",
  "relativeFrequency",
  "timestepLinear1",
  "relativeLinear1",
  "timestepActivation1",
  "relativeActivation1",
  "timestepEmbedding",
  "relativeEmbedding",
  "timestepActivation2",
  "relativeActivation2",
  "timestepProjection",
  "relativeProjection",
] as const satisfies readonly (keyof AceDitTimestepScratch)[]);

interface FakeRuntimeRecord {
  readonly method: string;
  readonly label: string;
  readonly arguments: readonly unknown[];
}

describe("ACE DiT production graph planning", () => {
  it("plans exact OPT-0034 batch1/batch8/batch16 physical accounting", () => {
    const cases = [
      [1, 2_553, 1],
      [8, 320, 1],
      [16, 160, 9],
    ] as const;
    for (const [physicalBatch, expectedBatches, expectedTail] of cases) {
      const batches = planAceDitPhysicalQuantumBatches(
        ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        physicalBatch,
      );
      expect(batches).toHaveLength(expectedBatches);
      expect(batches[0]).toMatchObject({
        batchIndex: 0,
        firstPhysicalIndex: 0,
        physicalQuantumCount: physicalBatch,
      });
      expect(batches.at(-1)).toMatchObject({
        batchIndex: expectedBatches - 1,
        lastPhysicalIndexExclusive:
          ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        physicalQuantumCount: expectedTail,
      });
      expect(batches.reduce(
        (total, batch) => total + batch.physicalQuantumCount,
        0,
      )).toBe(ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT);
      expect(batches.every((batch, index) =>
        index === 0 ||
        batch.firstPhysicalIndex ===
          batches[index - 1]!.lastPhysicalIndexExclusive
      )).toBe(true);
    }
    expect(() => planAceDitPhysicalQuantumBatches(2_553, 4 as 1)).toThrow(
      /must be 1, 8, or 16/,
    );
  });

  it("freezes one setup pass and exactly 24 layers across eight evaluations", () => {
    const quanta = createAceDitGraphQuantumPlan();
    expect(ACE_DIT_GRAPH_QUANTUM_COUNT).toBe(249);
    expect(quanta).toHaveLength(249);
    expect(quanta.map((quantum) => quantum.index)).toEqual(
      Array.from({ length: 249 }, (_, index) => index),
    );
    expect(quanta.filter((quantum) => quantum.kind === "condition-projection"))
      .toHaveLength(1);
    expect(quanta.filter((quantum) => quantum.kind === "cross-cache"))
      .toHaveLength(24);
    expect(quanta.filter((quantum) => quantum.kind === "timestep"))
      .toHaveLength(8);
    expect(quanta.filter((quantum) => quantum.kind === "input-projection"))
      .toHaveLength(8);
    expect(quanta.filter((quantum) => quantum.kind === "layer"))
      .toHaveLength(24 * 8);
    expect(quanta.filter((quantum) => quantum.kind === "output-projection"))
      .toHaveLength(8);
    expect(quanta.filter((quantum) => quantum.kind === "sampler"))
      .toHaveLength(8);

    for (let evaluation = 0; evaluation < 8; evaluation += 1) {
      const base = 25 + evaluation * 28;
      expect(quanta.slice(base, base + 28).map((quantum) => quantum.kind)).toEqual([
        "timestep",
        "input-projection",
        ...Array.from({ length: 24 }, () => "layer"),
        "output-projection",
        "sampler",
      ]);
    }
  });

  it("materializes immutable controls from the effective BF16 schedule", () => {
    const controls = createAceDitGraphControlData(
      { batch: 2, latentFrames: 5, conditionTokens: 3 },
      ACE_DIRECT_DCW_CONFIGURATION,
    );
    expect(controls.selfValidLengths).toEqual(new Uint32Array([3, 3, 3, 3]));
    expect(controls.crossValidLengths).toEqual(new Uint32Array([3, 3, 3, 3]));
    expect(controls.cosine).toHaveLength(3 * 128);
    expect(controls.sine).toHaveLength(3 * 128);
    expect(controls.timesteps).toHaveLength(8);
    expect(controls.timesteps.map((values) => values[0])).toEqual(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16,
    );
    expect(controls.timesteps.every((values) => values.length === 2)).toBe(true);
    expect(controls.relativeTimestepZero).toEqual(new Float32Array(2));
  });

  it("accounts for the retained K/V and reusable activation high-water mark", () => {
    const reference = planAceDitGraphMemory("reference-bf16", {
      batch: 1,
      latentFrames: 6_000,
      conditionTokens: 256,
    });
    const fp16 = planAceDitGraphMemory("raw-fp16", {
      batch: 1,
      latentFrames: 6_000,
      conditionTokens: 256,
    });
    expect(reference.activationElementBytes).toBe(4);
    expect(fp16.activationElementBytes).toBe(2);
    expect(reference.crossKeyValueBytes).toBe(2 * fp16.crossKeyValueBytes);
    expect(reference.layerScratchBytes).toBe(2 * fp16.layerScratchBytes);
    expect(reference.minimumGraphBytesExcludingWeights).toBeLessThan(
      reference.unaliasedGraphBytesExcludingWeights,
    );
    expect(reference.minimumGraphBytesExcludingWeights).toBeGreaterThan(
      reference.crossKeyValueBytes,
    );
    expect(reference.relativeZeroPolicy).toBe(
      "immutable-shared-input-recomputed-each-evaluation",
    );
    expect(reference.largestRequiredBindingBytes).toBeLessThanOrEqual(
      reference.minimumGraphBytesExcludingWeights,
    );
  });

  it("freezes exhaustive OPT-0018 family and physical-kernel attribution", () => {
    const condition = {
      index: 0,
      kind: "condition-projection",
      label: "ace-dit-condition-projection",
    } as const satisfies AceDitGraphQuantum;
    const crossCache = {
      index: 1,
      kind: "cross-cache",
      layer: 0,
      label: "ace-dit-cross-cache-0",
    } as const satisfies AceDitGraphQuantum;
    const timestep = {
      index: 25,
      kind: "timestep",
      evaluation: 0,
      label: "ace-dit-eval-0-timestep",
    } as const satisfies AceDitGraphQuantum;
    const input = {
      index: 26,
      kind: "input-projection",
      evaluation: 0,
      label: "ace-dit-eval-0-input-projection",
    } as const satisfies AceDitGraphQuantum;
    const layer = {
      index: 27,
      kind: "layer",
      evaluation: 0,
      layer: 0,
      label: "ace-dit-eval-0-layer-0",
    } as const satisfies AceDitGraphQuantum;
    const output = {
      index: 51,
      kind: "output-projection",
      evaluation: 0,
      label: "ace-dit-eval-0-output-projection",
    } as const satisfies AceDitGraphQuantum;
    const sampler = {
      index: 52,
      kind: "sampler",
      evaluation: 0,
      label: "ace-dit-eval-0-sampler",
    } as const satisfies AceDitGraphQuantum;
    const cases: readonly Readonly<{
      quantum: AceDitGraphQuantum;
      suffix: string;
      family: string;
      backend: string;
      kernel: string;
    }>[] = [
      {
        quantum: condition,
        suffix: "",
        family: "precompute",
        backend: "fixed32-subgroups",
        kernel: "fixed32-subgroup-gemm-n128-k32-v1",
      },
      ...[
        "key-projection",
        "value-projection",
      ].map((suffix) => ({
        quantum: crossCache,
        suffix,
        family: "cross-cache",
        backend: "fixed32-subgroups",
        kernel: "fixed32-subgroup-gemm-n128-k32-v1",
      })),
      ...[
        "split-key-heads",
        "split-value-heads",
      ].map((suffix) => ({
        quantum: crossCache,
        suffix,
        family: "cross-cache",
        backend: "reference-bf16",
        kernel: "reference-bf16-transformer-plumbing",
      })),
      {
        quantum: crossCache,
        suffix: "key-norm",
        family: "cross-cache",
        backend: "reference-bf16",
        kernel: "reference-bf16-rmsnorm",
      },
      ...["timestep-frequency", "relative-frequency"].map((suffix) => ({
        quantum: timestep,
        suffix,
        family: "timestep",
        backend: "reference-bf16",
        kernel: "reference-bf16-dit-plumbing",
      })),
      ...[
        "timestep-linear-1", "relative-linear-1",
        "timestep-linear-2", "relative-linear-2",
        "timestep-projection", "relative-projection",
      ].map((suffix) => ({
        quantum: timestep,
        suffix,
        family: "timestep",
        backend: "fixed32-subgroups",
        kernel: "fixed32-subgroup-gemm-n128-k32-v1",
      })),
      ...[
        "timestep-silu-1", "relative-silu-1",
        "timestep-silu-2", "relative-silu-2",
        "embedding-add", "projection-add",
      ].map((suffix) => ({
        quantum: timestep,
        suffix,
        family: "timestep",
        backend: "reference-bf16",
        kernel: "reference-bf16-transformer-plumbing",
      })),
      ...["concatenate", "patch"].map((suffix) => ({
        quantum: input,
        suffix,
        family: "input",
        backend: "reference-bf16",
        kernel: "reference-bf16-dit-plumbing",
      })),
      ...[
        "self-query-projection", "self-key-projection",
        "self-value-projection", "self-output-projection",
        "cross-query-projection", "cross-output-projection",
      ].map((suffix) => ({
        quantum: layer,
        suffix,
        family: "attention-projections",
        backend: "opt-0009-fp16-fp32",
        kernel: "opt-0009-n256-k32-fp16-fp32-v1",
      })),
      ...[
        ["self-full-attention", "self-full"],
        ["self-sliding-attention", "self-sliding"],
        ["cross-attention", "cross-attention"],
      ].map(([suffix, family]) => ({
        quantum: layer,
        suffix: suffix!,
        family: family!,
        backend: "fixed32-subgroup-query8",
        kernel: "fixed32-subgroup-query8",
      })),
      {
        quantum: layer,
        suffix: "mlp-norm",
        family: "feed-forward",
        backend: "reference-bf16",
        kernel: "reference-bf16-rmsnorm",
      },
      ...["mlp-gate-projection", "mlp-up-projection", "mlp-down-projection"]
        .map((suffix) => ({
          quantum: layer,
          suffix,
          family: "feed-forward",
          backend: "opt-0009-fp16-fp32",
          kernel: "opt-0009-n256-k32-fp16-fp32-v1",
        })),
      ...["mlp-adaln", "mlp-swiglu", "mlp-gated-residual"]
        .map((suffix) => ({
          quantum: layer,
          suffix,
          family: "feed-forward",
          backend: "reference-bf16",
          kernel: "reference-bf16-transformer-plumbing",
        })),
      {
        quantum: layer,
        suffix: "modulation",
        family: "plumbing",
        backend: "reference-bf16",
        kernel: "reference-bf16-dit-plumbing",
      },
      ...[
        "self-norm", "self-query-norm", "self-key-norm",
        "cross-norm", "cross-query-norm",
      ].map((suffix) => ({
        quantum: layer,
        suffix,
        family: "plumbing",
        backend: "reference-bf16",
        kernel: "reference-bf16-rmsnorm",
      })),
      ...["self-query-rope", "self-key-rope"].map((suffix) => ({
        quantum: layer,
        suffix,
        family: "plumbing",
        backend: "reference-bf16",
        kernel: "reference-bf16-rope",
      })),
      ...[
        "self-adaln", "self-split-query-heads", "self-split-key-heads",
        "self-split-value-heads", "self-merge-heads",
        "self-gated-residual", "cross-split-query-heads",
        "cross-merge-heads", "cross-residual",
      ].map((suffix) => ({
        quantum: layer,
        suffix,
        family: "plumbing",
        backend: "reference-bf16",
        kernel: "reference-bf16-transformer-plumbing",
      })),
      {
        quantum: output,
        suffix: "norm",
        family: "output",
        backend: "reference-bf16",
        kernel: "reference-bf16-rmsnorm",
      },
      ...["modulation", "unpatch"].map((suffix) => ({
        quantum: output,
        suffix,
        family: "output",
        backend: "reference-bf16",
        kernel: "reference-bf16-dit-plumbing",
      })),
      {
        quantum: output,
        suffix: "adaln",
        family: "output",
        backend: "reference-bf16",
        kernel: "reference-bf16-transformer-plumbing",
      },
      ...["sampler-update", "predicted-clean"].map((suffix) => ({
        quantum: sampler,
        suffix,
        family: "sampler-dcw",
        backend: "reference-bf16",
        kernel: "reference-bf16-dit-plumbing",
      })),
      {
        quantum: sampler,
        suffix: "dcw",
        family: "sampler-dcw",
        backend: "reference-bf16",
        kernel: "reference-bf16-haar-dcw",
      },
    ];
    for (const testCase of cases) {
      const label = testCase.suffix === ""
        ? testCase.quantum.label
        : `${testCase.quantum.label}-${testCase.suffix}`;
      expect(
        classifyAceOpt0018DitCommandMember(testCase.quantum, label),
        label,
      ).toEqual({
        family: testCase.family,
        backend: testCase.backend,
        kernel: testCase.kernel,
      });
    }
    expect(cases).toHaveLength(62);
    expect(() => classifyAceOpt0018DitCommandMember(
      layer,
      `${layer.label}-future-kernel`,
    )).toThrow(/cannot classify/);
    expect(() => classifyAceOpt0018DitCommandMember(
      layer,
      "escaped-layer-member",
    )).toThrow(/escaped logical command/);
  });
});

describe("ACE DiT production graph owner", () => {
  it("submits every quantum singly, reports sampler progress, and releases DiT", async () => {
    const events: string[] = [];
    const records: FakeRuntimeRecord[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const descriptorState = { calls: 0 };
    const runtime = createFakeRuntime(
      records,
      events,
      runtimeState,
      1,
      descriptorState,
    );
    const model = createFakeModel(modelState);
    const bindings = createBindings();
    const { device, submissions, drains, commandBuffers } = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      model,
      runtime,
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      bindings,
    );
    expect(commandBuffers.created).toBe(0);
    expect(commandBuffers.live).toBe(0);
    expect(owner.physicalCommandDescriptors).toBeUndefined();
    expect(descriptorState.calls).toBe(0);

    expect(records.filter((record) => record.method === "condition")).toHaveLength(1);
    expect(records.filter((record) => record.method === "cross-cache")).toHaveLength(24);
    expect(records.filter((record) => record.method === "timestep")).toHaveLength(8);
    expect(records.filter((record) => record.method === "input")).toHaveLength(8);
    expect(records.filter((record) => record.method === "layer")).toHaveLength(192);
    expect(records.filter((record) => record.method === "output")).toHaveLength(8);
    expect(records.filter((record) => record.method === "sampler")).toHaveLength(8);
    const layerRecords = records.filter((record) => record.method === "layer");
    expect(layerRecords.slice(0, 4).map((record) =>
      (record.arguments[2] as { readonly attentionMode: string }).attentionMode
    )).toEqual(["sliding", "full", "sliding", "full"]);

    const progress: Array<Readonly<{
      completed: number;
      evaluations: number;
      kind: string;
    }>> = [];
    let idles = 0;
    const result = await owner.run({
      signal: new AbortController().signal,
      yieldQueueIdle: async () => {
        idles += 1;
      },
      onProgress: (value) => {
        progress.push({
          completed: value.completedQuanta,
          evaluations: value.completedEvaluations,
          kind: value.quantum.kind,
        });
      },
    });

    expect(submissions).toHaveLength(249);
    expect(submissions.every((submitted) => submitted.length === 1)).toBe(true);
    expect(drains.value).toBe(249);
    expect(commandBuffers.maximumLive).toBe(1);
    expect(commandBuffers.live).toBe(0);
    expect(idles).toBe(248);
    expect(progress).toHaveLength(249);
    expect(progress.filter((value) => value.kind === "sampler").map(
      (value) => value.evaluations,
    )).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(result).toMatchObject({
      commandBuffersSubmitted: 249,
      queueDrains: 249,
      cooperativeIdleMs: 248,
      completedEvaluations: 8,
      finalLatent: bindings.latents[0],
    });
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
    expect(descriptorState.calls).toBe(0);
    await expect(owner.run({
      signal: new AbortController().signal,
    })).rejects.toThrow(/state finished/);
  });

  it("runs the capture-free production policy with dynamically derived phases", async () => {
    const encodedByPolicy = new Map<string, readonly string[]>();
    for (const policy of [
      "depth1-epoch1",
      "depth2-phase-epoch4",
    ] as const) {
      const events: string[] = [];
      const runtimeState = { destroys: 0 };
      const modelState = { destroys: 0 };
      const fixture = createFakeDevice(events);
      const owner = await AceDitGraphOwner.createWithRuntime(
        fixture.device,
        createFakeModel(modelState),
        createFakeRuntime([], events, runtimeState),
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        createBindings(),
      );
      expect(owner.commandBufferCount).toBe(249);
      expect(owner.physicalCommandDescriptors).toBeUndefined();
      const completedEvaluations: number[] = [];
      let idles = 0;
      const result = await owner.run({
        signal: new AbortController().signal,
        submissionPolicy: policy,
        yieldQueueIdle: async () => { idles += 1; },
        onProgress: (progress) => {
          completedEvaluations.push(progress.completedEvaluations);
        },
      });
      const candidate = policy === "depth2-phase-epoch4";
      expect(fixture.submissions).toHaveLength(249);
      expect(fixture.submissions.every((value) => value.length === 1)).toBe(true);
      expect(fixture.drains.value).toBe(249);
      expect(idles).toBe(candidate ? 62 : 248);
      expect(completedEvaluations).toHaveLength(249);
      expect(completedEvaluations.at(-1)).toBe(8);
      expect(result).toMatchObject({
        commandBuffersSubmitted: 249,
        queueDrains: candidate ? 63 : 249,
        cooperativeIdleMs: candidate ? 62 : 248,
        completedEvaluations: 8,
      });
      expect(result).not.toHaveProperty("opt0080Scheduling");
      expect(fixture.commandBuffers.live).toBe(0);
      expect(runtimeState.destroys).toBe(1);
      expect(modelState.destroys).toBe(1);
      encodedByPolicy.set(
        policy,
        events.filter((event) => event.startsWith("encode:")),
      );
    }
    expect(encodedByPolicy.get("depth2-phase-epoch4")).toEqual(
      encodedByPolicy.get("depth1-epoch1"),
    );
  });

  it("settles the capture-free successor before production cancellation releases", async () => {
    const events: string[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const fixture = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      fixture.device,
      createFakeModel(modelState),
      createFakeRuntime([], events, runtimeState),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    expect(owner.physicalCommandDescriptors).toBeUndefined();
    const controller = new AbortController();
    const progress: number[] = [];
    await expect(owner.run({
      signal: controller.signal,
      submissionPolicy: "depth2-phase-epoch4",
      yieldQueueIdle: async () => undefined,
      onProgress: ({ completedCommandBuffers }) => {
        progress.push(completedCommandBuffers);
        controller.abort();
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(progress).toEqual([1]);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.drains.value).toBe(2);
    expect(fixture.commandBuffers.live).toBe(0);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it("keeps OPT-0080 A/B descriptor ownership FIFO across the exact phase boundary", async () => {
    const encodedByProfile = new Map<string, readonly string[]>();
    for (const profile of [
      "depth1-epoch1",
      "opt-0080-depth2-epoch4",
    ] as const) {
      const events: string[] = [];
      const runtimeState = { destroys: 0 };
      const modelState = { destroys: 0 };
      const fixture = createFakeDevice(events);
      const owner = await AceDitGraphOwner.createWithRuntime(
        fixture.device,
        createFakeModel(modelState),
        createFakeRuntime(
          [],
          events,
          runtimeState,
          (method) => method === "layer" ? 13 : 1,
        ),
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        createBindings(),
      );
      expect(owner.commandBufferCount).toBe(
        ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
      );
      const completed: number[] = [];
      const phases: number[] = [];
      const epochs: number[] = [];
      let idles = 0;
      const result = await owner.run({
        signal: new AbortController().signal,
        opt0067EvaluationLimit: 1,
        opt0080SchedulingProfile: profile,
        yieldQueueIdle: async () => { idles += 1; },
        onProgress: (progress) => {
          completed.push(progress.completedQuanta);
        },
        onOpt0080PhaseStarted: (phase) => {
          phases.push(phase.firstCommandBufferIndex);
        },
        onOpt0080CompletionEpochDrained: (epoch) => {
          epochs.push(epoch.lastCommandBufferIndex);
        },
      });
      const candidate = profile === "opt-0080-depth2-epoch4";
      expect(fixture.submissions).toHaveLength(341);
      expect(fixture.submissions.every((value) => value.length === 1)).toBe(true);
      expect(fixture.drains.value).toBe(341);
      expect(completed).toEqual(Array.from({ length: 341 }, (_, index) => index + 1));
      expect(phases).toEqual([0, 25]);
      expect(epochs).toHaveLength(candidate ? 86 : 341);
      expect(epochs.at(-1)).toBe(340);
      expect(idles).toBe(candidate ? 85 : 340);
      expect(result).toMatchObject({
        commandBuffersSubmitted: 341,
        queueDrains: candidate ? 86 : 341,
        cooperativeIdleMs: candidate ? 85 : 340,
        completedEvaluations: 1,
        opt0080Scheduling: {
          profile,
          completionFenceRequestedCount: 341,
          completionFenceSettledCount: 341,
          completionFenceRejectedCount: 0,
          trueQueueDrainCount: candidate ? 86 : 341,
          completionEpochCount: candidate ? 86 : 341,
          requestedCooperativeIdleMs: candidate ? 85 : 340,
          cooperativeIdleTurns: candidate ? 85 : 340,
          maximumOutstandingCommandBuffers: candidate ? 2 : 1,
          maximumPendingDescriptorCount: candidate ? 2 : 1,
        },
      });
      expect(fixture.commandBuffers.live).toBe(0);
      expect(runtimeState.destroys).toBe(1);
      expect(modelState.destroys).toBe(1);
      encodedByProfile.set(
        profile,
        events.filter((event) => event.startsWith("encode:")),
      );
    }
    expect(encodedByProfile.get("opt-0080-depth2-epoch4")).toEqual(
      encodedByProfile.get("depth1-epoch1"),
    );
  });

  it("extends OPT-0080 unchanged across all nine full-graph phases", async () => {
    for (const profile of [
      "depth1-epoch1",
      "opt-0080-depth2-epoch4",
    ] as const) {
      const events: string[] = [];
      const runtimeState = { destroys: 0 };
      const modelState = { destroys: 0 };
      const fixture = createFakeDevice(events);
      const owner = await AceDitGraphOwner.createWithRuntime(
        fixture.device,
        createFakeModel(modelState),
        createFakeRuntime(
          [],
          events,
          runtimeState,
          (method) => method === "layer" ? 13 : 1,
        ),
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        createBindings(),
      );
      const phases: number[] = [];
      const epochs: number[] = [];
      let idles = 0;
      const result = await owner.run({
        signal: new AbortController().signal,
        opt0080SchedulingProfile: profile,
        opt0080FullGraph: true,
        yieldQueueIdle: async () => { idles += 1; },
        onOpt0080PhaseStarted: (phase) => {
          phases.push(phase.firstCommandBufferIndex);
        },
        onOpt0080CompletionEpochDrained: (epoch) => {
          epochs.push(epoch.lastCommandBufferIndex);
        },
      });
      const candidate = profile === "opt-0080-depth2-epoch4";
      expect(fixture.submissions).toHaveLength(2_553);
      expect(fixture.submissions.every((value) => value.length === 1)).toBe(true);
      expect(fixture.drains.value).toBe(2_553);
      expect(phases).toEqual([0, 25, 341, 657, 973, 1_289, 1_605, 1_921, 2_237]);
      expect(epochs).toHaveLength(candidate ? 639 : 2_553);
      expect(epochs.at(-1)).toBe(2_552);
      expect(idles).toBe(candidate ? 638 : 2_552);
      expect(result).toMatchObject({
        commandBuffersSubmitted: 2_553,
        queueDrains: candidate ? 639 : 2_553,
        cooperativeIdleMs: candidate ? 638 : 2_552,
        completedEvaluations: 8,
        opt0080Scheduling: {
          completionFenceRequestedCount: 2_553,
          completionFenceSettledCount: 2_553,
          completionFenceRejectedCount: 0,
          maximumOutstandingCommandBuffers: candidate ? 2 : 1,
          maximumPendingDescriptorCount: candidate ? 2 : 1,
        },
      });
      expect(fixture.commandBuffers.live).toBe(0);
      expect(runtimeState.destroys).toBe(1);
      expect(modelState.destroys).toBe(1);
    }
  });

  it("retains the prefetched OPT-0080 descriptor until abort settlement", async () => {
    const events: string[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const fixture = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      fixture.device,
      createFakeModel(modelState),
      createFakeRuntime(
        [],
        events,
        runtimeState,
        (method) => method === "layer" ? 13 : 1,
      ),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    const descriptor = Object.freeze({
      physicalIndex: 0,
      logicalIndex: 0,
      logicalKind: "condition-projection" as const,
      commandId: "ace-dit-condition-projection",
      subquantumIndex: 0,
      subquantumCount: 1,
      evaluation: null,
      layer: null,
      family: "precompute" as const,
      primitiveCount: 1,
      scheduledMultiplyAdds: 0,
      members: Object.freeze([]),
    });
    (owner as unknown as {
      physicalCommandDescriptors: AceDitPhysicalCommandDescriptorTable;
    }).physicalCommandDescriptors = Object.freeze({
      descriptors: Object.freeze([descriptor]),
      sha256: "test",
      serializedBytes: 0,
      memberCount: 0,
      preparationMs: 0,
    });
    const controller = new AbortController();
    const progress: number[] = [];
    const completions: number[] = [];
    await expect(owner.run({
      signal: controller.signal,
      opt0067EvaluationLimit: 1,
      opt0080SchedulingProfile: "opt-0080-depth2-epoch4",
      yieldQueueIdle: async () => undefined,
      onProgress: ({ completedCommandBuffers }) => {
        progress.push(completedCommandBuffers);
      },
      onOpt0080CommandBufferCompleted: (completion) => {
        completions.push(completion.descriptor.physicalIndex);
        expect(fixture.submissions).toHaveLength(2);
        expect(completion.schedulingProgress.outstandingCommandBuffers).toBe(1);
        controller.abort();
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(completions).toEqual([0]);
    expect(progress).toEqual([1]);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.drains.value).toBe(2);
    expect(fixture.commandBuffers.live).toBe(0);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it("suppresses the specialized OPT-0080 completion after progress aborts", async () => {
    const events: string[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const fixture = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      fixture.device,
      createFakeModel(modelState),
      createFakeRuntime(
        [],
        events,
        runtimeState,
        (method) => method === "layer" ? 13 : 1,
      ),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    const descriptor = Object.freeze({
      physicalIndex: 0,
      logicalIndex: 0,
      logicalKind: "condition-projection" as const,
      commandId: "ace-dit-condition-projection",
      subquantumIndex: 0,
      subquantumCount: 1,
      evaluation: null,
      layer: null,
      family: "precompute" as const,
      primitiveCount: 1,
      scheduledMultiplyAdds: 0,
      members: Object.freeze([]),
    });
    (owner as unknown as {
      physicalCommandDescriptors: AceDitPhysicalCommandDescriptorTable;
    }).physicalCommandDescriptors = Object.freeze({
      descriptors: Object.freeze([descriptor]),
      sha256: "test",
      serializedBytes: 0,
      memberCount: 0,
      preparationMs: 0,
    });
    const controller = new AbortController();
    const progress: number[] = [];
    const completions: number[] = [];
    await expect(owner.run({
      signal: controller.signal,
      opt0067EvaluationLimit: 1,
      opt0080SchedulingProfile: "opt-0080-depth2-epoch4",
      yieldQueueIdle: async () => undefined,
      onProgress: ({ completedCommandBuffers }) => {
        progress.push(completedCommandBuffers);
        controller.abort();
      },
      onOpt0080CommandBufferCompleted: (completion) => {
        completions.push(completion.descriptor.physicalIndex);
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(progress).toEqual([1]);
    expect(completions).toEqual([]);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.drains.value).toBe(2);
    expect(fixture.commandBuffers.live).toBe(0);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it("copies every sampler result in its existing command buffer", async () => {
    const events: string[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const bindings = createBindings();
    const fixture = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      fixture.device,
      createFakeModel(modelState),
      createFakeRuntime([], events, runtimeState),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      bindings,
    );
    const evaluationReadbacks = Array.from({ length: 8 }, () => ({
      size: owner.memoryPlan.latentBytes,
    } as GPUBuffer));
    const result = await owner.run({
      signal: new AbortController().signal,
      yieldQueueIdle: async () => undefined,
      evaluationReadbacks,
    });
    expect(result).toMatchObject({
      commandBuffersSubmitted: 249,
      queueDrains: 249,
      cooperativeIdleMs: 248,
      completedEvaluations: 8,
    });
    expect(fixture.submissions).toHaveLength(249);
    expect(fixture.drains.value).toBe(249);
    expect(fixture.copies).toHaveLength(8);
    expect(fixture.copies.map(({ source }) => source)).toEqual(
      Array.from({ length: 8 }, (_, evaluation) =>
        bindings.latents[(evaluation + 1) % 2]!.buffer
      ),
    );
    expect(fixture.copies.map(({ target }) => target)).toEqual(
      evaluationReadbacks,
    );
    expect(fixture.copies.every(({ bytes }) =>
      bytes === owner.memoryPlan.latentBytes
    )).toBe(true);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it("cancels only after the submitted quantum drains, then destroys weights", async () => {
    const events: string[] = [];
    const records: FakeRuntimeRecord[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const { device, submissions, drains } = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      createFakeModel(modelState),
      createFakeRuntime(records, events, runtimeState),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    const controller = new AbortController();
    await expect(owner.run({
      signal: controller.signal,
      yieldQueueIdle: async () => undefined,
      onProgress: ({ completedQuanta }) => {
        if (completedQuanta === 27) controller.abort();
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(submissions).toHaveLength(27);
    expect(drains.value).toBe(27);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it.each([8, 16] as const)(
    "coalesces consecutive physical quanta batch%i with one outstanding drain",
    async (physicalQuantaPerCommandBuffer) => {
      const events: string[] = [];
      const records: FakeRuntimeRecord[] = [];
      const runtimeState = { destroys: 0 };
      const modelState = { destroys: 0 };
      const { device, submissions, drains, commandBuffers } =
        createFakeDevice(events);
      const owner = await AceDitGraphOwner.createWithRuntime(
        device,
        createFakeModel(modelState),
        createFakeRuntime(records, events, runtimeState),
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        createBindings(),
      );
      const progress: Array<Readonly<{
        completedQuanta: number;
        completedCommandBuffers: number;
        batchFirstPhysicalQuantum: number;
        batchPhysicalQuantumCount: number;
        completedEvaluations: number;
      }>> = [];
      let idles = 0;
      const result = await owner.run({
        signal: new AbortController().signal,
        physicalQuantaPerCommandBuffer,
        yieldQueueIdle: async () => { idles += 1; },
        onProgress: (event) => progress.push({
          completedQuanta: event.completedQuanta,
          completedCommandBuffers: event.completedCommandBuffers,
          batchFirstPhysicalQuantum: event.batchFirstPhysicalQuantum,
          batchPhysicalQuantumCount: event.batchPhysicalQuantumCount,
          completedEvaluations: event.completedEvaluations,
        }),
      });
      const expectedBatches = Math.ceil(
        ACE_DIT_GRAPH_QUANTUM_COUNT / physicalQuantaPerCommandBuffer,
      );
      expect(submissions).toHaveLength(expectedBatches);
      expect(drains.value).toBe(expectedBatches);
      expect(idles).toBe(expectedBatches - 1);
      expect(commandBuffers.maximumLive).toBe(1);
      expect(commandBuffers.live).toBe(0);
      expect(events.filter((event) => event.startsWith("encode:")))
        .toHaveLength(ACE_DIT_GRAPH_QUANTUM_COUNT);
      expect(progress).toHaveLength(expectedBatches);
      expect(progress[0]).toMatchObject({
        completedQuanta: physicalQuantaPerCommandBuffer,
        completedCommandBuffers: 1,
        batchFirstPhysicalQuantum: 0,
        batchPhysicalQuantumCount: physicalQuantaPerCommandBuffer,
      });
      expect(progress.at(-1)).toMatchObject({
        completedQuanta: ACE_DIT_GRAPH_QUANTUM_COUNT,
        completedCommandBuffers: expectedBatches,
        completedEvaluations: 8,
      });
      expect(result).toMatchObject({
        commandBuffersSubmitted: expectedBatches,
        queueDrains: expectedBatches,
        cooperativeIdleMs: expectedBatches - 1,
        completedEvaluations: 8,
      });
      expect(runtimeState.destroys).toBe(1);
      expect(modelState.destroys).toBe(1);
    },
  );

  it("checks abort after a drained batch and reports only completed work", async () => {
    const events: string[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const { device, submissions, drains, commandBuffers } =
      createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      createFakeModel(modelState),
      createFakeRuntime([], events, runtimeState),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    const controller = new AbortController();
    const completed: number[] = [];
    await expect(owner.run({
      signal: controller.signal,
      physicalQuantaPerCommandBuffer: 8,
      yieldQueueIdle: async () => undefined,
      onProgress: (progress) => {
        completed.push(progress.completedQuanta);
        controller.abort();
      },
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(completed).toEqual([8]);
    expect(submissions).toHaveLength(1);
    expect(drains.value).toBe(1);
    expect(commandBuffers.maximumLive).toBe(1);
    expect(commandBuffers.live).toBe(0);
    expect(events.filter((event) => event.startsWith("encode:")))
      .toHaveLength(8);
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });

  it("flattens composite subquanta into independently drained command buffers", async () => {
    const events: string[] = [];
    const records: FakeRuntimeRecord[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const runtime = createFakeRuntime(records, events, runtimeState, 3);
    const { device, submissions, drains } = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      createFakeModel(modelState),
      runtime,
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    const progress: Array<{ sub: number; count: number; id: string }> = [];
    let idles = 0;
    const result = await owner.run({
      signal: new AbortController().signal,
      yieldQueueIdle: async () => { idles += 1; },
      onProgress: (value) => progress.push({
        sub: value.subquantumIndex,
        count: value.subquantumCount,
        id: value.commandId,
      }),
    });
    expect(submissions).toHaveLength(249 * 3);
    expect(drains.value).toBe(249 * 3);
    expect(idles).toBe(249 * 3 - 1);
    expect(result.commandBuffersSubmitted).toBe(249 * 3);
    expect(progress.slice(0, 3)).toEqual([
      { sub: 0, count: 3, id: "ace-dit-condition-projection-part-0" },
      { sub: 1, count: 3, id: "ace-dit-condition-projection-part-1" },
      { sub: 2, count: 3, id: "ace-dit-condition-projection-part-2" },
    ]);
    expect(events.filter((event) => event.startsWith("encode:"))).toHaveLength(249 * 3);
  });

  it("expands a bare ranged GEMM without eager command-buffer retention", async () => {
    const events: string[] = [];
    const records: FakeRuntimeRecord[] = [];
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const base = createFakeRuntime(records, events, runtimeState);
    const runtime = {
      ...base,
      async createConditionProjectionDispatch(label: string) {
        return {
          label,
          plan: {
            outputRanges: [
              { multiplyAdds: 1_000_000_000 },
              { multiplyAdds: 1_000_000_000 },
              { multiplyAdds: 1_000_000_000 },
            ],
          },
          rangeCount: 3,
          encodeRange(_pass: GPUComputePassEncoder, range: number) {
            events.push(`bare-range:${range}`);
          },
          encode() {
            throw new Error("bare ranged GEMM was collapsed");
          },
        };
      },
    } as unknown as Parameters<typeof AceDitGraphOwner.createWithRuntime>[2];
    const { device, submissions, commandBuffers } = createFakeDevice(events);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      createFakeModel(modelState),
      runtime,
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      createBindings(),
    );
    expect(owner.commandBufferCount).toBe(250);
    expect(commandBuffers.created).toBe(0);
    await owner.run({
      signal: new AbortController().signal,
      yieldQueueIdle: async () => undefined,
    });
    expect(submissions).toHaveLength(250);
    expect(events.filter((event) => event.startsWith("bare-range:"))).toEqual([
      "bare-range:0",
      "bare-range:1",
      "bare-range:2",
    ]);
    expect(commandBuffers.maximumLive).toBe(1);
    expect(commandBuffers.live).toBe(0);
  });

  it("fails closed on missing caches and corrupting lifetime aliases", async () => {
    const cases: Array<{
      readonly mutate: (bindings: MutableBindings) => void;
      readonly pattern: RegExp;
    }> = [
      {
        mutate: (bindings) => {
          bindings.crossCaches = bindings.crossCaches.slice(0, 23);
        },
        pattern: /exactly 24 cross/,
      },
      {
        mutate: (bindings) => {
          bindings.layerScratch.selfNormalized = bindings.context;
        },
        pattern: /context aliases layer scratch selfNormalized/,
      },
      {
        mutate: (bindings) => {
          bindings.controls.timesteps[7] = bindings.crossCacheScratch.keyFlat;
        },
        pattern: /cross cache scratch keyFlat aliases timestep 7/,
      },
    ];
    for (const testCase of cases) {
      const bindings = createMutableBindings();
      testCase.mutate(bindings);
      const runtimeState = { destroys: 0 };
      const modelState = { destroys: 0 };
      const { device } = createFakeDevice([]);
      await expect(AceDitGraphOwner.createWithRuntime(
        device,
        createFakeModel(modelState),
        createFakeRuntime([], [], runtimeState),
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        bindings,
      )).rejects.toThrow(testCase.pattern);
      expect(runtimeState.destroys).toBe(1);
      expect(modelState.destroys).toBe(1);
    }
  });

  it("permits an arena range to be reused after its last quantum", async () => {
    const bindings = createMutableBindings();
    bindings.layerScratch.selfNormalized = bindings.conditionInput;
    const runtimeState = { destroys: 0 };
    const modelState = { destroys: 0 };
    const { device } = createFakeDevice([]);
    const owner = await AceDitGraphOwner.createWithRuntime(
      device,
      createFakeModel(modelState),
      createFakeRuntime([], [], runtimeState),
      SHAPE,
      ACE_DIRECT_DCW_CONFIGURATION,
      bindings,
    );
    await owner.destroy();
    expect(runtimeState.destroys).toBe(1);
    expect(modelState.destroys).toBe(1);
  });
});

type MutableBindings = Omit<
  AceDitGraphBindings,
  "crossCaches" | "layerScratch" | "controls"
> & {
  crossCaches: AceDitGraphBindings["crossCaches"];
  layerScratch: { -readonly [Key in keyof AceDitLayerScratch]: GPUBufferBinding };
  controls: Omit<AceDitGraphBindings["controls"], "timesteps"> & {
    timesteps: GPUBufferBinding[];
  };
};

function createBindings(): AceDitGraphBindings {
  return createMutableBindings();
}

function createMutableBindings(): MutableBindings {
  const bindings = {
    conditionInput: binding(),
    projectedCondition: binding(),
    crossCaches: Array.from({ length: 24 }, () => ({
      key: binding(),
      value: binding(),
    })),
    crossCacheScratch: objectBindings([
      "keyFlat",
      "valueFlat",
      "keyHeads",
    ] as const satisfies readonly (keyof AceDitCrossCacheScratch)[]),
    context: binding(),
    latents: [binding(), binding(), binding()] as const,
    concatenatedInput: binding(),
    hidden: [binding(), binding()] as const,
    timestepScratch: objectBindings(TIMESTEP_SCRATCH_KEYS),
    timestepEmbedding: binding(),
    timestepProjection: binding(),
    layerScratch: objectBindings(LAYER_SCRATCH_KEYS),
    outputScratch: objectBindings([
      "normalized",
      "modulation",
      "modulated",
    ] as const satisfies readonly (keyof AceDitOutputScratch)[]),
    velocity: binding(),
    predictedCleanLatent: binding(),
    controls: {
      selfValidLengths: binding(),
      crossValidLengths: binding(),
      cosine: binding(),
      sine: binding(),
      timesteps: Array.from({ length: 8 }, binding),
      relativeTimestepZero: binding(),
    },
  };
  return bindings as MutableBindings;
}

function createFakeModel(state: { destroys: number }): AceDitGraphModel {
  return {
    modelProfile: "raw-fp16",
    residentBytes: 123,
    weights: createFakeWeights(),
    destroy() {
      state.destroys += 1;
    },
  };
}

function createFakeWeights(): AceDitResolvedPackageWeights {
  const branch = () => objectBindings([
    "linear1Weight",
    "linear1Bias",
    "linear2Weight",
    "linear2Bias",
    "projectionWeight",
    "projectionBias",
  ] as const);
  const layer = () => objectBindings([
    "scaleShiftTable",
    "selfAttentionNorm",
    "selfQueryProjection",
    "selfKeyProjection",
    "selfValueProjection",
    "selfQueryNorm",
    "selfKeyNorm",
    "selfOutputProjection",
    "crossAttentionNorm",
    "crossQueryProjection",
    "crossQueryNorm",
    "crossOutputProjection",
    "mlpNorm",
    "gateProjection",
    "upProjection",
    "downProjection",
  ] as const);
  return {
    conditionProjection: { weight: binding(), bias: binding() },
    inputProjection: { weight: binding(), bias: binding() },
    timestep: { timestep: branch(), relative: branch() },
    output: objectBindings(["norm", "scaleShiftTable", "projection", "bias"] as const),
    layers: Array.from({ length: 24 }, layer),
    crossCaches: Array.from({ length: 24 }, () =>
      objectBindings(["keyProjection", "valueProjection", "keyNorm"] as const)
    ),
  } as AceDitResolvedPackageWeights;
}

function createFakeRuntime(
  records: FakeRuntimeRecord[],
  events: string[],
  state: { destroys: number },
  cooperativeParts: number | ((method: string) => number) = 1,
  descriptorState?: { calls: number },
): Parameters<typeof AceDitGraphOwner.createWithRuntime>[2] {
  const create = (method: string) => async (...arguments_: readonly unknown[]) => {
    const label = arguments_[0] as string;
    records.push({ method, label, arguments: arguments_ });
    const partCount = typeof cooperativeParts === "number"
      ? cooperativeParts
      : cooperativeParts(method);
    const cooperativeQuanta = Array.from(
      { length: partCount },
      (_, part) => ({
        id: `${label}-part-${part}`,
        primitiveCount: 1,
        encode() {
          events.push(`encode:${label}:part-${part}`);
        },
      }),
    );
    const cooperativeSequence = Object.freeze({
      quantumCount: cooperativeQuanta.length,
      describeQuantum() {
        if (descriptorState !== undefined) descriptorState.calls += 1;
        throw new Error("normal graph path touched capture descriptors");
      },
      encodeQuantum(_pass: GPUComputePassEncoder, index: number) {
        cooperativeQuanta[index]!.encode();
      },
    });
    return {
      label,
      plan: Object.freeze({}),
      primitiveCount: 1,
      ...(descriptorState === undefined
        ? { cooperativeQuanta }
        : { cooperativeSequence }),
      encode() {
        events.push(`encode:${label}`);
      },
    };
  };
  return {
    createConditionProjectionDispatch: create("condition"),
    createCrossCacheDispatch: create("cross-cache"),
    createTimestepDispatch: create("timestep"),
    createInputProjectionDispatch: create("input"),
    createLayerDispatch: create("layer"),
    createOutputProjectionDispatch: create("output"),
    createSamplerStepDispatch: create("sampler"),
    destroy() {
      state.destroys += 1;
    },
  } as unknown as Parameters<typeof AceDitGraphOwner.createWithRuntime>[2];
}

function createFakeDevice(events: string[]): Readonly<{
  device: GPUDevice;
  submissions: GPUCommandBuffer[][];
  drains: { value: number };
  commandBuffers: { created: number; live: number; maximumLive: number };
  copies: readonly Readonly<{
    source: GPUBuffer;
    target: GPUBuffer;
    bytes: number;
  }>[];
}> {
  const submissions: GPUCommandBuffer[][] = [];
  const drains = { value: 0 };
  const commandBuffers = { created: 0, live: 0, maximumLive: 0 };
  const copies: Array<Readonly<{
    source: GPUBuffer;
    target: GPUBuffer;
    bytes: number;
  }>> = [];
  const device = {
    lost: new Promise<never>(() => undefined),
    queue: {
      submit(commandBuffers: Iterable<GPUCommandBuffer>) {
        const values = [...commandBuffers];
        submissions.push(values);
        events.push(`submit:${values[0]!.label}`);
      },
      async onSubmittedWorkDone() {
        drains.value += 1;
        commandBuffers.live -= 1;
      },
    },
    createCommandEncoder(descriptor: GPUCommandEncoderDescriptor) {
      let ended = false;
      return {
        beginComputePass() {
          return {
            end() {
              ended = true;
            },
          };
        },
        copyBufferToBuffer(
          source: GPUBuffer,
          _sourceOffset: number,
          target: GPUBuffer,
          _targetOffset: number,
          bytes: number,
        ) {
          expect(ended).toBe(true);
          copies.push(Object.freeze({ source, target, bytes }));
        },
        finish(commandDescriptor: GPUCommandBufferDescriptor) {
          expect(ended).toBe(true);
          commandBuffers.created += 1;
          commandBuffers.live += 1;
          commandBuffers.maximumLive = Math.max(
            commandBuffers.maximumLive,
            commandBuffers.live,
          );
          return { label: commandDescriptor.label ?? descriptor.label } as GPUCommandBuffer;
        },
      };
    },
  } as unknown as GPUDevice;
  return { device, submissions, drains, commandBuffers, copies };
}

function binding(): GPUBufferBinding {
  const size = 1_000_000_000;
  return {
    buffer: { size } as GPUBuffer,
    offset: 0,
    size,
  };
}

function objectBindings<const Keys extends readonly string[]>(
  keys: Keys,
): { readonly [Key in Keys[number]]: GPUBufferBinding } {
  return Object.fromEntries(keys.map((key) => [key, binding()])) as {
    readonly [Key in Keys[number]]: GPUBufferBinding;
  };
}
