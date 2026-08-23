import { describe, expect, it } from "vitest";

import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  ACE_OOBLECK_DECODER_CONFIG,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  resolveAceVaeDecoderTensorBindings,
  snapshotAceVaeDecoderQuantumWorkPolicy,
  summarizeAceVaeDecoderConv1dOperationSelection,
  type AceVaeDecoderConfig,
  type AceVaeDecoderOperation,
} from "../src/webgpu/vae-decoder.js";
import {
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
} from "../src/webgpu/kernels/vae-primitives.js";
import { ACE_REQUIRED_WEBGPU_LIMITS } from
  "../src/webgpu/capabilities.js";

describe("ACE decoder-only AutoencoderOobleck contract", () => {
  it("pins exact optimized Conv1D coverage under the production 16 KiB limit", () => {
    expect(ACE_REQUIRED_WEBGPU_LIMITS.maxComputeWorkgroupStorageSize)
      .toBe(16 * 1_024);
    const selections = summarizeAceVaeDecoderConv1dOperationSelection(
      planAceVaeDecoder(256),
      {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupStorageSize:
          ACE_REQUIRED_WEBGPU_LIMITS.maxComputeWorkgroupStorageSize!,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxStorageBufferBindingSize: 256 * 1_024 * 1_024,
      },
    );
    const tiled = selections.filter(({ selection }) => selection === "tiled");
    expect(tiled).toEqual([
      { label: "conv1", selection: "tiled", reason: "eligible", quantumCount: 1 },
      {
        label: "block-3-res-1-conv1",
        selection: "tiled",
        reason: "eligible",
        quantumCount: 120,
      },
      {
        label: "block-4-res-1-conv1",
        selection: "tiled",
        reason: "eligible",
        quantumCount: 240,
      },
      { label: "conv2", selection: "tiled", reason: "eligible", quantumCount: 4 },
    ]);
    expect(tiled.reduce((sum, operation) => sum + operation.quantumCount, 0))
      .toBe(365);
    const channelChunked = selections.filter(
      ({ selection }) => selection === "channel-chunked",
    );
    expect(channelChunked.map(({ label }) => label)).toEqual([
      "block-0-res-1-conv1",
      "block-0-res-2-conv1",
      "block-0-res-3-conv1",
      "block-1-res-1-conv1",
      "block-1-res-2-conv1",
      "block-1-res-3-conv1",
      "block-2-res-1-conv1",
      "block-2-res-2-conv1",
      "block-2-res-3-conv1",
      "block-3-res-2-conv1",
      "block-3-res-3-conv1",
      "block-4-res-2-conv1",
      "block-4-res-3-conv1",
    ]);
    expect(channelChunked.reduce(
      (sum, operation) => sum + operation.quantumCount,
      0,
    )).toBe(1_680);
    const portable = selections.filter(
      ({ selection }) => selection === "portable",
    );
    expect(portable.map(({ label }) => label)).toEqual([
      "block-0-res-1-conv2",
      "block-0-res-2-conv2",
      "block-0-res-3-conv2",
      "block-1-res-1-conv2",
      "block-1-res-2-conv2",
      "block-1-res-3-conv2",
      "block-2-res-1-conv2",
      "block-2-res-2-conv2",
      "block-2-res-3-conv2",
      "block-3-res-1-conv2",
      "block-3-res-2-conv2",
      "block-3-res-3-conv2",
      "block-4-res-1-conv2",
      "block-4-res-2-conv2",
      "block-4-res-3-conv2",
    ]);
    expect(portable.reduce(
      (sum, operation) => sum + operation.quantumCount,
      0,
    )).toBe(414);
    expect(selections.reduce(
      (sum, operation) => sum + operation.quantumCount,
      0,
    )).toBe(2_459);
    expect(selections).toHaveLength(32);
    expect(Object.isFrozen(selections)).toBe(true);
  });

  it("reports mixed per-quantum batch fallback instead of rejecting the graph", () => {
    const selections = summarizeAceVaeDecoderConv1dOperationSelection(
      planAceVaeDecoder(256, ACE_OOBLECK_DECODER_CONFIG, 2),
      {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupStorageSize: 16 * 1_024,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxStorageBufferBindingSize: 512 * 1_024 * 1_024,
      },
    );
    expect(selections.find(({ label }) => label === "conv2")).toEqual({
      label: "conv2",
      selection: "mixed",
      reason: "mixed-per-quantum",
      quantumCount: 8,
    });
  });

  it("pins the production non-causal 48 kHz stereo decoder geometry", () => {
    expect(ACE_OOBLECK_DECODER_CONFIG).toEqual({
      id: "ace-step-1.5-oobleck-decoder-v1",
      decoderInputChannels: 64,
      decoderChannels: 128,
      audioChannels: 2,
      channelMultiples: [1, 2, 4, 8, 16],
      downsamplingRatios: [2, 4, 4, 6, 10],
      sampleRateHz: 48_000,
    });
    const plan = planAceVaeDecoder(256);
    expect(plan).toMatchObject({
      inputFrames: 256,
      outputFrames: 491_520,
      hopLength: 1_920,
      maximumActivationElements: 62_914_560,
      workspaceBytes: 251_658_240,
      allWorkspaceBytes: 754_974_720,
      parameterBytes: 337_583_104,
      primitiveCount: 88,
    });
    expect(plan.requiredTensorNames).toHaveLength(145);
    expect(
      plan.operations
        .filter((operation) => operation.kind === "conv-transpose1d")
        .map((operation) => ({
          stride: operation.shape.stride,
          padding: operation.shape.padding,
          kernel: operation.shape.kernelSize,
        })),
    ).toEqual([
      { stride: 10, padding: 5, kernel: 20 },
      { stride: 6, padding: 3, kernel: 12 },
      { stride: 4, padding: 2, kernel: 8 },
      { stride: 4, padding: 2, kernel: 8 },
      { stride: 2, padding: 1, kernel: 4 },
    ]);
  });

  it("resolves exactly the canonical post-fusion package tensor names", () => {
    const plan = planAceVaeDecoder(1);
    expect(plan.requiredTensorNames).toContain("vae.decoder.conv1.weight");
    expect(plan.requiredTensorNames).toContain(
      "vae.decoder.block.0.conv_t1.weight",
    );
    expect(plan.requiredTensorNames).toContain(
      "vae.decoder.block.4.res_unit3.snake2.beta",
    );
    expect(plan.requiredTensorNames).toContain("vae.decoder.conv2.weight");
    expect(plan.requiredTensorNames).not.toContain("vae.decoder.conv2.bias");
    expect(
      plan.requiredTensorNames.some((name) =>
        name.endsWith(".weight_g") || name.endsWith(".weight_v")
      ),
    ).toBe(false);

    const fake = { size: 4 } as GPUBuffer;
    const canonical = Object.fromEntries(
      plan.requiredTensorNames.map((name) => [name, { buffer: fake }]),
    );
    const resolved = resolveAceVaeDecoderTensorBindings(plan, {
      ...canonical,
      "vae.encoder.must-not-leak": { buffer: fake },
    });
    expect(Object.keys(resolved)).toEqual(plan.requiredTensorNames);
    expect(Object.isFrozen(resolved)).toBe(true);
    expect(resolved["vae.decoder.conv1.weight"]).toEqual([
      { binding: { buffer: fake }, partStart: 0, partEnd: 1 },
    ]);

    const missing = { ...canonical };
    delete missing["vae.decoder.block.2.conv_t1.weight"];
    expect(() => resolveAceVaeDecoderTensorBindings(plan, missing)).toThrow(
      /missing VAE tensor vae\.decoder\.block\.2\.conv_t1\.weight/,
    );
  });

  it("requires contiguous output-axis parts for sharded ConvTranspose weights", () => {
    const plan = planAceVaeDecoder(1);
    const fake = { size: 4 } as GPUBuffer;
    const canonical: Record<string, GPUBufferBinding | readonly {
      readonly binding: GPUBufferBinding;
      readonly partStart: number;
      readonly partEnd: number;
    }[]> = Object.fromEntries(
      plan.requiredTensorNames.map((name) => [name, { buffer: fake }]),
    );
    canonical["vae.decoder.block.0.conv_t1.weight"] = [
      { binding: { buffer: fake }, partStart: 614, partEnd: 1_024 },
      { binding: { buffer: fake }, partStart: 0, partEnd: 614 },
    ];
    expect(resolveAceVaeDecoderTensorBindings(plan, canonical)[
      "vae.decoder.block.0.conv_t1.weight"
    ]).toEqual([
      { binding: { buffer: fake }, partStart: 0, partEnd: 614 },
      { binding: { buffer: fake }, partStart: 614, partEnd: 1_024 },
    ]);
    canonical["vae.decoder.block.0.conv_t1.weight"] = [
      { binding: { buffer: fake }, partStart: 0, partEnd: 613 },
      { binding: { buffer: fake }, partStart: 614, partEnd: 1_024 },
    ];
    expect(() => resolveAceVaeDecoderTensorBindings(plan, canonical)).toThrow(
      /cover logical axis 0 contiguously/,
    );
  });

  it("keeps every residual primitive explicit and uses three-slot liveness", () => {
    const toy: AceVaeDecoderConfig = {
      id: "toy-oobleck",
      decoderInputChannels: 1,
      decoderChannels: 1,
      audioChannels: 2,
      channelMultiples: [1],
      downsamplingRatios: [2],
      sampleRateHz: 48_000,
    };
    const plan = planAceVaeDecoder(3, toy);
    expect(plan.outputFrames).toBe(6);
    expect(plan.primitiveCount).toBe(20);
    expect(plan.maximumActivationElements).toBe(6);
    expect(plan.operations.map((operation) => operation.kind)).toEqual([
      "conv1d",
      "snake",
      "conv-transpose1d",
      "snake", "conv1d", "snake", "conv1d", "add",
      "snake", "conv1d", "snake", "conv1d", "add",
      "snake", "conv1d", "snake", "conv1d", "add",
      "snake",
      "conv1d",
    ]);
    for (const operation of plan.operations) {
      expect(operation.output).not.toBe(operation.input);
      if (operation.kind === "add") {
        expect(operation.output).not.toBe(operation.right);
      }
    }
  });

  it("covers every production operation exactly once in FIFO bounded quanta", () => {
    const graph = planAceVaeDecoder(256);
    const block0Weight = "vae.decoder.block.0.conv_t1.weight";
    const cooperative = planAceVaeDecoderQuanta(graph, {
      [block0Weight]: [
        { partStart: 0, partEnd: 614 },
        { partStart: 614, partEnd: 1_024 },
      ],
    });
    expect(cooperative.quantumWorkPolicy).toEqual(
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
    );
    expect(Object.isFrozen(cooperative.quantumWorkPolicy)).toBe(true);
    expect(cooperative.quantumCount).toBe(3_942);
    expect(cooperative.primitiveDispatchCount).toBe(3_988);
    expect(cooperative.quantumCount).toBe(cooperative.quanta.length);
    expect(cooperative.primitiveDispatchCount).toBe(
      cooperative.quanta.reduce(
        (total, quantum) => total + quantum.primitives.length,
        0,
      ),
    );
    expect(cooperative.quanta.every((quantum) =>
      quantum.logicalOutputCount <=
        ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY.maximumOutputElements &&
      quantum.estimatedMaximumMultiplyAccumulates <=
        ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY
          .maximumConvolutionMultiplyAccumulates
    )).toBe(true);
    expect([
      "conv1d",
      "conv-transpose1d",
      "snake",
      "add",
    ].map((kind) => cooperative.quanta.filter((quantum) =>
      quantum.operationKind === kind
    ).length)).toEqual([2_459, 322, 813, 348]);
    expect(cooperative.quanta.filter((quantum) =>
      quantum.operationKind === "snake" || quantum.operationKind === "add"
    ).every((quantum) =>
      quantum.estimatedMaximumMultiplyAccumulates === 0
    )).toBe(true);

    const controlRecords = cooperative.quanta.flatMap((quantum) =>
      quantum.primitives.map((primitive) => primitive.controlRecordIndex)
    );
    expect(controlRecords).toEqual(
      Array.from({ length: controlRecords.length }, (_, index) => index),
    );

    let previousOperation = -1;
    for (const [operationIndex, operation] of graph.operations.entries()) {
      const operationQuanta = cooperative.quanta.filter((quantum) =>
        quantum.operationIndex === operationIndex
      );
      expect(operationQuanta.length).toBeGreaterThan(0);
      expect(operationQuanta[0]!.logicalOutputBase).toBe(0);
      let cursor = 0;
      for (const quantum of operationQuanta) {
        expect(quantum.index).toBeGreaterThan(previousOperation);
        previousOperation = quantum.index;
        expect(quantum.operationLabel).toBe(operation.label);
        expect(quantum.operationKind).toBe(operation.kind);
        expect(quantum.logicalOutputBase).toBe(cursor);
        cursor += quantum.logicalOutputCount;
      }
      expect(cursor).toBe(operationOutputElements(operation));
    }

    const block0Quanta = cooperative.quanta.filter((quantum) =>
      quantum.operationLabel === "block-0-conv-t1"
    );
    expect(block0Quanta.length).toBeGreaterThan(1);
    for (const quantum of block0Quanta) {
      expect(quantum.primitives).toHaveLength(2);
      expect(quantum.primitives.map((primitive) => ({
        physicalPartIndex: primitive.physicalPartIndex,
        firstOutputChannel: primitive.firstOutputChannel,
        outputChannels: primitive.outputChannels,
      }))).toEqual([
        { physicalPartIndex: 0, firstOutputChannel: 0, outputChannels: 614 },
        { physicalPartIndex: 1, firstOutputChannel: 614, outputChannels: 410 },
      ]);
      const rowBase = quantum.logicalOutputBase / 1_024;
      const rowCount = quantum.logicalOutputCount / 1_024;
      for (const primitive of quantum.primitives) {
        expect(primitive.outputBase).toBe(rowBase * primitive.outputChannels);
        expect(primitive.outputCount).toBe(rowCount * primitive.outputChannels);
      }
      expect(
        quantum.primitives.reduce(
          (total, primitive) => total + primitive.outputCount,
          0,
        ),
      ).toBe(quantum.logicalOutputCount);
    }
  });

  it("fails closed for unsafe policies and malformed physical transpose grouping", () => {
    const graph = planAceVaeDecoder(1);
    expect(() => planAceVaeDecoderQuanta(graph, {}, {
      maximumConvolutionMultiplyAccumulates: 0,
      maximumOutputElements: 1,
    })).toThrow(/positive/);
    expect(() => planAceVaeDecoderQuanta(graph, {}, {
      maximumConvolutionMultiplyAccumulates: 1_000_000,
      maximumOutputElements: 1_023,
    })).toThrow(/cannot hold one complete block-0-conv-t1 output row/);
    expect(() => planAceVaeDecoderQuanta(graph, {
      "vae.decoder.block.0.conv_t1.weight": [
        { partStart: 0, partEnd: 613 },
        { partStart: 614, partEnd: 1_024 },
      ],
    })).toThrow(/contiguously in source order/);
    expect(() => planAceVaeDecoderQuanta(graph, {
      "vae.decoder.conv1.weight": [{ partStart: 0, partEnd: 2_048 }],
    })).toThrow(/non-transpose/);
  });

  it("uses the safe congruent-tap transpose bound for nontrivial dilation", () => {
    const production = planAceVaeDecoder(1);
    const template = production.operations.find((operation) =>
      operation.kind === "conv-transpose1d"
    );
    expect(template?.kind).toBe("conv-transpose1d");
    if (template?.kind !== "conv-transpose1d") throw new Error("missing transpose");
    const operation = Object.freeze({
      ...template,
      label: "gcd-transpose",
      shape: Object.freeze({
        batch: 1,
        inputFrames: 3,
        inputChannels: 3,
        outputChannels: 5,
        kernelSize: 7,
        stride: 4,
        dilation: 2,
        padding: 0,
        outputPadding: 0,
      }),
    });
    const graph = Object.freeze({
      ...production,
      operations: Object.freeze([operation]),
      primitiveCount: 1,
    });
    const cooperative = planAceVaeDecoderQuanta(graph, {}, {
      maximumConvolutionMultiplyAccumulates: 60,
      maximumOutputElements: 20,
    });
    expect(cooperative.quanta.every((quantum) =>
      quantum.logicalOutputCount === 5 &&
      quantum.estimatedMaximumMultiplyAccumulates === 60
    )).toBe(true);
  });

  it("falls back to every transpose tap when WGSL u32 arithmetic can wrap", () => {
    const production = planAceVaeDecoder(1);
    const template = production.operations.find((operation) =>
      operation.kind === "conv-transpose1d"
    );
    expect(template?.kind).toBe("conv-transpose1d");
    if (template?.kind !== "conv-transpose1d") throw new Error("missing transpose");
    const operation = Object.freeze({
      ...template,
      label: "wrapping-transpose",
      shape: Object.freeze({
        batch: 1,
        inputFrames: 1,
        inputChannels: 1,
        outputChannels: 1,
        kernelSize: 5,
        stride: 3,
        dilation: 0x8000_0000,
        padding: 0xffff_ffff,
        outputPadding: 0,
      }),
    });
    const graph = Object.freeze({
      ...production,
      operations: Object.freeze([operation]),
      primitiveCount: 1,
    });
    const cooperative = planAceVaeDecoderQuanta(graph, {}, {
      maximumConvolutionMultiplyAccumulates: 5,
      maximumOutputElements: 3,
    });
    expect(cooperative.quanta).toHaveLength(3);
    expect(cooperative.quanta.every((quantum) =>
      quantum.logicalOutputCount === 1 &&
      quantum.estimatedMaximumMultiplyAccumulates === 5
    )).toBe(true);
  });

  it("snapshots a caller-owned policy exactly once before planning", () => {
    const mutablePolicy = {
      maximumConvolutionMultiplyAccumulates: 234_881_024,
      maximumOutputElements: 32_768,
    };
    const snapshot = snapshotAceVaeDecoderQuantumWorkPolicy(mutablePolicy);
    mutablePolicy.maximumConvolutionMultiplyAccumulates = 1;
    mutablePolicy.maximumOutputElements = 1;
    expect(snapshot).toEqual({
      maximumConvolutionMultiplyAccumulates: 234_881_024,
      maximumOutputElements: 32_768,
    });
    expect(Object.isFrozen(snapshot)).toBe(true);
  });

  it("rejects odd ratios whose PyTorch transpose geometry does not multiply exactly", () => {
    expect(() => planAceVaeDecoder(3, {
      ...ACE_OOBLECK_DECODER_CONFIG,
      id: "invalid",
      channelMultiples: [1],
      downsamplingRatios: [3],
    })).toThrow(/even ratios/);
  });
});

function operationOutputElements(operation: AceVaeDecoderOperation): number {
  switch (operation.kind) {
    case "conv1d":
      return planAceVaeConv1d(operation.shape).outputElements;
    case "conv-transpose1d":
      return planAceVaeConvTranspose1d(operation.shape).outputElements;
    case "snake":
    case "add":
      return operation.shape.batch * operation.shape.frames * operation.shape.channels;
  }
}
