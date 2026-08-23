import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID,
  ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS,
  AceOpt0063VaeK1AddSnakeExactFusedKernel,
  aceOpt0063VaeK1AddSnakeExactFusedWgsl,
  aceOpt0063VaeK1PackedWeightIndex,
  planAceOpt0063VaeK1AddSnake,
  planAceOpt0063VaeK1AddSnakeOwner,
  type AceOpt0063VaeK1AddSnakeBindings,
} from "../src/webgpu/kernels/vae-k1-add-snake-exact-fused.js";
import {
  aceOpt0025VaeK1SubgroupGemmWgsl,
} from "../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  planAceVaeDecoder,
  type AceVaeDecoderConvOperation,
} from "../src/webgpu/vae-decoder.js";
import type { AceVaeConv1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";
import {
  buildOpt0063Cases,
  buildOpt0063TimingOrders,
  parseOpt0063ThermalGate,
  summarizeOpt0063Timing,
} from "./browser/opt-0063-vae-k1-add-snake-exact-fused.js";

describe("OPT-0063 exact K1/Add/Snake isolated fusion", () => {
  it("pins the eight-binding exact boundary contract", () => {
    const plan = planAceOpt0063VaeK1AddSnake(k1Shape(33, 128));
    expect(plan).toMatchObject({
      elements: 4_224,
      activationStorageBytes: 8_448,
      activationBindingBytes: 8_448,
      parameterStorageBytes: 256,
      parameterBindingBytes: 256,
      workgroupCount: 2,
      storageBindingCount: 8,
      formerBoundaries: [
        "producer-f32-to-f16",
        "add-f32-to-f16",
        "snake-f16-to-f32",
        "snake-f32-to-f16",
      ],
    });
    expect(ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS).toBe(8);
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.formerBoundaries)).toBe(true);
  });

  it.each([
    [33, 128],
    [65, 256],
  ] as const)(
    "gives every output exactly one owner for %i rows x %i channels",
    (frames, channels) => {
      const plan = planAceOpt0063VaeK1AddSnake(k1Shape(frames, channels));
      const owners = new Uint8Array(plan.elements);
      for (let workgroup = 0; workgroup < plan.workgroupCount; workgroup += 1) {
        for (let local = 0; local < 128; local += 1) {
          const owner = planAceOpt0063VaeK1AddSnakeOwner(
            plan,
            workgroup,
            local,
          );
          expect(owner.subgroup).toBe(Math.floor(local / 32));
          expect(owner.subgroupLane).toBe(local % 32);
          for (const index of owner.outputIndices) {
            expect(index).toBeGreaterThanOrEqual(0);
            expect(index).toBeLessThan(plan.elements);
            owners[index] = owners[index]! + 1;
          }
        }
      }
      expect(Array.from(owners).every((count) => count === 1)).toBe(true);
    },
  );

  it("exhaustively maps each packed K1 word once", () => {
    const plan = planAceOpt0063VaeK1AddSnake(k1Shape(1, 256));
    const packedOwners = new Uint8Array(plan.k1.weightElements);
    for (let inner = 0; inner < plan.k1.inner; inner += 1) {
      for (let column = 0; column < plan.k1.columns; column += 1) {
        const index = aceOpt0063VaeK1PackedWeightIndex(plan, inner, column);
        expect(index).toBeGreaterThanOrEqual(0);
        expect(index).toBeLessThan(plan.k1.weightElements);
        packedOwners[index] = packedOwners[index]! + 1;
      }
    }
    expect(Array.from(packedOwners).every((count) => count === 1)).toBe(true);
    expect(() => aceOpt0063VaeK1PackedWeightIndex(plan, -1, 0)).toThrow(
      /inner/,
    );
    expect(() => aceOpt0063VaeK1PackedWeightIndex(
      plan,
      0,
      plan.k1.columns,
    )).toThrow(/column/);
  });

  it("copies OPT-0025 accumulation and spells both eliminated f16 boundaries", () => {
    const shape = k1Shape(33, 128);
    const producer = aceOpt0025VaeK1SubgroupGemmWgsl(shape);
    const fused = aceOpt0063VaeK1AddSnakeExactFusedWgsl(shape);
    expect(ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID).toContain(
      "opt-0063",
    );
    for (const exactProducerLine of [
      "var acc0 = load_bias(column);",
      "inner_in_tile += 1u",
      "let a7 = subgroupBroadcast(lane_a, 7u);",
      "acc7 = acc7 + vec4<f32>(a7) * b;",
      "f32(input[lane_row * INNER + inner])",
      "f32(packed_weight[weight_base + 3u])",
    ]) {
      expect(producer).toContain(exactProducerLine);
      expect(fused).toContain(exactProducerLine);
    }
    expect(fused).toContain(
      "let producer_rounded: f16 = f16(producer_accumulator);",
    );
    expect(fused).toContain("let left_operand: f32 = f32(skip[index]);");
    expect(fused).toContain(
      "let right_operand: f32 = f32(producer_rounded);",
    );
    expect(fused).toContain("let sum: f32 = left_operand + right_operand;");
    expect(fused).toContain("let add_rounded: f16 = f16(sum);");
    expect(fused).toContain("add_output[index] = add_rounded;");
    expect(fused).toContain("let value: f32 = f32(add_rounded);");
    expect(fused).toContain("let alpha_value: f32 = exp(alpha_log_scale);");
    expect(fused).toContain("let beta_value: f32 = exp(beta_log_scale);");
    expect(fused).toContain("let periodic: f32 = sin(alpha_value * value);");
    expect(fused).toContain(
      "let reciprocal_beta: f32 = 1.0 / (beta_value + 1e-9);",
    );
    expect(fused).toContain(
      "value + reciprocal_beta * periodic * periodic;",
    );
    expect(fused).toContain("snake_output[index] = f16(result);");
    expect(fused.match(/@group\(0\) @binding\(/g)).toHaveLength(8);
    expect(fused).not.toContain("var<uniform>");
    expect(fused).not.toContain("var<workgroup>");

    const producerBoundary = fused.indexOf("producer_rounded: f16");
    const addBoundary = fused.indexOf("add_rounded: f16");
    const snakeLoad = fused.indexOf("value: f32 = f32(add_rounded)");
    const snakeStore = fused.indexOf("snake_output[index] = f16(result)");
    expect(producerBoundary).toBeLessThan(addBoundary);
    expect(addBoundary).toBeLessThan(snakeLoad);
    expect(snakeLoad).toBeLessThan(snakeStore);
  });

  it("proves all 15 rev7 graph chains are adjacent and shape congruent", () => {
    for (const inputFrames of [512, 2_314]) {
      const graph = planAceVaeDecoder(inputFrames);
      const k1Operations = graph.operations
        .map((operation, index) => ({ operation, index }))
        .filter((entry): entry is {
          readonly operation: AceVaeDecoderConvOperation;
          readonly index: number;
        } => entry.operation.kind === "conv1d" &&
          entry.operation.shape.kernelSize === 1);
      expect(k1Operations).toHaveLength(15);
      for (const { operation, index } of k1Operations) {
        const add = graph.operations[index + 1];
        const successor = graph.operations[index + 2];
        expect(add).toMatchObject({
          kind: "add",
          input: expect.any(String),
          right: operation.output,
          shape: {
            batch: operation.shape.batch,
            frames: operation.shape.inputFrames,
            channels: operation.shape.outputChannels,
          },
        });
        expect(successor).toMatchObject({
          kind: "snake",
          input: add?.output,
          shape: add?.shape,
        });
      }
    }
  });

  it("fails closed on unsupported geometry, dispatch size, limits, and aliases", async () => {
    expect(() => planAceOpt0063VaeK1AddSnake({
      ...k1Shape(33, 128),
      kernelSize: 7,
    })).toThrow(/K1/);
    expect(() => planAceOpt0063VaeK1AddSnake(
      k1Shape(65_535 * 32 + 1, 128),
    )).toThrow(/65,535/);

    const noBindings = fakeDevice({ maxStorageBuffersPerShaderStage: 7 });
    expect(() => AceOpt0063VaeK1AddSnakeExactFusedKernel.create(
      noBindings,
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    )).toThrow(/eight storage/);

    const device = fakeDevice();
    const kernel = AceOpt0063VaeK1AddSnakeExactFusedKernel.create(
      device,
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    );
    const plan = planAceOpt0063VaeK1AddSnake(k1Shape(33, 128));
    const bindings = bindingsFor(plan);
    bindings.snakeOutput = bindings.addOutput;
    await expect(kernel.createDispatch("aliases", plan.shape, bindings))
      .rejects.toThrow(/alias/);

    bindings.snakeOutput = binding(plan.activationBindingBytes);
    const dispatch = await kernel.createDispatch("valid", plan.shape, bindings);
    expect(dispatch.kernelId).toBe(
      ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID,
    );
    expect(device.createBindGroup).toHaveBeenCalledTimes(1);
    const descriptor = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect(Array.from(descriptor.entries)).toHaveLength(8);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch("destroyed", plan.shape, bindings))
      .rejects.toThrow(/destroyed/);
  });

  it("keeps the browser gate isolated from production routing", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0063-vae-k1-add-snake-exact-fused.html",
      import.meta.url,
    ), "utf8");
    const harness = readFileSync(new URL(
      "./browser/opt-0063-vae-k1-add-snake-exact-fused.ts",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("opt-0063-vae-k1-add-snake-exact-fused.ts");
    expect(harness).toContain("rawU16FormerAddBoundaryIdentity");
    expect(harness).toContain("rawU16FinalSnakeIdentity");
    expect(harness).toContain("requiredAffectedChainSpeedup: 1.15");
    expect(harness).not.toContain("vae-fp16-decoder");
    expect(harness).not.toContain("vae-fp16-profile");
  });

  it("pins ten bounded production ranges plus three adversarial fixtures", () => {
    const cases = buildOpt0063Cases();
    const production = cases.filter(({ kind }) => kind === "production-range");
    const adversarial = cases.filter(({ kind }) => kind !== "production-range");
    expect(production).toHaveLength(10);
    expect(adversarial.map(({ id, screenRows }) => ({ id, screenRows })))
      .toEqual([
        { id: "signed-zero-rne", screenRows: 33 },
        { id: "subnormal-cancellation", screenRows: 35 },
        { id: "transcendental", screenRows: 37 },
      ]);
    expect(production.map(({ geometry, logicalFrames, channels }) => ({
      geometry,
      logicalFrames,
      channels,
    }))).toEqual([
      { geometry: "c512", logicalFrames: 5_120, channels: 1_024 },
      { geometry: "c512", logicalFrames: 30_720, channels: 512 },
      { geometry: "c512", logicalFrames: 122_880, channels: 256 },
      { geometry: "c512", logicalFrames: 491_520, channels: 128 },
      { geometry: "c512", logicalFrames: 983_040, channels: 128 },
      { geometry: "c2314", logicalFrames: 23_140, channels: 1_024 },
      { geometry: "c2314", logicalFrames: 138_840, channels: 512 },
      { geometry: "c2314", logicalFrames: 555_360, channels: 256 },
      { geometry: "c2314", logicalFrames: 2_221_440, channels: 128 },
      { geometry: "c2314", logicalFrames: 4_442_880, channels: 128 },
    ]);
    expect(production.every(({ screenRows, channels }) =>
      screenRows * channels === 1_048_576
    )).toBe(true);
  });

  it("freezes ABBA, weighted C512/C2314 speedups, and one-check thermal input", () => {
    expect(buildOpt0063TimingOrders()).toEqual([
      ["unfused", "fused"],
      ["fused", "unfused"],
      ["fused", "unfused"],
      ["unfused", "fused"],
    ]);
    const timing = summarizeOpt0063Timing(
      buildOpt0063Cases()
        .filter(({ kind }) => kind === "production-range")
        .map(({ id }) => ({
          id,
          samples: {
            unfused: [12, 10, 14, 8],
            fused: [8, 6, 7, 5],
          },
        })),
    );
    expect(timing).toMatchObject({
      requiredAffectedChainSpeedup: 1.15,
      samplesPerArmPerCase: 4,
      aggregateSpeedup: 11 / 6.5,
      passed: true,
      geometry: {
        c512: { speedup: 11 / 6.5, passed: true },
        c2314: { speedup: 11 / 6.5, passed: true },
      },
    });

    const prepared = 1_000_000;
    expect(parseOpt0063ThermalGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "1000000",
      thermalCheckedAtEpochMilliseconds: "1030000",
      thermalObservations: "1",
      thermalObservedLevel: "0",
    }), prepared, 1_030_100)).toMatchObject({
      durationMilliseconds: 30_000,
      observationCount: 1,
      observedLevel: 0,
      launchDelayMilliseconds: 100,
    });
    expect(() => parseOpt0063ThermalGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "1000000",
      thermalCheckedAtEpochMilliseconds: "1029999",
      thermalObservations: "1",
      thermalObservedLevel: "0",
    }), prepared, 1_030_100)).toThrow(/30-second/);
  });
});

function k1Shape(frames: number, channels: number): AceVaeConv1dShape {
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

function fakeDevice(
  limitOverrides: Partial<GPUSupportedLimits> = {},
): GPUDevice & {
  createBindGroup: ReturnType<typeof vi.fn>;
} {
  const layout = {} as GPUBindGroupLayout;
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxStorageBuffersPerShaderStage: 8,
      minStorageBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1 << 30,
      ...limitOverrides,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => layout),
    })),
    createBindGroup: vi.fn(() => ({})),
  } as unknown as GPUDevice & {
    createBindGroup: ReturnType<typeof vi.fn>;
  };
}

function bindingsFor(
  plan: ReturnType<typeof planAceOpt0063VaeK1AddSnake>,
): AceOpt0063VaeK1AddSnakeBindings & {
  addOutput: GPUBufferBinding;
  snakeOutput: GPUBufferBinding;
} {
  return {
    input: binding(plan.k1.inputBytes),
    packedWeight: binding(plan.k1.weightBytes),
    bias: binding(plan.k1.biasBytes),
    skip: binding(plan.activationBindingBytes),
    alpha: binding(plan.parameterBindingBytes),
    beta: binding(plan.parameterBindingBytes),
    addOutput: binding(plan.activationBindingBytes),
    snakeOutput: binding(plan.activationBindingBytes),
  };
}

function binding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}
