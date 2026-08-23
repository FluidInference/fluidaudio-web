/// <reference types="@webgpu/types" />

import fusedCoreSource from
  "../../src/webgpu/kernels/vae-k1-add-snake-exact-fused.ts?raw";
import {
  AceOpt0063VaeK1AddSnakeExactFusedKernel,
  aceOpt0063VaeK1PackedWeightIndex,
  planAceOpt0063VaeK1AddSnake,
} from
  "../../src/webgpu/kernels/vae-k1-add-snake-exact-fused.js";
import {
  AceOpt0025VaeK1SubgroupGemmKernel,
} from "../../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import { AceFp16VaePointwiseKernel } from
  "../../src/webgpu/kernels/vae-pointwise-fp16.js";
import { AceFp16VaeSnakeKernel } from
  "../../src/webgpu/kernels/vae-snake-fp16.js";
import type { AceVaeConv1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

declare global {
  interface Window {
    __ACE_OPT0063_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0063Arm = "unfused" | "fused";
export type Opt0063Geometry = "c512" | "c2314";

export interface Opt0063CaseSpec {
  readonly id: string;
  readonly kind:
    | "production-range"
    | "signed-zero-rne"
    | "subnormal-cancellation"
    | "transcendental";
  readonly geometry?: Opt0063Geometry;
  readonly block?: 0 | 1 | 2 | 3 | 4;
  readonly logicalFrames: number;
  readonly firstLogicalRow: number;
  readonly screenRows: number;
  readonly channels: 128 | 256 | 512 | 1_024;
  readonly operationMultiplicity: 3 | 0;
  readonly shape: AceVaeConv1dShape;
}

export interface Opt0063TimingInput {
  readonly id: string;
  readonly samples: Readonly<Record<Opt0063Arm, readonly number[]>>;
}

export interface Opt0063ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly protocol: "wait-30s-then-one-level0-check";
  readonly startedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: 1;
  readonly observedLevel: 0;
  readonly launchDelayMilliseconds: number;
}

interface Encodable {
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly totalBytes: number;
  readonly logicalElements: number;
}

interface OutputSnapshot {
  readonly logical: Uint16Array<ArrayBuffer>;
  readonly sha256: string;
  readonly qNaNPrefillCount: number;
  readonly nonFiniteCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
}

interface PreparedCase {
  readonly spec: Opt0063CaseSpec;
  readonly arms: Readonly<Record<Opt0063Arm, readonly Encodable[]>>;
  readonly outputs: Readonly<Record<
    "unfusedAdd" | "unfusedSnake" | "fusedAdd" | "fusedSnake",
    GuardedOutput
  >>;
  readonly producer: GPUBuffer;
  readonly ownedBuffers: readonly GPUBuffer[];
  destroy(): void;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly cases: readonly PreparedCase[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly preparedAtEpochMilliseconds: number;
  readonly uncapturedErrors: readonly string[];
  destroy(): Readonly<Record<string, unknown>>;
}

const TARGET_SCREEN_ELEMENTS = 1_048_576;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_CANARY_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const REQUIRED_AFFECTED_CHAIN_SPEEDUP = 1.15;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 30_000;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["unfused", "fused"] as const),
  Object.freeze(["fused", "unfused"] as const),
  Object.freeze(["fused", "unfused"] as const),
  Object.freeze(["unfused", "fused"] as const),
]);
const PRODUCTION_INPUT = new Uint16Array([
  0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
]);
const PRODUCTION_WEIGHT = new Uint16Array([
  0x1000, 0x9000, 0x1400, 0x9400, 0x1800, 0x9800, 0x1a00, 0x9a00,
]);
const PRODUCTION_BIAS = new Uint16Array([
  0x0000, 0x8000, 0x1800, 0x9800, 0x1c00, 0x9c00, 0x2000, 0xa000,
]);
const PRODUCTION_SKIP = new Uint16Array([
  0x3000, 0xb000, 0x3400, 0xb400, 0x3800, 0xb800, 0x2c00, 0xac00,
]);
const ALPHA_PATTERN = new Uint16Array([
  0xbc00, 0xb800, 0x0000, 0x3800, 0x3c00, 0x4000, 0xc000, 0x4400,
]);
const BETA_PATTERN = new Uint16Array([
  0x3c00, 0x0000, 0xbc00, 0x4000, 0xc000, 0x3800, 0x4400, 0xc400,
]);
const ADVERSARIAL_VALUES = Object.freeze({
  "signed-zero-rne": new Uint16Array([
    0x0000, 0x8000, 0x3c00, 0xbc00, 0x3c01, 0xbc01, 0x1000, 0x9000,
  ]),
  "subnormal-cancellation": new Uint16Array([
    0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3555, 0xb555,
  ]),
  transcendental: new Uint16Array([
    0x3800, 0xb800, 0x3400, 0xb400, 0x3000, 0xb000, 0x3a00, 0xba00,
  ]),
});

export function buildOpt0063Cases(): readonly Opt0063CaseSpec[] {
  const geometries = Object.freeze([
    Object.freeze({
      id: "c512" as const,
      logical: Object.freeze([5_120, 30_720, 122_880, 491_520, 983_040]),
    }),
    Object.freeze({
      id: "c2314" as const,
      logical: Object.freeze([23_140, 138_840, 555_360, 2_221_440, 4_442_880]),
    }),
  ]);
  const channels = Object.freeze([1_024, 512, 256, 128, 128] as const);
  const production = geometries.flatMap(({ id, logical }) =>
    channels.map((channelCount, block) => {
      const logicalFrames = logical[block]!;
      const screenRows = TARGET_SCREEN_ELEMENTS / channelCount;
      const firstLogicalRow = id === "c512"
        ? Math.floor((logicalFrames - screenRows) / 2)
        : logicalFrames - screenRows;
      return Object.freeze({
        id: `${id}-block${block}-c${channelCount}`,
        kind: "production-range" as const,
        geometry: id,
        block: block as 0 | 1 | 2 | 3 | 4,
        logicalFrames,
        firstLogicalRow,
        screenRows,
        channels: channelCount,
        operationMultiplicity: 3 as const,
        shape: k1Shape(screenRows, channelCount),
      });
    })
  );
  const adversarial = Object.freeze([
    adversarialCase("signed-zero-rne", 33),
    adversarialCase("subnormal-cancellation", 35),
    adversarialCase("transcendental", 37),
  ]);
  if (
    production.length !== 10 ||
    production.some(({ screenRows, channels: channelCount }) =>
      screenRows * channelCount !== TARGET_SCREEN_ELEMENTS
    )
  ) {
    throw new Error("OPT-0063 production-range screen topology changed");
  }
  return Object.freeze([...production, ...adversarial]);
}

export function buildOpt0063TimingOrders(): typeof TIMING_ORDERS {
  return TIMING_ORDERS;
}

export function summarizeOpt0063Timing(
  inputs: readonly Opt0063TimingInput[],
): Readonly<Record<string, unknown>> {
  const specs = buildOpt0063Cases().filter(({ kind }) =>
    kind === "production-range"
  );
  if (
    inputs.length !== specs.length ||
    inputs.some((input, index) => input.id !== specs[index]!.id)
  ) {
    throw new Error("OPT-0063 timing inputs changed order or shape");
  }
  const cases = inputs.map((input, index) => {
    const spec = specs[index]!;
    const unfused = median4(input.samples.unfused);
    const fused = median4(input.samples.fused);
    const productionScale =
      spec.operationMultiplicity * spec.logicalFrames / spec.screenRows;
    return Object.freeze({
      id: spec.id,
      geometry: spec.geometry!,
      logicalFrames: spec.logicalFrames,
      firstLogicalRow: spec.firstLogicalRow,
      screenRows: spec.screenRows,
      channels: spec.channels,
      samples: input.samples,
      mediansMilliseconds: Object.freeze({ unfused, fused }),
      speedup: unfused / fused,
      productionScale,
      projectedMilliseconds: Object.freeze({
        unfused: unfused * productionScale,
        fused: fused * productionScale,
      }),
    });
  });
  const geometry = Object.freeze(Object.fromEntries(
    (["c512", "c2314"] as const).map((id) => {
      const selected = cases.filter(({ geometry: value }) => value === id);
      const unfused = selected.reduce(
        (sum, item) => sum + item.projectedMilliseconds.unfused,
        0,
      );
      const fused = selected.reduce(
        (sum, item) => sum + item.projectedMilliseconds.fused,
        0,
      );
      return [id, Object.freeze({
        projectedAffectedChainMilliseconds: Object.freeze({ unfused, fused }),
        speedup: unfused / fused,
        passed: unfused / fused >= REQUIRED_AFFECTED_CHAIN_SPEEDUP,
      })];
    }),
  ) as Record<Opt0063Geometry, Readonly<{
    projectedAffectedChainMilliseconds: Readonly<{
      unfused: number;
      fused: number;
    }>;
    speedup: number;
    passed: boolean;
  }>>);
  const aggregateUnfused = geometry.c512.projectedAffectedChainMilliseconds.unfused +
    geometry.c2314.projectedAffectedChainMilliseconds.unfused;
  const aggregateFused = geometry.c512.projectedAffectedChainMilliseconds.fused +
    geometry.c2314.projectedAffectedChainMilliseconds.fused;
  const aggregateSpeedup = aggregateUnfused / aggregateFused;
  return Object.freeze({
    samplesPerArmPerCase: 4,
    requiredAffectedChainSpeedup: 1.15,
    cases,
    geometry,
    aggregateSpeedup,
    passed: aggregateSpeedup >= REQUIRED_AFFECTED_CHAIN_SPEEDUP &&
      geometry.c512.passed && geometry.c2314.passed,
    decision: aggregateSpeedup >= REQUIRED_AFFECTED_CHAIN_SPEEDUP &&
        geometry.c512.passed && geometry.c2314.passed
      ? "positive-isolated-chain-screen"
      : "negative-stop-before-decoder-integration",
  });
}

export function parseOpt0063ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0063ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const startedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalCheckedAtEpochMilliseconds",
  );
  const observationCount = requiredNumber(parameters, "thermalObservations");
  const observedLevel = requiredNumber(parameters, "thermalObservedLevel");
  const durationMilliseconds = checkedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    source !== THERMAL_SOURCE || observationCount !== 1 || observedLevel !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    checkedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0063 requires one truthful level-0 notifyutil check after a 30-second wait",
    );
  }
  return Object.freeze({
    source: THERMAL_SOURCE,
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    protocol: "wait-30s-then-one-level0-check",
    startedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount: 1 as const,
    observedLevel: 0 as const,
    launchDelayMilliseconds,
  });
}

function adversarialCase(
  kind: Exclude<Opt0063CaseSpec["kind"], "production-range">,
  screenRows: number,
): Opt0063CaseSpec {
  return Object.freeze({
    id: kind,
    kind,
    logicalFrames: screenRows,
    firstLogicalRow: 0,
    screenRows,
    channels: 128,
    operationMultiplicity: 0,
    shape: k1Shape(screenRows, 128),
  });
}

function k1Shape(frames: number, channels: 128 | 256 | 512 | 1_024): AceVaeConv1dShape {
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

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private created = 0;
  private destroyed = 0;
  private peakBytes = 0;

  constructor(private readonly device: GPUDevice) {}

  create(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.peakBytes = Math.max(this.peakBytes, this.liveBytes());
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  snapshot(): Readonly<Record<string, number>> {
    return Object.freeze({
      createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes(),
      peakLiveBytes: this.peakBytes,
    });
  }

  private liveBytes(): number {
    let bytes = 0;
    for (const buffer of this.live) bytes += buffer.size;
    return bytes;
  }
}

async function prepareHarness(
  progress: HTMLElement,
): Promise<PreparedHarness> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter is available");
  for (const feature of ["shader-f16", "subgroups"] as const) {
    if (!adapter.features.has(feature)) {
      throw new Error(`OPT-0063 requires ${feature}`);
    }
  }
  const adapterInfo = adapter.info as GPUAdapterInfo & {
    readonly subgroupMinSize?: number;
    readonly subgroupMaxSize?: number;
  };
  if (adapterInfo.subgroupMinSize !== 32 || adapterInfo.subgroupMaxSize !== 32) {
    throw new Error("OPT-0063 requires fixed 32-lane subgroups");
  }
  const device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16", "subgroups"],
  });
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const tracker = new BufferTracker(device);
  const k1 = AceOpt0025VaeK1SubgroupGemmKernel.create(device, adapterInfo);
  const add = AceFp16VaePointwiseKernel.create(device);
  const snake = AceFp16VaeSnakeKernel.create(device);
  const fused = AceOpt0063VaeK1AddSnakeExactFusedKernel.create(
    device,
    adapterInfo,
  );
  const retainedCases: PreparedCase[] = [];
  const correctnessCases: Readonly<Record<string, unknown>>[] = [];
  let destroyed = false;
  try {
    const specs = buildOpt0063Cases();
    for (let index = 0; index < specs.length; index += 1) {
      const spec = specs[index]!;
      progress.textContent =
        `Correctness ${index + 1}/${specs.length}: ${spec.id}`;
      const prepared = await prepareCase(
        tracker,
        { k1, add, snake, fused },
        spec,
      );
      const correctness = await validatePreparedCase(device, tracker, prepared);
      correctnessCases.push(correctness);
      if (correctness.passed !== true) {
        prepared.destroy();
        throw new Error(`OPT-0063 ${spec.id} raw-U16 correctness failed`);
      }
      if (spec.kind === "production-range") {
        await runArm(device, prepared.arms.unfused);
        await runArm(device, prepared.arms.fused);
        retainedCases.push(prepared);
      } else {
        prepared.destroy();
      }
    }
    if (uncapturedErrors.length > 0) {
      throw new Error(`WebGPU errors: ${uncapturedErrors.join("; ")}`);
    }
    const identity = Object.freeze({
      experimentId: "OPT-0063",
      profile: "opt-0063-isolated-k1-add-snake-exact-screen-v1",
      fusedCoreSha256: await sha256Text(fusedCoreSource),
      baselineKernelIds: Object.freeze([
        "opt-0025-vae-k1-fp16-fixed32-subgroup-gemm-v1",
        "ace-vae-fp16-portable-add-v1",
        "ace-vae-fp16-portable-snake-v1",
      ]),
      fusedKernelId: "opt-0063-vae-k1-add-snake-exact-fused-v1",
      productionRuntimeSelectionPerformed: false,
      adapterInfo,
      adapterFeatures: Object.freeze([...adapter.features].sort()),
      adapterLimits: Object.freeze({
        maxStorageBuffersPerShaderStage:
          device.limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize:
          Number(device.limits.maxStorageBufferBindingSize),
        maxBufferSize: Number(device.limits.maxBufferSize),
        maxComputeInvocationsPerWorkgroup:
          device.limits.maxComputeInvocationsPerWorkgroup,
      }),
    });
    return Object.freeze({
      adapter,
      device,
      cases: Object.freeze(retainedCases),
      correctness: Object.freeze({
        rawU16FormerAddBoundaryIdentity: true,
        rawU16FinalSnakeIdentity: true,
        deterministicReruns: true,
        completeWritesAndCanaries: true,
        caseCount: correctnessCases.length,
        cases: Object.freeze(correctnessCases),
        passed: true,
      }),
      identity,
      preparedAtEpochMilliseconds: Date.now(),
      uncapturedErrors,
      destroy(): Readonly<Record<string, unknown>> {
        if (!destroyed) {
          destroyed = true;
          for (const prepared of retainedCases) prepared.destroy();
          k1.destroy();
          add.destroy();
          snake.destroy();
          fused.destroy();
          tracker.destroyAll();
          device.destroy();
        }
        return Object.freeze({
          ...tracker.snapshot(),
          deviceDestroyed: destroyed,
          idempotent: true,
        });
      },
    });
  } catch (error) {
    for (const prepared of retainedCases) prepared.destroy();
    k1.destroy();
    add.destroy();
    snake.destroy();
    fused.destroy();
    tracker.destroyAll();
    device.destroy();
    throw error;
  }
}

async function prepareCase(
  tracker: BufferTracker,
  kernels: Readonly<{
    k1: AceOpt0025VaeK1SubgroupGemmKernel;
    add: AceFp16VaePointwiseKernel;
    snake: AceFp16VaeSnakeKernel;
    fused: AceOpt0063VaeK1AddSnakeExactFusedKernel;
  }>,
  spec: Opt0063CaseSpec,
): Promise<PreparedCase> {
  const plan = planAceOpt0063VaeK1AddSnake(spec.shape);
  const fixture = buildFixture(spec, plan);
  const owned: GPUBuffer[] = [];
  const ownData = (label: string, data: Uint16Array): GPUBuffer => {
    const buffer = createDataBuffer(tracker, label, data, GPUBufferUsage.STORAGE);
    owned.push(buffer);
    return buffer;
  };
  const input = ownData(`${spec.id}-input`, fixture.input);
  const weight = ownData(`${spec.id}-packed-weight`, fixture.packedWeight);
  const bias = ownData(`${spec.id}-bias`, fixture.bias);
  const skip = ownData(`${spec.id}-skip`, fixture.skip);
  const alpha = ownData(`${spec.id}-alpha`, fixture.alpha);
  const beta = ownData(`${spec.id}-beta`, fixture.beta);
  const producer = tracker.create({
    label: `${spec.id}-unfused-k1-output`,
    size: plan.activationBindingBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  owned.push(producer);
  const outputs = Object.freeze({
    unfusedAdd: createGuardedOutput(tracker, `${spec.id}-unfused-add`, plan),
    unfusedSnake:
      createGuardedOutput(tracker, `${spec.id}-unfused-snake`, plan),
    fusedAdd: createGuardedOutput(tracker, `${spec.id}-fused-add`, plan),
    fusedSnake: createGuardedOutput(tracker, `${spec.id}-fused-snake`, plan),
  });
  owned.push(...Object.values(outputs).map(({ buffer }) => buffer));
  const rangeWords = new Uint32Array([0, plan.elements, 0, 0]);
  const rangeControl = createU32Buffer(
    tracker,
    `${spec.id}-range`,
    rangeWords,
    GPUBufferUsage.UNIFORM,
  );
  owned.push(rangeControl);
  const activation = (buffer: GPUBuffer): GPUBufferBinding => Object.freeze({
    buffer,
    size: plan.activationBindingBytes,
  });
  const parameter = (buffer: GPUBuffer): GPUBufferBinding => Object.freeze({
    buffer,
    size: plan.parameterBindingBytes,
  });
  const k1Dispatch = await kernels.k1.createDispatch(
    `${spec.id}-unfused-k1`,
    spec.shape,
    {
      input: { buffer: input, size: plan.k1.inputBytes },
      packedWeight: { buffer: weight, size: plan.k1.weightBytes },
      bias: { buffer: bias, size: plan.k1.biasBytes },
      output: activation(producer),
    },
  );
  const pointwiseShape = Object.freeze({
    batch: 1,
    frames: spec.screenRows,
    channels: spec.channels,
  });
  const range = Object.freeze({
    base: 0,
    count: plan.elements,
    control: Object.freeze({ buffer: rangeControl, size: 16 }),
  });
  const addDispatch = await kernels.add.createAddDispatch(
    `${spec.id}-unfused-add`,
    pointwiseShape,
    {
      left: activation(skip),
      right: activation(producer),
      output: outputs.unfusedAdd.binding,
    },
    range,
  );
  const snakeDispatch = await kernels.snake.createDispatch(
    `${spec.id}-unfused-snake`,
    pointwiseShape,
    {
      input: outputs.unfusedAdd.binding,
      alpha: parameter(alpha),
      beta: parameter(beta),
      output: outputs.unfusedSnake.binding,
    },
    range,
  );
  const fusedDispatch = await kernels.fused.createDispatch(
    `${spec.id}-fused`,
    spec.shape,
    {
      input: { buffer: input, size: plan.k1.inputBytes },
      packedWeight: { buffer: weight, size: plan.k1.weightBytes },
      bias: { buffer: bias, size: plan.k1.biasBytes },
      skip: activation(skip),
      alpha: parameter(alpha),
      beta: parameter(beta),
      addOutput: outputs.fusedAdd.binding,
      snakeOutput: outputs.fusedSnake.binding,
    },
  );
  let destroyed = false;
  return Object.freeze({
    spec,
    arms: Object.freeze({
      unfused: Object.freeze([k1Dispatch, addDispatch, snakeDispatch]),
      fused: Object.freeze([fusedDispatch]),
    }),
    outputs,
    producer,
    ownedBuffers: Object.freeze(owned),
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      for (const buffer of owned) tracker.destroy(buffer);
    },
  });
}

function buildFixture(
  spec: Opt0063CaseSpec,
  plan: ReturnType<typeof planAceOpt0063VaeK1AddSnake>,
): Readonly<{
  input: Uint16Array;
  packedWeight: Uint16Array;
  bias: Uint16Array;
  skip: Uint16Array;
  alpha: Uint16Array;
  beta: Uint16Array;
}> {
  const input = new Uint16Array(plan.elements);
  const packedWeight = new Uint16Array(plan.k1.weightElements);
  const bias = new Uint16Array(plan.k1.columns);
  const skip = new Uint16Array(plan.elements);
  const alpha = new Uint16Array(plan.k1.columns);
  const beta = new Uint16Array(plan.k1.columns);
  const seed = spec.firstLogicalRow + spec.channels + (spec.block ?? 0) * 17;
  if (spec.kind === "production-range") {
    fillPattern(input, PRODUCTION_INPUT, seed);
    fillPattern(packedWeight, PRODUCTION_WEIGHT, seed * 3);
    fillPattern(bias, PRODUCTION_BIAS, seed * 5);
    fillPattern(skip, PRODUCTION_SKIP, seed * 7);
  } else {
    const values = ADVERSARIAL_VALUES[spec.kind];
    fillPattern(input, values, seed);
    fillPattern(skip, values, seed + 3);
    for (let channel = 0; channel < plan.k1.columns; channel += 1) {
      packedWeight[
        aceOpt0063VaeK1PackedWeightIndex(plan, channel, channel)
      ] = 0x3c00;
      bias[channel] = channel % 2 === 0 ? 0x0000 : 0x8000;
    }
  }
  fillPattern(alpha, spec.kind === "transcendental"
    ? new Uint16Array([0x4500, 0xc500, 0x4000, 0xc000])
    : ALPHA_PATTERN, seed * 11);
  fillPattern(beta, spec.kind === "transcendental"
    ? new Uint16Array([0x4000, 0x3c00, 0xc000, 0x4400])
    : BETA_PATTERN, seed * 13);
  return Object.freeze({ input, packedWeight, bias, skip, alpha, beta });
}

async function validatePreparedCase(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedCase,
): Promise<Readonly<Record<string, unknown>>> {
  const executions: Readonly<Record<
    "unfusedAdd" | "unfusedSnake" | "fusedAdd" | "fusedSnake",
    OutputSnapshot
  >>[] = [];
  for (let repeat = 0; repeat < 2; repeat += 1) {
    const prefill = repeat === 0
      ? OUTPUT_PREFILL_QNAN_F16
      : OUTPUT_PREFILL_QNAN_F16 ^ 0x0100;
    for (const output of Object.values(prepared.outputs)) {
      prefillGuardedOutput(device, output, prefill);
    }
    const producerPrefill = new Uint16Array(prepared.producer.size / 2);
    producerPrefill.fill(prefill);
    device.queue.writeBuffer(prepared.producer, 0, producerPrefill);
    await runArm(device, prepared.arms.unfused);
    await runArm(device, prepared.arms.fused);
    executions.push(await snapshotOutputs(device, tracker, prepared.outputs, prefill));
  }
  const first = executions[0]!;
  const second = executions[1]!;
  const firstAddMismatches = mismatchCount(
    first.unfusedAdd.logical,
    first.fusedAdd.logical,
  );
  const secondAddMismatches = mismatchCount(
    second.unfusedAdd.logical,
    second.fusedAdd.logical,
  );
  const firstSnakeMismatches = mismatchCount(
    first.unfusedSnake.logical,
    first.fusedSnake.logical,
  );
  const secondSnakeMismatches = mismatchCount(
    second.unfusedSnake.logical,
    second.fusedSnake.logical,
  );
  const deterministicMismatches =
    mismatchCount(first.unfusedAdd.logical, second.unfusedAdd.logical) +
    mismatchCount(first.fusedAdd.logical, second.fusedAdd.logical) +
    mismatchCount(first.unfusedSnake.logical, second.unfusedSnake.logical) +
    mismatchCount(first.fusedSnake.logical, second.fusedSnake.logical);
  const snapshots = [...Object.values(first), ...Object.values(second)];
  const prefillCount = snapshots.reduce(
    (sum, snapshot) => sum + snapshot.qNaNPrefillCount,
    0,
  );
  const nonFiniteCount = snapshots.reduce(
    (sum, snapshot) => sum + snapshot.nonFiniteCount,
    0,
  );
  const canariesIntact = snapshots.every((snapshot) =>
    snapshot.prefixCanaryIntact && snapshot.suffixCanaryIntact
  );
  const passed = firstAddMismatches === 0 && secondAddMismatches === 0 &&
    firstSnakeMismatches === 0 && secondSnakeMismatches === 0 &&
    deterministicMismatches === 0 && prefillCount === 0 &&
    nonFiniteCount === 0 && canariesIntact;
  return Object.freeze({
    id: prepared.spec.id,
    kind: prepared.spec.kind,
    geometry: prepared.spec.geometry ?? null,
    block: prepared.spec.block ?? null,
    logicalFrames: prepared.spec.logicalFrames,
    firstLogicalRow: prepared.spec.firstLogicalRow,
    screenRows: prepared.spec.screenRows,
    channels: prepared.spec.channels,
    comparedWordsPerBoundaryPerExecution: prepared.outputs.fusedAdd.logicalElements,
    rawU16FormerAddBoundaryMismatchCount:
      firstAddMismatches + secondAddMismatches,
    rawU16FinalSnakeMismatchCount:
      firstSnakeMismatches + secondSnakeMismatches,
    deterministicMismatchCount: deterministicMismatches,
    qNaNPrefillCount: prefillCount,
    nonFiniteCount,
    canariesIntact,
    hashes: Object.freeze({
      first: Object.freeze(Object.fromEntries(Object.entries(first).map(
        ([name, snapshot]) => [name, snapshot.sha256],
      ))),
      second: Object.freeze(Object.fromEntries(Object.entries(second).map(
        ([name, snapshot]) => [name, snapshot.sha256],
      ))),
    }),
    passed,
  });
}

async function snapshotOutputs(
  device: GPUDevice,
  tracker: BufferTracker,
  outputs: PreparedCase["outputs"],
  prefill: number,
): Promise<Readonly<Record<
  "unfusedAdd" | "unfusedSnake" | "fusedAdd" | "fusedSnake",
  OutputSnapshot
>>> {
  const entries = Object.entries(outputs) as Array<[
    keyof PreparedCase["outputs"],
    GuardedOutput,
  ]>;
  const readbacks = entries.map(([name, output]) => [
    name,
    tracker.create({
      label: `${name}-readback`,
      size: output.totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  ] as const);
  const encoder = device.createCommandEncoder({ label: "opt-0063-readback" });
  for (let index = 0; index < entries.length; index += 1) {
    encoder.copyBufferToBuffer(
      entries[index]![1].buffer,
      0,
      readbacks[index]![1],
      0,
      entries[index]![1].totalBytes,
    );
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const result: Partial<Record<keyof PreparedCase["outputs"], OutputSnapshot>> = {};
  try {
    for (let index = 0; index < entries.length; index += 1) {
      const [name, output] = entries[index]!;
      const readback = readbacks[index]![1];
      await readback.mapAsync(GPUMapMode.READ);
      const detached = new Uint8Array(readback.getMappedRange()).slice();
      readback.unmap();
      const logical = new Uint16Array(
        detached.buffer,
        STORAGE_GUARD_BYTES,
        output.logicalElements,
      ).slice();
      result[name] = Object.freeze({
        logical,
        sha256: await sha256Bytes(
          new Uint8Array(logical.buffer, logical.byteOffset, logical.byteLength),
        ),
        qNaNPrefillCount: countWord(logical, prefill),
        nonFiniteCount: countNonFiniteF16(logical),
        prefixCanaryIntact: canaryIntact(
          detached,
          0,
          STORAGE_GUARD_BYTES,
        ),
        suffixCanaryIntact: canaryIntact(
          detached,
          output.totalBytes - STORAGE_GUARD_BYTES,
          output.totalBytes,
        ),
      });
    }
  } finally {
    for (const [, readback] of readbacks) tracker.destroy(readback);
  }
  return Object.freeze(result as Record<
    "unfusedAdd" | "unfusedSnake" | "fusedAdd" | "fusedSnake",
    OutputSnapshot
  >);
}

function createGuardedOutput(
  tracker: BufferTracker,
  label: string,
  plan: ReturnType<typeof planAceOpt0063VaeK1AddSnake>,
): GuardedOutput {
  const totalBytes = STORAGE_GUARD_BYTES + plan.activationBindingBytes +
    STORAGE_GUARD_BYTES;
  const buffer = tracker.create({
    label,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({
      buffer,
      offset: STORAGE_GUARD_BYTES,
      size: plan.activationBindingBytes,
    }),
    totalBytes,
    logicalElements: plan.elements,
  });
}

function prefillGuardedOutput(
  device: GPUDevice,
  output: GuardedOutput,
  prefill: number,
): void {
  const bytes = new Uint8Array(output.totalBytes);
  new Uint32Array(bytes.buffer).fill(STORAGE_CANARY_U32);
  new Uint16Array(
    bytes.buffer,
    STORAGE_GUARD_BYTES,
    output.logicalElements,
  ).fill(prefill);
  device.queue.writeBuffer(output.buffer, 0, bytes);
}

function createDataBuffer(
  tracker: BufferTracker,
  label: string,
  data: Uint16Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const size = align4(data.byteLength);
  const buffer = tracker.create({ label, size, usage, mappedAtCreation: true });
  new Uint16Array(buffer.getMappedRange(), 0, data.length).set(data);
  buffer.unmap();
  return buffer;
}

function createU32Buffer(
  tracker: BufferTracker,
  label: string,
  data: Uint32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = tracker.create({
    label,
    size: align4(data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange(), 0, data.length).set(data);
  buffer.unmap();
  return buffer;
}

async function runArm(
  device: GPUDevice,
  dispatches: readonly Encodable[],
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "opt-0063-execution" });
  const pass = encoder.beginComputePass({ label: "opt-0063-affected-chain" });
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function timeArm(
  device: GPUDevice,
  dispatches: readonly Encodable[],
): Promise<number> {
  const started = performance.now();
  await runArm(device, dispatches);
  return performance.now() - started;
}

async function runTimedGate(
  harness: PreparedHarness,
  thermal: Opt0063ThermalGate,
  progress: HTMLElement,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Opt0063Arm, number[]>>(
    harness.cases.map(({ spec }) => [
      spec.id,
      { unfused: [], fused: [] },
    ]),
  );
  const timedStartedAtEpochMilliseconds = Date.now();
  for (let round = 0; round < TIMING_ORDERS.length; round += 1) {
    const order = TIMING_ORDERS[round]!;
    for (let caseIndex = 0; caseIndex < harness.cases.length; caseIndex += 1) {
      const prepared = harness.cases[caseIndex]!;
      progress.textContent =
        `Timing round ${round + 1}/${TIMING_ORDERS.length}, ` +
        `${caseIndex + 1}/${harness.cases.length}: ${prepared.spec.id}`;
      for (const arm of order) {
        samples.get(prepared.spec.id)![arm].push(
          await timeArm(harness.device, prepared.arms[arm]),
        );
      }
    }
  }
  const timing = summarizeOpt0063Timing(harness.cases.map(({ spec }) => ({
    id: spec.id,
    samples: Object.freeze({
      unfused: Object.freeze(samples.get(spec.id)!.unfused.slice()),
      fused: Object.freeze(samples.get(spec.id)!.fused.slice()),
    }),
  })));
  const timedCompletedAtEpochMilliseconds = Date.now();
  if (harness.uncapturedErrors.length > 0) {
    throw new Error(`Timed WebGPU errors: ${harness.uncapturedErrors.join("; ")}`);
  }
  return Object.freeze({
    schemaVersion: 1,
    experimentId: "OPT-0063",
    status: timing.passed === true ? "passed" : "failed",
    identity: harness.identity,
    correctness: harness.correctness,
    protocol: Object.freeze({
      compilationAllocationUploadExcludedFromTiming: true,
      oneCommandBufferAndDrainPerAffectedChainArm: true,
      balancedOrder: TIMING_ORDERS,
      warmupPerArmPerCase: 1,
      thermal,
      timedStartedAtEpochMilliseconds,
      timedCompletedAtEpochMilliseconds,
    }),
    timing,
    evidence: Object.freeze({
      conclusion: timing.passed === true ? "positive" : "negative",
      productionIntegrationPerformed: false,
      decoderProfileChanged: false,
      packageChanged: false,
      nextGate: timing.passed === true
        ? "joint-revision7-c512-c2314-decoder-screen"
        : "stop-opt-0063-before-decoder-integration",
    }),
  });
}

function fillPattern(
  target: Uint16Array,
  pattern: Uint16Array,
  seed: number,
): void {
  for (let index = 0; index < target.length; index += 1) {
    target[index] = pattern[(index + seed) % pattern.length]!;
  }
}

function mismatchCount(left: Uint16Array, right: Uint16Array): number {
  if (left.length !== right.length) throw new Error("U16 comparison length changed");
  let mismatches = 0;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) mismatches += 1;
  }
  return mismatches;
}

function countWord(words: Uint16Array, expected: number): number {
  let count = 0;
  for (const word of words) if (word === expected) count += 1;
  return count;
}

function countNonFiniteF16(words: Uint16Array): number {
  let count = 0;
  for (const word of words) if ((word & 0x7c00) === 0x7c00) count += 1;
  return count;
}

function canaryIntact(bytes: Uint8Array, start: number, end: number): boolean {
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset + start, (end - start) / 4);
  return Array.from(words).every((word) => word === STORAGE_CANARY_U32);
}

function median4(values: readonly number[]): number {
  if (
    values.length !== 4 ||
    values.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("OPT-0063 timing requires four positive finite samples");
  }
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function align4(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  const owned = Uint8Array.from(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned.buffer);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

function thermalParameters(fieldset: HTMLFieldSetElement): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of fieldset.querySelectorAll<HTMLInputElement>("input[name]")) {
    parameters.set(input.name, input.value);
  }
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0063 thermal field ${name} is missing`);
  }
  return value;
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0063 thermal field ${name} is not finite`);
  }
  return value;
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

if (typeof document !== "undefined") installBrowserGate();

function installBrowserGate(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const result = requireElement<HTMLElement>("#result");
  const run = requireElement<HTMLButtonElement>("#run");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let owned: PreparedHarness | undefined;

  void prepareHarness(progress).then((harness) => {
    owned = harness;
    document.body.dataset.status = "ready";
    progress.textContent =
      "Correctness passed and both arms are warm. Wait 30 s, record one level-0 check, then run once.";
    thermalGate.disabled = false;
    run.disabled = false;
  }).catch((error: unknown) => {
    document.body.dataset.status = "failed";
    progress.textContent = "Preparation failed";
    result.textContent = String(
      error instanceof Error ? error.stack ?? error.message : error,
    );
  });

  run.addEventListener("click", () => {
    if (owned === undefined) return;
    run.disabled = true;
    thermalGate.disabled = true;
    const launchedAtEpochMilliseconds = Date.now();
    void (async () => {
      const thermal = parseOpt0063ThermalGate(
        thermalParameters(thermalGate),
        owned!.preparedAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
      const receipt = await runTimedGate(owned!, thermal, progress);
      const cleanup = owned!.destroy();
      owned = undefined;
      const complete = Object.freeze({ ...receipt, cleanup });
      window.__ACE_OPT0063_RESULT__ = complete;
      result.textContent = JSON.stringify(complete, null, 2);
      document.body.dataset.status = receipt.status === "passed"
        ? "passed"
        : "failed";
      progress.textContent = `OPT-0063 ${String(receipt.status)}`;
    })().catch((error: unknown) => {
      const cleanup = owned?.destroy();
      owned = undefined;
      document.body.dataset.status = "failed";
      progress.textContent = "Timed gate failed";
      result.textContent = JSON.stringify({
        error: String(error instanceof Error ? error.stack ?? error.message : error),
        cleanup,
      }, null, 2);
    });
  });

  window.addEventListener("beforeunload", () => {
    owned?.destroy();
    owned = undefined;
  });
}
