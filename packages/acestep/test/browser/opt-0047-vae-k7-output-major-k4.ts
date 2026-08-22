/// <reference types="@webgpu/types" />

import opt0024CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts?raw";
import candidateCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-output-major-k4.ts?raw";
import decoderCoreSource from "../../src/webgpu/vae-decoder.ts?raw";

import {
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID,
  AceOpt0047VaeConv1dOutputMajorK4Kernel,
  aceOpt0047VaeK7NativeWeightIndex,
  aceOpt0047VaeK7PackedWeightCoordinate,
  aceOpt0047VaeK7PackedWeightIndex,
  aceOpt0047VaeConv1dOutputMajorK4Wgsl,
  packAceOpt0047VaeK7WeightU16,
  unpackAceOpt0047VaeK7WeightU16,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-output-major-k4.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  aceOpt0024VaeConv1dDirectDot4SubgroupWgsl,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type { AceVaeConv1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";
import { planAceVaeDecoder } from "../../src/webgpu/vae-decoder.js";

declare global {
  interface Window {
    __ACE_OPT0047_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "opt0024" | "outputMajor";

export interface Opt0047CaseSpec {
  readonly id: string;
  readonly tier: "conv1" | "c1024" | "c512" | "c256" | "c128";
  readonly dilation: 1 | 3 | 9;
  readonly timingWeight: number;
  readonly shape: AceVaeConv1dShape;
}

export interface Opt0047TimingTierInput {
  readonly id: string;
  readonly weight: number;
  readonly opt0024SamplesMilliseconds: readonly number[];
  readonly outputMajorSamplesMilliseconds: readonly number[];
}

export interface Opt0047ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly protocol: "wait-30s-then-one-level0-check";
  readonly startedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: 1;
  readonly observedLevel: 0;
  readonly maximumObservationGapMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

interface EncodableDispatch {
  readonly kernelId: string;
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly totalBytes: number;
  readonly logicalElements: number;
}

interface PreparedCase {
  readonly spec: Opt0047CaseSpec;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly layoutProof: Readonly<Record<string, unknown>>;
  readonly arms: Readonly<Record<Arm, readonly EncodableDispatch[]>>;
  readonly outputs: Readonly<Record<Arm, GuardedOutput>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly cases: readonly PreparedCase[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly preparedAtEpochMilliseconds: number;
  readonly uncapturedErrors: string[];
  readonly deviceLosses: string[];
  readonly tracker: BufferTracker;
  readonly controlKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel;
  readonly candidateKernel: AceOpt0047VaeConv1dOutputMajorK4Kernel;
  destroy(): Readonly<Record<string, unknown>>;
}

interface OutputSnapshot {
  readonly logical: Uint16Array<ArrayBuffer>;
  readonly sha256: string;
  readonly qNaNPrefillCount: number;
  readonly nonFiniteCount: number;
  readonly classes: Readonly<{
    readonly zero: number;
    readonly subnormal: number;
    readonly normal: number;
    readonly nonFinite: number;
  }>;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
}

const FRAMES = 512;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const TIMING_REPEATS = 8;
const REQUIRED_WEIGHTED_SPEEDUP = 1.20;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 30_000;
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["opt0024", "outputMajor"] as const),
  Object.freeze(["outputMajor", "opt0024"] as const),
  Object.freeze(["outputMajor", "opt0024"] as const),
  Object.freeze(["opt0024", "outputMajor"] as const),
]);
const INPUT_PATTERN = new Uint16Array([
  0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
]);
const WEIGHT_PATTERN = new Uint16Array([
  0x1000, 0x9000, 0x1400, 0x9400, 0x1800, 0x9800, 0x1a00, 0x9a00,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400, 0x2800, 0xa800,
]);
const EXPECTED_OPT0024_CORE_SHA256 =
  "fe3bf8110cef1a3bb791006e9d376fe549e9f00fe30e4738d7429cb0daf65841";
const EXPECTED_CANDIDATE_CORE_SHA256 =
  "664a6786e12b6e03dab9857783ea29c7bd49f3ce55795dc918fdf20a229c30cc";
const EXPECTED_DECODER_CORE_SHA256 =
  "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e";
const EXPECTED_GENERATED_SHADER_AGGREGATE_SHA256 =
  "5506d0619347ae0f6ad5d4ee50af73e6ee69e90d19e5849bf6684d649da7f89c";

export function buildOpt0047Cases(): readonly Opt0047CaseSpec[] {
  const timing = new Map<string, number>([
    ["block-0-res-1-conv1", 282],
    ["block-1-res-2-conv1", 423],
    ["block-2-res-1-conv1", 423],
    ["block-4-res-3-conv1", 1_269],
  ]);
  const cases = planAceVaeDecoder(FRAMES).operations.flatMap((operation) => {
    if (operation.kind !== "conv1d" || operation.shape.kernelSize !== 7 ||
      operation.bias === undefined) return [];
    const dilation = operation.shape.dilation;
    if (dilation !== 1 && dilation !== 3 && dilation !== 9) {
      throw new Error(`OPT-0047 unexpected dilation in ${operation.label}`);
    }
    const channels = operation.shape.outputChannels;
    const tier = operation.label === "conv1"
      ? "conv1" as const
      : channels === 1_024
      ? "c1024" as const
      : channels === 512
      ? "c512" as const
      : channels === 256
      ? "c256" as const
      : channels === 128
      ? "c128" as const
      : undefined;
    if (tier === undefined) {
      throw new Error(`OPT-0047 unexpected tier in ${operation.label}`);
    }
    return [Object.freeze({
      id: operation.label,
      tier,
      dilation,
      timingWeight: timing.get(operation.label) ?? 0,
      shape: Object.freeze({ ...operation.shape, inputFrames: FRAMES }),
    })];
  });
  if (cases.length !== 16 ||
    cases.reduce((sum, spec) => sum + spec.timingWeight, 0) !== 2_397) {
    throw new Error("OPT-0047 production K7 topology changed");
  }
  return Object.freeze(cases);
}

export function buildOpt0047GeneratedShaderIdentityPayload(
  cases: readonly Opt0047CaseSpec[] = buildOpt0047Cases(),
): string {
  return cases.map((spec) => [
    spec.id,
    aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
      spec.shape,
      true,
      "float16",
    ),
    aceOpt0047VaeConv1dOutputMajorK4Wgsl(
      spec.shape,
      true,
      "float16",
    ),
  ].join("\0")).join("\0\n");
}

async function buildOpt0047Identity(
  cases: readonly Opt0047CaseSpec[],
): Promise<Readonly<Record<string, unknown>>> {
  const values = Object.freeze({
    opt0024CoreSha256: await sha256Text(opt0024CoreSource),
    candidateCoreSha256: await sha256Text(candidateCoreSource),
    decoderCoreSha256: await sha256Text(decoderCoreSource),
    generatedShaderAggregateSha256: await sha256Text(
      buildOpt0047GeneratedShaderIdentityPayload(cases),
    ),
  });
  const expected = Object.freeze({
    opt0024CoreSha256: EXPECTED_OPT0024_CORE_SHA256,
    candidateCoreSha256: EXPECTED_CANDIDATE_CORE_SHA256,
    decoderCoreSha256: EXPECTED_DECODER_CORE_SHA256,
    generatedShaderAggregateSha256:
      EXPECTED_GENERATED_SHADER_AGGREGATE_SHA256,
  });
  if (Object.entries(expected).some(([key, value]) =>
    values[key as keyof typeof values] !== value
  )) throw new Error("OPT-0047 static source identity changed");
  return Object.freeze({ ...values, expected, authenticated: true });
}

export function summarizeOpt0047Timing(
  tiers: readonly Opt0047TimingTierInput[],
): Readonly<Record<string, unknown>> {
  if (
    tiers.length !== 4 ||
    tiers.some((tier) =>
      tier.weight <= 0 || tier.opt0024SamplesMilliseconds.length !== 4 ||
      tier.outputMajorSamplesMilliseconds.length !== 4
    )
  ) throw new Error("OPT-0047 timing tier/sample topology changed");
  const summaries = tiers.map((tier) => {
    const controlMedian = median4(tier.opt0024SamplesMilliseconds);
    const candidateMedian = median4(
      tier.outputMajorSamplesMilliseconds,
    );
    return Object.freeze({
      id: tier.id,
      weight: tier.weight,
      opt0024SamplesMilliseconds: tier.opt0024SamplesMilliseconds,
      outputMajorSamplesMilliseconds:
        tier.outputMajorSamplesMilliseconds,
      opt0024MedianMilliseconds: controlMedian,
      outputMajorMedianMilliseconds: candidateMedian,
      speedup: controlMedian / candidateMedian,
      nonSlower: candidateMedian <= controlMedian,
    });
  });
  const weightedControl = summaries.reduce(
    (sum, tier) => sum + tier.weight * tier.opt0024MedianMilliseconds,
    0,
  );
  const weightedCandidate = summaries.reduce(
    (sum, tier) =>
      sum + tier.weight * tier.outputMajorMedianMilliseconds,
    0,
  );
  const weightedSpeedup = weightedControl / weightedCandidate;
  const everyTierNonSlower = summaries.every(({ nonSlower }) => nonSlower);
  const passed = everyTierNonSlower &&
    weightedSpeedup >= REQUIRED_WEIGHTED_SPEEDUP;
  return Object.freeze({
    tiers: Object.freeze(summaries),
    weightTotal: summaries.reduce((sum, tier) => sum + tier.weight, 0),
    weightedOpt0024Milliseconds: weightedControl,
    weightedOutputMajorK4Milliseconds: weightedCandidate,
    weightedSpeedup,
    everyTierNonSlower,
    requiredWeightedSpeedup: REQUIRED_WEIGHTED_SPEEDUP,
    passed,
    decision: passed
      ? "positive-primitive-qualifier"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0047ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0047ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const command = requiredParameter(parameters, "thermalCommand");
  const startedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalCheckedAtEpochMilliseconds",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalObservations",
  );
  const observedLevel = requiredFiniteParameter(
    parameters,
    "thermalObservedLevel",
  );
  const durationMilliseconds = checkedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const gapText = parameters.get("thermalMaximumObservationGapMilliseconds");
  const maximumObservationGapMilliseconds =
    gapText === null || gapText.trim() === ""
      ? durationMilliseconds
      : Number(gapText);
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    source !== THERMAL_SOURCE || command !== THERMAL_COMMAND ||
    observationCount !== 1 || observedLevel !== 0 ||
    !Number.isFinite(maximumObservationGapMilliseconds) ||
    maximumObservationGapMilliseconds !== durationMilliseconds ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    checkedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0047 requires one truthful level-0 notifyutil check after a 30-second wait",
    );
  }
  return Object.freeze({
    source: THERMAL_SOURCE,
    command: THERMAL_COMMAND,
    protocol: "wait-30s-then-one-level0-check",
    startedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount: 1,
    observedLevel: 0,
    maximumObservationGapMilliseconds,
    launchDelayMilliseconds,
  });
}

if (typeof document !== "undefined") install();

function install(): void {
  const progress = element<HTMLElement>("#progress");
  const run = element<HTMLButtonElement>("#run");
  const thermal = element<HTMLFieldSetElement>("#thermal-gate");
  let prepared: PreparedHarness | undefined;
  let started = false;
  void prepare((message) => progress.textContent = message).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — all 16 production K7 weights and C512 shapes are raw-U16 exact; timing has not run";
      thermal.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finish("failed", failure(error)),
  );
  run.addEventListener("click", () => {
    if (started || prepared === undefined) return;
    started = true;
    run.disabled = true;
    thermal.disabled = true;
    document.body.dataset.status = "running";
    const owned = prepared;
    prepared = undefined;
    void runTiming(owned, (message) => progress.textContent = message).then(
      (receipt) => finish(receipt.passed === true ? "passed" : "failed", receipt),
      (error: unknown) => {
        owned.destroy();
        finish("failed", failure(error));
      },
    );
  }, { once: true });
  window.addEventListener("beforeunload", () => {
    prepared?.destroy();
    prepared = undefined;
  });
}

async function prepare(
  update: (message: string) => void,
): Promise<PreparedHarness> {
  const specs = buildOpt0047Cases();
  update("authenticating OPT-0024, OPT-0047, and generated shader identities");
  const identity = await buildOpt0047Identity(specs);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const info = adapter.info;
  if (
    !adapter.features.has("shader-f16") || !adapter.features.has("subgroups") ||
    Number(info.subgroupMinSize) !== 32 || Number(info.subgroupMaxSize) !== 32
  ) throw new Error("OPT-0047 requires shader-f16 and fixed32 subgroups");
  const plans = specs.map(({ shape }) =>
    planAceFp16VaeConv1d(shape, "float16")
  );
  const requiredStorageBindingSize = Math.max(...plans.flatMap((plan) => [
    plan.inputBindingBytes,
    plan.weightBindingBytes,
    plan.outputBindingBytes,
  ]));
  const requiredBufferSize = Math.max(
    requiredStorageBindingSize,
    ...plans.map((plan) =>
      plan.outputBindingBytes + 2 * STORAGE_GUARD_BYTES
    ),
  );
  const device = await adapter.requestDevice({
    label: "ace-opt-0047-output-major-k4-gate",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: requiredBufferSize,
      maxStorageBufferBindingSize: requiredStorageBindingSize,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  void device.lost.then((loss) => {
    if (loss.reason !== "destroyed") deviceLosses.push(`${loss.reason}: ${loss.message}`);
  });
  const capability = Object.freeze({
    subgroupMinSize: Number(info.subgroupMinSize),
    subgroupMaxSize: Number(info.subgroupMaxSize),
  });
  const controlKernel =
    AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(device, capability);
  const candidateKernel =
    AceOpt0047VaeConv1dOutputMajorK4Kernel.create(device, capability);
  const cases: PreparedCase[] = [];
  try {
    for (const [index, spec] of specs.entries()) {
      update(`pack/compile ${index + 1}/16: ${spec.id}`);
      cases.push(await prepareCase(
        device,
        tracker,
        controlKernel,
        candidateKernel,
        spec,
        index,
      ));
      await browserYield();
    }
    let comparedU16Words = 0;
    let packedWeightU16Words = 0;
    const caseReceipts: Readonly<Record<string, unknown>>[] = [];
    for (const [index, preparedCase] of cases.entries()) {
      update(`exactness ${index + 1}/16: ${preparedCase.spec.id}`);
      const control0 = await runSnapshot(device, tracker, preparedCase, "opt0024");
      const candidate0 = await runSnapshot(
        device,
        tracker,
        preparedCase,
        "outputMajor",
      );
      const candidate1 = await runSnapshot(
        device,
        tracker,
        preparedCase,
        "outputMajor",
      );
      const control1 = await runSnapshot(device, tracker, preparedCase, "opt0024");
      requireValidSnapshot(control0, `${preparedCase.spec.id} control0`);
      requireValidSnapshot(control1, `${preparedCase.spec.id} control1`);
      requireValidSnapshot(candidate0, `${preparedCase.spec.id} candidate0`);
      requireValidSnapshot(candidate1, `${preparedCase.spec.id} candidate1`);
      const comparisons = [
        compareWords(control0.logical, candidate0.logical),
        compareWords(control1.logical, candidate1.logical),
        compareWords(control0.logical, control1.logical),
        compareWords(candidate0.logical, candidate1.logical),
      ];
      if (comparisons.some((mismatches) => mismatches !== 0)) {
        throw new Error(`${preparedCase.spec.id} raw-U16 identity failed`);
      }
      comparedU16Words += preparedCase.plan.outputElements * 4;
      caseReceipts.push(Object.freeze({
        id: preparedCase.spec.id,
        shape: preparedCase.spec.shape,
        c512BoundaryCoverage: Object.freeze({
          firstOutputTime: 0,
          lastOutputTime: FRAMES - 1,
          outputRowsCompared: FRAMES,
          includesLeftBoundary: true,
          includesInterior: true,
          includesRightBoundary: true,
        }),
        layoutProof: preparedCase.layoutProof,
        outputU16Words: preparedCase.plan.outputElements,
        comparedU16Words: preparedCase.plan.outputElements * 4,
        opt0024Sha256: Object.freeze([control0.sha256, control1.sha256]),
        candidateSha256: Object.freeze([candidate0.sha256, candidate1.sha256]),
        outputClasses: Object.freeze({
          opt0024: control0.classes,
          outputMajor: candidate0.classes,
        }),
        rawU16Exact: true,
        deterministicBothArms: true,
        completeFiniteCanarySafeWrites: true,
      }));
      packedWeightU16Words += preparedCase.plan.weightElements;
      await browserYield();
    }
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0047 GPU/device errors during preparation");
    }
    const preparedAtEpochMilliseconds = Date.now();
    let destroyed = false;
    return Object.freeze({
      adapter,
      device,
      cases: Object.freeze(cases),
      correctness: Object.freeze({
        caseCount: cases.length,
        productionBiasedK7OperationCount: 16,
        channelTiers: Object.freeze([1_024, 512, 256, 128]),
        dilations: Object.freeze([1, 3, 9]),
        fixedInputFramesPerPrimitive: FRAMES,
        packedWeightU16Words,
        exhaustivePackInverseMismatchCount: 0,
        executionsPerArmPerCase: 2,
        comparedU16Words,
        mismatchCount: 0,
        rawU16Exact: true,
        deterministicBothArms: true,
        qNaNPrefillCompleteWrites: true,
        allOutputsFinite: true,
        guardsUntouched: true,
        completedBeforeReady: true,
        cases: Object.freeze(caseReceipts),
      }),
      identity,
      preparedAtEpochMilliseconds,
      uncapturedErrors,
      deviceLosses,
      tracker,
      controlKernel,
      candidateKernel,
      destroy(): Readonly<Record<string, unknown>> {
        if (!destroyed) {
          destroyed = true;
          controlKernel.destroy();
          candidateKernel.destroy();
          tracker.destroyAll();
          device.destroy();
        }
        return Object.freeze({
          idempotent: true,
          createdBufferCount: tracker.createdCount,
          destroyedBufferCount: tracker.destroyedCount,
          liveBufferCount: tracker.liveCount,
          liveBytes: tracker.liveBytes,
          deviceDestroyed: true,
        });
      },
    });
  } catch (error) {
    controlKernel.destroy();
    candidateKernel.destroy();
    tracker.destroyAll();
    device.destroy();
    throw error;
  }
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  controlKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  candidateKernel: AceOpt0047VaeConv1dOutputMajorK4Kernel,
  spec: Opt0047CaseSpec,
  ordinal: number,
): Promise<PreparedCase> {
  const plan = planAceFp16VaeConv1d(spec.shape, "float16");
  const nativeWeightWords = deterministicWeightWords(
    plan.weightElements,
    0x0047_0001 ^ Math.imul(ordinal + 1, 0x9e37_79b9),
  );
  const packedWeightWords = packAceOpt0047VaeK7WeightU16(
    nativeWeightWords,
    plan.inputChannels,
    plan.outputChannels,
  );
  const inverseWeightWords = unpackAceOpt0047VaeK7WeightU16(
    packedWeightWords,
    plan.inputChannels,
    plan.outputChannels,
  );
  const packInverseMismatchCount = compareWords(
    nativeWeightWords,
    inverseWeightWords,
  );
  if (packInverseMismatchCount !== 0) {
    throw new Error(`${spec.id} native-to-packed-to-native U16 identity failed`);
  }
  const lastPackedIndex = plan.weightElements - 1;
  const firstCoordinate = aceOpt0047VaeK7PackedWeightCoordinate(
    plan.inputChannels,
    plan.outputChannels,
    0,
  );
  const lastCoordinate = aceOpt0047VaeK7PackedWeightCoordinate(
    plan.inputChannels,
    plan.outputChannels,
    lastPackedIndex,
  );
  const layoutProof = Object.freeze({
    layout:
      "k7-cin4-cout-tile128-lane32-output4-cin-element4",
    comparedU16Words: plan.weightElements,
    packInverseMismatchCount,
    firstCoordinate,
    firstNativeIndex: aceOpt0047VaeK7NativeWeightIndex(
      plan.inputChannels,
      plan.outputChannels,
      firstCoordinate,
    ),
    firstPackedIndex: aceOpt0047VaeK7PackedWeightIndex(
      plan.inputChannels,
      plan.outputChannels,
      firstCoordinate,
    ),
    lastCoordinate,
    lastNativeIndex: aceOpt0047VaeK7NativeWeightIndex(
      plan.inputChannels,
      plan.outputChannels,
      lastCoordinate,
    ),
    lastPackedIndex: aceOpt0047VaeK7PackedWeightIndex(
      plan.inputChannels,
      plan.outputChannels,
      lastCoordinate,
    ),
    packedOutsideTimedRegion: true,
  });
  const input = patternedBuffer(
    device,
    tracker,
    `${spec.id}-input`,
    plan.inputBindingBytes,
    INPUT_PATTERN,
    ordinal,
  );
  const nativeWeight = u16Buffer(
    device,
    tracker,
    `${spec.id}-weight-native-oki`,
    plan.weightBindingBytes,
    nativeWeightWords,
  );
  const packedWeight = u16Buffer(
    device,
    tracker,
    `${spec.id}-weight-output-major-k4`,
    plan.weightBindingBytes,
    packedWeightWords,
  );
  const bias = patternedBuffer(
    device,
    tracker,
    `${spec.id}-bias`,
    plan.biasBindingBytes,
    BIAS_PATTERN,
    ordinal,
  );
  const outputs = Object.freeze({
    opt0024: guardedOutput(device, tracker, `${spec.id}-opt0024-output`, plan),
    outputMajor: guardedOutput(
      device,
      tracker,
      `${spec.id}-candidate-output`,
      plan,
    ),
  });
  const shared = Object.freeze({
    input: binding(input, plan.inputBindingBytes),
    bias: binding(bias, plan.biasBindingBytes),
  });
  const fullControl = controlBuffer(
    device,
    tracker,
    `${spec.id}-full-control`,
    0,
    plan.outputElements,
  );
  const controlDispatch = await controlKernel.createDispatch(
    `${spec.id}-opt0024-full`,
    spec.shape,
    Object.freeze({
      ...shared,
      weight: binding(nativeWeight, plan.weightBindingBytes),
      output: outputs.opt0024.binding,
    }),
    "float16",
    { base: 0, count: plan.outputElements, control: binding(fullControl, 16) },
  );
  const candidateDispatch = await candidateKernel.createDispatch(
    `${spec.id}-output-major-k4-full`,
    spec.shape,
    Object.freeze({
      ...shared,
      weight: binding(packedWeight, plan.weightBindingBytes),
      output: outputs.outputMajor.binding,
    }),
    "float16",
    { base: 0, count: plan.outputElements, control: binding(fullControl, 16) },
  );
  if (
    controlDispatch.kernelId !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID ||
    candidateDispatch.kernelId !==
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID
  ) throw new Error(`${spec.id} dispatch-owner identity changed`);
  return Object.freeze({
    spec,
    plan,
    layoutProof,
    arms: Object.freeze({
      opt0024: Object.freeze([controlDispatch]),
      outputMajor: Object.freeze([candidateDispatch]),
    }),
    outputs,
  });
}

async function runSnapshot(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedCase,
  arm: Arm,
): Promise<OutputSnapshot> {
  prefillGuarded(device, prepared.outputs[arm]);
  const readback = tracker.create(device, {
    label: `${prepared.spec.id}-${arm}-readback`,
    size: prepared.outputs[arm].totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({
    label: `${prepared.spec.id}-${arm}-correctness`,
  });
  const pass = encoder.beginComputePass();
  for (const dispatch of prepared.arms[arm]) dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(
    prepared.outputs[arm].buffer,
    0,
    readback,
    0,
    prepared.outputs[arm].totalBytes,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange()).slice();
  readback.unmap();
  tracker.destroy(readback);
  const words32 = new Uint32Array(bytes.buffer);
  const guardWords = STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
  const logical = new Uint16Array(
    bytes.buffer.slice(
      STORAGE_GUARD_BYTES,
      STORAGE_GUARD_BYTES + prepared.plan.outputStorageBytes,
    ),
  );
  let qNaNPrefillCount = 0;
  const classes = { zero: 0, subnormal: 0, normal: 0, nonFinite: 0 };
  for (const word of logical) {
    if (word === OUTPUT_PREFILL_QNAN_F16) qNaNPrefillCount += 1;
    const exponent = word & 0x7c00;
    const fraction = word & 0x03ff;
    if (exponent === 0x7c00) classes.nonFinite += 1;
    else if (exponent !== 0) classes.normal += 1;
    else if (fraction !== 0) classes.subnormal += 1;
    else classes.zero += 1;
  }
  return Object.freeze({
    logical,
    sha256: await sha256(logical),
    qNaNPrefillCount,
    nonFiniteCount: classes.nonFinite,
    classes: Object.freeze(classes),
    prefixCanaryIntact: words32.slice(0, guardWords).every(
      (word) => word === STORAGE_GUARD_U32,
    ),
    suffixCanaryIntact: words32.slice(words32.length - guardWords).every(
      (word) => word === STORAGE_GUARD_U32,
    ),
  });
}

async function runTiming(
  prepared: PreparedHarness,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0047ThermalGate(
    formParameters(),
    prepared.preparedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const representatives = prepared.cases.filter(({ spec }) =>
    spec.timingWeight > 0
  );
  if (representatives.length !== 4) throw new Error("timing representatives changed");
  const samples = new Map<PreparedCase, Record<Arm, number[]>>();
  for (const candidate of representatives) {
    samples.set(candidate, { opt0024: [], outputMajor: [] });
  }
  for (const [round, order] of TIMING_ORDERS.entries()) {
    for (const candidate of representatives) {
      for (const arm of order) {
        update(`timing ${candidate.spec.id}, round ${round + 1}/4: ${arm}`);
        samples.get(candidate)![arm].push(
          await timeArm(prepared.device, candidate.arms[arm]),
        );
      }
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  if (
    prepared.uncapturedErrors.length !== 0 || prepared.deviceLosses.length !== 0
  ) throw new Error("OPT-0047 GPU/device error during timing");
  const timing = summarizeOpt0047Timing(representatives.map((candidate) => {
    const values = samples.get(candidate)!;
    return Object.freeze({
      id: candidate.spec.id,
      weight: candidate.spec.timingWeight,
      opt0024SamplesMilliseconds: Object.freeze(values.opt0024),
      outputMajorSamplesMilliseconds:
        Object.freeze(values.outputMajor),
    });
  }));
  const firstCleanup = prepared.destroy();
  const repeatedCleanup = prepared.destroy();
  if (firstCleanup["liveBufferCount"] !== 0 ||
    repeatedCleanup["liveBufferCount"] !== 0) {
    throw new Error("OPT-0047 cleanup left live GPU buffers");
  }
  const cleanup = Object.freeze({
    first: firstCleanup,
    repeated: repeatedCleanup,
    queueDrainedBeforeRelease: true,
    destroyIdempotenceExercised: true,
    zeroLiveResources: true,
  });
  const passed = timing["passed"] === true;
  return Object.freeze({
    schema: "ace-opt-0047-k7-output-major-k4-v1",
    experimentId: "OPT-0047",
    passed,
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      adapterInfo: prepared.adapter.info,
      stockWebGpuOnly: true,
      experimentalBrowserFlags: false,
      webNn: false,
    }),
    kernels: Object.freeze({
      control: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
      candidate: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID,
      opt0024ControlUnchanged: true,
    }),
    correctness: prepared.correctness,
    identity: prepared.identity,
    timing,
    protocol: Object.freeze({
      thermal,
      oneButtonAction: true,
      timingRepeatsPerSample: TIMING_REPEATS,
      samplesPerArmPerTier: 4,
      balancedOrder: TIMING_ORDERS,
      timedRetryPerformed: false,
    }),
    cleanup,
    disposition: passed
      ? "benchmark-only-positive-integration-not-authorized"
      : "benchmark-only-negative-stop",
    productionIntegrationAuthorized: false,
  });
}

async function timeArm(
  device: GPUDevice,
  dispatches: readonly EncodableDispatch[],
): Promise<number> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let repeat = 0; repeat < TIMING_REPEATS; repeat += 1) {
    for (const dispatch of dispatches) dispatch.encode(pass);
  }
  pass.end();
  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / TIMING_REPEATS;
}

class BufferTracker {
  private readonly buffers = new Map<GPUBuffer, number>();
  createdCount = 0;
  destroyedCount = 0;
  liveBytes = 0;

  get liveCount(): number {
    return this.buffers.size;
  }

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const bytes = Number(descriptor.size);
    this.buffers.set(buffer, bytes);
    this.createdCount += 1;
    this.liveBytes += bytes;
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    const bytes = this.buffers.get(buffer);
    if (bytes === undefined) return;
    buffer.destroy();
    this.buffers.delete(buffer);
    this.destroyedCount += 1;
    this.liveBytes -= bytes;
  }

  destroyAll(): void {
    for (const buffer of [...this.buffers.keys()]) this.destroy(buffer);
  }
}

function patternedBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  pattern: Uint16Array,
  offset: number,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const words = new Uint16Array(buffer.getMappedRange());
  for (let index = 0; index < words.length; index += 1) {
    words[index] = pattern[(index + offset) % pattern.length]!;
  }
  buffer.unmap();
  return buffer;
}

function deterministicWeightWords(length: number, seed: number): Uint16Array {
  const words = new Uint16Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < words.length; index += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    words[index] = WEIGHT_PATTERN[state & (WEIGHT_PATTERN.length - 1)]!;
  }
  return words;
}

function u16Buffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  payload: Uint16Array,
): GPUBuffer {
  if (payload.byteLength > bytes || bytes % 4 !== 0) {
    throw new RangeError(`${label} U16 payload does not fit its GPU buffer`);
  }
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint16Array(buffer.getMappedRange()).set(payload);
  buffer.unmap();
  return buffer;
}

function controlBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  base: number,
  count: number,
): GPUBuffer {
  const bytes = device.limits.minUniformBufferOffsetAlignment;
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set([base, count, 0, 0]);
  buffer.unmap();
  return buffer;
}

function guardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceFp16VaeConv1dPlan,
): GuardedOutput {
  const totalBytes = plan.outputBindingBytes + 2 * STORAGE_GUARD_BYTES;
  const buffer = tracker.create(device, {
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
      size: plan.outputBindingBytes,
    }),
    totalBytes,
    logicalElements: plan.outputElements,
  });
}

function prefillGuarded(device: GPUDevice, output: GuardedOutput): void {
  const bytes = new Uint8Array(output.totalBytes);
  new Uint32Array(bytes.buffer).fill(STORAGE_GUARD_U32);
  new Uint16Array(
    bytes.buffer,
    STORAGE_GUARD_BYTES,
    output.logicalElements,
  ).fill(OUTPUT_PREFILL_QNAN_F16);
  device.queue.writeBuffer(output.buffer, 0, bytes);
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function requireValidSnapshot(snapshot: OutputSnapshot, label: string): void {
  if (
    snapshot.qNaNPrefillCount !== 0 || snapshot.nonFiniteCount !== 0 ||
    !snapshot.prefixCanaryIntact || !snapshot.suffixCanaryIntact
  ) throw new Error(`${label} failed complete/finite/canary gate`);
}

function compareWords(left: Uint16Array, right: Uint16Array): number {
  if (left.length !== right.length) throw new Error("comparison length changed");
  let mismatches = 0;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) mismatches += 1;
  }
  return mismatches;
}

async function sha256(words: Uint16Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", words.slice().buffer);
  return hexadecimalDigest(digest);
}

async function sha256Text(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return hexadecimalDigest(digest);
}

function hexadecimalDigest(digest: ArrayBuffer): string {
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

function median4(values: readonly number[]): number {
  if (
    values.length !== 4 || values.some((value) =>
      !Number.isFinite(value) || value <= 0
    )
  ) throw new Error("OPT-0047 requires four positive finite samples");
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function formParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "") {
    throw new Error(`OPT-0047 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0047 thermal field ${name} is not finite`);
  }
  return value;
}

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (found === null) throw new Error(`missing ${selector}`);
  return found;
}

function finish(status: "passed" | "failed", receipt: Readonly<Record<string, unknown>>): void {
  window.__ACE_OPT0047_RESULT__ = receipt;
  document.body.dataset.status = status;
  element<HTMLElement>("#progress").textContent = status.toUpperCase();
  element<HTMLElement>("#result").textContent = JSON.stringify(receipt, null, 2);
}

function failure(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0047-k7-output-major-k4-v1",
    experimentId: "OPT-0047",
    passed: false,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
}

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
