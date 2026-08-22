/// <reference types="@webgpu/types" />

import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
  aceOpt0036VaeConvTranspose1dR4C8Wgsl,
  aceOpt0036VaeConvTranspose1dR8C4Wgsl,
  planAceOpt0036VaeConvTranspose1dR4C8,
  planAceOpt0036VaeConvTranspose1dR8C4,
  planAceOpt0036VaeConvTranspose1dRange,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-shape-selector.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
  aceOpt0048VaeConvTranspose1dK4Wgsl,
  packAceOpt0048VaeConvTranspose1dK4WeightU16,
  planAceOpt0048VaeConvTranspose1dK4,
  planAceOpt0048VaeConvTranspose1dK4Range,
  planAceOpt0048VaeConvTranspose1dK4Weight,
  unpackAceOpt0048VaeConvTranspose1dK4WeightU16,
  type AceOpt0048VaeConvTranspose1dK4WeightPlan,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  planAceFp16VaeConvTranspose1d,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

declare global {
  interface Window {
    __ACE_OPT0048_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "exact" | "candidate";

export interface Opt0048CaseSpec {
  readonly label:
    | "block-0-conv-t1"
    | "block-1-conv-t1"
    | "block-2-conv-t1"
    | "block-3-conv-t1"
    | "block-4-conv-t1";
  readonly reuseAxis: "channel" | "row";
  readonly shape: AceVaeConvTranspose1dShape;
}

export interface Opt0048ThermalGate {
  readonly command:
    "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly waitStartedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly waitDurationMilliseconds: number;
  readonly checkCount: 1;
  readonly thermalLevel: 0;
  readonly launchDelayMilliseconds: number;
}

interface CompiledArm {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly dispatch: readonly [number, number, number];
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly dispatch: readonly [number, number, number];
}

interface GuardedStorage {
  readonly label: string;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly payloadBytes: number;
}

interface PreparedCase {
  readonly spec: Opt0048CaseSpec;
  readonly outputElements: number;
  readonly arms: Readonly<Record<Arm, CompiledArm>>;
  readonly guarded: readonly GuardedStorage[];
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly cases: readonly PreparedCase[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly preparedAtEpochMilliseconds: number;
  readonly uncapturedErrors: string[];
}

interface NumericalAccumulator {
  count: number;
  finiteCount: number;
  controlNonFiniteCount: number;
  candidateNonFiniteCount: number;
  differingU16Count: number;
  deterministicMismatchCount: number;
  outputPrefillRemainingCount: number;
  signedZeroDifferenceCount: number;
  classChangeCount: number;
  finiteClassChangeCount: number;
  controlSum: number;
  candidateSum: number;
  controlSquareSum: number;
  candidateSquareSum: number;
  crossSum: number;
  errorSum: number;
  absoluteErrorSum: number;
  errorSquareSum: number;
  maximumAbsoluteControl: number;
  maximumAbsoluteError: number;
  maximumClassChangeMagnitude: number;
  firstDifference: Readonly<Record<string, unknown>> | null;
  tailControl: number;
  tailCandidate: number;
  tailRerun: number;
}

const EXPERIMENT_ID = "OPT-0048" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const MINIMUM_WAIT_MILLISECONDS = 30_000;
const MAXIMUM_WAIT_MILLISECONDS = 60_000;
const MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS = 5_000;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_CANARY = 0xa55a;
const OUTPUT_PREFILL = 0x7e55;
const READBACK_CHUNK_BYTES = 8 * 1_024 * 1_024;
const REQUIRED_SUMMED_SPEEDUP = 1.50;
const NRMSE_MAXIMUM = 0.02;
const SNR_DECIBELS_MINIMUM = 34;
const PEARSON_MINIMUM = 0.999;
const MAXIMUM_ABSOLUTE_ERROR = 0.25;
const MAXIMUM_CLASS_CHANGE_MAGNITUDE = 0.02;
const INPUT_PATTERN = new Uint16Array([
  0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
]);
const WEIGHT_PATTERN = new Uint16Array([
  0x1800, 0x9800, 0x1c00, 0x9c00, 0x2000, 0xa000, 0x2200, 0xa200,
  0x0000, 0x8000,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2800, 0xa800,
]);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["exact", "candidate"] as const),
  Object.freeze(["candidate", "exact"] as const),
  Object.freeze(["candidate", "exact"] as const),
  Object.freeze(["exact", "candidate"] as const),
  Object.freeze(["exact", "candidate"] as const),
  Object.freeze(["candidate", "exact"] as const),
]);

if (typeof document !== "undefined" && document.querySelector("#run") !== null) {
  installBrowserGate();
}

export function buildOpt0048Cases(): readonly Opt0048CaseSpec[] {
  return Object.freeze([
    operation("block-0-conv-t1", "channel", 300, 2_048, 1_024, 10),
    operation("block-1-conv-t1", "channel", 3_000, 1_024, 512, 6),
    operation("block-2-conv-t1", "channel", 18_000, 512, 256, 4),
    operation("block-3-conv-t1", "row", 72_000, 256, 128, 4),
    operation("block-4-conv-t1", "row", 288_000, 128, 128, 2),
  ]);
}

function installBrowserGate(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const fieldset = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const run = requireElement<HTMLButtonElement>("#run");
  let prepared: PreparedGate | undefined;
  let running: PreparedGate | undefined;
  void prepareGate((message) => progress.textContent = message).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — five complete production shapes passed pack, numerical, deterministic, tail, finite, and canary gates; begin one fresh 30-second idle wait";
      fieldset.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finishPage("failed", failureReceipt(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    run.disabled = true;
    fieldset.disabled = true;
    const owned = prepared;
    prepared = undefined;
    running = owned;
    const launchedAtEpochMilliseconds = Date.now();
    let thermal: Opt0048ThermalGate;
    try {
      thermal = parseOpt0048ThermalGate(
        fieldParameters("#thermal-gate"),
        owned.preparedAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
    } catch (error) {
      const cleanupReceipt = cleanup(owned);
      running = undefined;
      finishPage("failed", Object.freeze({
        ...failureReceipt(error),
        cleanup: cleanupReceipt,
      }));
      return;
    }
    document.body.dataset.status = "running";
    progress.textContent = "running six balanced exact/K4 permutations";
    void runTimedGate(owned, thermal, progress).then(
      (receipt) => {
        running = undefined;
        finishPage(receipt.passed === true ? "passed" : "failed", receipt);
      },
      (error: unknown) => {
        const cleanupReceipt = cleanup(owned);
        running = undefined;
        finishPage("failed", Object.freeze({
          ...failureReceipt(error),
          cleanup: cleanupReceipt,
        }));
      },
    );
  }, { once: true });
  window.addEventListener("beforeunload", () => {
    if (prepared !== undefined) {
      cleanup(prepared);
      prepared = undefined;
    }
    if (running !== undefined) {
      cleanup(running);
      running = undefined;
    }
  });
}

async function prepareGate(
  update: (message: string) => void,
): Promise<PreparedGate> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const requestedLimits = requiredDeviceLimits(adapter);
  const device = await adapter.requestDevice({
    label: "ace-opt-0048-convtranspose-k4-partials",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: requestedLimits,
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const preparedCases: PreparedCase[] = [];
  const started = performance.now();
  try {
    for (const [index, spec] of buildOpt0048Cases().entries()) {
      update(`preparing ${index + 1}/5 ${spec.label}`);
      preparedCases.push(await prepareCase(device, tracker, spec, index));
      await browserYield();
    }
    await device.queue.onSubmittedWorkDone();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`uncaptured GPU errors: ${uncapturedErrors.join("; ")}`);
    }
    const correctness = summarizeCorrectness(preparedCases, performance.now() - started);
    if (correctness["passed"] !== true) {
      throw new Error("OPT-0048 complete correctness envelope failed");
    }
    return Object.freeze({
      adapter,
      device,
      tracker,
      cases: Object.freeze(preparedCases),
      correctness,
      preparedAtEpochMilliseconds: Date.now(),
      uncapturedErrors,
    });
  } catch (error) {
    tracker.destroyAll();
    device.destroy();
    throw error;
  }
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0048CaseSpec,
  ordinal: number,
): Promise<PreparedCase> {
  const base = planAceFp16VaeConvTranspose1d(spec.shape);
  const candidatePlan = planAceOpt0048VaeConvTranspose1dK4(
    spec.label,
    spec.shape,
  );
  if (candidatePlan.reuseAxis !== spec.reuseAxis) {
    throw new Error(`${spec.label} changed OPT-0040 reuse-axis ownership`);
  }
  const exactPlan = spec.reuseAxis === "channel"
    ? planAceOpt0036VaeConvTranspose1dR4C8(spec.shape)
    : planAceOpt0036VaeConvTranspose1dR8C4(spec.shape);
  const fullRange = Object.freeze({ base: 0, count: base.outputElements });
  const exactRange = planAceOpt0036VaeConvTranspose1dRange(
    exactPlan,
    fullRange,
  );
  const candidateRange = planAceOpt0048VaeConvTranspose1dK4Range(
    candidatePlan,
    fullRange,
  );
  if (JSON.stringify(exactRange) !== JSON.stringify(candidateRange)) {
    throw new Error(`${spec.label} changed OPT-0040 output ownership`);
  }

  const input = createPatternedGuardedStorage(
    device,
    tracker,
    `${spec.label}-input`,
    base.inputBindingBytes,
    INPUT_PATTERN,
    ordinal,
  );
  const bias = createPatternedGuardedStorage(
    device,
    tracker,
    `${spec.label}-bias`,
    base.biasBindingBytes,
    BIAS_PATTERN,
    ordinal,
  );
  const logicalWeights = patternedWords(
    base.weightElements,
    WEIGHT_PATTERN,
    ordinal * 3,
  );
  const weightPlan = planAceOpt0048VaeConvTranspose1dK4Weight(
    spec.shape,
    spec.reuseAxis,
  );
  const packedWeights = packAceOpt0048VaeConvTranspose1dK4WeightU16(
    logicalWeights,
    weightPlan,
  );
  const packing = verifyPacking(logicalWeights, packedWeights, weightPlan);
  const exactWeight = createGuardedStorage(
    device,
    tracker,
    `${spec.label}-revision6-weight`,
    logicalWeights,
  );
  const candidateWeight = createGuardedStorage(
    device,
    tracker,
    `${spec.label}-opt0048-weight`,
    packedWeights,
  );
  const range = createRangeBuffer(
    device,
    tracker,
    `${spec.label}-range`,
    base.outputElements,
  );
  const exactCompiled = await compileKernel(
    device,
    `${spec.label}-opt0040-exact`,
    spec.reuseAxis === "channel"
      ? aceOpt0036VaeConvTranspose1dR4C8Wgsl(spec.shape)
      : aceOpt0036VaeConvTranspose1dR8C4Wgsl(spec.shape),
    [exactRange.workgroupsX, exactRange.workgroupsY, exactRange.workgroupsZ],
  );
  const candidateCompiled = await compileKernel(
    device,
    `${spec.label}-opt0048-k4`,
    aceOpt0048VaeConvTranspose1dK4Wgsl(spec.label, spec.shape),
    [
      candidateRange.workgroupsX,
      candidateRange.workgroupsY,
      candidateRange.workgroupsZ,
    ],
  );

  const exactOutput = createGuardedOutput(
    device,
    tracker,
    `${spec.label}-exact-output`,
    base.outputBindingBytes,
  );
  const candidateOutput = createGuardedOutput(
    device,
    tracker,
    `${spec.label}-candidate-output`,
    base.outputBindingBytes,
  );
  const rerunOutput = createGuardedOutput(
    device,
    tracker,
    `${spec.label}-candidate-rerun-output`,
    base.outputBindingBytes,
  );
  await executeArm(device, bindArm(
    device,
    `${spec.label}-exact-correctness`,
    exactCompiled,
    [input.binding, exactWeight.binding, bias.binding, exactOutput.binding, range],
  ));
  await executeArm(device, bindArm(
    device,
    `${spec.label}-candidate-correctness`,
    candidateCompiled,
    [input.binding, candidateWeight.binding, bias.binding, candidateOutput.binding, range],
  ));
  await executeArm(device, bindArm(
    device,
    `${spec.label}-candidate-rerun`,
    candidateCompiled,
    [input.binding, candidateWeight.binding, bias.binding, rerunOutput.binding, range],
  ));
  const numerical = await scanOutputs(
    device,
    tracker,
    exactOutput,
    candidateOutput,
    rerunOutput,
    base.outputElements,
  );
  const transientCanaries = await requireCanaries(
    device,
    tracker,
    [exactOutput, candidateOutput, rerunOutput],
  );
  tracker.destroy(exactOutput.buffer);
  tracker.destroy(candidateOutput.buffer);
  tracker.destroy(rerunOutput.buffer);

  const timingOutput = createGuardedOutput(
    device,
    tracker,
    `${spec.label}-shared-timing-output`,
    base.outputBindingBytes,
  );
  const guarded = Object.freeze([
    input,
    bias,
    exactWeight,
    candidateWeight,
    timingOutput,
  ]);
  const arms = Object.freeze({
    exact: bindArm(
      device,
      `${spec.label}-exact-timing`,
      exactCompiled,
      [input.binding, exactWeight.binding, bias.binding, timingOutput.binding, range],
    ),
    candidate: bindArm(
      device,
      `${spec.label}-candidate-timing`,
      candidateCompiled,
      [input.binding, candidateWeight.binding, bias.binding, timingOutput.binding, range],
    ),
  });
  await executeArm(device, arms.exact);
  await executeArm(device, arms.candidate);
  const retainedCanaries = await requireCanaries(device, tracker, guarded);
  const metrics = finishNumerical(numerical);
  const packingPassed = packing["exhaustiveForwardAndInverse"] === true;
  const correctness = Object.freeze({
    label: spec.label,
    reuseAxis: spec.reuseAxis,
    outputElements: base.outputElements,
    packing,
    numerical: metrics,
    deterministicRawU16: numerical.deterministicMismatchCount === 0,
    completeWrites: numerical.outputPrefillRemainingCount === 0,
    tailWritten: numerical.tailControl !== OUTPUT_PREFILL &&
      numerical.tailCandidate !== OUTPUT_PREFILL &&
      numerical.tailRerun !== OUTPUT_PREFILL,
    allFinite: numerical.controlNonFiniteCount === 0 &&
      numerical.candidateNonFiniteCount === 0,
    allCanariesIntact: transientCanaries && retainedCanaries,
    passed: packingPassed && metrics.passed &&
      numerical.deterministicMismatchCount === 0 &&
      numerical.outputPrefillRemainingCount === 0 &&
      numerical.tailControl !== OUTPUT_PREFILL &&
      numerical.tailCandidate !== OUTPUT_PREFILL &&
      numerical.tailRerun !== OUTPUT_PREFILL && transientCanaries &&
      retainedCanaries,
  });
  return Object.freeze({
    spec,
    outputElements: base.outputElements,
    arms,
    guarded,
    correctness,
  });
}

async function runTimedGate(
  prepared: PreparedGate,
  thermal: Opt0048ThermalGate,
  progress: HTMLElement,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Arm, number[]>>();
  for (const candidate of prepared.cases) {
    samples.set(candidate.spec.label, { exact: [], candidate: [] });
  }
  const started = performance.now();
  for (const [round, order] of TIMING_ORDERS.entries()) {
    const cases = round % 2 === 0
      ? prepared.cases
      : prepared.cases.slice().reverse();
    for (const [caseIndex, candidate] of cases.entries()) {
      for (const arm of order) {
        progress.textContent =
          `timing permutation ${round + 1}/${TIMING_ORDERS.length}, ` +
          `shape ${caseIndex + 1}/5, ${arm}`;
        samples.get(candidate.spec.label)![arm].push(
          await executeTimed(prepared.device, candidate.arms[arm]),
        );
      }
      await browserYield();
    }
  }
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error(
      `uncaptured GPU errors: ${prepared.uncapturedErrors.join("; ")}`,
    );
  }
  let allCanariesIntact = true;
  for (const candidate of prepared.cases) {
    allCanariesIntact = await requireCanaries(
      prepared.device,
      prepared.tracker,
      candidate.guarded,
    ) && allCanariesIntact;
  }
  const cases = prepared.cases.map((candidate) => {
    const values = samples.get(candidate.spec.label)!;
    const exact = median6(values.exact);
    const k4 = median6(values.candidate);
    return Object.freeze({
      label: candidate.spec.label,
      reuseAxis: candidate.spec.reuseAxis,
      samplesMilliseconds: Object.freeze({
        exact: Object.freeze(values.exact.slice()),
        candidate: Object.freeze(values.candidate.slice()),
      }),
      mediansMilliseconds: Object.freeze({ exact, candidate: k4 }),
      speedup: exact / k4,
      nonSlower: k4 <= exact,
    });
  });
  const summedMediansMilliseconds = Object.freeze({
    exact: cases.reduce((sum, entry) =>
      sum + entry.mediansMilliseconds.exact, 0),
    candidate: cases.reduce((sum, entry) =>
      sum + entry.mediansMilliseconds.candidate, 0),
  });
  const summedSpeedup = summedMediansMilliseconds.exact /
    summedMediansMilliseconds.candidate;
  const noSlowerEveryShape = cases.every(({ nonSlower }) => nonSlower);
  const timingPassed = noSlowerEveryShape &&
    summedSpeedup >= REQUIRED_SUMMED_SPEEDUP;
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const cleanupReceipt = cleanup(prepared);
  return Object.freeze({
    schema: "ace-opt-0048-vae-convtranspose-k4-partials-v1",
    experimentId: EXPERIMENT_ID,
    passed: prepared.correctness["passed"] === true && timingPassed &&
      allCanariesIntact,
    disposition: "benchmark-only",
    controlKernelId:
      ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
    exactRouteKernelIds: Object.freeze({
      channel: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
      row: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
    }),
    candidateKernelIds: Object.freeze({
      channel: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
      row: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
    }),
    candidateWeightLayout:
      ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
    packingOutsideTimedRegion: true,
    correctness: prepared.correctness,
    thermal,
    timing: Object.freeze({
      rounds: TIMING_ORDERS.length,
      orders: TIMING_ORDERS,
      orderContract:
        "six balanced A/B or B/A permutations; shape order alternates forward/reverse",
      sampleContract: "one submit and matching queue drain per sample",
      milliseconds: performance.now() - started,
      cases: Object.freeze(cases),
      summedMediansMilliseconds,
      summedSpeedup,
      requiredSummedSpeedup: REQUIRED_SUMMED_SPEEDUP,
      noSlowerEveryShape,
      allCanariesIntact,
      passed: timingPassed && allCanariesIntact,
    }),
    environment,
    cleanup: cleanupReceipt,
  });
}

function summarizeCorrectness(
  cases: readonly PreparedCase[],
  milliseconds: number,
): Readonly<Record<string, unknown>> {
  const passed = cases.length === 5 &&
    cases.every(({ correctness }) => correctness["passed"] === true);
  return Object.freeze({
    passed,
    operationCount: cases.length,
    outputElementsCompared: cases.reduce(
      (sum, candidate) => sum + candidate.outputElements,
      0,
    ),
    candidateComparisons: cases.reduce(
      (sum, candidate) => sum + candidate.outputElements * 2,
      0,
    ),
    cases: Object.freeze(cases.map(({ correctness }) => correctness)),
    deterministicRawU16: true,
    completeWrites: true,
    allCanariesIntact: true,
    allFinite: true,
    packingOutsideTimedRegion: true,
    milliseconds,
  });
}

async function scanOutputs(
  device: GPUDevice,
  tracker: BufferTracker,
  control: GuardedStorage,
  candidate: GuardedStorage,
  rerun: GuardedStorage,
  elements: number,
): Promise<NumericalAccumulator> {
  const accumulator = emptyNumericalAccumulator();
  const readBytes = Math.min(READBACK_CHUNK_BYTES, control.payloadBytes);
  const controlRead = tracker.create(device, {
    label: `${control.label}-chunk-readback`,
    size: readBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const candidateRead = tracker.create(device, {
    label: `${candidate.label}-chunk-readback`,
    size: readBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const rerunRead = tracker.create(device, {
    label: `${rerun.label}-chunk-readback`,
    size: readBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    for (let first = 0; first < elements;) {
      const count = Math.min(readBytes / 2, elements - first);
      const bytes = count * 2;
      const sourceOffset = STORAGE_GUARD_BYTES + first * 2;
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(control.buffer, sourceOffset, controlRead, 0, bytes);
      encoder.copyBufferToBuffer(
        candidate.buffer,
        sourceOffset,
        candidateRead,
        0,
        bytes,
      );
      encoder.copyBufferToBuffer(rerun.buffer, sourceOffset, rerunRead, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await Promise.all([
        tracker.mapRead(controlRead),
        tracker.mapRead(candidateRead),
        tracker.mapRead(rerunRead),
      ]);
      const controlWords = new Uint16Array(controlRead.getMappedRange(), 0, count);
      const candidateWords = new Uint16Array(
        candidateRead.getMappedRange(),
        0,
        count,
      );
      const rerunWords = new Uint16Array(rerunRead.getMappedRange(), 0, count);
      for (let local = 0; local < count; local += 1) {
        const index = first + local;
        accumulateWord(
          accumulator,
          controlWords[local]!,
          candidateWords[local]!,
          rerunWords[local]!,
          index,
        );
      }
      tracker.unmap(controlRead);
      tracker.unmap(candidateRead);
      tracker.unmap(rerunRead);
      first += count;
      await browserYield();
    }
  } finally {
    tracker.destroy(controlRead);
    tracker.destroy(candidateRead);
    tracker.destroy(rerunRead);
  }
  return accumulator;
}

function emptyNumericalAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    finiteCount: 0,
    controlNonFiniteCount: 0,
    candidateNonFiniteCount: 0,
    differingU16Count: 0,
    deterministicMismatchCount: 0,
    outputPrefillRemainingCount: 0,
    signedZeroDifferenceCount: 0,
    classChangeCount: 0,
    finiteClassChangeCount: 0,
    controlSum: 0,
    candidateSum: 0,
    controlSquareSum: 0,
    candidateSquareSum: 0,
    crossSum: 0,
    errorSum: 0,
    absoluteErrorSum: 0,
    errorSquareSum: 0,
    maximumAbsoluteControl: 0,
    maximumAbsoluteError: 0,
    maximumClassChangeMagnitude: 0,
    firstDifference: null,
    tailControl: OUTPUT_PREFILL,
    tailCandidate: OUTPUT_PREFILL,
    tailRerun: OUTPUT_PREFILL,
  };
}

function accumulateWord(
  result: NumericalAccumulator,
  controlBits: number,
  candidateBits: number,
  rerunBits: number,
  index: number,
): void {
  result.count += 1;
  if (controlBits === OUTPUT_PREFILL) result.outputPrefillRemainingCount += 1;
  if (candidateBits === OUTPUT_PREFILL) result.outputPrefillRemainingCount += 1;
  if (rerunBits === OUTPUT_PREFILL) result.outputPrefillRemainingCount += 1;
  if (candidateBits !== rerunBits) result.deterministicMismatchCount += 1;
  if (controlBits !== candidateBits) {
    result.differingU16Count += 1;
    if (result.firstDifference === null) {
      result.firstDifference = Object.freeze({
        index,
        controlU16: controlBits,
        candidateU16: candidateBits,
      });
    }
  }
  if ((controlBits & 0x7fff) === 0 && (candidateBits & 0x7fff) === 0 &&
    controlBits !== candidateBits) result.signedZeroDifferenceCount += 1;
  const control = f16ToF32(controlBits);
  const candidate = f16ToF32(candidateBits);
  if (f16Class(controlBits) !== f16Class(candidateBits)) {
    result.classChangeCount += 1;
    if (Number.isFinite(control) && Number.isFinite(candidate)) {
      result.maximumClassChangeMagnitude = Math.max(
        result.maximumClassChangeMagnitude,
        Math.abs(control),
        Math.abs(candidate),
      );
    }
  }
  const controlFinite = Number.isFinite(control);
  const candidateFinite = Number.isFinite(candidate);
  if (!controlFinite) result.controlNonFiniteCount += 1;
  if (!candidateFinite) result.candidateNonFiniteCount += 1;
  if (controlFinite !== candidateFinite) result.finiteClassChangeCount += 1;
  if (controlFinite && candidateFinite) {
    const error = candidate - control;
    const absoluteError = Math.abs(error);
    result.finiteCount += 1;
    result.controlSum += control;
    result.candidateSum += candidate;
    result.controlSquareSum += control * control;
    result.candidateSquareSum += candidate * candidate;
    result.crossSum += control * candidate;
    result.errorSum += error;
    result.absoluteErrorSum += absoluteError;
    result.errorSquareSum += error * error;
    result.maximumAbsoluteControl = Math.max(
      result.maximumAbsoluteControl,
      Math.abs(control),
    );
    result.maximumAbsoluteError = Math.max(
      result.maximumAbsoluteError,
      absoluteError,
    );
  }
  result.tailControl = controlBits;
  result.tailCandidate = candidateBits;
  result.tailRerun = rerunBits;
}

function finishNumerical(
  result: NumericalAccumulator,
): Readonly<Record<string, number | boolean | null | Readonly<Record<string, unknown>>>> {
  const count = Math.max(result.finiteCount, 1);
  const rmsError = Math.sqrt(result.errorSquareSum / count);
  const controlRms = Math.sqrt(result.controlSquareSum / count);
  const nrmse = rmsError / Math.max(controlRms, 1e-12);
  const snrDecibels = rmsError === 0
    ? Number.POSITIVE_INFINITY
    : 20 * Math.log10(controlRms / rmsError);
  const controlMean = result.controlSum / count;
  const candidateMean = result.candidateSum / count;
  const covariance = result.crossSum - count * controlMean * candidateMean;
  const controlVariance = result.controlSquareSum -
    count * controlMean * controlMean;
  const candidateVariance = result.candidateSquareSum -
    count * candidateMean * candidateMean;
  const denominator = Math.sqrt(Math.max(0, controlVariance) *
    Math.max(0, candidateVariance));
  const pearson = denominator === 0
    ? (result.differingU16Count === 0 ? 1 : 0)
    : covariance / denominator;
  const passed = result.controlNonFiniteCount === 0 &&
    result.candidateNonFiniteCount === 0 &&
    result.finiteClassChangeCount === 0 &&
    nrmse <= NRMSE_MAXIMUM && snrDecibels >= SNR_DECIBELS_MINIMUM &&
    pearson >= PEARSON_MINIMUM &&
    result.maximumAbsoluteError <= MAXIMUM_ABSOLUTE_ERROR &&
    result.maximumClassChangeMagnitude <= MAXIMUM_CLASS_CHANGE_MAGNITUDE;
  return Object.freeze({
    passed,
    count: result.count,
    finiteCount: result.finiteCount,
    differingU16Count: result.differingU16Count,
    deterministicMismatchCount: result.deterministicMismatchCount,
    outputPrefillRemainingCount: result.outputPrefillRemainingCount,
    controlNonFiniteCount: result.controlNonFiniteCount,
    candidateNonFiniteCount: result.candidateNonFiniteCount,
    signedZeroDifferenceCount: result.signedZeroDifferenceCount,
    classChangeCount: result.classChangeCount,
    finiteClassChangeCount: result.finiteClassChangeCount,
    meanError: result.errorSum / count,
    meanAbsoluteError: result.absoluteErrorSum / count,
    rmsError,
    nrmse,
    snrDecibels: Number.isFinite(snrDecibels) ? snrDecibels : 999,
    pearsonCorrelation: pearson,
    maximumAbsoluteControl: result.maximumAbsoluteControl,
    maximumAbsoluteError: result.maximumAbsoluteError,
    maximumClassChangeMagnitude: result.maximumClassChangeMagnitude,
    firstDifference: result.firstDifference,
    envelope: Object.freeze({
      nrmseMaximum: NRMSE_MAXIMUM,
      snrDecibelsMinimum: SNR_DECIBELS_MINIMUM,
      pearsonMinimum: PEARSON_MINIMUM,
      maximumAbsoluteError: MAXIMUM_ABSOLUTE_ERROR,
      maximumClassChangeMagnitude: MAXIMUM_CLASS_CHANGE_MAGNITUDE,
    }),
  });
}

function verifyPacking(
  logical: Uint16Array,
  packed: Uint16Array,
  plan: AceOpt0048VaeConvTranspose1dK4WeightPlan,
): Readonly<Record<string, number | boolean>> {
  const inverse = unpackAceOpt0048VaeConvTranspose1dK4WeightU16(packed, plan);
  let inverseMismatchCount = 0;
  for (let index = 0; index < logical.length; index += 1) {
    if (logical[index] !== inverse[index]) inverseMismatchCount += 1;
  }
  return Object.freeze({
    logicalWords: logical.length,
    packedWords: packed.length,
    physicalWordsVisited: packed.length,
    inverseWordsCompared: inverse.length,
    inverseMismatchCount,
    exhaustiveForwardAndInverse: inverseMismatchCount === 0 &&
      logical.length === packed.length,
  });
}

async function compileKernel(
  device: GPUDevice,
  label: string,
  code: string,
  dispatch: readonly [number, number, number],
): Promise<CompiledKernel> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(`${label} WGSL failed: ${errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`
    ).join("; ")}`);
  }
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, dispatch });
}

function bindArm(
  device: GPUDevice,
  label: string,
  compiled: CompiledKernel,
  bindings: readonly GPUBufferBinding[],
): CompiledArm {
  const bindGroup = device.createBindGroup({
    label: `${label}-bindings`,
    layout: compiled.pipeline.getBindGroupLayout(0),
    entries: bindings.map((resource, binding) => ({ binding, resource })),
  });
  return Object.freeze({ ...compiled, bindGroup });
}

async function executeArm(device: GPUDevice, arm: CompiledArm): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encodeArm(pass, arm);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeTimed(
  device: GPUDevice,
  arm: CompiledArm,
): Promise<number> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encodeArm(pass, arm);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function encodeArm(pass: GPUComputePassEncoder, arm: CompiledArm): void {
  pass.setPipeline(arm.pipeline);
  pass.setBindGroup(0, arm.bindGroup);
  pass.dispatchWorkgroups(...arm.dispatch);
}

function createPatternedGuardedStorage(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  payloadBytes: number,
  pattern: Uint16Array,
  patternOffset: number,
): GuardedStorage {
  return createGuardedStorage(
    device,
    tracker,
    label,
    patternedWords(payloadBytes / 2, pattern, patternOffset),
  );
}

function patternedWords(
  count: number,
  pattern: Uint16Array,
  offset: number,
): Uint16Array {
  const words = new Uint16Array(count);
  for (let index = 0; index < count; index += 1) {
    words[index] = pattern[(index + offset) % pattern.length]!;
  }
  return words;
}

function createGuardedStorage(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  payload: Uint16Array,
): GuardedStorage {
  const payloadBytes = payload.byteLength;
  if (payloadBytes % 4 !== 0) {
    throw new Error(`${label} payload is not four-byte aligned`);
  }
  const buffer = tracker.create(device, {
    label,
    size: payloadBytes + STORAGE_GUARD_BYTES * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint16Array(buffer.getMappedRange());
  words.fill(STORAGE_CANARY);
  words.set(payload, STORAGE_GUARD_BYTES / 2);
  tracker.unmap(buffer);
  return Object.freeze({
    label,
    buffer,
    binding: Object.freeze({
      buffer,
      offset: STORAGE_GUARD_BYTES,
      size: payloadBytes,
    }),
    payloadBytes,
  });
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  payloadBytes: number,
): GuardedStorage {
  const payload = new Uint16Array(payloadBytes / 2);
  payload.fill(OUTPUT_PREFILL);
  return createGuardedStorage(device, tracker, label, payload);
}

function createRangeBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  outputElements: number,
): GPUBufferBinding {
  const buffer = tracker.create(device, {
    label,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set([0, outputElements, 0, 0]);
  tracker.unmap(buffer);
  return Object.freeze({ buffer, offset: 0, size: 16 });
}

async function requireCanaries(
  device: GPUDevice,
  tracker: BufferTracker,
  guarded: readonly GuardedStorage[],
): Promise<boolean> {
  for (const storage of guarded) {
    const readback = tracker.create(device, {
      label: `${storage.label}-canary-readback`,
      size: STORAGE_GUARD_BYTES * 2,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      storage.buffer,
      0,
      readback,
      0,
      STORAGE_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      storage.buffer,
      STORAGE_GUARD_BYTES + storage.payloadBytes,
      readback,
      STORAGE_GUARD_BYTES,
      STORAGE_GUARD_BYTES,
    );
    device.queue.submit([encoder.finish()]);
    await tracker.mapRead(readback);
    const words = new Uint16Array(readback.getMappedRange());
    let intact = true;
    for (const word of words) {
      if (word !== STORAGE_CANARY) {
        intact = false;
        break;
      }
    }
    tracker.unmap(readback);
    tracker.destroy(readback);
    if (!intact) throw new Error(`${storage.label} canary changed`);
  }
  return true;
}

export function parseOpt0048ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0048ThermalGate {
  const command = requiredParameter(parameters, "thermalCommand");
  const waitStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "waitStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "checkedAtEpochMilliseconds",
  );
  const checkCount = requiredFiniteParameter(parameters, "checkCount");
  const thermalLevel = requiredFiniteParameter(parameters, "thermalLevel");
  const waitDurationMilliseconds = checkedAtEpochMilliseconds -
    waitStartedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    command !== THERMAL_COMMAND || checkCount !== 1 || thermalLevel !== 0 ||
    waitStartedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    waitDurationMilliseconds < MINIMUM_WAIT_MILLISECONDS ||
    waitDurationMilliseconds > MAXIMUM_WAIT_MILLISECONDS ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0048 requires exactly one level-0 notifyutil check after one fresh 30-second idle wait",
    );
  }
  return Object.freeze({
    command: THERMAL_COMMAND,
    waitStartedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    waitDurationMilliseconds,
    checkCount: 1,
    thermalLevel: 0,
    launchDelayMilliseconds,
  });
}

function requireAdapter(adapter: GPUAdapter): void {
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    Number(adapter.info.subgroupMinSize) !== 32 ||
    Number(adapter.info.subgroupMaxSize) !== 32
  ) {
    throw new Error("OPT-0048 requires shader-f16 and fixed32 subgroups");
  }
  if (Number(adapter.limits.minStorageBufferOffsetAlignment) >
    STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0048 storage guard is below adapter alignment");
  }
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
): Readonly<Record<string, number>> {
  return Object.freeze({
    maxBufferSize: Number(adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize:
      Number(adapter.limits.maxStorageBufferBindingSize),
    maxComputeWorkgroupsPerDimension:
      Number(adapter.limits.maxComputeWorkgroupsPerDimension),
  });
}

function environmentReceipt(
  adapter: GPUAdapter,
  device: GPUDevice,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    userAgent: navigator.userAgent,
    adapter: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    features: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new WeakMap<GPUBuffer, number>();
  createdBufferCount = 0;
  destroyedBufferCount = 0;
  mapCount = 0;
  unmapCount = 0;
  activeMapCount = 0;
  liveBytes = 0;
  maximumLiveBytes = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.createdBufferCount += 1;
    this.liveBytes += size;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    if (descriptor.mappedAtCreation === true) {
      this.mapCount += 1;
      this.activeMapCount += 1;
    }
    return buffer;
  }

  async mapRead(buffer: GPUBuffer): Promise<void> {
    await buffer.mapAsync(GPUMapMode.READ);
    this.mapCount += 1;
    this.activeMapCount += 1;
  }

  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") {
      throw new Error("OPT-0048 attempted an unbalanced unmap");
    }
    buffer.unmap();
    this.unmapCount += 1;
    this.activeMapCount -= 1;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    if (buffer.mapState === "mapped") this.unmap(buffer);
    buffer.destroy();
    this.destroyedBufferCount += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number | boolean>> {
    return Object.freeze({
      createdBufferCount: this.createdBufferCount,
      destroyedBufferCount: this.destroyedBufferCount,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes,
      maximumLiveBytes: this.maximumLiveBytes,
      mapCount: this.mapCount,
      unmapCount: this.unmapCount,
      activeMapCount: this.activeMapCount,
      mapsBalanced: this.mapCount === this.unmapCount &&
        this.activeMapCount === 0,
    });
  }
}

function cleanup(prepared: PreparedGate): Readonly<Record<string, unknown>> {
  prepared.tracker.destroyAll();
  prepared.device.destroy();
  return Object.freeze({
    ...prepared.tracker.receipt(),
    deviceDestroyed: true,
  });
}

function operation(
  label: Opt0048CaseSpec["label"],
  reuseAxis: Opt0048CaseSpec["reuseAxis"],
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  stride: number,
): Opt0048CaseSpec {
  return Object.freeze({
    label,
    reuseAxis,
    shape: Object.freeze({
      batch: 1,
      inputFrames,
      inputChannels,
      outputChannels,
      kernelSize: stride * 2,
      stride,
      dilation: 1,
      padding: stride / 2,
      outputPadding: 0,
    }),
  });
}

function median6(samples: readonly number[]): number {
  if (
    samples.length !== 6 ||
    samples.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("OPT-0048 requires six finite positive samples per arm");
  }
  const sorted = samples.slice().sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
}

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return fraction === 0 ? sign * 0 : sign * 2 ** -14 * fraction / 1_024;
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1_024);
}

function f16Class(bits: number): string {
  const magnitude = bits & 0x7fff;
  const negative = (bits & 0x8000) !== 0;
  if (magnitude === 0) return negative ? "negative-zero" : "positive-zero";
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0x1f) {
    if (fraction !== 0) return "nan";
    return negative ? "negative-infinity" : "positive-infinity";
  }
  return negative ? "negative-finite" : "positive-finite";
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0048 field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0048 field ${name} is invalid`);
  }
  return value;
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0048 element ${selector}`);
  return element;
}

function finishPage(
  state: "passed" | "failed",
  receipt: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = state;
  requireElement<HTMLElement>("#progress").textContent = state;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
  window.__ACE_OPT0048_RESULT__ = receipt;
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0048-vae-convtranspose-k4-partials-v1",
    experimentId: EXPERIMENT_ID,
    passed: false,
    error: error instanceof Error
      ? Object.freeze({
          name: error.name,
          message: error.message,
          stack: error.stack,
        })
      : String(error),
  });
}

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
