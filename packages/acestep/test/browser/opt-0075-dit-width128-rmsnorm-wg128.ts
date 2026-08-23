/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import rmsNormSource from "../../src/webgpu/kernels/rmsnorm.ts?raw";
import harnessHtmlSource from "./opt-0075-dit-width128-rmsnorm-wg128.html?raw";
import harnessSource from "./opt-0075-dit-width128-rmsnorm-wg128.ts?raw";
import {
  ACE_OPT_0075_WIDTH128_RMSNORM_KERNEL_ID,
  AceCorrectnessRmsNormKernel,
  AceOpt0075Width128RmsNormKernel,
  type AceRmsNormDispatch,
  type AceRmsNormShape,
} from "../../src/webgpu/kernels/rmsnorm.js";

declare global {
  interface Window {
    __ACE_OPT0075_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "current" | "candidate";
type FixtureKind = "full" | "signed-zero" | "boundary" |
  "alternating-magnitude" | "maximum-bf16-scale";

interface ShapeSpec {
  readonly id: "q-36000" | "k-18000" | "cross-cache-784";
  readonly rows: 36_000 | 18_000 | 784;
  readonly multiplicity: 2 | 1 | 0;
  readonly ordinal: number;
}

interface CaseSpec {
  readonly id: string;
  readonly shape: AceRmsNormShape;
  readonly kind: FixtureKind;
  readonly ordinal: number;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly readback: GPUBuffer;
  readonly prefill: Uint32Array<ArrayBuffer>;
  readonly outputElements: number;
  readonly outputWordOffset: number;
  readonly totalBytes: number;
}

interface Snapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly qNaNPrefillCount: number;
  readonly nonFiniteCount: number;
  readonly classes: Readonly<Record<string, number>>;
  readonly tailRowWritten: boolean;
}

interface PreparedShape {
  readonly spec: ShapeSpec;
  readonly dispatches: Readonly<Record<Arm, AceRmsNormDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceCorrectnessRmsNormKernel;
  readonly candidateKernel: AceOpt0075Width128RmsNormKernel;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  destroy(): Readonly<Record<string, unknown>>;
}

interface TimestampSample {
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly submitAtEpochMilliseconds: number;
  readonly fenceAtEpochMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly gpuElapsedNanoseconds: number;
  readonly gpuMilliseconds: number;
  readonly perDispatchGpuMilliseconds: number;
  readonly perDispatchWallMilliseconds: number;
  readonly gpuToWallRatio: number;
  readonly dispatchRepetitions: 8;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
  readonly timestampQueryPairCount: 1;
}

interface RawSample extends TimestampSample {
  readonly roundIndex: number;
  readonly shapePosition: number;
  readonly shapeId: ShapeSpec["id"];
  readonly armPosition: number;
  readonly arm: Arm;
}

interface TimingFailureEvidence {
  readonly rawSamples: RawSample[];
  runStartedAtEpochMilliseconds?: number;
  measurementCompletedAtEpochMilliseconds?: number;
}

const EXPERIMENT_ID = "OPT-0075" as const;
const RECEIPT_SCHEMA = "ace-opt-0075-width128-rmsnorm-wg128-v1";
const WIDTH = 128;
const EPSILON = 1e-6;
const GUARD_BYTES = 256;
const GUARD_WORD = 0xa55a_c33c;
const OUTPUT_QNAN_WORD = 0x7fc0_7555;
const QUERY_BYTES = 16;
const DISPATCH_REPETITIONS = 8 as const;
const REQUIRED_SPEEDUP = 1.25;
const REQUIRED_LAYER_MIX_SAVING_MILLISECONDS = 10.5;
const SHAPES = Object.freeze([
  shapeSpec("q-36000", 36_000, 2, 0),
  shapeSpec("k-18000", 18_000, 1, 1),
  shapeSpec("cross-cache-784", 784, 0, 2),
] as const);
const ADVERSARIAL_CASES = Object.freeze([
  caseSpec("signed-zero", 4, "signed-zero", 10),
  caseSpec("normal-subnormal-boundary", 4, "boundary", 11),
  caseSpec("alternating-magnitude", 4, "alternating-magnitude", 12),
  caseSpec("maximum-finite-bf16-scale", 4, "maximum-bf16-scale", 13),
] as const);
const TIMING_ROUNDS = Object.freeze([
  timingRound([0, 1, 2], ["current", "candidate"]),
  timingRound([2, 1, 0], ["candidate", "current"]),
  timingRound([1, 2, 0], ["current", "candidate"]),
  timingRound([0, 2, 1], ["candidate", "current"]),
  timingRound([2, 0, 1], ["current", "candidate"]),
  timingRound([1, 0, 2], ["candidate", "current"]),
  timingRound([0, 1, 2], ["candidate", "current"]),
  timingRound([2, 1, 0], ["current", "candidate"]),
] as const);

const progress = requireElement<HTMLElement>("#progress");
const runButton = requireElement<HTMLButtonElement>("#run");
const result = requireElement<HTMLElement>("#result");
let active: PreparedHarness | undefined;
let running: PreparedHarness | undefined;
let started = false;

void prepareHarness().then(
  (prepared) => {
    if (prepared.correctness["passed"] !== true) {
      const cleanupFirst = prepared.destroy();
      const cleanupSecond = prepared.destroy();
      publish(Object.freeze({
        schema: RECEIPT_SCHEMA,
        experiment: EXPERIMENT_ID,
        status: "correctness-stop",
        passed: false,
        identity: prepared.identity,
        correctness: prepared.correctness,
        cleanup: Object.freeze({ cleanupFirst, cleanupSecond }),
        decision: "negative-stop-raw-u32-identity",
      }), "failed");
      return;
    }
    active = prepared;
    document.body.dataset.status = "ready";
    progress.textContent =
      "READY — full and adversarial raw-U32 gates passed; timing has not run";
    runButton.disabled = false;
  },
  (error: unknown) => fail(error),
);

runButton.addEventListener("click", () => {
  if (started || active === undefined) return;
  started = true;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  running = active;
  active = undefined;
  const evidence: TimingFailureEvidence = { rawSamples: [] };
  void runTiming(running, evidence).catch(async (error: unknown) => {
    const prepared = running;
    running = undefined;
    if (prepared === undefined) {
      fail(error, Object.freeze({ timing: evidence }));
      return;
    }
    await prepared.device.queue.onSubmittedWorkDone().catch(() => undefined);
    const memoryBeforeCleanup = prepared.tracker.receipt();
    const cleanupFirst = prepared.destroy();
    const cleanupSecond = prepared.destroy();
    fail(error, Object.freeze({
      identity: prepared.identity,
      correctness: prepared.correctness,
      timing: Object.freeze({
        ...evidence,
        rawSamples: Object.freeze(evidence.rawSamples.slice()),
      }),
      uncapturedGpuErrors: Object.freeze([...prepared.uncapturedErrors]),
      memoryBeforeCleanup,
      cleanup: Object.freeze({ cleanupFirst, cleanupSecond }),
    }));
  });
});

window.addEventListener("beforeunload", () => {
  active?.destroy();
  running?.destroy();
  active = undefined;
  running = undefined;
});

async function prepareHarness(): Promise<PreparedHarness> {
  requireLittleEndianHost();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  if (!adapter.features.has("timestamp-query")) {
    throw new Error("OPT-0075 requires timestamp-query");
  }
  const largestOutputBytes = 36_000 * WIDTH * 4 + 2 * GUARD_BYTES;
  const device = await adapter.requestDevice({
    label: "ace-opt-0075-width128-rmsnorm-device",
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxBufferSize: largestOutputBytes,
      maxStorageBufferBindingSize: 36_000 * WIDTH * 4,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 1_024,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const currentKernel = AceCorrectnessRmsNormKernel.create(
    device,
    "reference-bf16",
  );
  const candidateKernel = AceOpt0075Width128RmsNormKernel.create(device);
  const querySet = device.createQuerySet({
    label: "opt0075-rmsnorm-timestamps",
    type: "timestamp",
    count: 2,
  });
  const queryResolve = tracker.create(device, {
    label: "opt0075-timestamp-resolve",
    size: QUERY_BYTES,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const queryReadback = tracker.create(device, {
    label: "opt0075-timestamp-readback",
    size: QUERY_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({
        ...tracker.receipt(),
        idempotent: true,
        repeatedCall: true,
      });
    }
    destroyed = true;
    currentKernel.destroy();
    candidateKernel.destroy();
    querySet.destroy();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      idempotent: true,
      repeatedCall: false,
      deviceDestroyed: true,
    });
  };
  try {
    const identity = await buildIdentity(adapter, device);
    const shapes: PreparedShape[] = [];
    for (const [index, spec] of SHAPES.entries()) {
      progress.textContent =
        `full raw-U32 current/candidate/rerun ${index + 1}/3: ${spec.id}`;
      shapes.push(await prepareFullShape(
        device,
        tracker,
        currentKernel,
        candidateKernel,
        spec,
      ));
      await yieldToBrowser();
    }
    const adversarial: Readonly<Record<string, unknown>>[] = [];
    for (const [index, spec] of ADVERSARIAL_CASES.entries()) {
      progress.textContent =
        `bounded edge raw-U32 screen ${index + 1}/4: ${spec.id}`;
      adversarial.push(await runAdversarialCase(
        device,
        tracker,
        currentKernel,
        candidateKernel,
        spec,
      ));
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    const fullCases = shapes.map((shape) => shape.correctness);
    const expectedFullWords = SHAPES.reduce(
      (sum, spec) => sum + spec.rows * WIDTH,
      0,
    );
    const comparedFullWords = fullCases.reduce(
      (sum, item) => sum + Number(item["comparedU32Count"]),
      0,
    );
    const correctnessPassed = comparedFullWords === expectedFullWords &&
      fullCases.every((item) => item["passed"] === true) &&
      adversarial.every((item) => item["passed"] === true) &&
      uncapturedErrors.length === 0;
    if (correctnessPassed) {
      progress.textContent = "preparing completed; timing remains click-gated";
    }
    const correctness = Object.freeze({
      fullShapeCount: fullCases.length,
      comparedFullU32CountPerCandidate: comparedFullWords,
      expectedFullU32CountPerCandidate: expectedFullWords,
      fullCases: Object.freeze(fullCases),
      adversarialCaseCount: adversarial.length,
      adversarialCases: Object.freeze(adversarial),
      rawU32IdentityRequired: true,
      deterministicCandidateRerunsRequired: true,
      finiteFullOutputsRequired: true,
      exactAdversarialClassIdentityRequired: true,
      uncapturedGpuErrorCount: uncapturedErrors.length,
      completedBeforeReady: true,
      passed: correctnessPassed,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      currentKernel,
      candidateKernel,
      querySet,
      queryResolve,
      queryReadback,
      shapes: Object.freeze(shapes),
      correctness,
      identity,
      uncapturedErrors,
      destroy,
    });
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    destroy();
    throw error;
  }
}

async function prepareFullShape(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceCorrectnessRmsNormKernel,
  candidateKernel: AceOpt0075Width128RmsNormKernel,
  spec: ShapeSpec,
): Promise<PreparedShape> {
  const caseSpecValue: CaseSpec = Object.freeze({
    id: spec.id,
    shape: Object.freeze({ rows: spec.rows, width: WIDTH, epsilon: EPSILON }),
    kind: "full",
    ordinal: spec.ordinal,
  });
  const resources = await createCaseResources(device, tracker, caseSpecValue);
  const currentOutput = createGuardedOutput(device, tracker, caseSpecValue, "current");
  const candidateOutput = createGuardedOutput(
    device,
    tracker,
    caseSpecValue,
    "candidate",
  );
  const dispatches = Object.freeze({
    current: await currentKernel.createDispatch(
      `opt0075-${spec.id}-current`,
      caseSpecValue.shape,
      {
        input: resources.inputBinding,
        weight: resources.weightBinding,
        output: currentOutput.binding,
      },
    ),
    candidate: await candidateKernel.createDispatch(
      `opt0075-${spec.id}-candidate`,
      caseSpecValue.shape,
      {
        input: resources.inputBinding,
        weight: resources.weightBinding,
        output: candidateOutput.binding,
      },
    ),
  });
  const correctness = await verifyCase(
    device,
    caseSpecValue,
    dispatches,
    currentOutput,
    candidateOutput,
    resources.inputHash,
    resources.weightHash,
    true,
  );
  return Object.freeze({ spec, dispatches, correctness });
}

async function runAdversarialCase(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceCorrectnessRmsNormKernel,
  candidateKernel: AceOpt0075Width128RmsNormKernel,
  spec: CaseSpec,
): Promise<Readonly<Record<string, unknown>>> {
  const resources = await createCaseResources(device, tracker, spec);
  const currentOutput = createGuardedOutput(device, tracker, spec, "current");
  const candidateOutput = createGuardedOutput(device, tracker, spec, "candidate");
  const dispatches = Object.freeze({
    current: await currentKernel.createDispatch(`opt0075-${spec.id}-current`, spec.shape, {
      input: resources.inputBinding,
      weight: resources.weightBinding,
      output: currentOutput.binding,
    }),
    candidate: await candidateKernel.createDispatch(
      `opt0075-${spec.id}-candidate`,
      spec.shape,
      {
        input: resources.inputBinding,
        weight: resources.weightBinding,
        output: candidateOutput.binding,
      },
    ),
  });
  const receipt = await verifyCase(
    device,
    spec,
    dispatches,
    currentOutput,
    candidateOutput,
    resources.inputHash,
    resources.weightHash,
    false,
  );
  tracker.destroy(resources.inputBuffer);
  tracker.destroy(resources.weightBuffer);
  tracker.destroy(currentOutput.buffer);
  tracker.destroy(currentOutput.readback);
  tracker.destroy(candidateOutput.buffer);
  tracker.destroy(candidateOutput.readback);
  return receipt;
}

async function createCaseResources(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
): Promise<Readonly<{
  inputBuffer: GPUBuffer;
  inputBinding: GPUBufferBinding;
  inputHash: string;
  weightBuffer: GPUBuffer;
  weightBinding: GPUBufferBinding;
  weightHash: string;
}>> {
  const input = createInputFixture(spec);
  const weights = createWeightFixture(spec);
  const inputBuffer = tracker.create(device, {
    label: `opt0075-${spec.id}-input`,
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(inputBuffer.getMappedRange()).set(input);
  inputBuffer.unmap();
  const weightBuffer = tracker.create(device, {
    label: `opt0075-${spec.id}-weight`,
    size: weights.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint32Array(weightBuffer.getMappedRange()).set(weights);
  weightBuffer.unmap();
  return Object.freeze({
    inputBuffer,
    inputBinding: binding(inputBuffer, input.byteLength),
    inputHash: await sha256Bytes(new Uint8Array(input.buffer)),
    weightBuffer,
    weightBinding: binding(weightBuffer, weights.byteLength),
    weightHash: await sha256Bytes(new Uint8Array(weights.buffer)),
  });
}

function createInputFixture(spec: CaseSpec): Float32Array<ArrayBuffer> {
  const values = new Float32Array(spec.shape.rows * WIDTH);
  const words = new Uint32Array(values.buffer);
  if (spec.kind === "signed-zero") {
    for (let index = 0; index < words.length; index += 1) {
      words[index] = index % 2 === 0 ? 0 : 0x8000_0000;
    }
    return values;
  }
  if (spec.kind === "boundary") {
    const pattern = [
      0x0000_0001, 0x007f_ffff, 0x0080_0000, 0x0080_0001,
      0x8000_0001, 0x807f_ffff, 0x8080_0000, 0x8080_0001,
      0x3380_0000, 0xb380_0000, 0x3f80_0000, 0xbf80_0000,
    ] as const;
    for (let index = 0; index < words.length; index += 1) {
      words[index] = pattern[index % pattern.length]!;
    }
    return values;
  }
  if (spec.kind === "alternating-magnitude") {
    const pattern = [
      2 ** -60, -(2 ** -60), 2 ** -20, -(2 ** -20),
      0.5, -0.5, 2 ** 20, -(2 ** 20), 2 ** 60, -(2 ** 60),
    ] as const;
    for (let index = 0; index < values.length; index += 1) {
      values[index] = pattern[index % pattern.length]!;
    }
    return values;
  }
  if (spec.kind === "maximum-bf16-scale") {
    for (let row = 0; row < spec.shape.rows; row += 1) {
      if (row === 0) {
        for (let column = 0; column < WIDTH; column += 1) {
          values[column] = column % 2 === 0 ? 1 : -1;
        }
        continue;
      }
      for (let column = 0; column < WIDTH; column += 1) {
        const index = row * WIDTH + column;
        words[index] = column % 2 === 0 ? 0 : 0x8000_0000;
      }
      values[row * WIDTH + (row * 29) % WIDTH] = row % 2 === 0 ? 1 : -1;
    }
    return values;
  }

  for (let index = 0; index < values.length; index += 1) {
    const mixed = mix32(index ^ Math.imul(spec.ordinal + 1, 0x9e37_79b9));
    values[index] = Math.fround(((mixed / 0xffff_ffff) * 2 - 1) * 3.25);
  }
  const edgeRows = Math.min(spec.shape.rows, 4);
  for (let row = 0; row < edgeRows; row += 1) {
    for (let column = 0; column < WIDTH; column += 1) {
      const index = row * WIDTH + column;
      if (row === 0) words[index] = column % 2 === 0 ? 0 : 0x8000_0000;
      else if (row === 1) words[index] = [
        0x0000_0001, 0x007f_ffff, 0x0080_0000, 0x8080_0000,
      ][column % 4]!;
      else if (row === 2) values[index] = Math.fround((column - 63.5) / 16);
      else values[index] = column % 2 === 0 ? 2 ** 20 : -(2 ** 20);
    }
  }
  return values;
}

function createWeightFixture(spec: CaseSpec): Uint32Array<ArrayBuffer> {
  const halves = new Uint16Array(WIDTH);
  if (spec.kind === "maximum-bf16-scale") {
    halves.fill(0x7f7f);
  } else if (spec.kind === "signed-zero") {
    for (let index = 0; index < halves.length; index += 1) {
      halves[index] = index % 3 === 0 ? 0xbf80 : 0x3f80;
    }
  } else if (spec.kind === "boundary") {
    const pattern = [0x0001, 0x007f, 0x0080, 0x3f00, 0xbf00, 0x3f80] as const;
    for (let index = 0; index < halves.length; index += 1) {
      halves[index] = pattern[index % pattern.length]!;
    }
  } else {
    const pattern = [0x3f00, 0x3f20, 0x3f40, 0x3f60, 0x3f80, 0xbf00] as const;
    for (let index = 0; index < halves.length; index += 1) {
      halves[index] = pattern[(index + spec.ordinal) % pattern.length]!;
    }
  }
  const packed = new Uint32Array(WIDTH / 2);
  for (let index = 0; index < packed.length; index += 1) {
    packed[index] = halves[index * 2]! | (halves[index * 2 + 1]! << 16);
  }
  return packed;
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
  arm: Arm,
): GuardedOutput {
  const outputElements = spec.shape.rows * WIDTH;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * GUARD_BYTES;
  const outputWordOffset = GUARD_BYTES / 4;
  const prefill = new Uint32Array(totalBytes / 4);
  prefill.fill(GUARD_WORD);
  prefill.fill(
    OUTPUT_QNAN_WORD,
    outputWordOffset,
    outputWordOffset + outputElements,
  );
  const buffer = tracker.create(device, {
    label: `opt0075-${spec.id}-${arm}-guarded-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(prefill);
  buffer.unmap();
  const readback = tracker.create(device, {
    label: `opt0075-${spec.id}-${arm}-readback`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({ buffer, offset: GUARD_BYTES, size: outputBytes }),
    readback,
    prefill,
    outputElements,
    outputWordOffset,
    totalBytes,
  });
}

async function verifyCase(
  device: GPUDevice,
  spec: CaseSpec,
  dispatches: Readonly<Record<Arm, AceRmsNormDispatch>>,
  currentOutput: GuardedOutput,
  candidateOutput: GuardedOutput,
  inputSha256: string,
  weightSha256: string,
  requireFinite: boolean,
): Promise<Readonly<Record<string, unknown>>> {
  await executeAndDrain(device, dispatches.current);
  const current = await readSnapshot(device, currentOutput);
  await executeAndDrain(device, dispatches.candidate);
  const candidate = await readSnapshot(device, candidateOutput);
  device.queue.writeBuffer(candidateOutput.buffer, 0, candidateOutput.prefill);
  await executeAndDrain(device, dispatches.candidate);
  const rerun = await readSnapshot(device, candidateOutput);
  const currentCandidate = compareWords(current.words, candidate.words);
  const candidateRerun = compareWords(candidate.words, rerun.words);
  const classesIdentical = JSON.stringify(current.classes) ===
    JSON.stringify(candidate.classes) && JSON.stringify(candidate.classes) ===
    JSON.stringify(rerun.classes);
  const guardsPassed = [current, candidate, rerun].every((snapshot) =>
    snapshot.prefixCanaryIntact && snapshot.suffixCanaryIntact &&
    snapshot.qNaNPrefillCount === 0 && snapshot.tailRowWritten
  );
  const finitePassed = !requireFinite || [current, candidate, rerun].every(
    (snapshot) => snapshot.nonFiniteCount === 0,
  );
  const edgeCoveragePassed = spec.kind !== "maximum-bf16-scale" ||
    current.nonFiniteCount > 0;
  const passed = currentCandidate.differingU32Count === 0 &&
    candidateRerun.differingU32Count === 0 && classesIdentical &&
    guardsPassed && finitePassed && edgeCoveragePassed;
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.kind,
    shape: spec.shape,
    inputSha256,
    weightSha256,
    outputSha256: Object.freeze({
      current: current.sha256,
      candidate: candidate.sha256,
      candidateRerun: rerun.sha256,
    }),
    comparedU32Count: current.words.length,
    currentCandidate,
    candidateRerun,
    outputClasses: Object.freeze({
      current: current.classes,
      candidate: candidate.classes,
      candidateRerun: rerun.classes,
    }),
    nonFiniteCounts: Object.freeze({
      current: current.nonFiniteCount,
      candidate: candidate.nonFiniteCount,
      candidateRerun: rerun.nonFiniteCount,
    }),
    classIdentityPassed: classesIdentical,
    completeWritesPassed: guardsPassed,
    finiteOutputPassed: finitePassed,
    edgeCoveragePassed,
    deterministicRerunPassed: candidateRerun.differingU32Count === 0,
    passed,
  });
}

async function executeAndDrain(
  device: GPUDevice,
  dispatch: AceRmsNormDispatch,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: `${dispatch.label}-encoder` });
  const pass = encoder.beginComputePass({ label: `${dispatch.label}-pass` });
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readSnapshot(
  device: GPUDevice,
  output: GuardedOutput,
): Promise<Snapshot> {
  const encoder = device.createCommandEncoder({ label: "opt0075-output-readback" });
  encoder.copyBufferToBuffer(
    output.buffer,
    0,
    output.readback,
    0,
    output.totalBytes,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await output.readback.mapAsync(GPUMapMode.READ);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const start = output.outputWordOffset;
    const end = start + output.outputElements;
    const words = new Uint32Array(output.outputElements);
    words.set(all.subarray(start, end));
    const classes: Record<string, number> = {};
    let qNaNPrefillCount = 0;
    let nonFiniteCount = 0;
    for (const word of words) {
      if (word === OUTPUT_QNAN_WORD) qNaNPrefillCount += 1;
      const name = f32Class(word);
      classes[name] = (classes[name] ?? 0) + 1;
      if (name.includes("infinity") || name === "nan") nonFiniteCount += 1;
    }
    const tailStart = Math.max(0, words.length - WIDTH);
    return Object.freeze({
      words,
      sha256: await sha256Bytes(new Uint8Array(words.buffer)),
      prefixCanaryIntact: all.subarray(0, start).every((word) => word === GUARD_WORD),
      suffixCanaryIntact: all.subarray(end).every((word) => word === GUARD_WORD),
      qNaNPrefillCount,
      nonFiniteCount,
      classes: Object.freeze(classes),
      tailRowWritten: words.subarray(tailStart).every(
        (word) => word !== OUTPUT_QNAN_WORD,
      ),
    });
  } finally {
    output.readback.unmap();
  }
}

function compareWords(
  control: Uint32Array,
  candidate: Uint32Array,
): Readonly<Record<string, unknown>> {
  if (control.length !== candidate.length) {
    throw new Error("OPT-0075 output lengths differ");
  }
  let differingU32Count = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  for (let index = 0; index < control.length; index += 1) {
    const left = control[index]!;
    const right = candidate[index]!;
    if (left === right) continue;
    differingU32Count += 1;
    firstDifference ??= Object.freeze({
      index,
      controlU32: left,
      candidateU32: right,
      controlClass: f32Class(left),
      candidateClass: f32Class(right),
    });
  }
  return Object.freeze({
    comparedU32Count: control.length,
    differingU32Count,
    firstDifference,
    passed: differingU32Count === 0,
  });
}

async function runTiming(
  prepared: PreparedHarness,
  failureEvidence: TimingFailureEvidence,
): Promise<void> {
  const warmupStartedAtEpochMilliseconds = Date.now();
  for (const [shapeIndex, shape] of prepared.shapes.entries()) {
    const order: readonly Arm[] = shapeIndex % 2 === 0
      ? ["current", "candidate"]
      : ["candidate", "current"];
    for (const arm of order) {
      progress.textContent = `click-boundary untimed wakeup: ${shape.spec.id} ${arm}`;
      await executeAndDrain(prepared.device, shape.dispatches[arm]);
    }
  }
  const warmupCompletedAtEpochMilliseconds = Date.now();
  const rawSamples = failureEvidence.rawSamples;
  const runStartedAtEpochMilliseconds = Date.now();
  failureEvidence.runStartedAtEpochMilliseconds = runStartedAtEpochMilliseconds;
  for (const [roundIndex, round] of TIMING_ROUNDS.entries()) {
    for (const [shapePosition, shapeIndex] of round.shapeOrder.entries()) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) throw new Error("OPT-0075 timing shape changed");
      for (const [armPosition, arm] of round.armOrder.entries()) {
        progress.textContent =
          `timing ${roundIndex + 1}/8: ${shape.spec.id} ${arm}`;
        const sample = await timeDispatch(
          prepared.device,
          shape.dispatches[arm],
          prepared.querySet,
          prepared.queryResolve,
          prepared.queryReadback,
        );
        rawSamples.push(Object.freeze({
          roundIndex,
          shapePosition,
          shapeId: shape.spec.id,
          armPosition,
          arm,
          ...sample,
        }));
      }
      await yieldToBrowser();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0075 observed an uncaptured timing GPU error");
  }
  const measurementCompletedAtEpochMilliseconds = Date.now();
  failureEvidence.measurementCompletedAtEpochMilliseconds =
    measurementCompletedAtEpochMilliseconds;
  const timing = summarizeOpt0075Timing(rawSamples);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupStartedAtEpochMilliseconds = Date.now();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  const cleanupCompletedAtEpochMilliseconds = Date.now();
  const cleanup = Object.freeze({
    cleanupStartedAtEpochMilliseconds,
    cleanupCompletedAtEpochMilliseconds,
    firstCall: cleanupFirst,
    secondCall: cleanupSecond,
    idempotent: true,
    zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&
      cleanupSecond["liveBufferCount"] === 0,
  });
  const inPagePassed = timing["passed"] === true && cleanup.zeroLiveBuffers;
  const receipt = Object.freeze({
    schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: inPagePassed ? "primitive-pass" : "inconclusive",
    passed: inPagePassed,
    identity: prepared.identity,
    correctness: prepared.correctness,
    protocol: Object.freeze({
      fullAndAdversarialCorrectnessBeforeReady: true,
      clickBoundaryUntimedDispatchPerArmPerShape: true,
      warmupStartedAtEpochMilliseconds,
      warmupCompletedAtEpochMilliseconds,
      timingButtonCount: 1,
      rounds: TIMING_ROUNDS.length,
      balancedOrders: TIMING_ROUNDS,
      samplesPerArmPerShape: TIMING_ROUNDS.length,
      dispatchRepetitionsPerSample: DISPATCH_REPETITIONS,
      oneComputePassPerSample: true,
      oneCommandBufferPerSample: true,
      oneSubmitAndMatchingQueueDrainPerSample: true,
      oneTimestampQueryPairPerSample: true,
      timestampWritesBracketComputePass: true,
      outputReadbackInsideTiming: false,
      weightedLayerMix: "2*T36000 + T18000",
      weightedMixNormalizedByMultiplicity: false,
      crossCache784IsDiagnosticOnly: true,
      externalThermalPollRequiredThroughCleanup: true,
      thermalClaim: null,
    }),
    timing: Object.freeze({
      ...timing,
      runStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(rawSamples.slice()),
    }),
    decision: Object.freeze({
      disposition: inPagePassed
        ? "positive-in-page-pending-external-thermal-audit"
        : "inconclusive-stop-primitive-gate",
      diagnosticDitProfileAuthorizedByPage: false,
      externalThermalGateAuditedByPage: false,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      listeningOrTrajectoryClaim: false,
    }),
    uncapturedGpuErrors: Object.freeze([...prepared.uncapturedErrors]),
    memoryBeforeCleanup,
    cleanup,
  });
  running = undefined;
  publish(receipt, receipt.passed ? "passed" : "failed");
  progress.textContent = receipt.passed
    ? "completed — primitive gates passed; external thermal audit pending"
    : "completed — candidate did not clear every frozen primitive gate";
}

async function timeDispatch(
  device: GPUDevice,
  dispatch: AceRmsNormDispatch,
  querySet: GPUQuerySet,
  queryResolve: GPUBuffer,
  queryReadback: GPUBuffer,
): Promise<TimestampSample> {
  const encoder = device.createCommandEncoder({ label: `${dispatch.label}-timed` });
  const pass = encoder.beginComputePass({
    label: `${dispatch.label}-timed-pass`,
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  for (let repeat = 0; repeat < DISPATCH_REPETITIONS; repeat += 1) {
    dispatch.encode(pass);
  }
  pass.end();
  encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
  encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, QUERY_BYTES);
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const wallMilliseconds = fenceAtPerformanceMilliseconds -
    submitAtPerformanceMilliseconds;
  await queryReadback.mapAsync(GPUMapMode.READ);
  let timestampBegin: bigint;
  let timestampEnd: bigint;
  try {
    const timestamps = new BigUint64Array(queryReadback.getMappedRange());
    timestampBegin = timestamps[0]!;
    timestampEnd = timestamps[1]!;
  } finally {
    queryReadback.unmap();
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) ||
    gpuElapsedNanoseconds <= 0 || !Number.isFinite(wallMilliseconds) ||
    wallMilliseconds <= 0) {
    throw new Error(`OPT-0075 ${dispatch.label} returned invalid timing`);
  }
  return Object.freeze({
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds:
      performance.timeOrigin + submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds:
      performance.timeOrigin + fenceAtPerformanceMilliseconds,
    wallMilliseconds,
    timestampBeginNanoseconds: timestampBegin.toString(),
    timestampEndNanoseconds: timestampEnd.toString(),
    gpuElapsedNanoseconds,
    gpuMilliseconds,
    perDispatchGpuMilliseconds: gpuMilliseconds / DISPATCH_REPETITIONS,
    perDispatchWallMilliseconds: wallMilliseconds / DISPATCH_REPETITIONS,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    dispatchRepetitions: DISPATCH_REPETITIONS,
    commandBufferCount: 1,
    queueDrainCount: 1,
    timestampQueryPairCount: 1,
  });
}

export function summarizeOpt0075Timing(
  rawSamples: readonly RawSample[],
): Readonly<Record<string, unknown>> {
  const expected = TIMING_ROUNDS.length * SHAPES.length * 2;
  if (rawSamples.length !== expected) {
    throw new Error(`OPT-0075 requires ${expected} raw timing samples`);
  }
  const findSample = (roundIndex: number, shapeId: ShapeSpec["id"], arm: Arm) => {
    const matches = rawSamples.filter((sample) =>
      sample.roundIndex === roundIndex && sample.shapeId === shapeId &&
      sample.arm === arm
    );
    if (matches.length !== 1) {
      throw new Error(`OPT-0075 missing unique ${roundIndex}/${shapeId}/${arm}`);
    }
    return matches[0]!;
  };
  const strata = SHAPES.map((spec) => {
    const arms = Object.fromEntries((["current", "candidate"] as const).map(
      (arm) => {
        const samples = TIMING_ROUNDS.map((_, roundIndex) =>
          findSample(roundIndex, spec.id, arm)
        );
        return [arm, summarizeArm(samples)];
      },
    ));
    const pairs = TIMING_ROUNDS.map((_, roundIndex) => {
      const current = findSample(roundIndex, spec.id, "current");
      const candidate = findSample(roundIndex, spec.id, "candidate");
      return Object.freeze({
        roundIndex,
        currentGpuMilliseconds: current.perDispatchGpuMilliseconds,
        candidateGpuMilliseconds: candidate.perDispatchGpuMilliseconds,
        gpuSpeedup: current.perDispatchGpuMilliseconds /
          candidate.perDispatchGpuMilliseconds,
        gpuWin: candidate.perDispatchGpuMilliseconds <
          current.perDispatchGpuMilliseconds,
        currentWallMilliseconds: current.perDispatchWallMilliseconds,
        candidateWallMilliseconds: candidate.perDispatchWallMilliseconds,
        wallSpeedup: current.perDispatchWallMilliseconds /
          candidate.perDispatchWallMilliseconds,
        wallWin: candidate.perDispatchWallMilliseconds <
          current.perDispatchWallMilliseconds,
      });
    });
    return Object.freeze({
      id: spec.id,
      shape: Object.freeze({ rows: spec.rows, width: WIDTH, epsilon: EPSILON }),
      layerMultiplicity: spec.multiplicity,
      diagnosticOnly: spec.multiplicity === 0,
      arms: Object.freeze(arms),
      pairedRounds: Object.freeze(pairs),
      everyPairedGpuWin: pairs.every((pair) => pair.gpuWin),
      everyPairedWallWin: pairs.every((pair) => pair.wallWin),
    });
  });
  const weightedRounds = TIMING_ROUNDS.map((_, roundIndex) => {
    const values = Object.fromEntries((["current", "candidate"] as const).map(
      (arm) => {
        const q = findSample(roundIndex, "q-36000", arm);
        const k = findSample(roundIndex, "k-18000", arm);
        return [arm, Object.freeze({
          gpuMilliseconds: 2 * q.perDispatchGpuMilliseconds +
            k.perDispatchGpuMilliseconds,
          wallMilliseconds: 2 * q.perDispatchWallMilliseconds +
            k.perDispatchWallMilliseconds,
        })];
      },
    )) as unknown as Readonly<Record<Arm, Readonly<{
      gpuMilliseconds: number;
      wallMilliseconds: number;
    }>>>;
    return Object.freeze({
      roundIndex,
      current: values.current,
      candidate: values.candidate,
      gpuSpeedup: values.current.gpuMilliseconds /
        values.candidate.gpuMilliseconds,
      wallSpeedup: values.current.wallMilliseconds /
        values.candidate.wallMilliseconds,
      gpuSavingMilliseconds: values.current.gpuMilliseconds -
        values.candidate.gpuMilliseconds,
      wallSavingMilliseconds: values.current.wallMilliseconds -
        values.candidate.wallMilliseconds,
      gpuWin: values.candidate.gpuMilliseconds < values.current.gpuMilliseconds,
      wallWin: values.candidate.wallMilliseconds < values.current.wallMilliseconds,
    });
  });
  const currentGpu = weightedRounds.map((round) => round.current.gpuMilliseconds);
  const candidateGpu = weightedRounds.map(
    (round) => round.candidate.gpuMilliseconds,
  );
  const currentWall = weightedRounds.map((round) => round.current.wallMilliseconds);
  const candidateWall = weightedRounds.map(
    (round) => round.candidate.wallMilliseconds,
  );
  const weightedSummary = Object.freeze({
    current: Object.freeze({
      meanGpuMilliseconds: mean(currentGpu),
      medianGpuMilliseconds: median(currentGpu),
      meanWallMilliseconds: mean(currentWall),
      medianWallMilliseconds: median(currentWall),
    }),
    candidate: Object.freeze({
      meanGpuMilliseconds: mean(candidateGpu),
      medianGpuMilliseconds: median(candidateGpu),
      meanWallMilliseconds: mean(candidateWall),
      medianWallMilliseconds: median(candidateWall),
    }),
  });
  const meanGpuSpeedup = weightedSummary.current.meanGpuMilliseconds /
    weightedSummary.candidate.meanGpuMilliseconds;
  const medianGpuSpeedup = weightedSummary.current.medianGpuMilliseconds /
    weightedSummary.candidate.medianGpuMilliseconds;
  const meanWallSpeedup = weightedSummary.current.meanWallMilliseconds /
    weightedSummary.candidate.meanWallMilliseconds;
  const medianWallSpeedup = weightedSummary.current.medianWallMilliseconds /
    weightedSummary.candidate.medianWallMilliseconds;
  const meanGpuSaving = weightedSummary.current.meanGpuMilliseconds -
    weightedSummary.candidate.meanGpuMilliseconds;
  const medianGpuSaving = weightedSummary.current.medianGpuMilliseconds -
    weightedSummary.candidate.medianGpuMilliseconds;
  const meanWallSaving = weightedSummary.current.meanWallMilliseconds -
    weightedSummary.candidate.meanWallMilliseconds;
  const medianWallSaving = weightedSummary.current.medianWallMilliseconds -
    weightedSummary.candidate.medianWallMilliseconds;
  const primaryStrata = strata.filter((stratum) => !stratum.diagnosticOnly);
  const gates = Object.freeze({
    everyPrimaryShapePairedGpuWin: primaryStrata.every(
      (stratum) => stratum.everyPairedGpuWin,
    ),
    everyPrimaryShapePairedWallWin: primaryStrata.every(
      (stratum) => stratum.everyPairedWallWin,
    ),
    everyWeightedRoundGpuWin: weightedRounds.every((round) => round.gpuWin),
    everyWeightedRoundWallWin: weightedRounds.every((round) => round.wallWin),
    weightedMeanGpuSpeedupPassed: meanGpuSpeedup >= REQUIRED_SPEEDUP,
    weightedMedianGpuSpeedupPassed: medianGpuSpeedup >= REQUIRED_SPEEDUP,
    weightedMeanWallSpeedupPassed: meanWallSpeedup >= REQUIRED_SPEEDUP,
    weightedMedianWallSpeedupPassed: medianWallSpeedup >= REQUIRED_SPEEDUP,
    weightedMeanGpuSavingPassed:
      meanGpuSaving >= REQUIRED_LAYER_MIX_SAVING_MILLISECONDS,
    weightedMedianGpuSavingPassed:
      medianGpuSaving >= REQUIRED_LAYER_MIX_SAVING_MILLISECONDS,
    weightedMeanWallSavingPassed:
      meanWallSaving >= REQUIRED_LAYER_MIX_SAVING_MILLISECONDS,
    weightedMedianWallSavingPassed:
      medianWallSaving >= REQUIRED_LAYER_MIX_SAVING_MILLISECONDS,
    weightedMeanWallGpuSavingAgreementPassed:
      meanWallSaving / meanGpuSaving >= 0.75 &&
      meanWallSaving / meanGpuSaving <= 1.25,
    weightedMedianWallGpuSavingAgreementPassed:
      medianWallSaving / medianGpuSaving >= 0.75 &&
      medianWallSaving / medianGpuSaving <= 1.25,
  });
  const passed = Object.values(gates).every((value) => value === true);
  return Object.freeze({
    strata: Object.freeze(strata),
    weightedLayerMix: weightedSummary,
    weightedRoundPairs: Object.freeze(weightedRounds),
    speedups: Object.freeze({
      meanGpu: meanGpuSpeedup,
      medianGpu: medianGpuSpeedup,
      meanWall: meanWallSpeedup,
      medianWall: medianWallSpeedup,
    }),
    savingsMillisecondsPerLayerMix: Object.freeze({
      meanGpu: meanGpuSaving,
      medianGpu: medianGpuSaving,
      meanWall: meanWallSaving,
      medianWall: medianWallSaving,
    }),
    projectedEightEvaluationSavingsMilliseconds: Object.freeze({
      meanGpu: meanGpuSaving * 24 * 8,
      medianGpu: medianGpuSaving * 24 * 8,
      meanWall: meanWallSaving * 24 * 8,
      medianWall: medianWallSaving * 24 * 8,
      excludesCrossCache: true,
      authoritativeOnlyWhenPassed: true,
    }),
    gates,
    passed,
  });
}

function summarizeArm(
  samples: readonly TimestampSample[],
): Readonly<Record<string, unknown>> {
  if (samples.length !== TIMING_ROUNDS.length) {
    throw new Error("OPT-0075 arm sample count changed");
  }
  const gpu = samples.map((sample) => sample.perDispatchGpuMilliseconds);
  const wall = samples.map((sample) => sample.perDispatchWallMilliseconds);
  return Object.freeze({
    samples: Object.freeze(samples),
    meanGpuMilliseconds: mean(gpu),
    medianGpuMilliseconds: median(gpu),
    minimumGpuMilliseconds: Math.min(...gpu),
    maximumGpuMilliseconds: Math.max(...gpu),
    meanWallMilliseconds: mean(wall),
    medianWallMilliseconds: median(wall),
    minimumWallMilliseconds: Math.min(...wall),
    maximumWallMilliseconds: Math.max(...wall),
    meanGpuToWallRatio: mean(gpu) / mean(wall),
  });
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<Readonly<Record<string, unknown>>> {
  const info = adapter.info;
  return Object.freeze({
    browserSurface: "Codex in-app Browser",
    browserUserAgent: navigator.userAgent,
    origin: location.origin,
    pathname: location.pathname,
    adapterVendor: info.vendor,
    adapterArchitecture: info.architecture,
    adapterDevice: info.device,
    adapterDescription: info.description,
    requestedFeatures: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
    }),
    currentKernel: "ace-correctness-rmsnorm-reference-bf16-wg256",
    candidateKernel: ACE_OPT_0075_WIDTH128_RMSNORM_KERNEL_ID,
    modelProfile: "reference-bf16",
    width: WIDTH,
    epsilon: EPSILON,
    kernelSourceSha256: await sha256Text(rmsNormSource),
    harnessSourceSha256: await sha256Text(harnessSource),
    harnessHtmlSha256: await sha256Text(harnessHtmlSource),
    fixtureGenerator: "opt0075-mix32-f32-edge-rows-bf16-pattern-v1",
    preparedAtEpochMilliseconds: Date.now(),
  });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  private created = 0;
  private destroyed = 0;
  private liveBytes = 0;
  private maximumLiveBytes = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.created += 1;
    this.liveBytes += size;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes,
      maximumLiveBytes: this.maximumLiveBytes,
    });
  }
}

function shapeSpec(
  id: ShapeSpec["id"],
  rows: ShapeSpec["rows"],
  multiplicity: ShapeSpec["multiplicity"],
  ordinal: number,
): ShapeSpec {
  return Object.freeze({ id, rows, multiplicity, ordinal });
}

function caseSpec(
  id: string,
  rows: number,
  kind: Exclude<FixtureKind, "full">,
  ordinal: number,
): CaseSpec {
  return Object.freeze({
    id,
    shape: Object.freeze({ rows, width: WIDTH, epsilon: EPSILON }),
    kind,
    ordinal,
  });
}

function timingRound(
  shapeOrder: readonly [number, number, number],
  armOrder: readonly [Arm, Arm],
): Readonly<{
  shapeOrder: readonly [number, number, number];
  armOrder: readonly [Arm, Arm];
}> {
  return Object.freeze({
    shapeOrder: Object.freeze(shapeOrder),
    armOrder: Object.freeze(armOrder),
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function f32Class(word: number): string {
  const absolute = word & 0x7fff_ffff;
  const negative = (word & 0x8000_0000) !== 0;
  if (absolute === 0) return negative ? "negative-zero" : "positive-zero";
  const exponent = word & 0x7f80_0000;
  const mantissa = word & 0x007f_ffff;
  if (exponent === 0x7f80_0000) {
    if (mantissa !== 0) return "nan";
    return negative ? "negative-infinity" : "positive-infinity";
  }
  if (exponent === 0) {
    return negative ? "negative-subnormal" : "positive-subnormal";
  }
  return negative ? "negative-normal" : "positive-normal";
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function mean(values: readonly number[]): number {
  requireFinitePositiveSamples(values);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  requireFinitePositiveSamples(values);
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length / 2;
  return Number.isInteger(middle)
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[Math.floor(middle)]!;
}

function requireFinitePositiveSamples(values: readonly number[]): void {
  if (values.length === 0 || values.some((value) =>
    !Number.isFinite(value) || value <= 0
  )) {
    throw new Error("OPT-0075 requires finite positive timing samples");
  }
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0075 fixtures require a little-endian host");
  }
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

async function sha256Text(value: string): Promise<string> {
  return await sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const copy = new Uint8Array(value.byteLength);
  copy.set(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", copy));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function settlePostDrainEvents(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function publish(
  receipt: Readonly<Record<string, unknown>>,
  status: "passed" | "failed",
): void {
  window.__ACE_OPT0075_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt);
  document.body.dataset.status = status;
}

function fail(
  error: unknown,
  evidence: Readonly<Record<string, unknown>> = Object.freeze({}),
): void {
  const receipt = Object.freeze({
    schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "failed",
    passed: false,
    ...evidence,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
  publish(receipt, "failed");
  progress.textContent = error instanceof Error ? error.message : String(error);
  runButton.disabled = true;
}
