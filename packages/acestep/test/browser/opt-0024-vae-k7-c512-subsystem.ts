/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />
/// <reference lib="webworker" />

import { createAceOpt0011LatentFixture } from
  "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import {
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import {
  submitAceCommandBufferFactoriesCooperatively,
  type AceGpuCommandBufferDrainTiming,
} from "../../src/runtime/scheduler.js";
import { requestAceWebGpuDevice, type AceWebGpuDeviceContext } from
  "../../src/webgpu/device.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  type AceFp16VaeConv1dSubgroupKernel,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8,
  ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  AceOpt0011Fp16VaeDecoderRuntime,
  type AceOpt0011Fp16VaeChunkDispatchSet,
  type AceOpt0011Fp16VaeWindowDispatch,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  resolveAceOpt0011Fp16VaePackageBindings,
  type AceOpt0011VaePackageBindings,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
} from "../../src/webgpu/vae-fp16-profile.js";
import { planAceVaeDecoder, type AceVaeDecoderOperation } from
  "../../src/webgpu/vae-decoder.js";

export const OPT_0024_C512_SCHEMA =
  "ace-opt-0024-vae-k7-c512-subsystem-v1" as const;
export const OPT_0024_C512_FIXTURE_SHA256 =
  "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899" as const;
export const OPT_0024_C512_FRAMES = 512 as const;
export const OPT_0024_C512_INPUT_ELEMENTS = 32_768 as const;
export const OPT_0024_C512_OUTPUT_ELEMENTS = 1_966_080 as const;
export const OPT_0024_C512_OUTPUT_BYTES = 7_864_320 as const;
export const OPT_0024_C512_K7_PURE_BATCHES = 500 as const;
export const OPT_0024_C512_K7_PURE_QUANTA = 3_999 as const;
export const OPT_0024_C512_K7_CANDIDATE_QUANTA = 4_082 as const;
export const OPT_0024_C512_K7_TOTAL_QUANTA = 4_090 as const;
export const OPT_0024_C512_WARMUP_ORDER = Object.freeze([
  "shipped",
  "candidate",
] as const);
export const OPT_0024_C512_TIMED_ORDER = Object.freeze([
  "shipped",
  "candidate",
  "candidate",
  "shipped",
] as const);

const EXPERIMENT_ID = "OPT-0024" as const;
const MANIFEST_PATH = "/model/files-fp16-vae-experimental/manifest.json";
const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = 4;
const QNAN_WORD = 0x7fc5_0024;
const REQUIRED_SUBGROUP_SIZE = 32;
const WORKER_QUERY = "opt0024Worker";
const RAW_RESULT_GLOBAL = "__ACE_OPT_0024_C512_RESULT_JSON__";
const NRMSE_LIMIT = 0.003;
const SNR_MINIMUM_DB = 50;
const PEARSON_MINIMUM = 0.9999;
const RELATIVE_RMS_DRIFT_LIMIT = 0.005;
const RELATIVE_ENERGY_DRIFT_LIMIT = 0.005;
const RELATIVE_PEAK_DRIFT_LIMIT = 0.01;
const RELATIVE_DC_DRIFT_LIMIT = 0.001;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT = 0.02;
const K7_SPEEDUP_MINIMUM = 1.25;
const K7_SAVING_MINIMUM_MS = 1_000;

export type Opt0024C512Arm = "shipped" | "candidate";
export type Opt0024C512Family =
  | "k7-conv1d"
  | "k1-conv1d"
  | "conv-transpose1d"
  | "snake"
  | "add"
  | "mixed";

export interface Opt0024C512FamilyTiming {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly submitThroughDrainMs: number;
}

export interface Opt0024C512Metrics {
  readonly count: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly pearson: number;
  readonly relativeRmsDrift: number;
  readonly relativeEnergyDrift: number;
  readonly relativePeakDrift: number;
  readonly relativeDcOffsetDrift: number;
  readonly relativeMaximumAbsoluteError: number;
  readonly maximumAbsoluteError: number;
  readonly controlRms: number;
  readonly candidateRms: number;
  readonly controlPeak: number;
  readonly candidatePeak: number;
  readonly controlMean: number;
  readonly candidateMean: number;
  readonly finite: boolean;
  readonly passed: boolean;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly residentBytes: number;
}

interface C512Resources {
  readonly stagingInput: GPUBuffer;
  readonly decoderInput: GPUBuffer;
  readonly workspaces: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  readonly output: GPUBuffer;
  readonly readback: GPUBuffer;
  destroy(): void;
}

interface PreparedGate {
  readonly context: AceWebGpuDeviceContext;
  readonly phase: AceGpuTensorPhase;
  readonly resources: C512Resources;
  readonly shippedRuntime: AceOpt0011Fp16VaeDecoderRuntime;
  readonly candidateRuntime: AceOpt0011Fp16VaeDecoderRuntime;
  readonly dispatches: Readonly<Record<Opt0024C512Arm,
    AceOpt0011Fp16VaeWindowDispatch>>;
  readonly fixture: Float32Array<ArrayBuffer>;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface ExecutionResult {
  readonly arm: Opt0024C512Arm;
  readonly output: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface WorkerCommand {
  readonly type: "prepare" | "run";
}

interface WorkerEvent {
  readonly type: "progress" | "prepared" | "result" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
}

/** Pure batch classifier used by the focused contract test and live profiler. */
export function classifyOpt0024C512Batch(
  operations: readonly (Readonly<{
    readonly operationIndex: number | null;
    readonly operationKind: string;
  }>)[],
  graphOperations: readonly (Readonly<{
    readonly kind: string;
    readonly shape?: unknown;
  }>)[],
): Opt0024C512Family {
  const families = new Set<Opt0024C512Family>();
  for (const quantum of operations) {
    if (quantum.operationIndex === null) return "mixed";
    const operation = graphOperations[quantum.operationIndex];
    if (operation === undefined || operation.kind !== quantum.operationKind) {
      return "mixed";
    }
    const family = operationFamily(operation as AceVaeDecoderOperation);
    if (family === null) return "mixed";
    families.add(family);
  }
  return families.size === 1 ? families.values().next().value! : "mixed";
}

/** Full-waveform metric envelope. Stride/offset select joint or one channel. */
export function compareOpt0024C512Waveforms(
  control: Float32Array,
  candidate: Float32Array,
  stride = 1,
  offset = 0,
): Opt0024C512Metrics {
  if (
    control.length !== candidate.length ||
    !Number.isSafeInteger(stride) || stride < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 || offset >= stride
  ) throw new RangeError("OPT-0024 C512 comparison geometry is invalid");
  let count = 0;
  let sumControl = 0;
  let sumCandidate = 0;
  let sumControlSquared = 0;
  let sumCandidateSquared = 0;
  let sumProduct = 0;
  let sumSquaredError = 0;
  let maximumAbsoluteError = 0;
  let controlPeak = 0;
  let candidatePeak = 0;
  let finite = true;
  for (let index = offset; index < control.length; index += stride) {
    const a = control[index]!;
    const b = candidate[index]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) finite = false;
    const error = b - a;
    const absoluteError = Math.abs(error);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
    controlPeak = Math.max(controlPeak, Math.abs(a));
    candidatePeak = Math.max(candidatePeak, Math.abs(b));
    sumControl += a;
    sumCandidate += b;
    sumControlSquared += a * a;
    sumCandidateSquared += b * b;
    sumProduct += a * b;
    sumSquaredError += error * error;
    count += 1;
  }
  if (count === 0) throw new RangeError("OPT-0024 C512 comparison is empty");
  const meanControl = sumControl / count;
  const meanCandidate = sumCandidate / count;
  const meanControlSquared = sumControlSquared / count;
  const meanCandidateSquared = sumCandidateSquared / count;
  const mse = sumSquaredError / count;
  const controlRms = Math.sqrt(meanControlSquared);
  const candidateRms = Math.sqrt(meanCandidateSquared);
  const rmse = Math.sqrt(mse);
  const nrmse = rmse / Math.max(controlRms, 1e-30);
  const snrDb = rmse === 0
    ? Number.POSITIVE_INFINITY
    : 20 * Math.log10(Math.max(controlRms, 1e-30) / rmse);
  const controlVariance = Math.max(0,
    meanControlSquared - meanControl * meanControl);
  const candidateVariance = Math.max(0,
    meanCandidateSquared - meanCandidate * meanCandidate);
  const covariance = sumProduct / count - meanControl * meanCandidate;
  const denominator = Math.sqrt(controlVariance * candidateVariance);
  const pearson = denominator === 0
    ? (mse === 0 ? 1 : 0)
    : covariance / denominator;
  const relativeRmsDrift = Math.abs(candidateRms - controlRms) /
    Math.max(controlRms, 1e-30);
  const relativeEnergyDrift = Math.abs(
    meanCandidateSquared - meanControlSquared,
  ) / Math.max(meanControlSquared, 1e-30);
  const relativePeakDrift = Math.abs(candidatePeak - controlPeak) /
    Math.max(controlPeak, 1e-30);
  const relativeDcOffsetDrift = Math.abs(meanCandidate - meanControl) /
    Math.max(controlRms, 1e-6);
  const relativeMaximumAbsoluteError = maximumAbsoluteError /
    Math.max(controlPeak, 1e-6);
  const passed = finite && nrmse <= NRMSE_LIMIT &&
    snrDb >= SNR_MINIMUM_DB && pearson >= PEARSON_MINIMUM &&
    relativeRmsDrift <= RELATIVE_RMS_DRIFT_LIMIT &&
    relativeEnergyDrift <= RELATIVE_ENERGY_DRIFT_LIMIT &&
    relativePeakDrift <= RELATIVE_PEAK_DRIFT_LIMIT &&
    relativeDcOffsetDrift <= RELATIVE_DC_DRIFT_LIMIT &&
    relativeMaximumAbsoluteError <= RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT;
  return Object.freeze({
    count,
    nrmse,
    snrDb,
    pearson,
    relativeRmsDrift,
    relativeEnergyDrift,
    relativePeakDrift,
    relativeDcOffsetDrift,
    relativeMaximumAbsoluteError,
    maximumAbsoluteError,
    controlRms,
    candidateRms,
    controlPeak,
    candidatePeak,
    controlMean: meanControl,
    candidateMean: meanCandidate,
    finite,
    passed,
  });
}

function operationFamily(
  operation: AceVaeDecoderOperation,
): Exclude<Opt0024C512Family, "mixed"> | null {
  switch (operation.kind) {
    case "conv1d":
      return operation.shape.kernelSize === 7
        ? "k7-conv1d"
        : operation.shape.kernelSize === 1 ? "k1-conv1d" : null;
    case "conv-transpose1d": return "conv-transpose1d";
    case "snake": return "snake";
    case "add": return "add";
  }
}

function isWorkerScope(): boolean {
  return typeof WorkerGlobalScope !== "undefined" &&
    globalThis instanceof WorkerGlobalScope &&
    new URL(globalThis.location.href).searchParams.get(WORKER_QUERY) === "1";
}

if (isWorkerScope()) {
  installWorker();
} else if (typeof document !== "undefined") {
  installPage();
}

function installPage(): void {
  const prepareButton = requireElement<HTMLButtonElement>("prepare");
  const runButton = requireElement<HTMLButtonElement>("run");
  const progress = requireElement<HTMLDivElement>("progress");
  const result = requireElement<HTMLPreElement>("result");
  let worker: Worker | undefined;

  const setStatus = (status: string, message: string): void => {
    document.body.dataset["status"] = status;
    progress.textContent = message;
  };
  const ensureWorker = (): Worker => {
    if (worker !== undefined) return worker;
    const url = new URL(import.meta.url);
    url.searchParams.set(WORKER_QUERY, "1");
    worker = new Worker(url, { type: "module" });
    worker.addEventListener("message", (event: MessageEvent<WorkerEvent>) => {
      const data = event.data;
      if (data.type === "progress") {
        setStatus(document.body.dataset["status"] ?? "working",
          data.message ?? "working");
      } else if (data.type === "prepared") {
        prepareButton.disabled = true;
        runButton.disabled = false;
        setStatus("prepared", data.message ?? "prepared and warm");
        result.textContent = JSON.stringify(data.receipt, null, 2);
      } else if (data.type === "result") {
        runButton.disabled = true;
        const raw = JSON.stringify(data.receipt, null, 2);
        (globalThis as unknown as Record<string, unknown>)[RAW_RESULT_GLOBAL] =
          raw;
        result.textContent = raw;
        setStatus(
          data.receipt?.["status"] === "passed" ? "passed" : "failed",
          data.receipt?.["status"] === "passed"
            ? "timed C512 AB/BA passed"
            : "timed C512 AB/BA completed with a failed gate",
        );
      } else if (data.type === "error") {
        prepareButton.disabled = true;
        runButton.disabled = true;
        setStatus("error", data.message ?? "worker failed");
        result.textContent = data.message ?? "worker failed";
      }
    });
    worker.addEventListener("error", (event) => {
      prepareButton.disabled = true;
      runButton.disabled = true;
      setStatus("error", event.message || "worker error");
    });
    return worker;
  };

  prepareButton.addEventListener("click", () => {
    prepareButton.disabled = true;
    setStatus("preparing", "starting dedicated-worker preparation");
    ensureWorker().postMessage({ type: "prepare" } satisfies WorkerCommand);
  });
  runButton.addEventListener("click", () => {
    runButton.disabled = true;
    setStatus("running", "running the one timed AB/BA gate");
    ensureWorker().postMessage({ type: "run" } satisfies WorkerCommand);
  });
}

function installWorker(): void {
  let prepared: PreparedGate | undefined;
  let operation: Promise<void> = Promise.resolve();
  globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
    operation = operation.then(async () => {
      if (event.data.type === "prepare") {
        if (prepared !== undefined) throw new Error("C512 is already prepared");
        prepared = await prepareGate();
        postWorker({
          type: "prepared",
          message: "prepared and warm; awaiting external thermal gate",
          receipt: prepared.preparationReceipt,
        });
      } else if (event.data.type === "run") {
        if (prepared === undefined) throw new Error("C512 is not prepared");
        const retained = prepared;
        prepared = undefined;
        try {
          const receipt = await runTimedGate(retained);
          postWorker({ type: "result", receipt });
        } finally {
          retained.destroy();
        }
      }
    }).catch((error) => {
      prepared?.destroy();
      prepared = undefined;
      postWorker({ type: "error", message: errorText(error) });
    });
  });
}

async function prepareGate(): Promise<PreparedGate> {
  postProgress("authenticating C512 fixture and current FP16 VAE package");
  const fixtureBytes = createAceOpt0011LatentFixture(OPT_0024_C512_FRAMES);
  if (
    fixtureBytes.byteLength !== OPT_0024_C512_INPUT_ELEMENTS * FLOAT32_BYTES ||
    await sha256Hex(fixtureBytes) !== OPT_0024_C512_FIXTURE_SHA256
  ) throw new Error("OPT-0024 C512 fixture identity changed");
  const fixture = new Float32Array(OPT_0024_C512_INPUT_ELEMENTS);
  fixture.set(new Float32Array(
    fixtureBytes.buffer,
    fixtureBytes.byteOffset,
    OPT_0024_C512_INPUT_ELEMENTS,
  ));
  const pkg = await authenticatePackage();
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    requiredFeatures: ["subgroups"],
  });
  let phase: AceGpuTensorPhase | undefined;
  let resources: C512Resources | undefined;
  let shippedRuntime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
  let candidateRuntime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
  try {
    requireFixed32Subgroups(context);
    const files = await acquirePackageFiles(pkg);
    postProgress("uploading the one resident VAE package");
    phase = await AceGpuTensorPhase.load(
      context.device,
      pkg.loaded.manifest,
      files,
      ["vae"],
      { onProgress: (progress) => postProgress(
        `uploading VAE ${progress.phaseFileIndex + 1}/` +
          `${progress.phaseFileCount}: ${progress.loadedPhaseBytes}/` +
          `${progress.totalPhaseBytes} bytes`,
      ) },
    );
    if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
      throw new Error("OPT-0024 resident VAE byte count changed");
    }
    const packageBindings = resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      pkg.loaded,
      phase,
    );
    resources = createC512Resources(context.device);
    const bindings = runtimeBindings(resources, packageBindings);
    const runtimeOptions = Object.freeze({
      runtimeProfileId:
        "opt-0028-mixed-fp16-fixed32-exact-packed-v1" as const,
      subgroupMinSize: 32 as const,
      subgroupMaxSize: 32 as const,
    });
    shippedRuntime = AceOpt0011Fp16VaeDecoderRuntime.create(
      context.device,
      runtimeOptions,
    );
    candidateRuntime = AceOpt0011Fp16VaeDecoderRuntime.create(
      context.device,
      runtimeOptions,
    );
    replaceK7Kernel(context.device, candidateRuntime);
    postProgress("building both complete C512 dispatch graphs");
    const shippedSet = await shippedRuntime.createChunkDispatchSet(
      "opt-0024-c512-shipped",
      OPT_0024_C512_FRAMES,
      OPT_0024_C512_FRAMES,
      bindings,
    );
    const candidateSet = await candidateRuntime.createChunkDispatchSet(
      "opt-0024-c512-candidate",
      OPT_0024_C512_FRAMES,
      OPT_0024_C512_FRAMES,
      bindings,
    );
    const shipped = validateSet(shippedSet, "shipped");
    const candidate = validateSet(candidateSet, "candidate");
    const dispatches = Object.freeze({ shipped, candidate });
    postProgress("running complete shipped C512 warmup");
    const warmShipped = await executeC512(
      context.device,
      resources,
      fixture,
      shipped,
      "shipped",
      false,
    );
    postProgress("running complete candidate C512 warmup");
    const warmCandidate = await executeC512(
      context.device,
      resources,
      fixture,
      candidate,
      "candidate",
      false,
    );
    const warmMetrics = compareAllChannels(
      warmShipped.output,
      warmCandidate.output,
    );
    if (!allMetricsPassed(warmMetrics)) {
      throw new Error("OPT-0024 C512 warmup waveform envelope failed");
    }
    const preparationReceipt = Object.freeze({
      schema: OPT_0024_C512_SCHEMA,
      status: "prepared",
      experimentId: EXPERIMENT_ID,
      fixture: Object.freeze({
        frames: OPT_0024_C512_FRAMES,
        elements: OPT_0024_C512_INPUT_ELEMENTS,
        byteLength: fixtureBytes.byteLength,
        sha256: OPT_0024_C512_FIXTURE_SHA256,
      }),
      package: Object.freeze({
        manifestSha256: pkg.loaded.manifestSha256,
        manifestByteLength: pkg.loaded.manifestByteLength,
        converterRevision:
          pkg.loaded.manifest.provenance.converterRevision,
        residentBytes: phase.residentBytes,
      }),
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        adapterInfo: context.capabilities.adapterInfo,
        deviceFeatures: context.capabilities.deviceFeatures,
        deviceLimits: context.capabilities.deviceLimits,
      }),
      graph: compactGraph(dispatches),
      warmup: Object.freeze({
        order: OPT_0024_C512_WARMUP_ORDER,
        completeRunsPerArm: 1,
        shippedOutputSha256: await sha256Float32(warmShipped.output),
        candidateOutputSha256: await sha256Float32(warmCandidate.output),
        metrics: warmMetrics,
        waveformEnvelopePassed: true,
        performanceClaim: null,
      }),
      stockWebGpuOnly: true,
      experimentalBrowserFlags: false,
      readyForExternalThermalGate: true,
    });
    let destroyed = false;
    return Object.freeze({
      context,
      phase,
      resources,
      shippedRuntime,
      candidateRuntime,
      dispatches,
      fixture,
      preparationReceipt,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        shippedRuntime!.destroy();
        candidateRuntime!.destroy();
        resources!.destroy();
        phase!.destroy();
        context.destroy();
      },
    });
  } catch (error) {
    shippedRuntime?.destroy();
    candidateRuntime?.destroy();
    resources?.destroy();
    phase?.destroy();
    context.destroy();
    throw error;
  }
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const executions: ExecutionResult[] = [];
  for (const [index, arm] of OPT_0024_C512_TIMED_ORDER.entries()) {
    postProgress(
      `timed C512 ${index + 1}/${OPT_0024_C512_TIMED_ORDER.length}: ${arm}`,
    );
    executions.push(await executeC512(
      prepared.context.device,
      prepared.resources,
      prepared.fixture,
      prepared.dispatches[arm],
      arm,
      true,
    ));
  }
  const ab = pairedGate(executions[0]!, executions[1]!, "AB");
  const ba = pairedGate(executions[3]!, executions[2]!, "BA");
  const candidateRepeat = compareRawU32(
    executions[1]!.output,
    executions[2]!.output,
  );
  const shippedRepeat = compareRawU32(
    executions[0]!.output,
    executions[3]!.output,
  );
  const outputHashes = await Promise.all(executions.map((execution) =>
    sha256Float32(execution.output)
  ));
  const runtimeEvents: unknown[] = [];
  const passed = ab.passed === true && ba.passed === true &&
    candidateRepeat.mismatchCount === 0 && shippedRepeat.mismatchCount === 0;
  return Object.freeze({
    schema: OPT_0024_C512_SCHEMA,
    status: passed ? "passed" : "failed",
    experimentId: EXPERIMENT_ID,
    classification:
      "complete-C512-control-candidate-balanced-ABBA-stock-WebGPU",
    environment: prepared.preparationReceipt["environment"],
    fixture: prepared.preparationReceipt["fixture"],
    package: prepared.preparationReceipt["package"],
    graph: prepared.preparationReceipt["graph"],
    protocol: Object.freeze({
      warmupOrder: OPT_0024_C512_WARMUP_ORDER,
      timedOrder: OPT_0024_C512_TIMED_ORDER,
      externalThermalGateRequiredBeforeTimedAction: true,
      completeRunsPerTimedArm: 2,
      quantaPerCommandBuffer:
        ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyMillisecondsBetweenCommandBuffers: 1,
      stockWebGpuOnly: true,
      wasmOrJavascriptHostOnly: true,
      experimentalBrowserFlags: false,
      webNn: false,
    }),
    executions: Object.freeze(executions.map((execution, index) =>
      Object.freeze({
        index,
        arm: execution.arm,
        outputSha256: outputHashes[index],
        ...execution.receipt,
      })
    )),
    pairs: Object.freeze({ AB: ab, BA: ba }),
    deterministic: Object.freeze({
      candidate: candidateRepeat,
      shipped: shippedRepeat,
      passed: candidateRepeat.mismatchCount === 0 &&
        shippedRepeat.mismatchCount === 0,
    }),
    runtimeEvents,
    cleanup: Object.freeze({
      cleanupRunsAfterReceiptConstruction: true,
      bothRuntimesOnePackageAndSharedWorkspaceOwned: true,
    }),
  });
}

function pairedGate(
  shipped: ExecutionResult,
  candidate: ExecutionResult,
  order: "AB" | "BA",
): Readonly<Record<string, unknown>> {
  if (shipped.arm !== "shipped" || candidate.arm !== "candidate") {
    throw new Error(`OPT-0024 ${order} pair arm order changed`);
  }
  const shippedTiming = shipped.receipt["timing"] as
    Readonly<Record<string, unknown>>;
  const candidateTiming = candidate.receipt["timing"] as
    Readonly<Record<string, unknown>>;
  const shippedFamilies = shippedTiming["families"] as
    Readonly<Record<Opt0024C512Family, Opt0024C512FamilyTiming>>;
  const candidateFamilies = candidateTiming["families"] as
    Readonly<Record<Opt0024C512Family, Opt0024C512FamilyTiming>>;
  const shippedK7 = shippedFamilies["k7-conv1d"].submitThroughDrainMs;
  const candidateK7 = candidateFamilies["k7-conv1d"].submitThroughDrainMs;
  const k7Speedup = shippedK7 / candidateK7;
  const k7SavingMs = shippedK7 - candidateK7;
  const shippedDecoder = Number(shippedTiming["decoderSubmitThroughDrainMs"]);
  const candidateDecoder = Number(candidateTiming["decoderSubmitThroughDrainMs"]);
  const shippedOuter = Number(shippedTiming["outerWindowWallMs"]);
  const candidateOuter = Number(candidateTiming["outerWindowWallMs"]);
  const metrics = compareAllChannels(shipped.output, candidate.output);
  const waveformPassed = allMetricsPassed(metrics);
  const passed = waveformPassed && k7Speedup >= K7_SPEEDUP_MINIMUM &&
    k7SavingMs >= K7_SAVING_MINIMUM_MS &&
    candidateDecoder <= shippedDecoder && candidateOuter <= shippedOuter;
  return Object.freeze({
    order,
    timing: Object.freeze({
      shippedK7Ms: shippedK7,
      candidateK7Ms: candidateK7,
      k7Speedup,
      k7SavingMs,
      shippedDecoderSubmitThroughDrainMs: shippedDecoder,
      candidateDecoderSubmitThroughDrainMs: candidateDecoder,
      shippedOuterWindowWallMs: shippedOuter,
      candidateOuterWindowWallMs: candidateOuter,
      k7SpeedupPassed: k7Speedup >= K7_SPEEDUP_MINIMUM,
      k7SavingPassed: k7SavingMs >= K7_SAVING_MINIMUM_MS,
      decoderNoSlowerPassed: candidateDecoder <= shippedDecoder,
      outerNoSlowerPassed: candidateOuter <= shippedOuter,
    }),
    metrics,
    waveformPassed,
    passed,
  });
}

async function executeC512(
  device: GPUDevice,
  resources: C512Resources,
  fixture: Float32Array<ArrayBuffer>,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  arm: Opt0024C512Arm,
  reportTiming: boolean,
): Promise<ExecutionResult> {
  if (
    fixture.length !== OPT_0024_C512_INPUT_ELEMENTS ||
    dispatch.activeOutputBytes !== OPT_0024_C512_OUTPUT_BYTES
  ) throw new Error("OPT-0024 C512 execution geometry changed");
  const outputPoison = new Uint32Array(OPT_0024_C512_OUTPUT_ELEMENTS);
  outputPoison.fill(QNAN_WORD);
  const outerStarted = performance.now();
  device.queue.writeBuffer(resources.stagingInput, 0, fixture);
  device.queue.writeBuffer(resources.output, 0, outputPoison);
  const batchProfiles = profileBatches(dispatch);
  const drained: AceGpuCommandBufferDrainTiming[] = [];
  const decoderBatchCount = Math.ceil(
    dispatch.quanta.length /
      ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  );
  const scheduling = await submitAceCommandBufferFactoriesCooperatively({
    queue: device.queue,
    commandBufferCount: decoderBatchCount + 1,
    createCommandBuffer: (index) => index < decoderBatchCount
      ? encodeBatch(device, dispatch, index, arm)
      : encodeReadback(device, resources, arm),
    signal: new AbortController().signal,
    onCommandBufferDrained: (timing) => drained.push(timing),
  });
  const mapStarted = performance.now();
  await resources.readback.mapAsync(
    GPUMapMode.READ,
    0,
    OPT_0024_C512_OUTPUT_BYTES,
  );
  let output: Float32Array<ArrayBuffer>;
  try {
    output = new Float32Array(OPT_0024_C512_OUTPUT_ELEMENTS);
    output.set(new Float32Array(
      resources.readback.getMappedRange(0, OPT_0024_C512_OUTPUT_BYTES),
    ));
  } finally {
    resources.readback.unmap();
  }
  const mapMs = performance.now() - mapStarted;
  const outerWindowWallMs = performance.now() - outerStarted;
  const scan = scanOutput(output);
  if (
    scan.elementCount !== OPT_0024_C512_OUTPUT_ELEMENTS ||
    scan.nonFiniteCount !== 0 || scan.qNaNSentinelCount !== 0 ||
    scan.nonzeroCount === 0 || scan.stereoDifferenceFrameCount === 0
  ) throw new Error(`OPT-0024 ${arm} produced an invalid C512 waveform`);
  if (drained.length !== decoderBatchCount + 1) {
    throw new Error(`OPT-0024 ${arm} drain timing was incomplete`);
  }
  const familyTiming = emptyFamilyTimings();
  let decoderSubmitThroughDrainMs = 0;
  for (let index = 0; index < decoderBatchCount; index += 1) {
    const timing = drained[index]!.submitThroughDrainMs;
    decoderSubmitThroughDrainMs += timing;
    const profile = batchProfiles[index]!;
    const total = familyTiming[profile.family];
    total.batchCount += 1;
    total.quantumCount += profile.quantumCount;
    total.submitThroughDrainMs += timing;
  }
  const readbackSubmitThroughDrainMs = drained[decoderBatchCount]!
    .submitThroughDrainMs;
  const families = freezeFamilyTimings(familyTiming);
  if (
    families["k7-conv1d"].batchCount !== OPT_0024_C512_K7_PURE_BATCHES ||
    families["k7-conv1d"].quantumCount !== OPT_0024_C512_K7_PURE_QUANTA
  ) throw new Error(`OPT-0024 ${arm} C512 K7 batch topology changed`);
  return Object.freeze({
    arm,
    output,
    receipt: Object.freeze({
      scan,
      timing: Object.freeze({
        measured: reportTiming,
        quantaPerCommandBuffer:
          ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
        decoderBatchCount,
        decoderQuantumCount: dispatch.quanta.length,
        decoderSubmitThroughDrainMs,
        readbackSubmitThroughDrainMs,
        mapMs,
        requestedCooperativeIdleMs: scheduling.cooperativeIdleMs,
        outerWindowWallMs,
        families,
      }),
    }),
  });
}

function profileBatches(
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
): readonly Readonly<{
  family: Opt0024C512Family;
  quantumCount: number;
}>[] {
  const qpc = ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
  return Object.freeze(Array.from(
    { length: Math.ceil(dispatch.quanta.length / qpc) },
    (_, index) => {
      const first = index * qpc;
      const quanta = dispatch.quanta.slice(
        first,
        Math.min(first + qpc, dispatch.quanta.length),
      );
      return Object.freeze({
        family: classifyOpt0024C512Batch(quanta, dispatch.plan.operations),
        quantumCount: quanta.length,
      });
    },
  ));
}

function encodeBatch(
  device: GPUDevice,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  batchIndex: number,
  arm: Opt0024C512Arm,
): GPUCommandBuffer {
  const qpc = ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
  const first = batchIndex * qpc;
  const end = Math.min(first + qpc, dispatch.quanta.length);
  const encoder = device.createCommandEncoder({
    label: `opt-0024-c512-${arm}-batch-${batchIndex}`,
  });
  const pass = encoder.beginComputePass({
    label: `opt-0024-c512-${arm}-batch-${batchIndex}-pass`,
  });
  for (let index = first; index < end; index += 1) {
    dispatch.quanta[index]!.encode(pass);
  }
  pass.end();
  return encoder.finish();
}

function encodeReadback(
  device: GPUDevice,
  resources: C512Resources,
  arm: Opt0024C512Arm,
): GPUCommandBuffer {
  const encoder = device.createCommandEncoder({
    label: `opt-0024-c512-${arm}-readback`,
  });
  encoder.copyBufferToBuffer(
    resources.output,
    0,
    resources.readback,
    0,
    OPT_0024_C512_OUTPUT_BYTES,
  );
  return encoder.finish();
}

function validateSet(
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  arm: Opt0024C512Arm,
): AceOpt0011Fp16VaeWindowDispatch {
  if (
    set.dispatches.length !== 1 || set.windows.length !== 1 ||
    set.windows[0]!.window.latentWindowFrames !== OPT_0024_C512_FRAMES
  ) throw new Error(`OPT-0024 ${arm} did not build one exact C512 window`);
  const dispatch = set.dispatches[0]!;
  if (
    dispatch.operationCount !== 88 ||
    dispatch.graphQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT ||
    dispatch.quanta.length !==
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
    dispatch.commandBufferCountAtBatch8 !==
      ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8 ||
    dispatch.activeOutputBytes !== OPT_0024_C512_OUTPUT_BYTES
  ) throw new Error(`OPT-0024 ${arm} C512 topology changed`);
  const k7 = dispatch.quanta.filter((quantum) =>
    quantum.operationIndex !== null &&
    dispatch.plan.operations[quantum.operationIndex]?.kind === "conv1d" &&
    (dispatch.plan.operations[quantum.operationIndex] as Extract<
      AceVaeDecoderOperation,
      { kind: "conv1d" }
    >).shape.kernelSize === 7
  );
  const expectedKernelId = arm === "shipped"
    ? ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
    : ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID;
  if (k7.length !== OPT_0024_C512_K7_TOTAL_QUANTA) {
    throw new Error(
      `OPT-0024 ${arm} K7 quantum count changed: ${k7.length}`,
    );
  }
  if (arm === "shipped") {
    if (k7.some((quantum) => quantum.kernelId !== expectedKernelId)) {
      throw new Error("OPT-0024 shipped K7 dispatch ownership changed");
    }
  } else {
    const changed = k7.filter((quantum) =>
      quantum.kernelId === expectedKernelId
    );
    const fallback = k7.filter((quantum) =>
      quantum.kernelId === ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
    );
    const changedOperations = new Set(changed.map((quantum) =>
      quantum.operationIndex
    ));
    const invalidFallbackCount = fallback.filter((quantum) => {
      const operation = dispatch.plan.operations[quantum.operationIndex!];
      return operation?.kind !== "conv1d" || operation.bias !== undefined;
    }).length;
    if (
      changedOperations.size !== 16 ||
      changed.length !== OPT_0024_C512_K7_CANDIDATE_QUANTA ||
      fallback.length !==
        OPT_0024_C512_K7_TOTAL_QUANTA -
          OPT_0024_C512_K7_CANDIDATE_QUANTA ||
      changed.length + fallback.length !== k7.length ||
      invalidFallbackCount !== 0
    ) throw new Error(
      "OPT-0024 candidate biased-K7 ownership changed: " +
        `changed=${changed.length}, fallback=${fallback.length}, ` +
        `operations=${changedOperations.size}, invalidFallback=` +
        `${invalidFallbackCount}`,
    );
  }
  return dispatch;
}

function replaceK7Kernel(
  device: GPUDevice,
  runtime: AceOpt0011Fp16VaeDecoderRuntime,
): void {
  type Replaceable = {
    subgroupConv1d: AceFp16VaeConv1dSubgroupKernel;
  };
  const replaceable = runtime as unknown as Replaceable;
  const shipped = replaceable.subgroupConv1d;
  const candidate = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
    device,
    {
      subgroupMinSize: REQUIRED_SUBGROUP_SIZE,
      subgroupMaxSize: REQUIRED_SUBGROUP_SIZE,
    },
  );
  type CreateDispatch = AceFp16VaeConv1dSubgroupKernel["createDispatch"];
  let destroyed = false;
  replaceable.subgroupConv1d = Object.freeze({
    async createDispatch(
      ...args: Parameters<CreateDispatch>
    ): Promise<Awaited<ReturnType<CreateDispatch>>> {
      const bindings = args[2];
      const dispatch = bindings.bias === undefined
        ? await shipped.createDispatch(...args)
        : await candidate.createDispatch(...args);
      return dispatch as unknown as Awaited<ReturnType<CreateDispatch>>;
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      shipped.destroy();
      candidate.destroy();
    },
  }) as unknown as AceFp16VaeConv1dSubgroupKernel;
}

function createC512Resources(device: GPUDevice): C512Resources {
  const graph = planAceVaeDecoder(OPT_0024_C512_FRAMES);
  const create = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const stagingInput = create(
    "opt-0024-c512-staging-input",
    graph.inputElements * FLOAT32_BYTES,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const decoderInput = create(
    "opt-0024-c512-decoder-input",
    graph.inputElements * FLOAT16_BYTES,
    GPUBufferUsage.STORAGE,
  );
  const workspaces = Object.freeze([0, 1, 2].map((index) => create(
    `opt-0024-c512-workspace-${index}`,
    ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    GPUBufferUsage.STORAGE,
  ))) as readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  const output = create(
    "opt-0024-c512-output",
    OPT_0024_C512_OUTPUT_BYTES,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const readback = create(
    "opt-0024-c512-readback",
    OPT_0024_C512_OUTPUT_BYTES,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  );
  let destroyed = false;
  return Object.freeze({
    stagingInput,
    decoderInput,
    workspaces,
    output,
    readback,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      stagingInput.destroy();
      decoderInput.destroy();
      for (const workspace of workspaces) workspace.destroy();
      output.destroy();
      readback.destroy();
    },
  });
}

function runtimeBindings(
  resources: C512Resources,
  packageBindings: AceOpt0011VaePackageBindings,
) {
  const binding = (buffer: GPUBuffer): GPUBufferBinding =>
    Object.freeze({ buffer, offset: 0, size: Number(buffer.size) });
  return Object.freeze({
    stagingInput: binding(resources.stagingInput),
    decoderInput: binding(resources.decoderInput),
    workspaces: Object.freeze(resources.workspaces.map(binding)) as readonly [
      GPUBufferBinding,
      GPUBufferBinding,
      GPUBufferBinding,
    ],
    output: binding(resources.output),
    package: packageBindings,
  });
}

async function authenticatePackage(): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(MANIFEST_PATH, globalThis.location.href).href,
    expectedManifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
  });
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) =>
    shardNames.has(file.name)
  );
  const residentBytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  if (
    loaded.manifestSha256 !== ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES ||
    loaded.manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== 7 || shardNames.size !== 7 ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) =>
      !shardNames.has(name)
    )
  ) throw new Error("OPT-0024 authenticated package identity changed");
  return Object.freeze({ loaded, files: Object.freeze(files), residentBytes });
}

async function acquirePackageFiles(
  prepared: PreparedPackage,
): Promise<ReadonlyMap<string, File>> {
  const cache = await AceOpfsModelCache.open();
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({
      ...prepared.loaded.manifest,
      files: prepared.files,
    }),
    manifestUrl: prepared.loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => postProgress(
      `acquiring VAE ${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== prepared.files.length ||
    acquired.plan.runtimeBytes !== prepared.residentBytes
  ) throw new Error("OPT-0024 package acquisition accounting changed");
  return acquired.files;
}

function requireFixed32Subgroups(context: AceWebGpuDeviceContext): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== REQUIRED_SUBGROUP_SIZE ||
    info.subgroupMaxSize !== REQUIRED_SUBGROUP_SIZE
  ) throw new Error("OPT-0024 requires stock fixed 32-lane WebGPU subgroups");
}

function compactGraph(
  dispatches: Readonly<Record<Opt0024C512Arm,
    AceOpt0011Fp16VaeWindowDispatch>>,
): Readonly<Record<string, unknown>> {
  const batches = profileBatches(dispatches.shipped);
  const familyCounts = emptyFamilyTimings();
  for (const batch of batches) {
    familyCounts[batch.family].batchCount += 1;
    familyCounts[batch.family].quantumCount += batch.quantumCount;
  }
  return Object.freeze({
    operationCount: dispatches.shipped.operationCount,
    graphQuantumCount: dispatches.shipped.graphQuantumCount,
    sequenceQuantumCount: dispatches.shipped.quanta.length,
    commandBufferCountAtBatch8:
      dispatches.shipped.commandBufferCountAtBatch8,
    activeOutputBytes: dispatches.shipped.activeOutputBytes,
    maximumWorkspaceBytes: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    familyBatches: freezeFamilyTimings(familyCounts),
    shippedK7KernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    candidateK7KernelId:
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
    onlyK7DispatchOwnerChanged: true,
  });
}

function compareAllChannels(
  control: Float32Array,
  candidate: Float32Array,
): Readonly<Record<"joint" | "left" | "right", Opt0024C512Metrics>> {
  return Object.freeze({
    joint: compareOpt0024C512Waveforms(control, candidate),
    left: compareOpt0024C512Waveforms(control, candidate, 2, 0),
    right: compareOpt0024C512Waveforms(control, candidate, 2, 1),
  });
}

function allMetricsPassed(
  metrics: Readonly<Record<"joint" | "left" | "right",
    Opt0024C512Metrics>>,
): boolean {
  return metrics.joint.passed && metrics.left.passed && metrics.right.passed;
}

function compareRawU32(
  a: Float32Array,
  b: Float32Array,
): Readonly<{ comparedElementCount: number; mismatchCount: number }> {
  if (a.length !== b.length) throw new RangeError("C512 repeat length changed");
  const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let mismatchCount = 0;
  for (let index = 0; index < aBits.length; index += 1) {
    if (aBits[index] !== bBits[index]) mismatchCount += 1;
  }
  return Object.freeze({ comparedElementCount: a.length, mismatchCount });
}

function scanOutput(output: Float32Array): Readonly<Record<string, unknown>> {
  const bits = new Uint32Array(output.buffer, output.byteOffset, output.length);
  let nonFiniteCount = 0;
  let qNaNSentinelCount = 0;
  let nonzeroCount = 0;
  let stereoDifferenceFrameCount = 0;
  let clampBoundaryCount = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let peak = 0;
  for (let index = 0; index < output.length; index += 1) {
    const value = output[index]!;
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    if (bits[index] === QNAN_WORD) qNaNSentinelCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (value === -1 || value === 1) clampBoundaryCount += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    peak = Math.max(peak, Math.abs(value));
    if (index % 2 === 0 && output[index + 1] !== value) {
      stereoDifferenceFrameCount += 1;
    }
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    nonFiniteCount,
    qNaNSentinelCount,
    nonzeroCount,
    stereoDifferenceFrameCount,
    clampBoundaryCount,
    minimum,
    maximum,
    peak,
  });
}

type MutableFamilyTiming = Record<
  Opt0024C512Family,
  { batchCount: number; quantumCount: number; submitThroughDrainMs: number }
>;

function emptyFamilyTimings(): MutableFamilyTiming {
  const empty = () => ({
    batchCount: 0,
    quantumCount: 0,
    submitThroughDrainMs: 0,
  });
  return {
    "k7-conv1d": empty(),
    "k1-conv1d": empty(),
    "conv-transpose1d": empty(),
    snake: empty(),
    add: empty(),
    mixed: empty(),
  };
}

function freezeFamilyTimings(
  mutable: MutableFamilyTiming,
): Readonly<Record<Opt0024C512Family, Opt0024C512FamilyTiming>> {
  return Object.freeze(Object.fromEntries(
    Object.entries(mutable).map(([family, timing]) => [
      family,
      Object.freeze({ ...timing }),
    ]),
  )) as Readonly<Record<Opt0024C512Family, Opt0024C512FamilyTiming>>;
}

async function sha256Float32(values: Float32Array): Promise<string> {
  return await sha256Hex(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest(
    "SHA-256",
    bytes as Uint8Array<ArrayBuffer>,
  ));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function postProgress(message: string): void {
  postWorker({ type: "progress", message });
}

function postWorker(event: WorkerEvent): void {
  (globalThis as unknown as DedicatedWorkerGlobalScope).postMessage(event);
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
}

function errorText(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}\n${error.stack ?? ""}`
    : String(error);
}
