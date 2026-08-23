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
import type { AceVaeChunkGpuBackendProgress } from
  "../../src/webgpu/vae-backend.js";
import {
  ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  AceOpt0011Fp16VaeChunkGpuBackend,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
} from "../../src/webgpu/vae-chunks.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";

declare global {
  interface Window {
    __ACE_OPT0027_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export const OPT_0027_SCHEMA =
  "ace-opt-0027-vae-batch64-c512-ab-v1" as const;
export const OPT_0027_C512_FRAMES = 512 as const;
export const OPT_0027_C512_INPUT_ELEMENTS = 32_768 as const;
export const OPT_0027_C512_OUTPUT_ELEMENTS = 1_966_080 as const;
export const OPT_0027_C512_FIXTURE_SHA256 =
  "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899" as const;
export const OPT_0027_C512_ACCEPTED_OUTPUT_SHA256 =
  "893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47" as const;
export const OPT_0027_QUANTA_PER_COMMAND_BUFFER = Object.freeze({
  batch8: ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  batch64: ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
} as const);
export const OPT_0027_WARMUP_ORDER = Object.freeze([
  "batch8",
  "batch64",
] as const);
export const OPT_0027_TIMED_ORDER = Object.freeze([
  "batch8",
  "batch64",
  "batch64",
  "batch8",
] as const);

const EXPERIMENT_ID = "OPT-0027" as const;
const MANIFEST_PATH = "/model/files-fp16-vae-experimental/manifest.json";
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const REQUIRED_SUBGROUP_SIZE = 32;
const WORKER_QUERY = "opt0027Worker";

export type Opt0027Arm = keyof typeof OPT_0027_QUANTA_PER_COMMAND_BUFFER;

export interface Opt0027SchedulingPlan {
  readonly quantaPerCommandBuffer: 8 | 64;
  readonly decoderQuantumCount: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  readonly queueDrainCount: number;
  readonly requestedCooperativeIdleMs: number;
}

export interface Opt0027RawComparison {
  readonly comparedU32WordCount: number;
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
  readonly firstControlWord: number | null;
  readonly firstCandidateWord: number | null;
  readonly rawFp32U32Exact: boolean;
}

export interface Opt0027ExecutionTiming {
  readonly outerWindowWallMs: number;
  readonly decoderSubmitThroughDrainMs: number;
  readonly decoderCommandBufferCount: number;
  readonly totalCommandBufferCount: number;
  readonly queueDrainCount: number;
  readonly requestedCooperativeIdleMs: number;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly residentBytes: number;
}

interface ObserverSummary {
  readonly progressEventCount: number;
  readonly finalProgress: AceVaeChunkGpuBackendProgress;
  readonly familyProfile: AceOpt0011Fp16VaeWindowFamilyProfile;
}

interface ExecutionResult {
  readonly arm: Opt0027Arm;
  readonly output: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<{
    arm: Opt0027Arm;
    measured: boolean;
    scan: Readonly<Record<string, number>>;
    timing: Opt0027ExecutionTiming;
    familyProfile: AceOpt0011Fp16VaeWindowFamilyProfile;
  }>;
}

interface PreparedGate {
  readonly context: AceWebGpuDeviceContext;
  readonly backends: Readonly<Record<
    Opt0027Arm,
    AceOpt0011Fp16VaeChunkGpuBackend
  >>;
  readonly observers: Readonly<Record<Opt0027Arm, RunObserver>>;
  readonly window: AceVaeDecodeWindow;
  readonly warmReference: Float32Array<ArrayBuffer>;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  destroy(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

type WorkerCommand = Readonly<{ readonly type: "prepare" | "run" | "dispose" }>;

interface WorkerEvent {
  readonly type: "progress" | "prepared" | "result" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
}

export function planOpt0027Scheduling(
  quantaPerCommandBuffer: number,
): Opt0027SchedulingPlan {
  if (
    quantaPerCommandBuffer !==
      ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER &&
    quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER
  ) {
    throw new RangeError("OPT-0027 scheduling requires batch8 or batch64");
  }
  const decoderCommandBufferCount = Math.ceil(
    ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT /
      quantaPerCommandBuffer,
  );
  const totalCommandBufferCount = decoderCommandBufferCount + 1;
  return Object.freeze({
    quantaPerCommandBuffer,
    decoderQuantumCount:
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
    decoderCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount,
    queueDrainCount: totalCommandBufferCount,
    requestedCooperativeIdleMs: totalCommandBufferCount - 1,
  });
}

export function compareOpt0027RawFp32(
  control: Float32Array,
  candidate: Float32Array,
): Opt0027RawComparison {
  if (control.length !== candidate.length) {
    throw new RangeError("OPT-0027 output lengths differ");
  }
  const controlBits = new Uint32Array(
    control.buffer,
    control.byteOffset,
    control.length,
  );
  const candidateBits = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < controlBits.length; index += 1) {
    if (controlBits[index] === candidateBits[index]) continue;
    mismatchCount += 1;
    if (firstMismatchIndex === null) firstMismatchIndex = index;
  }
  return Object.freeze({
    comparedU32WordCount: controlBits.length,
    mismatchCount,
    firstMismatchIndex,
    firstControlWord: firstMismatchIndex === null
      ? null
      : controlBits[firstMismatchIndex]!,
    firstCandidateWord: firstMismatchIndex === null
      ? null
      : candidateBits[firstMismatchIndex]!,
    rawFp32U32Exact: mismatchCount === 0,
  });
}

export function summarizeOpt0027Pair(
  batch8: Opt0027ExecutionTiming,
  batch64: Opt0027ExecutionTiming,
  order: "batch8-batch64" | "batch64-batch8",
): Readonly<Record<string, number | string>> {
  for (const [label, timing] of [["batch8", batch8], ["batch64", batch64]] as const) {
    for (const [field, value] of Object.entries(timing)) {
      if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`OPT-0027 ${label} ${field} timing is invalid`);
      }
    }
  }
  return Object.freeze({
    order,
    batch8OuterWindowWallMs: batch8.outerWindowWallMs,
    batch64OuterWindowWallMs: batch64.outerWindowWallMs,
    outerWindowWallSpeedup:
      batch8.outerWindowWallMs / batch64.outerWindowWallMs,
    outerWindowWallSavingMs:
      batch8.outerWindowWallMs - batch64.outerWindowWallMs,
    batch8DecoderSubmitThroughDrainMs:
      batch8.decoderSubmitThroughDrainMs,
    batch64DecoderSubmitThroughDrainMs:
      batch64.decoderSubmitThroughDrainMs,
    decoderSubmitThroughDrainSpeedup:
      batch8.decoderSubmitThroughDrainMs /
        batch64.decoderSubmitThroughDrainMs,
    decoderSubmitThroughDrainSavingMs:
      batch8.decoderSubmitThroughDrainMs -
        batch64.decoderSubmitThroughDrainMs,
    decoderCommandBufferReduction:
      batch8.decoderCommandBufferCount - batch64.decoderCommandBufferCount,
    requestedCooperativeIdleReductionMs:
      batch8.requestedCooperativeIdleMs -
        batch64.requestedCooperativeIdleMs,
  });
}

class RunObserver {
  private active: {
    readonly label: string;
    progressEventCount: number;
    finalProgress?: AceVaeChunkGpuBackendProgress;
    familyProfileCount: number;
    familyProfile?: AceOpt0011Fp16VaeWindowFamilyProfile;
  } | undefined;

  begin(label: string): void {
    if (this.active !== undefined) {
      throw new Error("OPT-0027 observer already owns a run");
    }
    this.active = { label, progressEventCount: 0, familyProfileCount: 0 };
  }

  readonly onProgress = (progress: AceVaeChunkGpuBackendProgress): void => {
    const active = this.active;
    if (active === undefined) return;
    active.progressEventCount += 1;
    active.finalProgress = Object.freeze({ ...progress });
  };

  readonly onFamilyProfile = (
    profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  ): void => {
    const active = this.active;
    if (active === undefined) return;
    active.familyProfileCount += 1;
    active.familyProfile = profile;
  };

  finish(label: string): ObserverSummary {
    const active = this.active;
    this.active = undefined;
    if (
      active === undefined || active.label !== label ||
      active.finalProgress === undefined ||
      active.familyProfileCount !== 1 || active.familyProfile === undefined
    ) {
      throw new Error(`OPT-0027 ${label} observation is incomplete`);
    }
    return Object.freeze({
      progressEventCount: active.progressEventCount,
      finalProgress: active.finalProgress,
      familyProfile: active.familyProfile,
    });
  }

  cancel(label: string): void {
    if (this.active?.label === label) this.active = undefined;
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
  const runButton = requireElement<HTMLButtonElement>("run");
  const progress = requireElement<HTMLParagraphElement>("progress");
  const result = requireElement<HTMLPreElement>("result");
  const workerUrl = new URL(import.meta.url);
  workerUrl.searchParams.set(WORKER_QUERY, "1");
  const worker = new Worker(workerUrl, { type: "module" });

  const setStatus = (status: string, message: string): void => {
    document.body.dataset["status"] = status;
    progress.textContent = message;
  };
  worker.addEventListener("message", (event: MessageEvent<WorkerEvent>) => {
    const data = event.data;
    if (data.type === "progress") {
      setStatus(document.body.dataset["status"] ?? "preparing",
        data.message ?? "working");
      return;
    }
    if (data.type === "prepared") {
      setStatus("ready",
        "READY — exact C512 outputs match; timing has not run");
      result.textContent = JSON.stringify(data.receipt, null, 2);
      runButton.disabled = false;
      return;
    }
    if (data.type === "result") {
      const receipt: Readonly<Record<string, unknown>> =
        data.receipt ?? Object.freeze({});
      window.__ACE_OPT0027_RESULT__ = receipt;
      result.textContent = JSON.stringify(receipt, null, 2);
      setStatus(receipt["status"] === "passed" ? "passed" : "failed",
        receipt["status"] === "passed"
          ? "OPT-0027 balanced timing passed"
          : "OPT-0027 balanced timing failed a gate");
      worker.terminate();
      return;
    }
    runButton.disabled = true;
    const receipt = data.receipt ?? Object.freeze({
      schema: OPT_0027_SCHEMA,
      status: "failed",
      experimentId: EXPERIMENT_ID,
      error: data.message ?? "worker failed",
    });
    window.__ACE_OPT0027_RESULT__ = receipt;
    result.textContent = JSON.stringify(receipt, null, 2);
    setStatus("failed", data.message ?? "worker failed");
    worker.terminate();
  });
  worker.addEventListener("error", (event) => {
    runButton.disabled = true;
    setStatus("failed", event.message || "OPT-0027 worker failed");
  });
  runButton.addEventListener("click", () => {
    runButton.disabled = true;
    setStatus("running", "running balanced batch8/batch64 timing");
    worker.postMessage({ type: "run" } satisfies WorkerCommand);
  }, { once: true });
  window.addEventListener("beforeunload", () => {
    worker.postMessage({ type: "dispose" } satisfies WorkerCommand);
  }, { once: true });
  worker.postMessage({ type: "prepare" } satisfies WorkerCommand);
}

function installWorker(): void {
  let prepared: PreparedGate | undefined;
  let operation = Promise.resolve();
  globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
    operation = operation.then(async () => {
      const command = event.data;
      if (command.type === "prepare") {
        if (prepared !== undefined) {
          throw new Error("OPT-0027 is already prepared");
        }
        prepared = await prepareGate();
        postWorker({
          type: "prepared",
          message: "READY — exact C512 outputs match; timing has not run",
          receipt: prepared.preparationReceipt,
        });
        return;
      }
      if (command.type === "dispose") {
        const retained = prepared;
        prepared = undefined;
        await retained?.destroy(new DOMException(
          "OPT-0027 page disposed",
          "AbortError",
        ));
        return;
      }
      if (prepared === undefined) {
        throw new Error("OPT-0027 timing requested before READY");
      }
      const retained = prepared;
      let timed: Readonly<Record<string, unknown>>;
      let cleanup: Readonly<Record<string, unknown>>;
      try {
        timed = await runTimedGate(retained);
      } finally {
        cleanup = await retained.destroy();
        if (prepared === retained) prepared = undefined;
      }
      const passed = timed["status"] === "passed" &&
        cleanup["passed"] === true;
      postWorker({
        type: "result",
        receipt: Object.freeze({
          ...timed,
          status: passed ? "passed" : "failed",
          cleanup,
        }),
      });
    }).catch(async (error: unknown) => {
      const retained = prepared;
      prepared = undefined;
      const cleanup = await retained?.destroy(error);
      postWorker({
        type: "error",
        message: errorText(error),
        receipt: Object.freeze({
          schema: OPT_0027_SCHEMA,
          status: "failed",
          experimentId: EXPERIMENT_ID,
          error: errorText(error),
          ...(cleanup === undefined ? {} : { cleanup }),
        }),
      });
    });
  });
}

async function prepareGate(): Promise<PreparedGate> {
  const preparationStarted = performance.now();
  postProgress("authenticating exact C512 fixture and revision-6 package");
  const fixtureBytes = createAceOpt0011LatentFixture(OPT_0027_C512_FRAMES);
  if (
    fixtureBytes.byteLength !==
      OPT_0027_C512_INPUT_ELEMENTS * FLOAT32_BYTES ||
    await sha256Hex(fixtureBytes) !== OPT_0027_C512_FIXTURE_SHA256
  ) {
    throw new Error("OPT-0027 C512 fixture identity changed");
  }
  const fixture = new Float32Array(OPT_0027_C512_INPUT_ELEMENTS);
  fixture.set(new Float32Array(
    fixtureBytes.buffer,
    fixtureBytes.byteOffset,
    OPT_0027_C512_INPUT_ELEMENTS,
  ));
  const pkg = await authenticatePackage();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let batch8Phase: AceGpuTensorPhase | undefined;
  let batch64Phase: AceGpuTensorPhase | undefined;
  let batch8Backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  let batch64Backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });
    requireFixed32Subgroups(context);
    const files = await acquirePackageFiles(pkg);
    batch8Phase = await loadVaePhase(context.device, pkg, files, "batch8");
    batch64Phase = await loadVaePhase(context.device, pkg, files, "batch64");
    const residentBytes = Object.freeze({
      batch8: batch8Phase.residentBytes,
      batch64: batch64Phase.residentBytes,
    });
    const plan = planAceVaeChunkedDecode(OPT_0027_C512_FRAMES, {
      chunkFrames: OPT_0027_C512_FRAMES,
      overlapFrames: 64,
    });
    const window = requireOneC512Window(plan);
    const batch8Observer = new RunObserver();
    const batch64Observer = new RunObserver();

    postProgress("building independent OPT-0028 batch8 production backend");
    const ownedBatch8Phase = batch8Phase;
    batch8Phase = undefined;
    batch8Backend = await createProductionBackend(
      context,
      pkg,
      plan,
      fixture,
      ownedBatch8Phase,
      "batch8",
      batch8Observer,
    );
    postProgress("building independent OPT-0028 batch64 production backend");
    const ownedBatch64Phase = batch64Phase;
    batch64Phase = undefined;
    batch64Backend = await createProductionBackend(
      context,
      pkg,
      plan,
      fixture,
      ownedBatch64Phase,
      "batch64",
      batch64Observer,
    );
    const backends = Object.freeze({
      batch8: batch8Backend,
      batch64: batch64Backend,
    });
    const observers = Object.freeze({
      batch8: batch8Observer,
      batch64: batch64Observer,
    });
    const backendContracts = Object.freeze({
      batch8: validateProductionBackend(batch8Backend, "batch8"),
      batch64: validateProductionBackend(batch64Backend, "batch64"),
    });

    postProgress("untimed warmup 1/2: exact production batch8");
    const warmBatch8 = await executeArm(
      backends.batch8,
      observers.batch8,
      window,
      "batch8",
      false,
      "warmup-batch8",
    );
    postProgress("untimed warmup 2/2: exact production batch64");
    const warmBatch64 = await executeArm(
      backends.batch64,
      observers.batch64,
      window,
      "batch64",
      false,
      "warmup-batch64",
    );
    const warmComparison = compareOpt0027RawFp32(
      warmBatch8.output,
      warmBatch64.output,
    );
    const [batch8Hash, batch64Hash] = await Promise.all([
      sha256Float32(warmBatch8.output),
      sha256Float32(warmBatch64.output),
    ]);
    if (
      !warmComparison.rawFp32U32Exact ||
      warmComparison.comparedU32WordCount !== OPT_0027_C512_OUTPUT_ELEMENTS ||
      batch8Hash !== OPT_0027_C512_ACCEPTED_OUTPUT_SHA256 ||
      batch64Hash !== OPT_0027_C512_ACCEPTED_OUTPUT_SHA256 ||
      runtimeEvents.length !== 0
    ) {
      throw new Error(
        "OPT-0027 warmups differ from each other or the accepted C512 oracle",
      );
    }
    const preparationReceipt = Object.freeze({
      schema: OPT_0027_SCHEMA,
      status: "ready",
      experimentId: EXPERIMENT_ID,
      classification:
        "exact-revision6-OPT0028-nativeK7-C512-batch8-vs-batch64",
      fixture: Object.freeze({
        frames: OPT_0027_C512_FRAMES,
        elements: OPT_0027_C512_INPUT_ELEMENTS,
        byteLength: fixtureBytes.byteLength,
        sha256: OPT_0027_C512_FIXTURE_SHA256,
      }),
      package: Object.freeze({
        manifestSha256: pkg.loaded.manifestSha256,
        manifestByteLength: pkg.loaded.manifestByteLength,
        converterRevision:
          pkg.loaded.manifest.provenance.converterRevision,
        residentBytesPerArm: residentBytes,
        independentOwnedWeightPhases: true,
      }),
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        adapterInfo: context.capabilities.adapterInfo,
        deviceFeatures: context.capabilities.deviceFeatures,
        deviceLimits: context.capabilities.deviceLimits,
      }),
      production: Object.freeze({
        runtimeProfileId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
        kernelSetId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId,
        kernelTopology:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
        nativeK7KernelId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY.conv1dK7,
        opt0024ApproximateK7Used: false,
        backendContracts,
      }),
      scheduling: Object.freeze({
        batch8: planOpt0027Scheduling(
          OPT_0027_QUANTA_PER_COMMAND_BUFFER.batch8,
        ),
        batch64: planOpt0027Scheduling(
          OPT_0027_QUANTA_PER_COMMAND_BUFFER.batch64,
        ),
      }),
      warmup: Object.freeze({
        order: OPT_0027_WARMUP_ORDER,
        completeRunsPerArm: 1,
        batch8: Object.freeze({
          outputSha256: batch8Hash,
          ...warmBatch8.receipt,
        }),
        batch64: Object.freeze({
          outputSha256: batch64Hash,
          ...warmBatch64.receipt,
        }),
        comparison: warmComparison,
        acceptedOutputSha256: OPT_0027_C512_ACCEPTED_OUTPUT_SHA256,
        performanceClaim: null,
      }),
      setup: Object.freeze({
        dedicatedWorker: true,
        distinctProductionBackendOwners: true,
        distinctWeightAndActivationOwnership: true,
        sharedAuthenticatedPackageFilesOnly: true,
        cleanupOnPreparationFailure: true,
        cleanupOnTimedFailure: true,
        beforeUnloadDisposeCommand: true,
      }),
      preparationWallMs: performance.now() - preparationStarted,
      runtimeEvents: Object.freeze([...runtimeEvents]),
      externalNominalThermalGateRequiredBeforeButton: true,
      readyForTiming: true,
    });
    return createPreparedGate({
      context,
      backends,
      observers,
      window,
      warmReference: warmBatch8.output,
      runtimeEvents,
      preparationReceipt,
    });
  } catch (error) {
    await Promise.allSettled([
      batch8Backend?.destroy(error) ?? Promise.resolve(),
      batch64Backend?.destroy(error) ?? Promise.resolve(),
    ]);
    batch8Phase?.destroy();
    batch64Phase?.destroy();
    context?.destroy();
    throw error;
  }
}

function createPreparedGate(
  input: Omit<PreparedGate, "destroy">,
): PreparedGate {
  let destroyPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  return Object.freeze({
    ...input,
    destroy(reason: unknown = new DOMException(
      "OPT-0027 timing complete",
      "AbortError",
    )): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        const started = performance.now();
        const settled = await Promise.allSettled([
          input.backends.batch8.destroy(reason),
          input.backends.batch64.destroy(reason),
        ]);
        input.context.destroy();
        const failures = settled.flatMap((result, index) =>
          result.status === "fulfilled"
            ? []
            : [`${index === 0 ? "batch8" : "batch64"}: ${errorText(result.reason)}`]
        );
        return Object.freeze({
          passed: failures.length === 0,
          bothBackendOwnersDestroyed: failures.length === 0,
          bothOwnedWeightPhasesDestroyedByBackends: failures.length === 0,
          activationAndReadbackBuffersDestroyedByBackends:
            failures.length === 0,
          deviceContextDestroyed: true,
          idempotentDestroyPromise: true,
          failureMessages: Object.freeze(failures),
          wallMs: performance.now() - started,
        });
      })();
      return destroyPromise;
    },
  });
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const executions: Array<Readonly<{
    readonly index: number;
    readonly arm: Opt0027Arm;
    readonly outputSha256: string;
    readonly comparisonToWarmReference: Opt0027RawComparison;
    readonly receipt: ExecutionResult["receipt"];
  }>> = [];
  for (const [index, arm] of OPT_0027_TIMED_ORDER.entries()) {
    postProgress(
      `timed C512 ${index + 1}/${OPT_0027_TIMED_ORDER.length}: ${arm}`,
    );
    const execution = await executeArm(
      prepared.backends[arm],
      prepared.observers[arm],
      prepared.window,
      arm,
      true,
      `timed-${index}-${arm}`,
    );
    const comparisonToWarmReference = compareOpt0027RawFp32(
      prepared.warmReference,
      execution.output,
    );
    const outputSha256 = await sha256Float32(execution.output);
    executions.push(Object.freeze({
      index,
      arm,
      outputSha256,
      comparisonToWarmReference,
      receipt: execution.receipt,
    }));
    await browserYield();
  }
  const forward = summarizeOpt0027Pair(
    executions[0]!.receipt.timing,
    executions[1]!.receipt.timing,
    "batch8-batch64",
  );
  const reverse = summarizeOpt0027Pair(
    executions[3]!.receipt.timing,
    executions[2]!.receipt.timing,
    "batch64-batch8",
  );
  const outputPassed = executions.every((execution) =>
    execution.outputSha256 === OPT_0027_C512_ACCEPTED_OUTPUT_SHA256 &&
    execution.comparisonToWarmReference.rawFp32U32Exact &&
    execution.comparisonToWarmReference.comparedU32WordCount ===
      OPT_0027_C512_OUTPUT_ELEMENTS
  );
  const passed = outputPassed && prepared.runtimeEvents.length === 0;
  return Object.freeze({
    schema: OPT_0027_SCHEMA,
    status: passed ? "passed" : "failed",
    experimentId: EXPERIMENT_ID,
    classification:
      "balanced-ABBA-exact-production-C512-batch8-vs-batch64-stock-Chrome",
    environment: prepared.preparationReceipt["environment"],
    fixture: prepared.preparationReceipt["fixture"],
    package: prepared.preparationReceipt["package"],
    production: prepared.preparationReceipt["production"],
    protocol: Object.freeze({
      warmupOrder: OPT_0027_WARMUP_ORDER,
      timedOrder: OPT_0027_TIMED_ORDER,
      completeUntimedWarmupsPerArm: 1,
      completeTimedRunsPerArm: 2,
      exactSameDispatchMathAndBindings: true,
      onlyQuantaPerCommandBufferDiffers: true,
      quantaPerCommandBuffer: OPT_0027_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyMillisecondsBetweenCommandBuffers: 1,
      outerWindowWallIncludesUploadSubmissionDrainReadbackAndMap: true,
      decoderSubmitThroughDrainExcludesReadbackAndMap: true,
      externalNominalThermalGateRequiredBeforeTimedAction: true,
      dedicatedWorker: true,
      stockChromeWebGpuOnly: true,
      experimentalBrowserFlags: false,
      timestampQueries: false,
      webNn: false,
    }),
    executions: Object.freeze(executions),
    pairs: Object.freeze({ forward, reverse }),
    aggregate: aggregateTimings(executions),
    correctness: Object.freeze({
      acceptedOutputSha256: OPT_0027_C512_ACCEPTED_OUTPUT_SHA256,
      rawFp32U32ExactForEveryWarmupAndTimedRun: outputPassed,
      comparedU32WordsPerExecution: OPT_0027_C512_OUTPUT_ELEMENTS,
      performanceChangesArithmetic: false,
      listeningRequired: false,
      passed: outputPassed,
    }),
    runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
    performanceGate: null,
  });
}

async function createProductionBackend(
  context: AceWebGpuDeviceContext,
  pkg: PreparedPackage,
  plan: AceVaeChunkedDecodePlan,
  fixture: Float32Array,
  phase: AceGpuTensorPhase,
  arm: Opt0027Arm,
  observer: RunObserver,
): Promise<AceOpt0011Fp16VaeChunkGpuBackend> {
  return await AceOpt0011Fp16VaeChunkGpuBackend.create({
    device: context.device,
    plan,
    finalLatents: fixture,
    authenticatedPackage: pkg.loaded,
    ownedVaeWeights: phase,
    maximumWindowFrames: OPT_0027_C512_FRAMES,
    runtimeProfileId:
      "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
    quantaPerCommandBuffer: OPT_0027_QUANTA_PER_COMMAND_BUFFER[arm],
    onProgress: observer.onProgress,
    onFamilyProfile: observer.onFamilyProfile,
  });
}

function validateProductionBackend(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  arm: Opt0027Arm,
): Readonly<Record<string, unknown>> {
  const expectedTopology =
    ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY;
  const topology = backend.kernelTopology;
  const expectedScheduling = planOpt0027Scheduling(
    OPT_0027_QUANTA_PER_COMMAND_BUFFER[arm],
  );
  if (
    backend.runtimeProfileId !==
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id ||
    backend.kernelSetId !==
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId ||
    topology.id !== expectedTopology.id ||
    topology.backend !== expectedTopology.backend ||
    topology.ingress !== expectedTopology.ingress ||
    topology.conv1dK1 !== expectedTopology.conv1dK1 ||
    topology.conv1dK7 !== expectedTopology.conv1dK7 ||
    topology.convTranspose1d !== expectedTopology.convTranspose1d ||
    topology.snake !== expectedTopology.snake ||
    topology.add !== expectedTopology.add ||
    backend.memory.maximumWindowFrames !== OPT_0027_C512_FRAMES ||
    backend.memory.quantaPerCommandBuffer !==
      expectedScheduling.quantaPerCommandBuffer
  ) {
    throw new Error(`OPT-0027 ${arm} is not the exact production backend`);
  }
  return Object.freeze({
    arm,
    runtimeProfileId: backend.runtimeProfileId,
    kernelSetId: backend.kernelSetId,
    kernelTopology: topology,
    quantaPerCommandBuffer: backend.memory.quantaPerCommandBuffer,
    accountedGpuBytes: backend.memory.accountedGpuBytes,
    boundedCpuBytes: backend.memory.boundedCpuBytes,
  });
}

async function executeArm(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  observer: RunObserver,
  window: AceVaeDecodeWindow,
  arm: Opt0027Arm,
  measured: boolean,
  label: string,
): Promise<ExecutionResult> {
  observer.begin(label);
  try {
    const started = performance.now();
    const output = await backend.decodeWindow(window);
    const outerWindowWallMs = performance.now() - started;
    const observed = observer.finish(label);
    const expected = planOpt0027Scheduling(
      OPT_0027_QUANTA_PER_COMMAND_BUFFER[arm],
    );
    const progress = observed.finalProgress;
    const profile = observed.familyProfile;
    if (
      output.length !== OPT_0027_C512_OUTPUT_ELEMENTS ||
      profile.windowIndex !== window.index ||
      profile.inputFrames !== OPT_0027_C512_FRAMES ||
      profile.quantaPerCommandBuffer !== expected.quantaPerCommandBuffer ||
      profile.decoderQuantumCount !== expected.decoderQuantumCount ||
      profile.decoderBatchCount !== expected.decoderCommandBufferCount ||
      observed.progressEventCount !== expected.totalCommandBufferCount ||
      progress.windowIndex !== window.index ||
      progress.completedDecoderQuanta !== expected.decoderQuantumCount ||
      progress.totalDecoderQuanta !== expected.decoderQuantumCount ||
      progress.completedCommandBuffers !== expected.totalCommandBufferCount ||
      progress.totalCommandBuffers !== expected.totalCommandBufferCount ||
      progress.queueDrains !== expected.queueDrainCount ||
      progress.cooperativeIdleMs !== expected.requestedCooperativeIdleMs ||
      progress.stage !== "readback"
    ) {
      throw new Error(`OPT-0027 ${label} scheduling topology changed`);
    }
    const scan = scanOutput(output);
    if (
      scan["elementCount"] !== OPT_0027_C512_OUTPUT_ELEMENTS ||
      scan["nonFiniteCount"] !== 0 || scan["nonzeroCount"] === 0 ||
      scan["stereoDifferenceFrameCount"] === 0
    ) {
      throw new Error(`OPT-0027 ${label} output is incomplete`);
    }
    return Object.freeze({
      arm,
      output: output as Float32Array<ArrayBuffer>,
      receipt: Object.freeze({
        arm,
        measured,
        scan,
        timing: Object.freeze({
          outerWindowWallMs,
          decoderSubmitThroughDrainMs:
            profile.decoderSubmitThroughDrainMs,
          decoderCommandBufferCount: profile.decoderBatchCount,
          totalCommandBufferCount: progress.totalCommandBuffers,
          queueDrainCount: progress.queueDrains,
          requestedCooperativeIdleMs: progress.cooperativeIdleMs,
        }),
        familyProfile: profile,
      }),
    });
  } catch (error) {
    observer.cancel(label);
    throw error;
  }
}

function requireOneC512Window(plan: AceVaeChunkedDecodePlan): AceVaeDecodeWindow {
  const window = plan.windows[0];
  if (
    plan.latentFrames !== OPT_0027_C512_FRAMES ||
    plan.maximumWindowFrames !== OPT_0027_C512_FRAMES ||
    plan.windows.length !== 1 || window === undefined ||
    window.index !== 0 ||
    window.latentWindowFrames !== OPT_0027_C512_FRAMES ||
    window.discardPrefixLatentFrames !== 0 ||
    window.discardSuffixLatentFrames !== 0
  ) {
    throw new Error("OPT-0027 plan is not one exact C512 window");
  }
  return window;
}

function aggregateTimings(
  executions: readonly Readonly<{
    readonly arm: Opt0027Arm;
    readonly receipt: ExecutionResult["receipt"];
  }>[] ,
): Readonly<Record<Opt0027Arm, Readonly<Record<string, number>>>> {
  const aggregateArm = (arm: Opt0027Arm) => {
    const timings = executions.filter((execution) => execution.arm === arm)
      .map((execution) => execution.receipt.timing);
    if (timings.length !== 2) {
      throw new Error(`OPT-0027 ${arm} balanced sample count changed`);
    }
    return Object.freeze({
      sampleCount: timings.length,
      meanOuterWindowWallMs: mean(timings.map((timing) =>
        timing.outerWindowWallMs
      )),
      meanDecoderSubmitThroughDrainMs: mean(timings.map((timing) =>
        timing.decoderSubmitThroughDrainMs
      )),
      decoderCommandBufferCount: timings[0]!.decoderCommandBufferCount,
      totalCommandBufferCount: timings[0]!.totalCommandBufferCount,
      requestedCooperativeIdleMs:
        timings[0]!.requestedCooperativeIdleMs,
    });
  };
  return Object.freeze({
    batch8: aggregateArm("batch8"),
    batch64: aggregateArm("batch64"),
  });
}

function mean(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError("OPT-0027 mean requires finite values");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function scanOutput(output: Float32Array): Readonly<Record<string, number>> {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let stereoDifferenceFrameCount = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let peak = 0;
  for (let index = 0; index < output.length; index += 1) {
    const value = output[index]!;
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (index % 2 === 0 && output[index + 1] !== value) {
      stereoDifferenceFrameCount += 1;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    peak = Math.max(peak, Math.abs(value));
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    nonFiniteCount,
    nonzeroCount,
    stereoDifferenceFrameCount,
    minimum,
    maximum,
    peak,
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
    files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    shardNames.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) {
    throw new Error("OPT-0027 authenticated package identity changed");
  }
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    residentBytes,
  });
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
  ) {
    throw new Error("OPT-0027 package acquisition accounting changed");
  }
  return acquired.files;
}

async function loadVaePhase(
  device: GPUDevice,
  pkg: PreparedPackage,
  files: ReadonlyMap<string, File>,
  arm: Opt0027Arm,
): Promise<AceGpuTensorPhase> {
  postProgress(`uploading independent ${arm} VAE weight phase`);
  const phase = await AceGpuTensorPhase.load(
    device,
    pkg.loaded.manifest,
    files,
    ["vae"],
    {
      onProgress: (progress) => postProgress(
        `uploading ${arm} VAE ${progress.phaseFileIndex + 1}/` +
          `${progress.phaseFileCount}: ${progress.loadedPhaseBytes}/` +
          `${progress.totalPhaseBytes} bytes`,
      ),
    },
  );
  if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
    phase.destroy();
    throw new Error(`OPT-0027 ${arm} resident VAE bytes changed`);
  }
  return phase;
}

function requireFixed32Subgroups(context: AceWebGpuDeviceContext): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== REQUIRED_SUBGROUP_SIZE ||
    info.subgroupMaxSize !== REQUIRED_SUBGROUP_SIZE
  ) {
    throw new Error("OPT-0027 requires stock fixed 32-lane WebGPU subgroups");
  }
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

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
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
