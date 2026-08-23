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
  type AceOpt0011Fp16VaeDispatchTopologyReceipt,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
} from "../../src/webgpu/vae-fp16-decoder.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
} from "../../src/webgpu/kernels/vae-pointwise-fp16.js";
import { ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import { ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import { ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import { ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-snake-fp16.js";
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
import {
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import type { Opt0040ThermalGate } from
  "./opt-0040-vae-convtranspose-shape-selector-contract.js";

export const OPT_0040_BROWSER_SCHEMA =
  "ace-opt-0040-vae-convtranspose-shape-selector-c512-abba-v1" as const;
export const OPT_0040_C512_FRAMES = 512 as const;
export const OPT_0040_C512_INPUT_ELEMENTS = 32_768 as const;
export const OPT_0040_C512_OUTPUT_ELEMENTS = 1_966_080 as const;
export const OPT_0040_C512_OUTPUT_U16_WORDS = 3_932_160 as const;
export const OPT_0040_C512_FIXTURE_SHA256 =
  "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899" as const;
export const OPT_0040_C512_ACCEPTED_OUTPUT_SHA256 =
  "893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47" as const;
export const OPT_0040_WARMUP_ORDER = Object.freeze([
  "control",
  "selector",
  "selector",
  "control",
] as const);
export const OPT_0040_TIMED_ORDER = Object.freeze([
  "control",
  "selector",
  "selector",
  "control",
] as const);
export const OPT_0040_REQUIRED_CONV_TRANSPOSE_SPEEDUP = 1.10 as const;

const EXPERIMENT_ID = "OPT-0040" as const;
const MANIFEST_PATH = "/model/files-fp16-vae-experimental/manifest.json";
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const REQUIRED_SUBGROUP_SIZE = 32;
const DECODER_COMMAND_BUFFER_COUNT = 123;
const TOTAL_COMMAND_BUFFER_COUNT = 124;
const REQUESTED_COOPERATIVE_IDLE_MS = 123;
const CANCEL_AFTER_QUANTA = 64;
const CANCELLATION_REASON = new DOMException(
  "OPT-0040 bounded cancellation probe",
  "AbortError",
);

export type Opt0040Arm = "control" | "selector";

type WorkerCommand =
  | Readonly<{
      readonly type: "initialize";
      readonly identity: Opt0018RunIdentity;
    }>
  | Readonly<{
      readonly type: "run";
      readonly thermalGate: Opt0040ThermalGate;
    }>
  | Readonly<{ readonly type: "dispose" }>;

interface WorkerEvent {
  readonly type:
    | "progress"
    | "ready-for-thermal-gate"
    | "comparison-complete"
    | "failed";
  readonly message?: string;
  readonly readyAtEpochMilliseconds?: number;
  readonly preparation?: Readonly<Record<string, unknown>>;
  readonly result?: Readonly<Record<string, unknown>>;
  readonly error?: Readonly<Record<string, unknown>>;
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
  readonly arm: Opt0040Arm;
  readonly output: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<{
    readonly arm: Opt0040Arm;
    readonly measured: boolean;
    readonly scan: Readonly<Record<string, number>>;
    readonly timing: Readonly<{
      readonly outerWindowWallMs: number;
      readonly convTransposeSubmitThroughDrainMs: number;
      readonly decoderSubmitThroughDrainMs: number;
      readonly decoderCommandBufferCount: number;
      readonly totalCommandBufferCount: number;
      readonly queueDrainCount: number;
      readonly requestedCooperativeIdleMs: number;
    }>;
    readonly familyProfile: AceOpt0011Fp16VaeWindowFamilyProfile;
  }>;
}

interface PreparedGate {
  readonly identity: Opt0018RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly backends: Readonly<Record<
    Opt0040Arm,
    AceOpt0011Fp16VaeChunkGpuBackend
  >>;
  readonly observers: Readonly<Record<Opt0040Arm, RunObserver>>;
  readonly window: AceVaeDecodeWindow;
  readonly warmReference: Float32Array<ArrayBuffer>;
  readonly familyTopology: Readonly<Record<string, unknown>>;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly readyAtEpochMilliseconds: number;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  destroy(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

export interface Opt0040RawComparison {
  readonly comparedU32WordCount: number;
  readonly u32MismatchCount: number;
  readonly firstU32MismatchIndex: number | null;
  readonly comparedU16WordCount: number;
  readonly u16MismatchCount: number;
  readonly firstU16MismatchIndex: number | null;
  readonly rawU32Exact: boolean;
  readonly rawU16Exact: boolean;
}

class RunObserver {
  private active: {
    readonly label: string;
    progressEventCount: number;
    finalProgress?: AceVaeChunkGpuBackendProgress;
    familyProfileCount: number;
    familyProfile?: AceOpt0011Fp16VaeWindowFamilyProfile;
    abortAfterFirstProgress?: AbortController;
  } | undefined;

  begin(label: string, abortAfterFirstProgress?: AbortController): void {
    if (this.active !== undefined) {
      throw new Error("OPT-0040 observer already owns a run");
    }
    this.active = {
      label,
      progressEventCount: 0,
      familyProfileCount: 0,
      ...(abortAfterFirstProgress === undefined
        ? {}
        : { abortAfterFirstProgress }),
    };
  }

  readonly onProgress = (progress: AceVaeChunkGpuBackendProgress): void => {
    const active = this.active;
    if (active === undefined) return;
    active.progressEventCount += 1;
    active.finalProgress = Object.freeze({ ...progress });
    if (
      active.progressEventCount === 1 &&
      active.abortAfterFirstProgress !== undefined
    ) {
      active.abortAfterFirstProgress.abort(CANCELLATION_REASON);
    }
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
    const active = this.take(label);
    if (
      active.finalProgress === undefined ||
      active.familyProfileCount !== 1 || active.familyProfile === undefined ||
      active.abortAfterFirstProgress !== undefined
    ) {
      throw new Error(`OPT-0040 ${label} observation is incomplete`);
    }
    return Object.freeze({
      progressEventCount: active.progressEventCount,
      finalProgress: active.finalProgress,
      familyProfile: active.familyProfile,
    });
  }

  finishCancellation(label: string): Readonly<{
    progressEventCount: number;
    finalProgress: AceVaeChunkGpuBackendProgress;
  }> {
    const active = this.take(label);
    if (
      active.abortAfterFirstProgress === undefined ||
      active.finalProgress === undefined || active.familyProfileCount !== 0
    ) {
      throw new Error(`OPT-0040 ${label} cancellation observation is invalid`);
    }
    return Object.freeze({
      progressEventCount: active.progressEventCount,
      finalProgress: active.finalProgress,
    });
  }

  cancel(label: string): void {
    if (this.active?.label === label) this.active = undefined;
  }

  private take(label: string): NonNullable<RunObserver["active"]> {
    const active = this.active;
    this.active = undefined;
    if (active === undefined || active.label !== label) {
      throw new Error(`OPT-0040 ${label} observer ownership changed`);
    }
    return active;
  }
}

let prepared: PreparedGate | undefined;
let operation = Promise.resolve();

globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
  operation = operation.then(async () => {
    const command = event.data;
    if (command.type === "initialize") {
      if (prepared !== undefined) throw new Error("OPT-0040 is already prepared");
      prepared = await prepareGate(validateOpt0018RunIdentity(command.identity));
      postWorker({
        type: "ready-for-thermal-gate",
        readyAtEpochMilliseconds: prepared.readyAtEpochMilliseconds,
        preparation: prepared.preparationReceipt,
      });
      return;
    }
    if (command.type === "dispose") {
      const retained = prepared;
      prepared = undefined;
      await retained?.destroy(new DOMException(
        "OPT-0040 page disposed",
        "AbortError",
      ));
      return;
    }
    if (prepared === undefined) {
      throw new Error("OPT-0040 timing requested before READY");
    }
    validateThermalGate(
      command.thermalGate,
      prepared.readyAtEpochMilliseconds,
    );
    const retained = prepared;
    let timed: Readonly<Record<string, unknown>>;
    let cleanup: Readonly<Record<string, unknown>>;
    try {
      timed = await runTimedGate(retained, command.thermalGate);
    } finally {
      cleanup = await retained.destroy();
      if (prepared === retained) prepared = undefined;
    }
    const clean = cleanup["passed"] === true;
    const timedStatus = timed["status"];
    postWorker({
      type: "comparison-complete",
      result: Object.freeze({
        ...timed,
        status: clean && (timedStatus === "passed" || timedStatus === "negative")
          ? timedStatus
          : "failed",
        cleanup,
      }),
    });
  }).catch(async (error: unknown) => {
    const retained = prepared;
    prepared = undefined;
    let cleanup: Readonly<Record<string, unknown>> | undefined;
    try {
      cleanup = await retained?.destroy(error);
    } catch (cleanupError) {
      postWorker({
        type: "failed",
        error: serializeOpt0018Failure(error, cleanupError),
      });
      return;
    }
    postWorker({
      type: "failed",
      error: Object.freeze({
        ...serializeOpt0018Failure(error),
        ...(cleanup === undefined ? {} : { cleanup }),
      }),
    });
  });
});

async function prepareGate(identity: Opt0018RunIdentity): Promise<PreparedGate> {
  const started = performance.now();
  postProgress("authenticating OPT-0027 C512 fixture and revision-6 package");
  const fixtureBytes = createAceOpt0011LatentFixture(OPT_0040_C512_FRAMES);
  if (
    fixtureBytes.byteLength !== OPT_0040_C512_INPUT_ELEMENTS * FLOAT32_BYTES ||
    await sha256Hex(fixtureBytes) !== OPT_0040_C512_FIXTURE_SHA256
  ) {
    throw new Error("OPT-0040 C512 fixture identity changed");
  }
  const fixture = new Float32Array(OPT_0040_C512_INPUT_ELEMENTS);
  fixture.set(new Float32Array(
    fixtureBytes.buffer,
    fixtureBytes.byteOffset,
    OPT_0040_C512_INPUT_ELEMENTS,
  ));
  const pkg = await authenticatePackage();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let controlPhase: AceGpuTensorPhase | undefined;
  let selectorPhase: AceGpuTensorPhase | undefined;
  let controlBackend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  let selectorBackend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });
    requireFixed32Subgroups(context);
    const files = await acquirePackageFiles(pkg);
    controlPhase = await loadVaePhase(context.device, pkg, files, "control");
    selectorPhase = await loadVaePhase(context.device, pkg, files, "selector");
    const residentBytes = Object.freeze({
      control: controlPhase.residentBytes,
      selector: selectorPhase.residentBytes,
    });
    const plan = planAceVaeChunkedDecode(OPT_0040_C512_FRAMES, {
      chunkFrames: OPT_0040_C512_FRAMES,
      overlapFrames: 64,
    });
    const window = requireOneC512Window(plan);
    const controlObserver = new RunObserver();
    const selectorObserver = new RunObserver();

    postProgress("building independent batch64 OPT-0028 control backend");
    const ownedControl = controlPhase;
    controlPhase = undefined;
    controlBackend = await createBackend(
      context,
      pkg,
      plan,
      fixture,
      ownedControl,
      "control",
      controlObserver,
    );
    postProgress("building independent batch64 OPT-0040 selector backend");
    const ownedSelector = selectorPhase;
    selectorPhase = undefined;
    selectorBackend = await createBackend(
      context,
      pkg,
      plan,
      fixture,
      ownedSelector,
      "selector",
      selectorObserver,
    );
    const backends = Object.freeze({
      control: controlBackend,
      selector: selectorBackend,
    });
    const observers = Object.freeze({
      control: controlObserver,
      selector: selectorObserver,
    });
    const topology = validateTopology(
      controlBackend.captureDispatchTopology(),
      selectorBackend.captureDispatchTopology(),
      controlBackend,
      selectorBackend,
    );

    const warmExecutions: ExecutionResult[] = [];
    let familyTopology: Readonly<Record<string, unknown>> | undefined;
    for (const [index, arm] of OPT_0040_WARMUP_ORDER.entries()) {
      postProgress(
        `untimed exactness/determinism ${index + 1}/4: ${arm}`,
      );
      const execution = await executeArm(
        backends[arm],
        observers[arm],
        window,
        arm,
        false,
        `warm-${index}-${arm}`,
      );
      const observedTopology = summarizeFamilyTopology(
        execution.receipt.familyProfile,
      );
      if (familyTopology === undefined) {
        familyTopology = observedTopology;
      } else if (
        JSON.stringify(observedTopology) !== JSON.stringify(familyTopology)
      ) {
        throw new Error("OPT-0040 batch64 family topology differs by arm/run");
      }
      warmExecutions.push(execution);
    }
    const warmReference = warmExecutions[0]!.output;
    const warmCorrectness = await compareExecutionSet(
      warmExecutions,
      warmReference,
    );
    const controlDeterminism = compareRaw(
      warmExecutions[0]!.output,
      warmExecutions[3]!.output,
    );
    const selectorDeterminism = compareRaw(
      warmExecutions[1]!.output,
      warmExecutions[2]!.output,
    );
    const crossArm = compareRaw(
      warmExecutions[0]!.output,
      warmExecutions[1]!.output,
    );
    if (
      !warmCorrectness.passed ||
      !comparisonPassed(controlDeterminism) ||
      !comparisonPassed(selectorDeterminism) ||
      !comparisonPassed(crossArm) ||
      runtimeEvents.length !== 0 || familyTopology === undefined
    ) {
      throw new Error("OPT-0040 untimed C512 correctness gate failed");
    }
    const readyAtEpochMilliseconds = Date.now();
    const preparationReceipt = Object.freeze({
      schema: OPT_0040_BROWSER_SCHEMA,
      status: "ready",
      experimentId: EXPERIMENT_ID,
      identity,
      fixture: Object.freeze({
        frames: OPT_0040_C512_FRAMES,
        elements: OPT_0040_C512_INPUT_ELEMENTS,
        byteLength: fixtureBytes.byteLength,
        sha256: OPT_0040_C512_FIXTURE_SHA256,
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
      topology,
      scheduling: Object.freeze({
        quantaPerCommandBuffer:
          ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
        decoderQuantumCount:
          ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
        decoderCommandBufferCount: DECODER_COMMAND_BUFFER_COUNT,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: TOTAL_COMMAND_BUFFER_COUNT,
        queueDrainCount: TOTAL_COMMAND_BUFFER_COUNT,
        requestedCooperativeIdleMs: REQUESTED_COOPERATIVE_IDLE_MS,
        familyTopology,
      }),
      correctness: Object.freeze({
        warmupOrder: OPT_0040_WARMUP_ORDER,
        completeRunsPerArm: 2,
        warmCorrectness,
        controlDeterminism,
        selectorDeterminism,
        crossArm,
        acceptedOutputSha256: OPT_0040_C512_ACCEPTED_OUTPUT_SHA256,
        performanceClaim: null,
        passed: true,
      }),
      lifecycle: Object.freeze({
        dedicatedWorker: true,
        distinctBackendOwners: true,
        distinctWeightAndActivationOwnership: true,
        cleanupOnPreparationFailure: true,
        cleanupOnTimedFailure: true,
        perCallCancellationProbeRunsAfterTiming: true,
      }),
      runtimeEvents: Object.freeze([...runtimeEvents]),
      preparationWallMs: performance.now() - started,
      readyAtEpochMilliseconds,
      readyForThermalGate: true,
    });
    return createPreparedGate({
      identity,
      context,
      backends,
      observers,
      window,
      warmReference,
      familyTopology,
      runtimeEvents,
      readyAtEpochMilliseconds,
      preparationReceipt,
    });
  } catch (error) {
    await Promise.allSettled([
      controlBackend?.destroy(error) ?? Promise.resolve(),
      selectorBackend?.destroy(error) ?? Promise.resolve(),
    ]);
    controlPhase?.destroy();
    selectorPhase?.destroy();
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
      "OPT-0040 comparison complete",
      "AbortError",
    )): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        const started = performance.now();
        const controlFirst = input.backends.control.destroy(reason);
        const controlSecond = input.backends.control.destroy(reason);
        const selectorFirst = input.backends.selector.destroy(reason);
        const selectorSecond = input.backends.selector.destroy(reason);
        const idempotentPromises = controlFirst === controlSecond &&
          selectorFirst === selectorSecond;
        const settled = await Promise.allSettled([
          controlFirst,
          selectorFirst,
        ]);
        input.context.destroy();
        const failureMessages = settled.flatMap((result, index) =>
          result.status === "fulfilled"
            ? []
            : [
                `${index === 0 ? "control" : "selector"}: ${errorText(result.reason)}`,
              ]
        );
        return Object.freeze({
          passed: failureMessages.length === 0 && idempotentPromises,
          bothBackendOwnersDestroyed: failureMessages.length === 0,
          bothOwnedWeightPhasesDestroyedByBackends:
            failureMessages.length === 0,
          activationControlAndReadbackBuffersDestroyedByBackends:
            failureMessages.length === 0,
          deviceContextDestroyed: true,
          idempotentDestroyPromises: idempotentPromises,
          failureMessages: Object.freeze(failureMessages),
          wallMs: performance.now() - started,
          completedAtEpochMilliseconds: Date.now(),
        });
      })();
      return destroyPromise;
    },
  });
}

async function runTimedGate(
  prepared: PreparedGate,
  thermalGate: Opt0040ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  const executions: ExecutionResult[] = [];
  for (const [index, arm] of OPT_0040_TIMED_ORDER.entries()) {
    postProgress(`timed C512 ${index + 1}/4: ${arm}`);
    const execution = await executeArm(
      prepared.backends[arm],
      prepared.observers[arm],
      prepared.window,
      arm,
      true,
      `timed-${index}-${arm}`,
    );
    if (
      JSON.stringify(summarizeFamilyTopology(
        execution.receipt.familyProfile,
      )) !== JSON.stringify(prepared.familyTopology)
    ) {
      throw new Error("OPT-0040 timed batch64 family topology changed");
    }
    executions.push(execution);
    await browserYield();
  }
  const correctness = await compareExecutionSet(
    executions,
    prepared.warmReference,
  );
  const forward = pairedGate(executions[0]!, executions[1]!, "control-selector");
  const reverse = pairedGate(executions[3]!, executions[2]!, "selector-control");
  const performancePassed = forward.passed && reverse.passed;
  postProgress("running post-timing batch64 cancellation probe");
  const cancellation = await runCancellationProbe(
    prepared.backends.control,
    prepared.observers.control,
    prepared.window,
  );
  const passed = correctness.passed && performancePassed &&
    cancellation.passed && prepared.runtimeEvents.length === 0;
  const correctnessAndLifecyclePassed = correctness.passed &&
    cancellation.passed && prepared.runtimeEvents.length === 0;
  return Object.freeze({
    schema: OPT_0040_BROWSER_SCHEMA,
    status: passed
      ? "passed"
      : correctnessAndLifecyclePassed
        ? "negative"
        : "failed",
    decision: passed
      ? "positive-complete-c512-gate-passed"
      : correctnessAndLifecyclePassed
        ? "negative-below-complete-c512-speed-gate"
        : "failed-correctness-or-lifecycle-gate",
    experimentId: EXPERIMENT_ID,
    identity: prepared.identity,
    environment: prepared.preparationReceipt["environment"],
    fixture: prepared.preparationReceipt["fixture"],
    package: prepared.preparationReceipt["package"],
    topology: prepared.preparationReceipt["topology"],
    protocol: Object.freeze({
      warmupOrder: OPT_0040_WARMUP_ORDER,
      timedOrder: OPT_0040_TIMED_ORDER,
      completeUntimedRunsPerArm: 2,
      completeTimedRunsPerArm: 2,
      quantaPerCommandBuffer:
        ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyMillisecondsBetweenCommandBuffers: 1,
      identicalScheduling: true,
      onlyConvTransposeKernelOwnerDiffers: true,
      outerWindowWallIncludesUploadSubmissionDrainReadbackAndMap: true,
      familyTimingUsesHomogeneousCommandBuffers: true,
      mixedFamilyCommandBuffersReportedSeparately: true,
      timingRunsBeforeCancellationProbe: true,
      dedicatedWorker: true,
      stockChromeWebGpuOnly: true,
      experimentalBrowserFlags: false,
      timestampQueries: false,
      webNn: false,
      thermalGate,
    }),
    executions: Object.freeze(await Promise.all(executions.map(
      async (execution, index) => Object.freeze({
        index,
        outputSha256: await sha256Float32(execution.output),
        comparisonToUntimedReference: compareRaw(
          prepared.warmReference,
          execution.output,
        ),
        ...execution.receipt,
      }),
    ))),
    pairs: Object.freeze({ forward, reverse }),
    aggregate: aggregateTimings(executions),
    performanceGate: Object.freeze({
      requiredConvTransposeFamilySpeedup:
        OPT_0040_REQUIRED_CONV_TRANSPOSE_SPEEDUP,
      requiresNoCompleteDecoderRegressionInBothPairedOrders: true,
      outerWallReportedButNotGating: true,
      forwardPassed: forward.passed,
      reversePassed: reverse.passed,
      passed: performancePassed,
    }),
    correctness,
    cancellation,
    runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
    productionDefaultChanged: false,
    under60SecondClaim: false,
  });
}

async function createBackend(
  context: AceWebGpuDeviceContext,
  pkg: PreparedPackage,
  plan: AceVaeChunkedDecodePlan,
  fixture: Float32Array,
  phase: AceGpuTensorPhase,
  arm: Opt0040Arm,
  observer: RunObserver,
): Promise<AceOpt0011Fp16VaeChunkGpuBackend> {
  return await AceOpt0011Fp16VaeChunkGpuBackend.create({
    device: context.device,
    plan,
    finalLatents: fixture,
    authenticatedPackage: pkg.loaded,
    ownedVaeWeights: phase,
    maximumWindowFrames: OPT_0040_C512_FRAMES,
    runtimeProfileId: arm === "control"
      ? "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
      : "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
    quantaPerCommandBuffer:
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
    onProgress: observer.onProgress,
    onFamilyProfile: observer.onFamilyProfile,
  });
}

async function executeArm(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  observer: RunObserver,
  window: AceVaeDecodeWindow,
  arm: Opt0040Arm,
  measured: boolean,
  label: string,
): Promise<ExecutionResult> {
  observer.begin(label);
  try {
    const started = performance.now();
    const output = await backend.decodeWindow(window);
    const outerWindowWallMs = performance.now() - started;
    const observed = observer.finish(label);
    const progress = observed.finalProgress;
    const profile = observed.familyProfile;
    if (
      output.length !== OPT_0040_C512_OUTPUT_ELEMENTS ||
      profile.windowIndex !== window.index ||
      profile.inputFrames !== OPT_0040_C512_FRAMES ||
      profile.quantaPerCommandBuffer !==
        ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
      profile.decoderQuantumCount !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      profile.decoderBatchCount !== DECODER_COMMAND_BUFFER_COUNT ||
      observed.progressEventCount !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.windowIndex !== window.index ||
      progress.completedDecoderQuanta !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      progress.totalDecoderQuanta !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      progress.completedCommandBuffers !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.totalCommandBuffers !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.queueDrains !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.cooperativeIdleMs !== REQUESTED_COOPERATIVE_IDLE_MS ||
      progress.stage !== "readback"
    ) {
      throw new Error(`OPT-0040 ${label} scheduling topology changed`);
    }
    const scan = scanOutput(output);
    if (
      scan["elementCount"] !== OPT_0040_C512_OUTPUT_ELEMENTS ||
      scan["nonFiniteCount"] !== 0 || scan["nonzeroCount"] === 0 ||
      scan["stereoDifferenceFrameCount"] === 0
    ) {
      throw new Error(`OPT-0040 ${label} output is incomplete`);
    }
    const convTranspose = profile.families["conv-transpose1d"];
    if (
      convTranspose.batchCount < 1 || convTranspose.quantumCount < 1 ||
      !(convTranspose.submitThroughDrainMs > 0) ||
      !(profile.decoderSubmitThroughDrainMs > 0) ||
      !(outerWindowWallMs > 0)
    ) {
      throw new Error(`OPT-0040 ${label} family timing is incomplete`);
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
          convTransposeSubmitThroughDrainMs:
            convTranspose.submitThroughDrainMs,
          decoderSubmitThroughDrainMs: profile.decoderSubmitThroughDrainMs,
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

function pairedGate(
  control: ExecutionResult,
  selector: ExecutionResult,
  order: "control-selector" | "selector-control",
): Readonly<{
  order: "control-selector" | "selector-control";
  convTransposeFamilySpeedup: number;
  completeDecoderSpeedup: number;
  outerWindowSpeedup: number;
  convTransposeFamilyPassed: boolean;
  completeDecoderNoRegressionPassed: boolean;
  passed: boolean;
  control: ExecutionResult["receipt"]["timing"];
  selector: ExecutionResult["receipt"]["timing"];
}> {
  if (control.arm !== "control" || selector.arm !== "selector") {
    throw new Error(`OPT-0040 ${order} pair arm ownership changed`);
  }
  const controlTiming = control.receipt.timing;
  const selectorTiming = selector.receipt.timing;
  const convTransposeFamilySpeedup =
    controlTiming.convTransposeSubmitThroughDrainMs /
      selectorTiming.convTransposeSubmitThroughDrainMs;
  const completeDecoderSpeedup = controlTiming.decoderSubmitThroughDrainMs /
    selectorTiming.decoderSubmitThroughDrainMs;
  const outerWindowSpeedup = controlTiming.outerWindowWallMs /
    selectorTiming.outerWindowWallMs;
  const convTransposeFamilyPassed = convTransposeFamilySpeedup >=
    OPT_0040_REQUIRED_CONV_TRANSPOSE_SPEEDUP;
  const completeDecoderNoRegressionPassed =
    selectorTiming.decoderSubmitThroughDrainMs <=
      controlTiming.decoderSubmitThroughDrainMs;
  return Object.freeze({
    order,
    convTransposeFamilySpeedup,
    completeDecoderSpeedup,
    outerWindowSpeedup,
    convTransposeFamilyPassed,
    completeDecoderNoRegressionPassed,
    passed: convTransposeFamilyPassed && completeDecoderNoRegressionPassed,
    control: controlTiming,
    selector: selectorTiming,
  });
}

async function runCancellationProbe(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  observer: RunObserver,
  window: AceVaeDecodeWindow,
): Promise<Readonly<Record<string, unknown>>> {
  const controller = new AbortController();
  const label = "post-timing-control-cancellation";
  observer.begin(label, controller);
  const started = performance.now();
  let rejection: unknown;
  try {
    await backend.decodeWindow(window, controller.signal);
  } catch (error) {
    rejection = error;
  }
  const observed = observer.finishCancellation(label);
  const progress = observed.finalProgress;
  const backendTopologyAfterAbort = backend.captureDispatchTopology();
  const passed = rejection instanceof DOMException &&
    rejection.name === "AbortError" && controller.signal.aborted &&
    observed.progressEventCount === 1 &&
    progress.completedCommandBuffers === 1 && progress.queueDrains === 1 &&
    progress.completedDecoderQuanta === CANCEL_AFTER_QUANTA &&
    progress.totalDecoderQuanta ===
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT &&
    progress.totalCommandBuffers === TOTAL_COMMAND_BUFFER_COUNT &&
    progress.stage === "decoder" &&
    backendTopologyAfterAbort.windows[0]?.sequenceQuantumCount ===
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT;
  return Object.freeze({
    passed,
    requestedAfterFirstDrainedCommandBuffer: true,
    maximumCompletedCommandBuffers: progress.completedCommandBuffers,
    maximumCompletedDecoderQuanta: progress.completedDecoderQuanta,
    configuredQuantaPerCommandBuffer:
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
    rejectionName: rejection instanceof Error ? rejection.name : null,
    rejectionMessage: rejection instanceof Error ? rejection.message : null,
    backendRemainedLiveAfterPerCallAbort: true,
    wallMs: performance.now() - started,
  });
}

function validateTopology(
  control: AceOpt0011Fp16VaeDispatchTopologyReceipt,
  selector: AceOpt0011Fp16VaeDispatchTopologyReceipt,
  controlBackend: AceOpt0011Fp16VaeChunkGpuBackend,
  selectorBackend: AceOpt0011Fp16VaeChunkGpuBackend,
): Readonly<Record<string, unknown>> {
  const controlWindow = control.windows[0];
  const selectorWindow = selector.windows[0];
  const expectedControlKernels = Object.freeze({
    [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
    [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 4_090,
    [ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID]: 819,
    [ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID]: 644,
    [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
    [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
  });
  const expectedSelectorKernels = Object.freeze({
    [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
    [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: 4_090,
    [ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID]: 819,
    [ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID]: 368,
    [ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID]: 276,
    [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
    [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
  });
  if (
    controlBackend.runtimeProfileId !==
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id ||
    selectorBackend.runtimeProfileId !==
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id ||
    controlBackend.kernelSetId !==
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId ||
    selectorBackend.kernelSetId !==
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.kernelSetId ||
    controlBackend.memory.quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
    selectorBackend.memory.quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
    control.uniqueWindowFrames.length !== 1 ||
    control.uniqueWindowFrames[0] !== OPT_0040_C512_FRAMES ||
    selector.uniqueWindowFrames.length !== 1 ||
    selector.uniqueWindowFrames[0] !== OPT_0040_C512_FRAMES ||
    controlWindow === undefined || selectorWindow === undefined ||
    controlWindow.operationCount !== 88 || selectorWindow.operationCount !== 88 ||
    controlWindow.graphQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT ||
    selectorWindow.graphQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT ||
    controlWindow.sequenceQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
    selectorWindow.sequenceQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
    !sameCounts(controlWindow.kernelQuantumCounts, expectedControlKernels) ||
    !sameCounts(selectorWindow.kernelQuantumCounts, expectedSelectorKernels)
  ) {
    throw new Error("OPT-0040 complete C512 kernel topology changed");
  }
  const controlTopology = controlBackend.kernelTopology;
  const selectorTopology = selectorBackend.kernelTopology;
  for (const field of ["ingress", "conv1dK1", "conv1dK7", "snake", "add"] as const) {
    if (controlTopology[field] !== selectorTopology[field]) {
      throw new Error(`OPT-0040 changed non-ConvTranspose owner ${field}`);
    }
  }
  if (
    controlTopology.convTranspose1d !==
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY.convTranspose1d ||
    selectorTopology.convTranspose1d !==
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY.convTranspose1d
  ) {
    throw new Error("OPT-0040 selector topology identity changed");
  }
  const expectedTranspose = [
    ["block-0-conv-t1", 92, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
    ["block-1-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
    ["block-2-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
    ["block-3-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID],
    ["block-4-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID],
  ] as const;
  const controlTranspose = controlWindow.operationQuantumCounts.filter(
    (entry) => entry.operationKind === "conv-transpose1d",
  );
  const selectorTranspose = selectorWindow.operationQuantumCounts.filter(
    (entry) => entry.operationKind === "conv-transpose1d",
  );
  if (
    expectedTranspose.some(([label, count, kernelId], index) =>
      controlTranspose[index]?.operationLabel !== label ||
      controlTranspose[index]?.quantumCount !== count ||
      controlTranspose[index]?.kernelId !==
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID ||
      selectorTranspose[index]?.operationLabel !== label ||
      selectorTranspose[index]?.quantumCount !== count ||
      selectorTranspose[index]?.kernelId !== kernelId
    ) ||
    controlWindow.operationQuantumCounts.some((entry, index) =>
      entry.operationKind === "conv-transpose1d"
        ? false
        : selectorWindow.operationQuantumCounts[index]?.kernelId !==
            entry.kernelId ||
          selectorWindow.operationQuantumCounts[index]?.quantumCount !==
            entry.quantumCount
    )
  ) {
    throw new Error("OPT-0040 operation-level routing changed");
  }
  return Object.freeze({
    control,
    selector,
    identicalNonConvTransposeOwners: true,
    selectorRouting: Object.freeze(expectedTranspose.map(
      ([operationLabel, quantumCount, kernelId]) => Object.freeze({
        operationLabel,
        quantumCount,
        kernelId,
      }),
    )),
    controlConvTransposeQuantumCount: 644,
    selectorChannelReuseQuantumCount: 368,
    selectorRowReuseQuantumCount: 276,
  });
}

function sameCounts(
  actual: Readonly<Record<string, number>>,
  expected: Readonly<Record<string, number>>,
): boolean {
  const names = Object.keys(expected);
  return Object.keys(actual).length === names.length &&
    names.every((name) => actual[name] === expected[name]);
}

export function compareOpt0040Raw(
  control: Float32Array,
  candidate: Float32Array,
): Opt0040RawComparison {
  return compareRaw(control, candidate);
}

function compareRaw(
  control: Float32Array,
  candidate: Float32Array,
): Opt0040RawComparison {
  if (
    control.length !== candidate.length ||
    control.byteLength !== candidate.byteLength
  ) {
    throw new RangeError("OPT-0040 output lengths differ");
  }
  const controlU32 = new Uint32Array(
    control.buffer,
    control.byteOffset,
    control.length,
  );
  const candidateU32 = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  const controlU16 = new Uint16Array(
    control.buffer,
    control.byteOffset,
    control.byteLength / Uint16Array.BYTES_PER_ELEMENT,
  );
  const candidateU16 = new Uint16Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.byteLength / Uint16Array.BYTES_PER_ELEMENT,
  );
  let u32MismatchCount = 0;
  let firstU32MismatchIndex: number | null = null;
  for (let index = 0; index < controlU32.length; index += 1) {
    if (controlU32[index] === candidateU32[index]) continue;
    u32MismatchCount += 1;
    if (firstU32MismatchIndex === null) firstU32MismatchIndex = index;
  }
  let u16MismatchCount = 0;
  let firstU16MismatchIndex: number | null = null;
  for (let index = 0; index < controlU16.length; index += 1) {
    if (controlU16[index] === candidateU16[index]) continue;
    u16MismatchCount += 1;
    if (firstU16MismatchIndex === null) firstU16MismatchIndex = index;
  }
  return Object.freeze({
    comparedU32WordCount: controlU32.length,
    u32MismatchCount,
    firstU32MismatchIndex,
    comparedU16WordCount: controlU16.length,
    u16MismatchCount,
    firstU16MismatchIndex,
    rawU32Exact: u32MismatchCount === 0,
    rawU16Exact: u16MismatchCount === 0,
  });
}

async function compareExecutionSet(
  executions: readonly ExecutionResult[],
  reference: Float32Array,
): Promise<Readonly<{
  passed: boolean;
  comparisonKind:
    "complete-final-fp32-byte-pattern-raw-u32-and-u16-views";
  comparedExecutions: number;
  comparedU32WordsPerExecution: number;
  comparedU16WordsPerExecution: number;
  acceptedOutputSha256: string;
  comparisons: readonly Readonly<{
    index: number;
    arm: Opt0040Arm;
    outputSha256: string;
    comparison: Opt0040RawComparison;
  }>[];
}>> {
  const comparisons = await Promise.all(executions.map(async (
    execution,
    index,
  ) => Object.freeze({
    index,
    arm: execution.arm,
    outputSha256: await sha256Float32(execution.output),
    comparison: compareRaw(reference, execution.output),
  })));
  const passed = comparisons.every((entry) =>
    entry.outputSha256 === OPT_0040_C512_ACCEPTED_OUTPUT_SHA256 &&
    comparisonPassed(entry.comparison)
  );
  return Object.freeze({
    passed,
    comparisonKind:
      "complete-final-fp32-byte-pattern-raw-u32-and-u16-views",
    comparedExecutions: comparisons.length,
    comparedU32WordsPerExecution: OPT_0040_C512_OUTPUT_ELEMENTS,
    comparedU16WordsPerExecution: OPT_0040_C512_OUTPUT_U16_WORDS,
    acceptedOutputSha256: OPT_0040_C512_ACCEPTED_OUTPUT_SHA256,
    comparisons: Object.freeze(comparisons),
  });
}

function comparisonPassed(comparison: Opt0040RawComparison): boolean {
  return comparison.comparedU32WordCount === OPT_0040_C512_OUTPUT_ELEMENTS &&
    comparison.comparedU16WordCount === OPT_0040_C512_OUTPUT_U16_WORDS &&
    comparison.rawU32Exact && comparison.rawU16Exact;
}

function summarizeFamilyTopology(
  profile: AceOpt0011Fp16VaeWindowFamilyProfile,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    quantaPerCommandBuffer: profile.quantaPerCommandBuffer,
    decoderBatchCount: profile.decoderBatchCount,
    decoderQuantumCount: profile.decoderQuantumCount,
    homogeneousBatchCount: profile.homogeneousBatchCount,
    homogeneousQuantumCount: profile.homogeneousQuantumCount,
    mixedBatchCount: profile.mixedBatchCount,
    mixedQuantumCount: profile.mixedQuantumCount,
    families: Object.freeze(Object.fromEntries(Object.entries(
      profile.families,
    ).map(([family, timing]) => [
      family,
      Object.freeze({
        batchCount: timing.batchCount,
        quantumCount: timing.quantumCount,
      }),
    ]))),
  });
}

function aggregateTimings(
  executions: readonly ExecutionResult[],
): Readonly<Record<string, unknown>> {
  const arm = (name: Opt0040Arm) => {
    const timings = executions.filter((execution) => execution.arm === name)
      .map((execution) => execution.receipt.timing);
    if (timings.length !== 2) {
      throw new Error(`OPT-0040 ${name} balanced sample count changed`);
    }
    return Object.freeze({
      sampleCount: 2,
      meanConvTransposeSubmitThroughDrainMs: mean(timings.map((timing) =>
        timing.convTransposeSubmitThroughDrainMs
      )),
      meanDecoderSubmitThroughDrainMs: mean(timings.map((timing) =>
        timing.decoderSubmitThroughDrainMs
      )),
      meanOuterWindowWallMs: mean(timings.map((timing) =>
        timing.outerWindowWallMs
      )),
    });
  };
  const control = arm("control");
  const selector = arm("selector");
  return Object.freeze({
    control,
    selector,
    meanConvTransposeFamilySpeedup:
      control.meanConvTransposeSubmitThroughDrainMs /
        selector.meanConvTransposeSubmitThroughDrainMs,
    meanCompleteDecoderSpeedup: control.meanDecoderSubmitThroughDrainMs /
      selector.meanDecoderSubmitThroughDrainMs,
    meanOuterWindowSpeedup: control.meanOuterWindowWallMs /
      selector.meanOuterWindowWallMs,
  });
}

function requireOneC512Window(plan: AceVaeChunkedDecodePlan): AceVaeDecodeWindow {
  const window = plan.windows[0];
  if (
    plan.latentFrames !== OPT_0040_C512_FRAMES ||
    plan.maximumWindowFrames !== OPT_0040_C512_FRAMES ||
    plan.windows.length !== 1 || window === undefined ||
    window.index !== 0 ||
    window.latentWindowFrames !== OPT_0040_C512_FRAMES ||
    window.discardPrefixLatentFrames !== 0 ||
    window.discardSuffixLatentFrames !== 0
  ) {
    throw new Error("OPT-0040 plan is not one exact C512 window");
  }
  return window;
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
    throw new Error("OPT-0040 authenticated package identity changed");
  }
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    residentBytes,
  });
}

async function acquirePackageFiles(
  preparedPackage: PreparedPackage,
): Promise<ReadonlyMap<string, File>> {
  const cache = await AceOpfsModelCache.open();
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({
      ...preparedPackage.loaded.manifest,
      files: preparedPackage.files,
    }),
    manifestUrl: preparedPackage.loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => postProgress(
      `acquiring VAE ${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== preparedPackage.files.length ||
    acquired.plan.runtimeBytes !== preparedPackage.residentBytes
  ) {
    throw new Error("OPT-0040 package acquisition accounting changed");
  }
  return acquired.files;
}

async function loadVaePhase(
  device: GPUDevice,
  pkg: PreparedPackage,
  files: ReadonlyMap<string, File>,
  arm: Opt0040Arm,
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
    throw new Error(`OPT-0040 ${arm} resident VAE bytes changed`);
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
    throw new Error("OPT-0040 requires stock fixed 32-lane WebGPU subgroups");
  }
}

function validateThermalGate(
  gate: Opt0040ThermalGate,
  readyAtEpochMilliseconds: number,
): void {
  if (
    gate.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    gate.command !== "notifyutil -g com.apple.system.thermalpressurelevel" ||
    gate.protocol !== "wait-30s-then-one-level0-check" ||
    gate.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    gate.checkedAtEpochMilliseconds < gate.startedAtEpochMilliseconds ||
    gate.durationMilliseconds < 30_000 || gate.observationCount !== 1 ||
    gate.observedLevel !== 0 ||
    gate.maximumObservationGapMilliseconds !== gate.durationMilliseconds
  ) {
    throw new Error("OPT-0040 worker rejected the one-check thermal gate");
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

function mean(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError("OPT-0040 mean requires finite values");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

function errorText(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}\n${error.stack ?? ""}`
    : String(error);
}
