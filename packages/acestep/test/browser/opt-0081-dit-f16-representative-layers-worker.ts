/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  assertAceGenerationRequest,
  type AceGenerationRequest,
} from "../../src/api.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from
  "../../src/runtime/diagnostics.js";
import type { AceWorkerConfiguration } from
  "../../src/runtime/protocol.js";
import type {
  AceGenerationProgress,
  AceInitializationProgress,
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0081RepresentativeConditioningAuthority,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS,
  AceOpt0081RepresentativeCheckpointExecutionError,
  type AceOpt0081RepresentativeCheckpointSnapshot,
  type AceOpt0081RepresentativeDeviceLossCleanupEvidence,
  type AceOpt0081RepresentativeDenseTap,
  type AceOpt0081RepresentativeDitSession,
  type AceOpt0081RepresentativeSetupCleanupEvidence,
  type AceOpt0081RepresentativeTap,
  type AceOpt0081RepresentativeTimedSliceResult,
  type AceOpt0081RepresentativeTopology as CoreTopology,
} from "../../src/webgpu/dit-backend.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
} from "../../src/webgpu/vae-window-profile.js";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  createOpt0018Request,
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES,
  OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
  OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES,
  OPT_0081_REPRESENTATIVE_EPOCHS,
  OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
  buildOpt0081RepresentativeRounds,
  inspectOpt0081RepresentativeCorrectness,
  numberToOpt0081Float16Bits,
  requireOpt0081RepresentativeCancellation,
  requireOpt0081RepresentativeReceivedThermalLaunch,
  requireOpt0081RepresentativeTopology,
  summarizeOpt0081RepresentativeTiming,
  type Opt0081RepresentativeCancellationEvidence,
  type Opt0081RepresentativeCorrectness,
  type Opt0081RepresentativeLifecycleEvidence,
  type Opt0081RepresentativeRawCheckpoint,
  type Opt0081RepresentativeThermalLaunch,
  type Opt0081RepresentativeTimingSample,
  type Opt0081RepresentativeTopology,
} from "./opt-0081-dit-f16-representative-layers-contract.js";
import {
  createOpt0081RepresentativeConditioningEvidence,
  createOpt0081RepresentativeIdentity,
  type Opt0081RepresentativeArenaEvidence,
  type Opt0081RepresentativeConditioningEvidence,
  type Opt0081RepresentativePreparationEvidence,
  type Opt0081RepresentativeRunEvidence,
} from "./opt-0081-dit-f16-representative-layers-result.js";

const MAIN_MANIFEST_PATH = "/model/files-reference/manifest.json" as const;
const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
const DENSE_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
const DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
const DENSE_MANIFEST_BYTES = 254_357 as const;
const DENSE_RUNTIME_PROFILE = "opt-0009-fp16-fp32-dense-v1" as const;
const DENSE_KERNEL_SET_ID = "opt-0009-n256-k32-fp16-fp32-v1" as const;
const VAE_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json" as const;
const VAE_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7" as const;
const CONDITIONING_DIAGNOSTIC_CODE =
  "ACE_PLANNER_CONDITIONING_RESOLVED" as const;

const DENSE_TAP_MAP = Object.freeze([
  Object.freeze({ receipt: "selfQuery" as const,
    core: "selfQueryFlat" as const }),
  Object.freeze({ receipt: "selfKey" as const,
    core: "selfKeyFlat" as const }),
  Object.freeze({ receipt: "selfValue" as const,
    core: "selfValueFlat" as const }),
  Object.freeze({ receipt: "selfOutput" as const,
    core: "selfProjectedAttention" as const }),
  Object.freeze({ receipt: "crossQuery" as const,
    core: "crossQueryFlat" as const }),
  Object.freeze({ receipt: "crossOutput" as const,
    core: "crossProjectedAttention" as const }),
  Object.freeze({ receipt: "mlpGate" as const, core: "gate" as const }),
  Object.freeze({ receipt: "mlpUp" as const, core: "up" as const }),
  Object.freeze({ receipt: "mlpDown" as const,
    core: "projectedMlp" as const }),
] as const);

interface PrepareMessage {
  readonly type: "prepare";
  readonly identity: Opt0018RunIdentity;
}

interface RunMessage {
  readonly type: "run";
  readonly thermalLaunch: Opt0081RepresentativeThermalLaunch;
}

type IncomingMessage = PrepareMessage | RunMessage;

interface PreparationCapture {
  readonly correctness: Opt0081RepresentativeCorrectness;
  readonly cancellation: Opt0081RepresentativeCancellationEvidence;
  readonly arena: Opt0081RepresentativeArenaEvidence;
  readonly preparation: Opt0081RepresentativePreparationEvidence;
}

interface SetupPreflightCapture {
  readonly cleanup: AceOpt0081RepresentativeSetupCleanupEvidence;
  readonly conditioningAuthority:
    AceOpt0081RepresentativeConditioningAuthority;
}

interface DiagnosticCounts {
  fatal: number;
  validation: number;
  deviceLoss: number;
}

let state: "idle" | "preparing" | "ready" | "timing" | "settled" = "idle";
let runIdentity: Opt0018RunIdentity | undefined;
let timingAuthorization: PromiseWithResolvers<
  Opt0081RepresentativeThermalLaunch
> | undefined;
let readyAtEpochMilliseconds = 0;
let activeFailurePhase = "preparation";

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "prepare" && state === "idle") {
    state = "preparing";
    void executeGate(message.identity).catch((error) =>
      fail(activeFailurePhase, error));
    return;
  }
  if (
    message.type === "run" && state === "ready" &&
    timingAuthorization !== undefined
  ) {
    try {
      const launch = requireOpt0081RepresentativeReceivedThermalLaunch(
        message.thermalLaunch,
        readyAtEpochMilliseconds,
        Date.now(),
      );
      state = "timing";
      activeFailurePhase = "timing";
      timingAuthorization.resolve(launch);
    } catch (error) {
      state = "timing";
      activeFailurePhase = "thermal-launch";
      timingAuthorization.reject(error);
    }
  }
});

async function executeGate(identityInput: Opt0018RunIdentity): Promise<void> {
  const identity = validateOpt0018RunIdentity(identityInput);
  runIdentity = identity;
  const request = validateCanonicalRequest();
  postProgress("running the isolated after-readback setup-failure cleanup preflight");
  const setupPreflight = await runSetupFailurePreflight(request);
  const setupFailureCleanup = setupPreflight.cleanup;
  postProgress("initializing the fresh correctness and timing device");
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({
    kind: "ace-opt-0081-representative-complete",
  });
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioningDiagnostic: Readonly<Record<string, unknown>> | undefined;
  let conditioning: Opt0081RepresentativeConditioningEvidence | undefined;
  let preparation: PreparationCapture | undefined;
  let timingSamples: readonly Opt0081RepresentativeTimingSample[] | undefined;
  let thermalLaunch: Opt0081RepresentativeThermalLaunch | undefined;
  let ownerForPostDestroy: AceOpt0081RepresentativeDitSession | undefined;
  let deviceLossCleanup:
    AceOpt0081RepresentativeDeviceLossCleanupEvidence | undefined;
  const diagnosticCounts: DiagnosticCounts = {
    fatal: 0,
    validation: 0,
    deviceLoss: 0,
  };
  let intentionalDeviceLossStarted = false;
  let forbiddenProgress = false;
  let cleanupCompletedAtEpochMilliseconds = 0;
  let primaryError: unknown;
  let cleanupError: unknown;
  let primaryCause: Readonly<Record<string, unknown>> | undefined;
  let cleanupCause: Readonly<Record<string, unknown>> | undefined;
  let checkpointFailureContext: Readonly<{
    readonly arm: "A" | "B";
    readonly layer: 0 | 1;
    readonly tap: AceOpt0081RepresentativeTap;
    readonly substage: string;
  }> | undefined;
  try {
    diagnostics = await backend.initialize(configuration(), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => undefined,
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        recordDiagnostic(
          diagnostic,
          diagnosticCounts,
          intentionalDeviceLossStarted,
        );
      },
    });
    validateDiagnostics(diagnostics);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        if (isForbiddenProgress(progress)) forbiddenProgress = true;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        recordDiagnostic(
          diagnostic,
          diagnosticCounts,
          intentionalDeviceLossStarted,
        );
        if (
          diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE &&
          diagnostic.details !== undefined
        ) {
          if (conditioningDiagnostic !== undefined) {
            throw new Error("OPT-0081 conditioning authority repeated");
          }
          conditioningDiagnostic = validateConditioning(diagnostic);
        }
      },
      opt0081RepresentativeRun: {
        mode: "run",
        verifiedSetupFailureCleanup: setupFailureCleanup,
        async run(
          owner,
          verifiedSetupFailureCleanup,
          mainConditioningAuthority,
        ) {
          if (ownerForPostDestroy !== undefined) {
            throw new Error("OPT-0081 representative owner repeated");
          }
          requireSameSetupCleanupEvidence(
            setupFailureCleanup,
            verifiedSetupFailureCleanup,
          );
          conditioning = createOpt0081RepresentativeConditioningEvidence(
            setupPreflight.conditioningAuthority,
            mainConditioningAuthority,
          );
          ownerForPostDestroy = owner;
          preparation = await prepareOwner(owner, diagnosticCounts);
          timingAuthorization = Promise.withResolvers();
          readyAtEpochMilliseconds = preparation.preparation
            .readyAtEpochMilliseconds;
          state = "ready";
          self.postMessage({
            type: "ready",
            readyAtEpochMilliseconds,
            preparation,
          });
          thermalLaunch = await timingAuthorization.promise;
          timingSamples = await runTiming(owner);
          summarizeOpt0081RepresentativeTiming(timingSamples);
          intentionalDeviceLossStarted = true;
          deviceLossCleanup = await owner.runDeviceLossCleanupPreflight();
          controller.abort(privateStop);
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error("OPT-0081 representative pipeline continued past owner");
    } catch (error) {
      if (error !== privateStop) throw error;
    }
  } catch (error) {
    primaryError = error;
    primaryCause = serializeOpt0018Failure(error);
    if (error instanceof AceOpt0081RepresentativeCheckpointExecutionError) {
      checkpointFailureContext = Object.freeze({
        arm: error.arm,
        layer: error.layer,
        tap: error.tap,
        substage: error.substage,
      });
    }
  } finally {
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
      cleanupCause = serializeOpt0018Failure(error);
    }
    cleanupCompletedAtEpochMilliseconds = Date.now();
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    const combined = primaryError === undefined || cleanupError === undefined
      ? primaryError ?? cleanupError
      : new AggregateError(
        [primaryError, cleanupError],
        "OPT-0081 representative execution and cleanup both failed",
      );
    const lifecycleSnapshot = ownerForPostDestroy?.lifecycleSnapshot();
    throw new Opt0081RepresentativeExecutionFailure(
      combined,
      Object.freeze({
        identity: createOpt0081RepresentativeIdentity(identity),
        ...(primaryCause === undefined ? {} : { primaryCause }),
        ...(cleanupCause === undefined ? {} : { cleanupCause }),
        ...(checkpointFailureContext === undefined
          ? {}
          : { checkpointFailureContext }),
        ...(conditioning === undefined ? {} : { conditioning }),
        ...(preparation === undefined
          ? primaryError instanceof Opt0081RepresentativeCorrectnessFailure
          ? { correctness: primaryError.correctness }
          : {}
          : {
              arena: preparation.arena,
              preparation: preparation.preparation,
              correctness: preparation.correctness,
              cancellation: preparation.cancellation,
            }),
        failureCleanup: Object.freeze({
          setupFailureCleanup,
          ...(lifecycleSnapshot === undefined
            ? {}
            : { ownerLifecycle: lifecycleSnapshot }),
          cleanupCompletedAtEpochMilliseconds,
        }),
      }),
    );
  }
  if (
    diagnostics === undefined || conditioningDiagnostic === undefined ||
    conditioning === undefined ||
    preparation === undefined || timingSamples === undefined ||
    thermalLaunch === undefined || ownerForPostDestroy === undefined ||
    deviceLossCleanup === undefined || diagnosticCounts.fatal !== 0 ||
    forbiddenProgress
  ) throw new Error("OPT-0081 representative pipeline evidence is incomplete");

  const lifecycle = requirePipelineLifecycleEvidence(
    ownerForPostDestroy,
    setupFailureCleanup,
    deviceLossCleanup,
  );
  const evidence: Opt0081RepresentativeRunEvidence = Object.freeze({
    identity: createOpt0081RepresentativeIdentity(identity),
    conditioning,
    arena: preparation.arena,
    preparation: preparation.preparation,
    correctness: preparation.correctness,
    cancellation: preparation.cancellation,
    timingSamples,
    cleanup: lifecycle,
    cleanupCompletedAtEpochMilliseconds,
    thermalLaunch,
  });
  state = "settled";
  self.postMessage({ type: "measurement-complete", evidence });
}

async function runSetupFailurePreflight(
  request: AceGenerationRequest,
): Promise<SetupPreflightCapture> {
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({
    kind: "ace-opt-0081-representative-setup-failure-complete",
  });
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioningDiagnostic: Readonly<Record<string, unknown>> | undefined;
  let conditioningAuthority:
    AceOpt0081RepresentativeConditioningAuthority | undefined;
  let evidence: AceOpt0081RepresentativeSetupCleanupEvidence | undefined;
  let fatalDiagnosticCount = 0;
  let forbiddenProgress = false;
  let primaryError: unknown;
  let cleanupError: unknown;
  try {
    diagnostics = await backend.initialize(configuration(), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => undefined,
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
      },
    });
    validateDiagnostics(diagnostics);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        if (isForbiddenProgress(progress)) forbiddenProgress = true;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
        if (
          diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE &&
          diagnostic.details !== undefined
        ) {
          if (conditioningDiagnostic !== undefined) {
            throw new Error("OPT-0081 setup conditioning authority repeated");
          }
          conditioningDiagnostic = validateConditioning(diagnostic);
        }
      },
      opt0081RepresentativeRun: {
        mode: "setup-failure",
        onEvidence(value, authority) {
          if (evidence !== undefined) {
            throw new Error("OPT-0081 setup cleanup evidence repeated");
          }
          evidence = requireSetupCleanupEvidence(value);
          conditioningAuthority = authority;
          controller.abort(privateStop);
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error("OPT-0081 setup preflight continued past failpoint");
    } catch (error) {
      if (error !== privateStop) throw error;
    }
  } catch (error) {
    primaryError = error;
  } finally {
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
    }
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    throw primaryError === undefined || cleanupError === undefined
      ? primaryError ?? cleanupError
      : new AggregateError(
        [primaryError, cleanupError],
        "OPT-0081 setup preflight and cleanup both failed",
      );
  }
  if (
    diagnostics === undefined || conditioningDiagnostic === undefined ||
    conditioningAuthority === undefined || evidence === undefined ||
    fatalDiagnosticCount !== 0 || forbiddenProgress
  ) throw new Error("OPT-0081 setup-failure preflight evidence is incomplete");
  return Object.freeze({ cleanup: evidence, conditioningAuthority });
}

async function prepareOwner(
  owner: AceOpt0081RepresentativeDitSession,
  diagnosticCounts: DiagnosticCounts,
): Promise<PreparationCapture> {
  validateCoreTopology(owner.topology);
  const arena = requireArenaEvidence(owner);
  postProgress("running the ordinary 25-command precompute for A and B");
  await owner.runPrecompute();
  postProgress("running A1/A2/B1/B2 raw checkpoints for all two-layer taps");
  await owner.beginCorrectnessCheckpointing();
  let correctness: Opt0081RepresentativeCorrectness | undefined;
  let correctnessError: unknown;
  let checkpointCleanupError: unknown;
  try {
    correctness = await runCorrectness(owner, diagnosticCounts);
  } catch (error) {
    correctnessError = error;
  }
  try {
    await owner.endCorrectnessCheckpointing();
  } catch (error) {
    checkpointCleanupError = error;
  }
  if (correctnessError !== undefined || checkpointCleanupError !== undefined) {
    if (correctnessError !== undefined && checkpointCleanupError !== undefined) {
      throw new AggregateError(
        [correctnessError, checkpointCleanupError],
        "OPT-0081 correctness execution and arena cleanup both failed",
      );
    }
    throw correctnessError ?? checkpointCleanupError;
  }
  if (correctness === undefined) {
    throw new Error("OPT-0081 correctness completed without evidence");
  }
  const inspection = inspectOpt0081RepresentativeCorrectness(correctness);
  if (!inspection.structurallyValid || !inspection.passed) {
    throw new Opt0081RepresentativeCorrectnessFailure(correctness);
  }
  requireCorrectnessOwnerSnapshot(owner);
  postProgress("running the candidate one-successor cancellation preflight");
  const coreCancellation = await owner.runCancellationPreflight("B");
  const cancellation = requireOpt0081RepresentativeCancellation(
    normalizeCancellation(coreCancellation),
  );
  postProgress("warming one terminally drained selected slice per arm");
  await owner.warmup("A");
  await owner.warmup("B");
  const readyAt = Date.now();
  const preparation = Object.freeze({
    correctnessCompletedBeforeReady: true,
    precomputeCompletedBeforeReady: true,
    precomputeCommandBufferCountPerArm: 25,
    warmupArmOrder: Object.freeze(["A", "B"] as const),
    warmupSelectedSliceOccurrencesPerArm: 1,
    everyWarmupTerminallyDrained: true,
    conditioningAuthorityCapturedBeforeReady: true,
    setupAndMainConditioningAuthorityExact: true,
    readyAtEpochMilliseconds: readyAt,
    passed: true,
  } satisfies Opt0081RepresentativePreparationEvidence);
  return Object.freeze({ correctness, cancellation, arena, preparation });
}

async function runCorrectness(
  owner: AceOpt0081RepresentativeDitSession,
  diagnosticCounts: DiagnosticCounts,
): Promise<Opt0081RepresentativeCorrectness> {
  const controlBoundaryRepeat: Opt0081RepresentativeRawCheckpoint[] = [];
  const candidateCastB1: Opt0081RepresentativeRawCheckpoint[] = [];
  const candidateCastB2: Opt0081RepresentativeRawCheckpoint[] = [];
  const candidateBoundaryRepeat: Opt0081RepresentativeRawCheckpoint[] = [];
  const aaDense: Opt0081RepresentativeRawCheckpoint[] = [];
  const abDense: Opt0081RepresentativeRawCheckpoint[] = [];
  const bbDense: Opt0081RepresentativeRawCheckpoint[] = [];
  const aaLayer: Opt0081RepresentativeRawCheckpoint[] = [];
  const abLayer: Opt0081RepresentativeRawCheckpoint[] = [];
  const bbLayer: Opt0081RepresentativeRawCheckpoint[] = [];
  let completed = 0;
  const total = 2 * (
    OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES.length +
    DENSE_TAP_MAP.length + 1
  );
  for (const layer of [0, 1] as const) {
    for (const { tap, words } of OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES) {
      const [a1, a2, b1, b2] = await snapshotFour(owner, layer, tap);
      requireSnapshot(a1, "A", layer, tap, "u32", words);
      requireSnapshot(a2, "A", layer, tap, "u32", words);
      requireSnapshot(b1, "B", layer, tap, "u16", words);
      requireSnapshot(b2, "B", layer, tap, "u16", words);
      controlBoundaryRepeat.push(rawComparison(layer, tap, a1, a2));
      candidateCastB1.push(castComparison(layer, tap, a1, b1));
      candidateCastB2.push(castComparison(layer, tap, a2, b2));
      candidateBoundaryRepeat.push(rawComparison(layer, tap, b1, b2));
      completed += 1;
      postProgress(`raw checkpoint ${completed}/${total}: layer ${layer} ${tap}`);
    }
    for (const mapping of DENSE_TAP_MAP) {
      const role = OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES.find(
        ({ tap }) => tap === mapping.receipt,
      );
      if (role === undefined) throw new Error("OPT-0081 dense tap escaped inventory");
      const [a1, a2, b1, b2] = await snapshotFour(
        owner,
        layer,
        mapping.core,
      );
      for (const [snapshot, arm] of [
        [a1, "A"], [a2, "A"], [b1, "B"], [b2, "B"],
      ] as const) requireSnapshot(
        snapshot,
        arm,
        layer,
        mapping.core,
        "u32",
        role.words,
      );
      aaDense.push(rawComparison(layer, mapping.receipt, a1, a2));
      abDense.push(rawComparison(layer, mapping.receipt, a1, b1));
      bbDense.push(rawComparison(layer, mapping.receipt, b1, b2));
      completed += 1;
      postProgress(
        `raw checkpoint ${completed}/${total}: layer ${layer} ${mapping.receipt}`,
      );
    }
    const [a1, a2, b1, b2] = await snapshotFour(
      owner,
      layer,
      "layerOutput",
    );
    for (const [snapshot, arm] of [
      [a1, "A"], [a2, "A"], [b1, "B"], [b2, "B"],
    ] as const) requireSnapshot(
      snapshot,
      arm,
      layer,
      "layerOutput",
      "u32",
      4_608_000,
    );
    aaLayer.push(rawComparison(layer, "layerOutput", a1, a2));
    abLayer.push(rawComparison(layer, "layerOutput", a1, b1));
    bbLayer.push(rawComparison(layer, "layerOutput", b1, b2));
    completed += 1;
    postProgress(`raw checkpoint ${completed}/${total}: layer ${layer} output`);
  }
  const candidate = {
    completedBeforeReady: true,
    runOrder: Object.freeze(["A", "A", "B", "B"] as const),
    controlBoundaryRepeat: Object.freeze(controlBoundaryRepeat),
    boundaryRuns: Object.freeze([
      Object.freeze({ run: "B1" as const,
        checkpoints: Object.freeze(candidateCastB1) }),
      Object.freeze({ run: "B2" as const,
        checkpoints: Object.freeze(candidateCastB2) }),
    ]),
    candidateBoundaryRepeat: Object.freeze(candidateBoundaryRepeat),
    comparisons: Object.freeze([
      Object.freeze({ comparison: "A/A" as const,
        denseOutputs: Object.freeze(aaDense),
        layerOutputs: Object.freeze(aaLayer) }),
      Object.freeze({ comparison: "A/B" as const,
        denseOutputs: Object.freeze(abDense),
        layerOutputs: Object.freeze(abLayer) }),
      Object.freeze({ comparison: "B/B" as const,
        denseOutputs: Object.freeze(bbDense),
        layerOutputs: Object.freeze(bbLayer) }),
    ]),
    boundaryWordsPerCandidateRun: 73_728_000,
    denseOutputWordsPerComparison: 110_592_000,
    layerOutputWordsPerComparison: 9_216_000,
    uncapturedGpuErrorCount: diagnosticCounts.fatal,
    validationErrorCount: diagnosticCounts.validation,
    deviceLossCount: diagnosticCounts.deviceLoss,
    passed: true,
  } satisfies Opt0081RepresentativeCorrectness;
  return Object.freeze({
    ...candidate,
    passed: allCheckpointGates(candidate) && diagnosticCounts.fatal === 0 &&
      diagnosticCounts.validation === 0 && diagnosticCounts.deviceLoss === 0,
  });
}

async function snapshotFour(
  owner: AceOpt0081RepresentativeDitSession,
  layer: 0 | 1,
  tap: AceOpt0081RepresentativeTap,
): Promise<readonly [
  AceOpt0081RepresentativeCheckpointSnapshot,
  AceOpt0081RepresentativeCheckpointSnapshot,
  AceOpt0081RepresentativeCheckpointSnapshot,
  AceOpt0081RepresentativeCheckpointSnapshot,
]> {
  const a1 = await owner.snapshotCheckpoint({ arm: "A", layer, tap });
  const a2 = await owner.snapshotCheckpoint({ arm: "A", layer, tap });
  const b1 = await owner.snapshotCheckpoint({ arm: "B", layer, tap });
  const b2 = await owner.snapshotCheckpoint({ arm: "B", layer, tap });
  return Object.freeze([a1, a2, b1, b2]);
}

function rawComparison(
  layer: 0 | 1,
  tap: string,
  left: AceOpt0081RepresentativeCheckpointSnapshot,
  right: AceOpt0081RepresentativeCheckpointSnapshot,
): Opt0081RepresentativeRawCheckpoint {
  if (left.storage !== right.storage || left.elementCount !== right.elementCount) {
    throw new Error("OPT-0081 raw comparison storage changed");
  }
  let differingWordCount = 0;
  for (let index = 0; index < left.words.length; index += 1) {
    if (left.words[index] !== right.words[index]) differingWordCount += 1;
  }
  return checkpointReceipt(layer, tap, left, right, differingWordCount,
    left.positiveZeroCount === right.positiveZeroCount &&
      left.negativeZeroCount === right.negativeZeroCount);
}

function castComparison(
  layer: 0 | 1,
  tap: string,
  control: AceOpt0081RepresentativeCheckpointSnapshot,
  candidate: AceOpt0081RepresentativeCheckpointSnapshot,
): Opt0081RepresentativeRawCheckpoint {
  if (
    control.storage !== "u32" || candidate.storage !== "u16" ||
    control.elementCount !== candidate.elementCount
  ) throw new Error("OPT-0081 typed boundary storage changed");
  const floats = new Float32Array(
    control.words.buffer,
    control.words.byteOffset,
    control.words.length,
  );
  let differingWordCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  for (let index = 0; index < floats.length; index += 1) {
    const expected = numberToOpt0081Float16Bits(floats[index]!);
    if (expected !== candidate.words[index]) differingWordCount += 1;
    if (expected === 0) positiveZeroCount += 1;
    else if (expected === 0x8000) negativeZeroCount += 1;
  }
  return checkpointReceipt(
    layer,
    tap,
    control,
    candidate,
    differingWordCount,
    positiveZeroCount === candidate.positiveZeroCount &&
      negativeZeroCount === candidate.negativeZeroCount,
  );
}

function checkpointReceipt(
  layer: 0 | 1,
  tap: string,
  left: AceOpt0081RepresentativeCheckpointSnapshot,
  right: AceOpt0081RepresentativeCheckpointSnapshot,
  differingWordCount: number,
  signedZeroExact: boolean,
): Opt0081RepresentativeRawCheckpoint {
  return Object.freeze({
    layer,
    tap,
    comparedWords: right.elementCount,
    differingWordCount,
    unwrittenWordCount: right.qNaNPrefillCount,
    exact: differingWordCount === 0,
    signedZeroExact,
    finite: left.nonFiniteCount === 0 && right.nonFiniteCount === 0,
    qNaNPrefillOverwritten:
      left.completeQNaNOverwrite && right.completeQNaNOverwrite,
    firstWordCovered: left.firstValidWritten && right.firstValidWritten,
    lastWordCovered: left.lastValidWritten && right.lastValidWritten,
    tailRows2240Through2249Covered:
      left.rows2240Through2249Written && right.rows2240Through2249Written,
    prefixGuardIntact:
      left.readbackPrefixGuardIntact && right.readbackPrefixGuardIntact,
    suffixGuardIntact:
      left.readbackSuffixGuardIntact && right.readbackSuffixGuardIntact,
    adjacentGuardsIntact:
      left.adjacentBeforeGuardIntact && left.adjacentAfterGuardIntact &&
      right.adjacentBeforeGuardIntact && right.adjacentAfterGuardIntact,
    sha256: aceSha256Hex(right.bytes),
  });
}

async function runTiming(
  owner: AceOpt0081RepresentativeDitSession,
): Promise<readonly Opt0081RepresentativeTimingSample[]> {
  const samples: Opt0081RepresentativeTimingSample[] = [];
  for (const round of buildOpt0081RepresentativeRounds()) {
    for (let occurrenceIndex = 0; occurrenceIndex < 2; occurrenceIndex += 1) {
      const arm = round.armOrder[occurrenceIndex]!;
      postProgress(
        `timing round ${round.roundIndex + 1}/8 ${round.armOrder.join("")} arm ${arm}`,
      );
      const progressIndices: number[] = [];
      const result = await owner.runTimedSlice(arm, {}, {
        onCommandBufferCompleted(timing, progress) {
          if (
            progress.completedCommandBuffers !== timing.commandBufferIndex + 1 ||
            progress.totalCommandBuffers !== 28
          ) throw new Error("OPT-0081 completion progress order changed");
          progressIndices.push(timing.commandBufferIndex);
        },
      });
      samples.push(normalizeTimingSample(
        result,
        round.roundIndex,
        occurrenceIndex as 0 | 1,
        round.direction,
        progressIndices,
      ));
    }
  }
  return Object.freeze(samples);
}

function normalizeTimingSample(
  result: AceOpt0081RepresentativeTimedSliceResult,
  roundIndex: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7,
  occurrenceIndex: 0 | 1,
  direction: "forward" | "reverse",
  progressIndices: readonly number[],
): Opt0081RepresentativeTimingSample {
  if (
    result.maximumPendingDescriptorCount !== 2 ||
    result.pendingDescriptorCountAfterRun !== 0
  ) throw new Error(
    "OPT-0081 owner omitted fenced boundaries or pending-descriptor evidence",
  );
  const topology = normalizeTopology(
    result.topology,
    result,
    progressIndices,
    result.maximumPendingDescriptorCount,
    result.pendingDescriptorCountAfterRun,
  );
  return Object.freeze({
    roundIndex,
    occurrenceIndex,
    arm: result.arm,
    direction,
    sliceStartedAtPerformanceMilliseconds:
      result.startedAtPerformanceMs,
    sliceDrainedAtPerformanceMilliseconds:
      result.drainedAtPerformanceMs,
    sliceWallMilliseconds: result.wallMs,
    epochs: Object.freeze(OPT_0081_REPRESENTATIVE_EPOCHS.map(
      (epoch, index) => Object.freeze({
        epochIndex: epoch.epochIndex,
        firstCommandIndex: epoch.firstCommandIndex,
        lastCommandIndex: epoch.lastCommandIndex,
        submitThroughTrueDrainMilliseconds:
          result.completionEpochWallsMs[index]!,
      }),
    )),
    completionFenceLatenciesMilliseconds: Object.freeze([
      ...result.submitThroughCompletionFenceMs,
    ]),
    topology,
  });
}

function normalizeTopology(
  core: CoreTopology,
  result: AceOpt0081RepresentativeTimedSliceResult,
  progressIndices: readonly number[],
  maximumPendingDescriptorCount: 2,
  pendingDescriptorCountAfterRun: 0,
): Opt0081RepresentativeTopology {
  validateCoreTopology(core);
  if (
    progressIndices.length !== 28 ||
    progressIndices.some((value, index) => value !== index)
  ) throw new Error("OPT-0081 timed progress order changed");
  const labels = physicalLabels(core.controlPhysicalQuanta);
  const topology = Object.freeze({
    firstCommandIndex: 25,
    lastCommandIndex: 52,
    commandBufferCount: result.commandBuffersSubmitted,
    completionFenceRequestedCount: result.completionFenceRequestedCount,
    completionFenceSettledCount: result.completionFenceSettledCount,
    completionFenceRejectedCount: result.completionFenceRejectedCount,
    completionEpochCount: result.completionEpochCount,
    trueQueueDrainCount: result.trueQueueDrainCount,
    cooperativeIdleTurnCount: result.cooperativeIdleTurns,
    requestedCooperativeIdleMilliseconds: result.requestedCooperativeIdleMs,
    maximumOutstandingCommandBuffers: result.maximumOutstandingCommandBuffers,
    maximumPendingDescriptorCount,
    pendingDescriptorCountAfterRun,
    timestepCommandCount: core.timestepCommandBufferCount,
    inputProjectionCommandCount: core.inputProjectionCommandBufferCount,
    slidingLayerCommandCount: core.layerCommandBufferCounts[0],
    fullLayerCommandCount: core.layerCommandBufferCounts[1],
    producerStoreCount: core.producerStoreCount,
    denseConsumerCount: core.denseConsumerCount,
    layerAttentionRoutes: core.layerAttentionRoutes,
    descriptorOrTapCommandCount: core.descriptorOrTapCommandCount,
    timestampQueryCount: core.timestampQueryCount,
    measurementReadbackCount: core.measurementReadbackCount,
    measurementMapCount: core.measurementMapCount,
    attentionRuntimeProfile: core.attentionRuntimeProfile,
    epochs: OPT_0081_REPRESENTATIVE_EPOCHS,
    physicalCommandLabels: labels,
    progressLabels: Object.freeze(progressIndices.map((index) => labels[index]!)),
  } satisfies Opt0081RepresentativeTopology);
  return requireOpt0081RepresentativeTopology(topology);
}

function validateCoreTopology(value: CoreTopology): void {
  const controlLabels = physicalLabels(value.controlPhysicalQuanta);
  const candidateLabels = physicalLabels(value.candidatePhysicalQuanta);
  if (
    value.schema !== "ace-opt-0081-representative-topology-v1" ||
    value.precomputeCommandBufferCountPerArm !== 25 ||
    value.timedCommandBufferCount !== 28 ||
    value.phaseCommandBufferCounts[0] !== 28 ||
    value.timestepCommandBufferCount !== 1 ||
    value.inputProjectionCommandBufferCount !== 1 ||
    value.layerCommandBufferCounts[0] !== 11 ||
    value.layerCommandBufferCounts[1] !== 15 ||
    value.completionEpochCount !== 7 || value.trueQueueDrainCount !== 7 ||
    value.cooperativeIdleTurns !== 6 ||
    value.maximumOutstandingCommandBuffers !== 2 ||
    value.maximumPendingDescriptorCount !== 2 ||
    value.attentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    JSON.stringify(value.layerAttentionRoutes) !== JSON.stringify([
      { layer: 0, self: "query8-sliding", cross: "query8-cross" },
      { layer: 1, self: "quad-query32-full", cross: "query8-cross" },
    ]) ||
    value.producerStoreCount !== 12 || value.denseConsumerCount !== 18 ||
    value.descriptorOrTapCommandCount !== 0 ||
    value.timestampQueryCount !== 0 || value.measurementReadbackCount !== 0 ||
    value.measurementMapCount !== 0 ||
    JSON.stringify(value.fourthEpochProductionPhysicalIndices) !==
      "[37,38,39,40]" ||
    controlLabels.length !== 28 ||
    JSON.stringify(controlLabels) !== JSON.stringify(candidateLabels) ||
    value.controlPhysicalQuanta.some((quantum, index) =>
      quantum.slicePhysicalIndex !== index ||
      quantum.productionPhysicalIndex !== index + 25
    ) || value.candidatePhysicalQuanta.some((quantum, index) =>
      quantum.slicePhysicalIndex !== index ||
      quantum.productionPhysicalIndex !== index + 25
    )
  ) throw new Error("OPT-0081 core representative topology changed");
}

function physicalLabels(
  values: CoreTopology["controlPhysicalQuanta"],
): readonly string[] {
  return Object.freeze(values.map((value) =>
    `${value.logicalLabel}#${value.logicalSubquantumIndex}/` +
      `${value.logicalSubquantumCount}`
  ));
}

function requireArenaEvidence(
  owner: AceOpt0081RepresentativeDitSession,
): Opt0081RepresentativeArenaEvidence {
  const memory = owner.memory;
  const passed = memory.controlArenaBytes === 674_815_488 &&
    memory.candidateArenaBytes === 601_087_488 &&
    memory.arenaSavingBytes === 73_728_000 &&
    memory.controlArenaBytes - memory.candidateArenaBytes ===
      memory.arenaSavingBytes &&
    memory.controlRoleCount === 123 && memory.candidateRoleCount === 123 &&
    memory.controlSlotCount === 98 && memory.candidateSlotCount === 98 &&
    JSON.stringify(memory.candidateSelectedDenseInputSlotIndices) ===
      "[61,69,72,77,81,84]" &&
    memory.timedCastOrCopyAuxiliaryBufferCount === 0 &&
    memory.largestArenaBindingIncreaseBytes === 0 &&
    memory.candidateLargestArenaBindingBytes <=
      memory.controlLargestArenaBindingBytes;
  if (!passed) throw new Error("OPT-0081 representative arena changed");
  return Object.freeze({
    controlArenaBytes: memory.controlArenaBytes,
    candidateArenaBytes: memory.candidateArenaBytes,
    arenaSavingBytes: memory.arenaSavingBytes,
    controlRoleCount: memory.controlRoleCount,
    candidateRoleCount: memory.candidateRoleCount,
    controlSlotCount: memory.controlSlotCount,
    candidateSlotCount: memory.candidateSlotCount,
    candidateSelectedDenseInputSlotIndices: Object.freeze([
      ...memory.candidateSelectedDenseInputSlotIndices,
    ]),
    controlLargestArenaBindingBytes:
      memory.controlLargestArenaBindingBytes,
    candidateLargestArenaBindingBytes:
      memory.candidateLargestArenaBindingBytes,
    largestArenaBindingIncreaseBytes:
      memory.largestArenaBindingIncreaseBytes,
    castOrCopyBufferCount: memory.timedCastOrCopyAuxiliaryBufferCount,
    passed,
  });
}

function normalizeCancellation(
  value: Awaited<ReturnType<
    AceOpt0081RepresentativeDitSession["runCancellationPreflight"]
  >>,
): Opt0081RepresentativeCancellationEvidence {
  const passed = value.schema ===
      "ace-opt-0081-representative-cancellation-v1" &&
    value.arm === "B" && value.residentArmReused === true &&
    value.observedCommandBufferIndex === 0 &&
    value.successorAlreadySubmitted && value.commandBuffersCreated === 2 &&
    value.completionCallbackCount === 1 &&
    value.completionCallbackCountAfterAbort === 0 &&
    value.noBackfillAfterObservation && value.originalReasonPreserved &&
    value.submittedFencesSettledBeforeReturn &&
    value.cleanupWithinOneSecond && value.settlementWallMs <= 1_000 &&
    value.maximumPendingDescriptorCount === 2 &&
    value.pendingDescriptorCountAfterRun === 0 &&
    value.temporaryCreatedGraphBufferCount === 0 &&
    value.temporaryDestroyedGraphBufferCount === 0 &&
    value.temporaryLiveGraphBufferCountAfterCleanup === 0 &&
    value.temporaryLiveGraphByteCountAfterCleanup === 0 &&
    value.temporaryRuntimeOwnerCount === 0 &&
    value.temporaryDestroyedRuntimeOwnerCount === 0;
  return Object.freeze({
    arm: "B",
    residentArmReused: value.residentArmReused,
    successorSubmittedBeforeObservation: true,
    outstandingSuccessorCountAtObservation: 1,
    backfillAfterObservationCount: 0,
    progressAfterObservationCount: 0,
    allSubmittedFencesSettledBeforeRelease:
      value.submittedFencesSettledBeforeReturn,
    originalErrorPreserved: value.originalReasonPreserved,
    cleanupMilliseconds: value.settlementWallMs,
    pendingDescriptorCountAfterCleanup: value.pendingDescriptorCountAfterRun,
    temporaryCreatedBufferCount: value.temporaryCreatedGraphBufferCount,
    temporaryDestroyedBufferCount: value.temporaryDestroyedGraphBufferCount,
    temporaryLiveBufferCountAfterCleanup:
      value.temporaryLiveGraphBufferCountAfterCleanup,
    temporaryLiveByteCountAfterCleanup:
      value.temporaryLiveGraphByteCountAfterCleanup,
    temporaryRuntimeOwnerCount: value.temporaryRuntimeOwnerCount,
    temporaryDestroyedRuntimeOwnerCount:
      value.temporaryDestroyedRuntimeOwnerCount,
    passed,
  });
}

function requirePipelineLifecycleEvidence(
  owner: AceOpt0081RepresentativeDitSession,
  setupFailure: AceOpt0081RepresentativeSetupCleanupEvidence,
  deviceLoss: AceOpt0081RepresentativeDeviceLossCleanupEvidence,
): Opt0081RepresentativeLifecycleEvidence {
  const value = owner.lifecycleSnapshot();
  const verifiedSetup = requireSetupCleanupEvidence(setupFailure);
  const devicePassed = deviceLoss.schema ===
      "ace-opt-0081-representative-device-loss-cleanup-v1" &&
    deviceLoss.deviceLossInduced && deviceLoss.deviceLossObserved &&
    deviceLoss.ownerDestroyedAfterLoss &&
    deviceLoss.liveGraphBufferCount === 0 &&
    deviceLoss.liveGraphByteCount === 0 && deviceLoss.liveMapCount === 0 &&
    deviceLoss.pendingDescriptorCount === 0 &&
    deviceLoss.activeCallbackCount === 0 && deviceLoss.activeLeaseCount === 0 &&
    deviceLoss.idempotentDestroyVerified && deviceLoss.postDestroyRejected;
  const passed = value.schema === "ace-opt-0081-representative-lifecycle-v1" &&
    value.state === "destroyed" && value.createdGraphBufferCount > 0 &&
    value.createdGraphBufferCount === value.destroyedGraphBufferCount &&
    value.maximumLiveGraphByteCount > 0 && value.mappedRangeCount > 0 &&
    value.mappedRangeCount === value.unmappedRangeCount &&
    value.liveGraphBufferCount === 0 && value.liveGraphByteCount === 0 &&
    value.liveMapCount === 0 && value.pendingDescriptorCount === 0 &&
    value.maximumPendingDescriptorCount === 2 &&
    value.activeCallbackCount === 0 && value.activeLeaseCount === 0 &&
    value.maximumActiveLeaseCount === 1 &&
    value.correctnessTargetCount === 0 &&
    value.maximumCorrectnessTargetCount === 1 &&
    value.correctnessRuntimeCount === 0 &&
    value.maximumCorrectnessRuntimeCount === 1 &&
    value.maximumCorrectnessTargetBytes ===
      value.maximumDetachedCheckpointBytes + 512 &&
    value.correctnessTargetCompilationCount === 64 &&
    value.correctnessTargetReuseCount === 64 &&
    value.checkpointSnapshotCount === 128 &&
    value.maximumDetachedCheckpointBytes > 0 && value.runtimeOwnerCount > 0 &&
    value.resetOrPrefillCount > 0 && value.profileSwitchCount > 0 &&
    value.snapshotMapCount === 128 &&
    value.guardedTargetReleaseCount === 64 && value.armReleaseCount === 2 &&
    value.drainOrderViolationCount === 0 &&
    value.profileSwitchWhilePendingCount === 0 &&
    value.runtimeOwnerCount === 66 &&
    value.destroyedRuntimeOwnerCount === 66 &&
    value.residentModelDestroyed && value.precomputeCompleted &&
    value.destroyCallCount >= 2 &&
    value.postDestroyRejectedOperationCount >= 1 &&
    value.setupFailureCleanupExecuted && value.deviceLossCleanupExecuted &&
    devicePassed;
  const evidence = Object.freeze({
    createdBufferCount: value.createdGraphBufferCount,
    destroyedBufferCount: value.destroyedGraphBufferCount,
    maximumLiveByteCount: value.maximumLiveGraphByteCount,
    mappedRangeCount: value.mappedRangeCount,
    unmappedRangeCount: value.unmappedRangeCount,
    liveBufferCount: value.liveGraphBufferCount,
    liveByteCount: value.liveGraphByteCount,
    liveMapCount: value.liveMapCount,
    pendingDescriptorCount: value.pendingDescriptorCount,
    maximumPendingDescriptorCount: value.maximumPendingDescriptorCount,
    callbackCount: value.activeCallbackCount,
    leaseCount: value.activeLeaseCount,
    maximumActiveLeaseCount: value.maximumActiveLeaseCount,
    correctnessTargetCount: value.correctnessTargetCount,
    maximumCorrectnessTargetCount: value.maximumCorrectnessTargetCount,
    correctnessRuntimeCount: value.correctnessRuntimeCount,
    maximumCorrectnessRuntimeCount: value.maximumCorrectnessRuntimeCount,
    maximumCorrectnessTargetBytes: value.maximumCorrectnessTargetBytes,
    correctnessTargetCompilationCount:
      value.correctnessTargetCompilationCount,
    correctnessTargetReuseCount: value.correctnessTargetReuseCount,
    checkpointSnapshotCount: value.checkpointSnapshotCount,
    maximumDetachedCheckpointBytes: value.maximumDetachedCheckpointBytes,
    resetOrPrefillCount: value.resetOrPrefillCount,
    profileSwitchCount: value.profileSwitchCount,
    snapshotMapCount: value.snapshotMapCount,
    guardedTargetReleaseCount: value.guardedTargetReleaseCount,
    armReleaseCount: value.armReleaseCount,
    drainOrderViolationCount: value.drainOrderViolationCount,
    profileSwitchWhilePendingCount: value.profileSwitchWhilePendingCount,
    runtimeOwnerCount: value.runtimeOwnerCount,
    destroyedRuntimeOwnerCount: value.destroyedRuntimeOwnerCount,
    residentModelDestroyed: value.residentModelDestroyed,
    precomputeCompleted: value.precomputeCompleted,
    destroyCallCount: value.destroyCallCount,
    postDestroyRejectedOperationCount:
      value.postDestroyRejectedOperationCount,
    drainBeforeEveryResetSwitchMapAndRelease:
      value.drainOrderViolationCount === 0,
    profileSwitchOnlyAfterTerminalDrain:
      value.profileSwitchWhilePendingCount === 0,
    postDestroyRejected: deviceLoss.postDestroyRejected,
    idempotentDestroy: deviceLoss.idempotentDestroyVerified,
    setupFailureCleanupPassed: value.setupFailureCleanupExecuted,
    deviceLossCleanupPassed: devicePassed && value.deviceLossCleanupExecuted,
    deviceDestroyed: deviceLoss.deviceLossInduced,
    setupFailureCleanup: Object.freeze({
      schema: verifiedSetup.schema,
      createdBufferCount: verifiedSetup.createdGraphBufferCount,
      destroyedBufferCount: verifiedSetup.destroyedGraphBufferCount,
      liveBufferCount: verifiedSetup.liveGraphBufferCount,
      liveByteCount: verifiedSetup.liveGraphByteCount,
      runtimeOwnerCount: verifiedSetup.runtimeOwnerCount,
      destroyedRuntimeOwnerCount: verifiedSetup.destroyedRuntimeOwnerCount,
      residentModelDestroyed: verifiedSetup.residentModelDestroyed,
      mappedRangeCount: verifiedSetup.mappedRangeCount,
      unmappedRangeCount: verifiedSetup.unmappedRangeCount,
      liveMapCount: verifiedSetup.liveMapCount,
      pendingDescriptorCount: verifiedSetup.pendingDescriptorCount,
      callbackCount: verifiedSetup.activeCallbackCount,
      leaseCount: verifiedSetup.activeLeaseCount,
      armReleaseCount: verifiedSetup.armReleaseCount,
      drainOrderViolationCount: verifiedSetup.drainOrderViolationCount,
      passed: true,
    }),
    deviceLossCleanup: Object.freeze({
      schema: deviceLoss.schema,
      deviceLossInduced: deviceLoss.deviceLossInduced,
      deviceLossObserved: deviceLoss.deviceLossObserved,
      ownerDestroyedAfterLoss: deviceLoss.ownerDestroyedAfterLoss,
      liveBufferCount: deviceLoss.liveGraphBufferCount,
      liveByteCount: deviceLoss.liveGraphByteCount,
      liveMapCount: deviceLoss.liveMapCount,
      pendingDescriptorCount: deviceLoss.pendingDescriptorCount,
      callbackCount: deviceLoss.activeCallbackCount,
      leaseCount: deviceLoss.activeLeaseCount,
      idempotentDestroyVerified: deviceLoss.idempotentDestroyVerified,
      postDestroyRejected: deviceLoss.postDestroyRejected,
      passed: devicePassed,
    }),
    passed,
  } satisfies Opt0081RepresentativeLifecycleEvidence);
  if (!passed) throw new Error("OPT-0081 owner lifecycle evidence changed");
  return evidence;
}

function requireCorrectnessOwnerSnapshot(
  owner: AceOpt0081RepresentativeDitSession,
): void {
  const value = owner.lifecycleSnapshot();
  if (
    value.schema !== "ace-opt-0081-representative-lifecycle-v1" ||
    value.state !== "ready" || !value.precomputeCompleted ||
    value.correctnessTargetCount !== 1 ||
    value.maximumCorrectnessTargetCount !== 1 ||
    value.correctnessRuntimeCount !== 1 ||
    value.maximumCorrectnessRuntimeCount !== 1 ||
    value.correctnessTargetCompilationCount !== 64 ||
    value.correctnessTargetReuseCount !== 64 ||
    value.runtimeOwnerCount !== 66 ||
    value.destroyedRuntimeOwnerCount !== 63 ||
    value.checkpointSnapshotCount !== 128 ||
    value.mappedRangeCount !== 128 || value.unmappedRangeCount !== 128 ||
    value.liveMapCount !== 0 || value.pendingDescriptorCount !== 0 ||
    value.maximumCorrectnessTargetBytes <= 0 ||
    value.maximumDetachedCheckpointBytes <= 0 ||
    value.maximumCorrectnessTargetBytes !==
      value.maximumDetachedCheckpointBytes + 512
  ) throw new Error("OPT-0081 bounded correctness owner evidence changed");
}

function requireSetupCleanupEvidence(
  value: AceOpt0081RepresentativeSetupCleanupEvidence,
): AceOpt0081RepresentativeSetupCleanupEvidence {
  if (
    value.schema !== "ace-opt-0081-representative-setup-cleanup-v1" ||
    value.createdGraphBufferCount <= 0 ||
    value.createdGraphBufferCount !== value.destroyedGraphBufferCount ||
    value.liveGraphBufferCount !== 0 || value.liveGraphByteCount !== 0 ||
    value.runtimeOwnerCount !== 2 ||
    value.destroyedRuntimeOwnerCount !== 2 ||
    !value.residentModelDestroyed || value.mappedRangeCount !== 0 ||
    value.unmappedRangeCount !== 0 || value.liveMapCount !== 0 ||
    value.pendingDescriptorCount !== 0 || value.activeCallbackCount !== 0 ||
    value.activeLeaseCount !== 0 || value.armReleaseCount <= 0 ||
    value.drainOrderViolationCount !== 0
  ) throw new Error("OPT-0081 setup failure cleanup evidence changed");
  return value;
}

function requireSameSetupCleanupEvidence(
  left: AceOpt0081RepresentativeSetupCleanupEvidence,
  right: AceOpt0081RepresentativeSetupCleanupEvidence,
): void {
  requireSetupCleanupEvidence(left);
  requireSetupCleanupEvidence(right);
  if (JSON.stringify(left) !== JSON.stringify(right)) {
    throw new Error("OPT-0081 setup evidence changed between devices");
  }
}

function requireSnapshot(
  value: AceOpt0081RepresentativeCheckpointSnapshot,
  arm: "A" | "B",
  layer: 0 | 1,
  tap: AceOpt0081RepresentativeTap,
  storage: "u16" | "u32",
  words: number,
): void {
  if (
    value.schema !== "ace-opt-0081-representative-checkpoint-v1" ||
    value.arm !== arm || value.layer !== layer || value.tap !== tap ||
    value.storage !== storage || value.elementCount !== words ||
    value.words.length !== words || value.byteLength !==
      words * (storage === "u16" ? 2 : 4) ||
    value.bytes.byteLength !== value.byteLength
  ) throw new Error(`OPT-0081 ${arm} layer ${layer} ${tap} snapshot changed`);
}

function allCheckpointGates(value: Opt0081RepresentativeCorrectness): boolean {
  return [
    ...value.controlBoundaryRepeat,
    ...value.boundaryRuns.flatMap(({ checkpoints }) => checkpoints),
    ...value.candidateBoundaryRepeat,
    ...value.comparisons.flatMap(({ denseOutputs, layerOutputs }) =>
      [...denseOutputs, ...layerOutputs]),
  ].every((checkpoint) =>
    checkpoint.exact && checkpoint.differingWordCount === 0 &&
    checkpoint.unwrittenWordCount === 0 && checkpoint.signedZeroExact &&
    checkpoint.finite && checkpoint.qNaNPrefillOverwritten &&
    checkpoint.firstWordCovered && checkpoint.lastWordCovered &&
    checkpoint.tailRows2240Through2249Covered &&
    checkpoint.prefixGuardIntact && checkpoint.suffixGuardIntact &&
    checkpoint.adjacentGuardsIntact
  );
}

function configuration(): AceWorkerConfiguration {
  return Object.freeze({
    manifestUrl: absoluteUrl(MAIN_MANIFEST_PATH),
    manifestSha256: MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage: Object.freeze({
      manifestUrl: absoluteUrl(DENSE_MANIFEST_PATH),
      manifestSha256: DENSE_MANIFEST_SHA256,
      runtimeProfile: DENSE_RUNTIME_PROFILE,
    }),
    ditAttentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    vaePackage: Object.freeze({
      manifestUrl: absoluteUrl(VAE_MANIFEST_PATH),
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      windowRuntimeProfile: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      maxWindowFrames: ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
    }),
  });
}

function validateDiagnostics(diagnostics: AceRuntimeDiagnostics): void {
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== DENSE_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestByteLength !== DENSE_MANIFEST_BYTES ||
    diagnostics.ditDenseRuntimeProfile !== DENSE_RUNTIME_PROFILE ||
    diagnostics.ditDenseKernelSetId !== DENSE_KERNEL_SET_ID ||
    diagnostics.ditAttentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    diagnostics.ditAttentionKernelSetId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !==
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE ||
    diagnostics.vaeKernelSetId !==
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID ||
    diagnostics.vaeWindowRuntimeProfile !==
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE ||
    diagnostics.vaeMaxWindowFrames !==
      ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.adapterInfo.subgroupMinSize !== 32 ||
    diagnostics.capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !diagnostics.capabilities.deviceFeatures.includes("subgroups") ||
    !diagnostics.capabilities.deviceFeatures.includes("shader-f16")
  ) throw new Error("OPT-0081 authenticated runtime changed");
}

function recordDiagnostic(
  diagnostic: AceDiagnostic,
  counts: DiagnosticCounts,
  intentionalDeviceLossStarted: boolean,
): void {
  if (diagnostic.severity !== "error") return;
  const expectedDeviceLoss = intentionalDeviceLossStarted &&
    diagnostic.code === "WEBGPU_DEVICE_LOST";
  if (expectedDeviceLoss) return;
  counts.fatal += 1;
  if (
    diagnostic.code === "WEBGPU_UNCAPTURED_ERROR" &&
    diagnostic.details?.["errorType"] === "validation"
  ) counts.validation += 1;
  if (diagnostic.code === "WEBGPU_DEVICE_LOST") counts.deviceLoss += 1;
}

function validateConditioning(
  diagnostic: AceDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details!;
  if (
    diagnostic.stage !== "semantic-planner" || details.plannerEnabled !== false ||
    details.instrumental !== true ||
    details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.textTokenCount !== 82 || details.lyricTokenCount !== 15 ||
    details.textTokenSha256 !== OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256 ||
    details.lyricTokenSha256 !== OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256
  ) throw new Error("OPT-0081 conditioning authority changed");
  return Object.freeze({
    textTokenCount: 82,
    textTokenSha256: OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
    lyricTokenCount: 15,
    lyricTokenSha256: OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
    conditionTokenCount: 98,
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0081 canonical request identity changed");
  const value = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(value);
  if (JSON.stringify(value) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0081 canonical request values changed");
  }
  return Object.freeze(value);
}

function isForbiddenProgress(progress: AceGenerationProgress): boolean {
  return new Set([
    "vae-load", "vae-decode", "wav-encode", "cleanup", "done",
  ]).has(progress.stage);
}

class Opt0081RepresentativeCorrectnessFailure extends Error {
  constructor(readonly correctness: Opt0081RepresentativeCorrectness) {
    super("OPT-0081 representative raw checkpoint gate failed");
    this.name = "Opt0081RepresentativeCorrectnessFailure";
  }
}

class Opt0081RepresentativeExecutionFailure extends Error {
  constructor(
    cause: unknown,
    readonly evidence: Readonly<Record<string, unknown>>,
  ) {
    super("OPT-0081 representative execution stopped after cleanup", { cause });
    this.name = "Opt0081RepresentativeExecutionFailure";
  }
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function fail(phase: string, error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  self.postMessage({
    type: "failed",
    phase,
    error: serializeOpt0018Failure(error),
    ...(error instanceof Opt0081RepresentativeExecutionFailure
      ? { evidence: error.evidence }
      : error instanceof Opt0081RepresentativeCorrectnessFailure
      ? { evidence: Object.freeze({
          ...(runIdentity === undefined
            ? {}
            : { identity: createOpt0081RepresentativeIdentity(runIdentity) }),
          correctness: error.correctness,
        }) }
      : {}),
  });
}

function absoluteUrl(path: string): string {
  return new URL(path, self.location.href).href;
}

void ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS;
void (null as unknown as AceOpt0081RepresentativeDenseTap);
