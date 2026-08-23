/// <reference lib="webworker" />

import {
  createAceOpt0010PlannerTokenAttribution,
  summarizeAceOpt0010PlannerTokenTrace,
  validateAceOpt0010PlannerTokenTrace,
  type AceOpt0010PlannerCopyCommand,
  type AceOpt0010PlannerPhysicalDispatch,
  type AceOpt0010PlannerProductionQuantumTag,
  type AceOpt0010PlannerReadbackProgressPayload,
  type AceOpt0010PlannerTokenAttribution,
  type AceOpt0010PlannerTokenTrace,
  type AceOpt0010PlannerModelProgressPayload,
} from "../../benchmark/opt-0010-planner-token-profiler.js";
import { DEFAULT_ACE_PLANNER_CONFIGURATION } from "../../src/api.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_PACKAGE_CONVERTER_REVISION,
  type AcePackageFileRecord,
  type AcePackageManifest,
} from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  createAcePlannerCodePrompts,
  createAcePlannerCotPrompt,
  type AcePlannerPrefillBatch,
} from "../../src/runtime/planner.js";
import { AcePlannerMetadataConstraintController } from
  "../../src/runtime/planner-metadata-fsm.js";
import {
  AcePlannerSamplingCursor,
  type AcePlannerAllowedTokens,
} from "../../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../../src/runtime/seed.js";
import {
  ACE_QWEN_IM_END_TOKEN_ID,
  ACE_QWEN_PAD_TOKEN_ID,
  loadPinnedAceTokenizer,
  type AceQwenBpeTokenizer,
} from "../../src/tokenizer/index.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  AcePlannerGpuExecutor,
  type AcePlannerGpuExecutorProgress,
} from "../../src/webgpu/planner-executor.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
} from "../../src/webgpu/qwen3.js";
import type {
  Opt0010RunIdentity,
  Opt0010ThermalGateMetadata,
} from "./opt-0010-package-native-planner-token-profiler.js";

export const OPT_0010_FP16_MANIFEST_SHA256 =
  "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b";
export const OPT_0010_FP16_MANIFEST_PATH = "/model/files-fp16/manifest.json";
export const OPT_0010_EXPECTED_PRODUCTION_COMMIT =
  "00dfd4732aa019bbbb238ae40265fe86cb38f27b";
export const OPT_0010_PLANNER_TENSOR_COUNT = 314;
export const OPT_0010_PLANNER_WEIGHT_FILE_COUNT = 33;
export const OPT_0010_PLANNER_RESIDENT_BYTES = 1_325_768_704;
export const OPT_0010_TOKENIZER_FILE_COUNT = 3;
export const OPT_0010_ACQUIRED_FILE_COUNT = 36;
export const OPT_0010_MODEL_QUANTUM_COUNT = 33;
export const OPT_0010_MODEL_DISPATCH_PRIMITIVE_COUNT = 624;
export const OPT_0010_PHYSICAL_DISPATCH_COUNT = 628;
export const OPT_0010_COMMAND_BUFFER_COUNT = 34;
export const OPT_0010_READBACK_COPY_COUNT = 6;
export const OPT_0010_ACCEPTED_RECEIPT_SHA256 =
  "554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678";
export const OPT_0010_ACCEPTED_SEMANTIC_CODE_SHA256 =
  "42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be";
export const OPT_0010_COT_DIAGNOSTIC_TRAJECTORY_SHA256 =
  "476515e1db6ebc30e1622eb30ac02a8ef4289d89ca12e34c64b5f911bc960da2";

const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const WORKER_HEARTBEAT_INTERVAL_MILLISECONDS = 10;
const ACCEPTED_SEED = canonicalizeSeed("000000000badc0de");
const ACCEPTED_PROMPT =
  "Bright neo-soul with elastic bass, clipped rhythm guitar, crisp pocket " +
  "drums, warm keys, and a confident mezzo-soprano vocal.";
const ACCEPTED_LYRICS =
  "[Verse]\nOpen the curtains, let the whole day in\n" +
  "Dust in the sunlight starts to spin\n\n[Chorus]\n" +
  "We found the rhythm under our feet\n" +
  "Turn up the room and follow the beat";
const ACCEPTED_RESOLVED_CAPTION =
  "A clean electric guitar plays a gentle, melodic chord progression with a " +
  "slightly funky, neo-soul feel. The tone is warm and direct, with a light " +
  "touch of reverb adding a touch of space to the arpeggiated chords. This " +
  "short, looping instrumental piece feels like a thoughtful intro or a " +
  "mellow interlude.";
const ACCEPTED_COT_TEXT = `<think>
bpm: 100
caption: A clean electric guitar plays a gentle, melodic chord progression with a
  slightly funky, neo-soul feel. The tone is warm and direct, with a light touch of
  reverb adding a touch of space to the arpeggiated chords. This short, looping instrumental
  piece feels like a thoughtful intro or a mellow interlude.
duration: 30
keyscale: B minor
language: unknown
timesignature: 2
</think>`;
export const OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS = Object.freeze([
  44_244, 2_430, 7_541, 38_339, 43_500, 14_023, 63_719, 16_071,
  63_855, 58_755, 37_828, 20_336, 52_689, 20_856, 53_201, 11_240,
  17_873, 15_217, 22_037, 12_976, 20_848, 47_248, 54_461, 28_656,
  28_812, 12_669, 10_110, 63_939, 48_579, 19_199, 63_718, 28_871,
  51_055, 45_955, 27_133, 22_832, 53_202, 20_920, 53_200, 9_720,
  22_992, 17_785, 15_536, 12_797, 12_715, 40_782, 28_800, 41_083,
  28_411, 11_130, 21_929, 29_754, 10_008, 33_298, 35_986, 25_049,
  7_613, 57_560, 45_964, 14_910, 34_822, 18_567, 53_535, 28_710,
  7_230, 25_184, 7_480, 25_456, 25_572, 9_702, 61_421, 62_781,
  12_723, 63_314, 13_779, 57_352, 2_705, 51_234, 61_459, 50_923,
  28_968, 15_224, 19_416, 26_752, 26_688, 26_752, 39_560, 42_256,
  57_104, 57_176, 56_728, 56_800, 5_544, 8_112, 23_480, 24_568,
  28_820, 12_669, 10_101, 63_426, 58_828, 16_583, 18_435, 16_709,
  53_300, 20_421, 62_950, 36_528, 53_202, 10_236, 53_196, 37_950,
  57_820, 33_075, 31_276, 10_168, 10_171, 40_847, 26_434, 28_159,
  41_684, 12_669, 10_101, 63_426, 58_860, 16_583, 38_467, 903,
  53_357, 31_175, 62_934, 35_992, 22_938, 23_158, 1_621, 23_534,
  2_501, 24_576, 18_421, 12_279, 25_460, 51_159, 25_986, 50_117,
  33_807, 35_847, 35_847, 35_847, 35_847, 35_847,
]);

export const OPT_0010_CASE_SPECS = Object.freeze([
  Object.freeze({
    id: "cot-m1-short",
    mode: "cot-m1" as const,
    position: "short" as const,
    cachedTokensBeforeAppend: 120,
    cacheCapacity: 512,
    drawIndex: 16,
  }),
  Object.freeze({
    id: "cot-m1-mid",
    mode: "cot-m1" as const,
    position: "mid" as const,
    cachedTokensBeforeAppend: 160,
    cacheCapacity: 1_024,
    drawIndex: 56,
  }),
  Object.freeze({
    id: "cot-m1-long",
    mode: "cot-m1" as const,
    position: "long" as const,
    cachedTokensBeforeAppend: 212,
    cacheCapacity: 2_048,
    drawIndex: 108,
  }),
  Object.freeze({
    id: "semantic-m2-short",
    mode: "semantic-m2" as const,
    position: "short" as const,
    cachedTokensBeforeAppend: 268,
    cacheCapacity: 768,
    drawIndex: 125,
  }),
  Object.freeze({
    id: "semantic-m2-mid",
    mode: "semantic-m2" as const,
    position: "mid" as const,
    cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280,
    drawIndex: 185,
  }),
  Object.freeze({
    id: "semantic-m2-long",
    mode: "semantic-m2" as const,
    position: "long" as const,
    cachedTokensBeforeAppend: 401,
    cacheCapacity: 2_048,
    drawIndex: 258,
  }),
]);

type CaseSpec = (typeof OPT_0010_CASE_SPECS)[number];

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0010RunIdentity;
}

interface RunTimedMessage {
  readonly type: "run-timed";
  readonly thermal: Opt0010ThermalGateMetadata;
}

type IncomingMessage = InitializeMessage | RunTimedMessage;

interface WorkerHeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly timerTickCount: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly summary: Readonly<Record<string, unknown>>;
}

interface PlannerCaseFixture {
  readonly spec: CaseSpec;
  readonly attribution: AceOpt0010PlannerTokenAttribution;
  readonly prefill: AcePlannerPrefillBatch;
  readonly decodeTokenId: number;
  readonly seenTokenIds: readonly number[];
  readonly acceptedPromptTokenIds: readonly number[];
  readonly acceptedEmittedTokenIdsIncludingDecode: readonly number[];
  readonly promptLengths: readonly number[];
  readonly tokenizer: AceQwenBpeTokenizer;
}

interface LogitSummary {
  readonly rows: 1 | 2;
  readonly rowElements: number;
  readonly totalElements: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly sha256: string;
}

interface SampleSummary {
  readonly tokenId: number;
  readonly word: number;
  readonly drawIndex: string;
  readonly drawEnd: string;
  readonly positiveCandidateCount: number;
}

interface SamplingConstraintState {
  readonly cot: AcePlannerMetadataConstraintController | null;
}

interface ReferenceCase {
  readonly fixture: PlannerCaseFixture;
  readonly logits: readonly Float32Array[];
  readonly logitsSummary: LogitSummary;
  readonly sample: SampleSummary;
  readonly prefillWallMilliseconds: number;
  readonly decodeWallMilliseconds: number;
}

interface PreparedSession {
  readonly runIdentity: Opt0010RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly observer: PlannerDeviceObserver;
  readonly progressRouter: ProgressRouter;
  readonly abortController: AbortController;
  readonly executor: AcePlannerGpuExecutor;
  readonly preparedPackage: PreparedPackage;
  readonly references: readonly ReferenceCase[];
  readonly packageAcquisitionWallMilliseconds: number;
  readonly phaseUploadWallMilliseconds: number;
  readonly executorCompileWallMilliseconds: number;
  readonly warmupCompletedAtEpochMilliseconds: number;
}

interface RawPhysicalDispatch {
  readonly indexInCommand: number;
  readonly pipelineLabel: string;
  readonly bindGroupLabel: string;
  readonly workgroups: readonly [number, number, number];
}

interface RawCopyCommand {
  readonly index: number;
  readonly sourceBufferLabel: string;
  readonly sourceOffset: number;
  readonly destinationBufferLabel: string;
  readonly destinationOffset: number;
  readonly copiedBytes: number;
}

interface MutableCommandRecord {
  readonly kind: "model" | "readback";
  readonly commandLabel: string;
  readonly quantumId: string | null;
  readonly passLabels: string[];
  readonly rawPhysicalDispatches: RawPhysicalDispatch[];
  readonly rawCopyCommands: RawCopyCommand[];
  readonly encodeStartedAt: number;
  encodeEndedAt?: number;
  submitStartedAt?: number;
  submitReturnedAt?: number;
  drainStartedAt?: number;
  drainEndedAt?: number;
  idleStartedAt?: number;
  progressReportedAt?: number;
  idleEndedAt?: number;
  nextEncodeStartedAt?: number;
  mapStartedAt?: number;
  mapEndedAt?: number;
  reconstructStartedAt?: number;
  reconstructEndedAt?: number;
  invocationResolvedAt?: number;
  progress?: AceOpt0010PlannerModelProgressPayload |
    AceOpt0010PlannerReadbackProgressPayload;
}

interface BufferRecord {
  readonly label: string;
  readonly size: number;
  destroyCallCount: number;
  destroyed: boolean;
}

interface CancellationSummary {
  readonly rejectionName: string;
  readonly rejectionMessage: string;
  readonly encodedCommandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly completedIdleCount: number;
  readonly laterEncodingPrevented: boolean;
  readonly laterSubmissionPrevented: boolean;
  readonly firstQuantumFullyDrained: boolean;
  readonly realIdleCompletedBeforeRejection: boolean;
}

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let session: PreparedSession | undefined;
let workerHeartbeat: ReturnType<typeof startWorkerHeartbeat> | undefined;

if (typeof self !== "undefined") self.addEventListener(
  "message",
  (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "initialize") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    workerHeartbeat = startWorkerHeartbeat();
    void initializeSession(event.data.identity).then(
      (prepared) => {
        if (lifecycle !== "preparing") return;
        session = prepared;
        lifecycle = "ready";
        self.postMessage({
          type: "ready-for-thermal-gate",
          warmupCompletedAtEpochMilliseconds:
            prepared.warmupCompletedAtEpochMilliseconds,
          preparation: publicPreparationSummary(prepared),
        });
      },
      (error: unknown) => void failAndCleanup(error),
    );
    return;
  }
  if (event.data.type === "run-timed" && lifecycle === "ready") {
    lifecycle = "running";
    const active = session!;
    session = undefined;
    void runTimedAndCleanup(active, event.data.thermal).then(
      (result) => {
        lifecycle = "settled";
        self.postMessage({ type: "passed", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
  },
);

async function initializeSession(identity: unknown): Promise<PreparedSession> {
  const runIdentity = validateRunIdentity(identity);
  postProgress("authenticating the converter-revision-4 raw-FP16 manifest");
  const acquisitionStarted = performance.now();
  const preparedPackage = await preparePackage();
  const packageAcquisitionWallMilliseconds =
    performance.now() - acquisitionStarted;
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  postProgress("requesting the shipped raw-FP16 cooperative device");
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (runtimeEvent) => runtimeEvents.push(runtimeEvent),
  });
  const observer = new PlannerDeviceObserver(context.device);
  const progressRouter = new ProgressRouter(observer);
  const abortController = new AbortController();
  let phase: AceGpuTensorPhase | undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  try {
    postProgress("loading the authenticated 1.235 GiB planner phase");
    const uploadStarted = performance.now();
    let lastStatusAt = 0;
    phase = await AceGpuTensorPhase.load(
      observer.device,
      preparedPackage.manifest,
      preparedPackage.acquiredFiles,
      ["planner"],
      {
        signal: abortController.signal,
        onProgress: (progress) => {
          const now = performance.now();
          if (
            now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
            progress.loadedPhaseBytes === progress.totalPhaseBytes
          ) {
            lastStatusAt = now;
            postProgress(
              `uploading planner shard ${progress.phaseFileIndex + 1}/` +
                `${progress.phaseFileCount} (${formatBytes(progress.loadedPhaseBytes)}/` +
                `${formatBytes(progress.totalPhaseBytes)})`,
            );
          }
        },
      },
    );
    if (
      phase.phases.length !== 1 ||
      phase.phases[0] !== "planner" ||
      phase.residentBytes !== OPT_0010_PLANNER_RESIDENT_BYTES
    ) throw new Error("OPT-0010 loaded planner phase identity changed");
    const phaseUploadWallMilliseconds = performance.now() - uploadStarted;

    const tokenizerLoaded = await loadPinnedAceTokenizer("planner", {
      tokenizerJson: requirePackageFile(
        preparedPackage.acquiredFiles,
        "assets/planner/tokenizer.json",
      ),
      tokenizerConfigJson: requirePackageFile(
        preparedPackage.acquiredFiles,
        "assets/planner/tokenizer_config.json",
      ),
      chatTemplate: requirePackageFile(
        preparedPackage.acquiredFiles,
        "assets/planner/chat_template.jinja",
      ),
    });
    const fixtures = createCaseFixtures(tokenizerLoaded.tokenizer);
    const compileStarted = performance.now();
    executor = AcePlannerGpuExecutor.create({
      device: observer.device,
      modelProfile: "raw-fp16",
      ownedPlannerWeights: phase,
      signal: abortController.signal,
      onProgress: (progress) => progressRouter.accept(progress),
    });
    phase = undefined;

    const references: ReferenceCase[] = [];
    for (let index = 0; index < fixtures.length; index += 1) {
      const fixture = fixtures[index]!;
      postProgress(
        `reference ${index + 1}/${fixtures.length}: ${fixture.spec.id}`,
      );
      const prefillStarted = performance.now();
      await executor.prefill(fixture.prefill);
      const prefillWallMilliseconds = performance.now() - prefillStarted;
      const samplingState = createSamplingConstraintState(fixture);
      const decodeStarted = performance.now();
      const returnedLogits = await executor.decode(createDecodeBatch(fixture));
      const decodeWallMilliseconds = performance.now() - decodeStarted;
      const logits = requireFloat32Logits(returnedLogits);
      const logitsSummary = await summarizeLogits(logits, fixture.attribution.rows);
      validateCompleteLogits(logitsSummary, `${fixture.spec.id} reference`);
      const allowedTokens = resolveCaseAllowedTokens(
        fixture,
        logits,
        samplingState,
      );
      const sample = sampleCase(fixture, logits, allowedTokens);
      validateSampleCursor(fixture, sample, "reference");
      references.push(Object.freeze({
        fixture,
        logits,
        logitsSummary,
        sample,
        prefillWallMilliseconds,
        decodeWallMilliseconds,
      }));
    }
    const executorCompileWallMilliseconds = performance.now() - compileStarted;
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0010 preparation observed a WebGPU runtime event");
    }
    return Object.freeze({
      runIdentity,
      context,
      runtimeEvents,
      observer,
      progressRouter,
      abortController,
      executor,
      preparedPackage,
      references: Object.freeze(references),
      packageAcquisitionWallMilliseconds,
      phaseUploadWallMilliseconds,
      executorCompileWallMilliseconds,
      warmupCompletedAtEpochMilliseconds: Date.now(),
    });
  } catch (error) {
    if (executor !== undefined) await executor.destroy(error);
    else phase?.destroy();
    context.destroy();
    throw error;
  }
}

async function runTimedAndCleanup(
  prepared: PreparedSession,
  thermal: Opt0010ThermalGateMetadata,
): Promise<Readonly<Record<string, unknown>>> {
  validateThermalGate(thermal, prepared.warmupCompletedAtEpochMilliseconds);
  const timedStartedAtEpochMilliseconds = Date.now();
  const cases: Array<Readonly<Record<string, unknown>>> = [];
  let destroyed = false;
  try {
    for (let index = 0; index < prepared.references.length; index += 1) {
      const reference = prepared.references[index]!;
      const fixture = reference.fixture;
      postProgress(
        `timed ${index + 1}/${prepared.references.length}: ${fixture.spec.id}`,
      );
      await prepared.executor.prefill(fixture.prefill);
      const samplingState = createSamplingConstraintState(fixture);
      prepared.observer.beginTimedTrace(fixture.attribution);
      prepared.progressRouter.begin("timed");
      const returnedLogits = await prepared.executor.decode(
        createDecodeBatch(fixture),
      );
      const invocationResolvedAt = performance.now();
      prepared.observer.noteInvocationResolved(invocationResolvedAt);
      const logits = requireFloat32Logits(returnedLogits);
      const constraintStartedAt = performance.now();
      const allowedTokens = resolveCaseAllowedTokens(
        fixture,
        logits,
        samplingState,
      );
      const constraintEndedAt = performance.now();
      const samplingStartedAt = performance.now();
      const sample = sampleCase(fixture, logits, allowedTokens);
      validateSampleCursor(fixture, sample, "timed");
      const samplingEndedAt = performance.now();
      const progress = prepared.progressRouter.end("timed");
      const observed = prepared.observer.endTimedTrace(
        constraintStartedAt,
        constraintEndedAt,
        samplingStartedAt,
        samplingEndedAt,
      );
      validateProgressSequence(fixture.attribution, progress);
      const validated = validateAceOpt0010PlannerTokenTrace(
        fixture.attribution,
        observed.trace,
      );
      const summary = summarizeAceOpt0010PlannerTokenTrace(
        fixture.attribution,
        observed.trace,
      );
      const logitsSummary = await summarizeLogits(
        logits,
        fixture.attribution.rows,
      );
      validateCompleteLogits(logitsSummary, `${fixture.spec.id} timed`);
      const comparison = compareLogits(reference.logits, logits);
      if (!comparison.bitExact || !sameSample(reference.sample, sample)) {
        throw new Error(`OPT-0010 ${fixture.spec.id} correctness changed`);
      }
      if (
        validated.commandBufferCount !== OPT_0010_COMMAND_BUFFER_COUNT ||
        validated.queueDrainCount !== OPT_0010_COMMAND_BUFFER_COUNT ||
        validated.completedIdleCount !== OPT_0010_COMMAND_BUFFER_COUNT ||
        observed.rawPhysicalDispatchCount !==
          OPT_0010_PHYSICAL_DISPATCH_COUNT ||
        observed.rawCopyCommands.length !== OPT_0010_READBACK_COPY_COUNT
      ) throw new Error(`OPT-0010 ${fixture.spec.id} accounting changed`);
      cases.push(Object.freeze({
        spec: fixture.spec,
        fixture: Object.freeze({
          rows: fixture.attribution.rows,
          promptLengths: fixture.promptLengths,
          decodeTokenId: fixture.decodeTokenId,
          acceptedEmittedBeforeDecode:
            fixture.acceptedEmittedTokenIdsIncludingDecode.length - 1,
          acceptedSamplingDrawIndex: fixture.spec.drawIndex,
          scheduledCapacityClassification:
            "explicit-short-mid-long-diagnostic",
          trajectoryAuthority: fixture.spec.mode === "cot-m1"
            ? "accepted-receipt-text-derived-fsm-admitted-longest-token-diagnostic"
            : "accepted-receipt-semantic-code-ids",
          acceptedReceiptSha256: OPT_0010_ACCEPTED_RECEIPT_SHA256,
          acceptedSemanticCodeSha256: OPT_0010_ACCEPTED_SEMANTIC_CODE_SHA256,
        }),
        attribution: fixture.attribution,
        reference: Object.freeze({
          logits: reference.logitsSummary,
          sample: reference.sample,
          prefillWallMilliseconds: reference.prefillWallMilliseconds,
          decodeWallMilliseconds: reference.decodeWallMilliseconds,
        }),
        timed: Object.freeze({
          logits: logitsSummary,
          sample,
          trace: observed.trace,
          summary,
          rawCommands: observed.rawCommands,
          rawPhysicalDispatchCount: observed.rawPhysicalDispatchCount,
          rawCopyCommands: observed.rawCopyCommands,
        }),
        correctness: comparison,
      }));
    }
    const timedCompletedAtEpochMilliseconds = Date.now();

    postProgress("running post-drain cancellation proof");
    const cancellationFixture = prepared.references[0]!.fixture;
    await prepared.executor.prefill(cancellationFixture.prefill);
    const cancellation = await runCancellationProof(
      prepared,
      cancellationFixture,
    );
    await prepared.executor.destroy();
    await prepared.executor.destroy();
    destroyed = true;
    const resources = prepared.observer.resourceSummary();
    if (!resources.destructionTrackingSupported) {
      throw new Error("OPT-0010 GPU buffer destruction tracking is unavailable");
    }
    if (resources.liveTrackedBufferCount !== 0) {
      throw new Error("OPT-0010 cleanup retained tracked GPU buffers");
    }
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0010 observed a WebGPU runtime error or loss");
    }
    const heartbeat = workerHeartbeat!.stop();
    workerHeartbeat = undefined;
    validateWorkerHeartbeat(heartbeat);
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    prepared.context.destroy();
    return Object.freeze({
      schemaVersion: 1,
      experimentId: "OPT-0010",
      kind: "ace-opt-0010-package-native-planner-token-profiler-raw",
      identity: Object.freeze({
        ...prepared.runIdentity,
        manifestSha256: OPT_0010_FP16_MANIFEST_SHA256,
        modelProfile: "raw-fp16",
        schedulingProfile: "cooperative",
        capabilities: prepared.context.capabilities,
      }),
      protocol: Object.freeze({
        benchmarkOnly: true,
        arithmeticChanged: false,
        packagePath: OPT_0010_FP16_MANIFEST_PATH,
        caseOrder: OPT_0010_CASE_SPECS.map((entry) => entry.id),
        referenceCreation: "one package-native prefill/decode before thermal gate",
        timedExecution: "fresh equivalent prefill then one observed decode",
        fullLogitBitComparison: true,
        samplingOracle: "ace-browser-softmax-v1",
        oneOutstandingCommandBuffer: true,
        performanceTimeOriginEpochMilliseconds: performance.timeOrigin,
        traceTimestamps: "worker performance.now milliseconds",
      }),
      package: prepared.preparedPackage.summary,
      preparation: Object.freeze({
        packageAcquisitionWallMilliseconds:
          prepared.packageAcquisitionWallMilliseconds,
        phaseUploadWallMilliseconds: prepared.phaseUploadWallMilliseconds,
        executorCompileAndReferenceWallMilliseconds:
          prepared.executorCompileWallMilliseconds,
        warmupCompletedAtEpochMilliseconds:
          prepared.warmupCompletedAtEpochMilliseconds,
      }),
      thermal: Object.freeze({
        preGate: thermal,
        preGateOnly: true,
        status: "pending-external-artifact-join",
        browserReceiptClaimsPlanValidThermalCoverage: false,
        continuousLoggerRequiredThroughEpochMilliseconds:
          cleanupCompletedAtEpochMilliseconds,
      }),
      timedStartedAtEpochMilliseconds,
      timedCompletedAtEpochMilliseconds,
      cases: Object.freeze(cases),
      cancellation,
      cleanup: Object.freeze({
        idempotentExecutorDestroy: true,
        resources,
        runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
        deviceDestroyedAfterEventCheck: true,
        cleanupCompletedAtEpochMilliseconds,
      }),
      workerHeartbeat: heartbeat,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      completedAt: new Date(cleanupCompletedAtEpochMilliseconds).toISOString(),
    });
  } finally {
    if (!destroyed) {
      try {
        await prepared.executor.destroy();
      } finally {
        prepared.context.destroy();
      }
    }
  }
}

class ProgressRouter {
  private mode: "none" | "timed" | "cancellation" = "none";
  private events: AcePlannerGpuExecutorProgress[] = [];
  private hook: ((event: AcePlannerGpuExecutorProgress) => void) | undefined;

  constructor(private readonly observer: PlannerDeviceObserver) {}

  begin(
    mode: "timed" | "cancellation",
    hook?: (event: AcePlannerGpuExecutorProgress) => void,
  ): void {
    if (this.mode !== "none") {
      throw new DOMException("OPT-0010 progress collection overlaps", "InvalidStateError");
    }
    this.mode = mode;
    this.events = [];
    this.hook = hook;
  }

  accept(event: AcePlannerGpuExecutorProgress): void {
    if (this.mode === "none") return;
    const snapshot = Object.freeze({ ...event });
    this.events.push(snapshot);
    this.observer.noteProgress(snapshot);
    this.hook?.(snapshot);
  }

  end(expected: "timed" | "cancellation"): readonly AcePlannerGpuExecutorProgress[] {
    if (this.mode !== expected) {
      throw new DOMException(
        `OPT-0010 expected ${expected} progress, got ${this.mode}`,
        "InvalidStateError",
      );
    }
    const snapshot = Object.freeze([...this.events]);
    this.mode = "none";
    this.events = [];
    this.hook = undefined;
    return snapshot;
  }

  abandon(): void {
    this.mode = "none";
    this.events = [];
    this.hook = undefined;
  }
}

class PlannerDeviceObserver {
  readonly device: GPUDevice;

  private readonly queue: GPUQueue;
  private readonly records: MutableCommandRecord[] = [];
  private readonly commandRecords = new WeakMap<GPUCommandBuffer, MutableCommandRecord>();
  private readonly buffers = new Map<GPUBuffer, BufferRecord>();
  private activeMode: "off" | "timed" | "cancellation" = "off";
  private attribution: AceOpt0010PlannerTokenAttribution | undefined;
  private pendingSubmission: MutableCommandRecord | undefined;
  private destructionTrackingSupported = true;
  private mapTrackingSupported = true;
  private submissionCount = 0;
  private drainCount = 0;

  constructor(private readonly target: GPUDevice) {
    this.queue = this.createQueueProxy(target.queue);
    this.device = new Proxy(target, {
      get: (device, property) => {
        if (property === "queue") return this.queue;
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer =>
            this.createTrackedBuffer(descriptor);
        }
        if (property === "createCommandEncoder") {
          return (descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder =>
            this.createObservedCommandEncoder(descriptor);
        }
        const value = Reflect.get(device, property, device) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(device)
          : value;
      },
    }) as GPUDevice;
  }

  beginTimedTrace(attribution: AceOpt0010PlannerTokenAttribution): void {
    this.beginTrace("timed", attribution);
  }

  beginCancellationTrace(attribution: AceOpt0010PlannerTokenAttribution): void {
    this.beginTrace("cancellation", attribution);
  }

  noteProgress(event: AcePlannerGpuExecutorProgress): void {
    if (this.activeMode === "off") {
      throw new Error("OPT-0010 observed progress without an active trace");
    }
    const record = this.records[event.completedCommandBuffers - 1];
    if (record === undefined) {
      throw new Error("OPT-0010 progress references an absent command buffer");
    }
    const expectedKind = event.stage === "model" ? "model" : "readback";
    if (record.kind !== expectedKind || record.drainEndedAt === undefined) {
      throw new Error("OPT-0010 progress did not follow the matching drain");
    }
    record.idleStartedAt = record.drainEndedAt;
    record.progressReportedAt = performance.now();
    record.progress = progressPayload(event);
  }

  noteInvocationResolved(invocationResolvedAt: number): void {
    if (this.activeMode !== "timed") {
      throw new DOMException("OPT-0010 timed trace is not active", "InvalidStateError");
    }
    const readback = this.records.at(-1);
    if (readback?.kind !== "readback") {
      throw new Error("OPT-0010 invocation omitted its readback command");
    }
    readback.idleEndedAt = invocationResolvedAt;
    readback.invocationResolvedAt = invocationResolvedAt;
    readback.reconstructEndedAt = invocationResolvedAt;
  }

  endTimedTrace(
    constraintStartedAt: number,
    constraintEndedAt: number,
    samplingStartedAt: number,
    samplingEndedAt: number,
  ): Readonly<{
    trace: AceOpt0010PlannerTokenTrace;
    rawCommands: readonly Readonly<Record<string, unknown>>[];
    rawPhysicalDispatchCount: number;
    rawCopyCommands: readonly RawCopyCommand[];
  }> {
    if (this.activeMode !== "timed" || this.attribution === undefined) {
      throw new DOMException("OPT-0010 timed trace is not active", "InvalidStateError");
    }
    this.requireDrainedCompleteTrace();
    const attribution = this.attribution;
    const modelRecords = this.records.slice(0, -1);
    const readbackRecord = this.records.at(-1)!;
    if (
      modelRecords.length !== attribution.quanta.length ||
      readbackRecord.kind !== "readback"
    ) throw new Error("OPT-0010 actual command topology changed");
    const quanta = modelRecords.map((record, index) => {
      const expected = attribution.quanta[index]!;
      const progress = requireModelProgress(record.progress, index);
      const physicalDispatches = normalizePhysicalDispatches(record, expected);
      return Object.freeze({
        index,
        productionQuantum: productionTag(progress.quantum),
        progress,
        physicalDispatches,
        encodeStartedAt: record.encodeStartedAt,
        encodeEndedAt: requiredTimestamp(record.encodeEndedAt, "encode end"),
        submitStartedAt: requiredTimestamp(record.submitStartedAt, "submit start"),
        submitReturnedAt: requiredTimestamp(record.submitReturnedAt, "submit return"),
        drainStartedAt: requiredTimestamp(record.drainStartedAt, "drain start"),
        drainEndedAt: requiredTimestamp(record.drainEndedAt, "drain end"),
        idleStartedAt: requiredTimestamp(record.idleStartedAt, "idle start"),
        progressReportedAt: requiredTimestamp(
          record.progressReportedAt,
          "progress report",
        ),
        idleEndedAt: requiredTimestamp(record.idleEndedAt, "idle end"),
        nextEncodeStartedAt: requiredTimestamp(
          record.nextEncodeStartedAt,
          "next encode start",
        ),
        commandBufferCount: 1,
        submissionCount: 1,
        queueDrainCount: 1,
        completedIdleCount: 1,
        requestedIdleMilliseconds: 1,
      });
    });
    const copyCommands = normalizeCopyCommands(readbackRecord);
    const readbackProgress = requireReadbackProgress(readbackRecord.progress);
    const trace = Object.freeze({
      quanta: Object.freeze(quanta),
      readback: Object.freeze({
        progress: readbackProgress,
        encodeStartedAt: readbackRecord.encodeStartedAt,
        encodeEndedAt: requiredTimestamp(
          readbackRecord.encodeEndedAt,
          "readback encode end",
        ),
        submitStartedAt: requiredTimestamp(
          readbackRecord.submitStartedAt,
          "readback submit start",
        ),
        submitReturnedAt: requiredTimestamp(
          readbackRecord.submitReturnedAt,
          "readback submit return",
        ),
        drainStartedAt: requiredTimestamp(
          readbackRecord.drainStartedAt,
          "readback drain start",
        ),
        drainEndedAt: requiredTimestamp(
          readbackRecord.drainEndedAt,
          "readback drain end",
        ),
        idleStartedAt: requiredTimestamp(
          readbackRecord.idleStartedAt,
          "readback idle start",
        ),
        progressReportedAt: requiredTimestamp(
          readbackRecord.progressReportedAt,
          "readback progress report",
        ),
        mapStartedAt: requiredTimestamp(
          readbackRecord.mapStartedAt,
          "readback map start",
        ),
        mapEndedAt: requiredTimestamp(
          readbackRecord.mapEndedAt,
          "readback map end",
        ),
        reconstructStartedAt: requiredTimestamp(
          readbackRecord.reconstructStartedAt,
          "readback reconstruct start",
        ),
        reconstructEndedAt: requiredTimestamp(
          readbackRecord.reconstructEndedAt,
          "readback reconstruct end",
        ),
        idleEndedAt: requiredTimestamp(
          readbackRecord.idleEndedAt,
          "readback idle end",
        ),
        invocationResolvedAt: requiredTimestamp(
          readbackRecord.invocationResolvedAt,
          "invocation resolution",
        ),
        commandBufferCount: 1,
        submissionCount: 1,
        queueDrainCount: 1,
        completedIdleCount: 1,
        requestedIdleMilliseconds: 1,
        copyCommands,
      }),
      constraintStartedAt,
      constraintEndedAt,
      samplingStartedAt,
      samplingEndedAt,
    }) satisfies AceOpt0010PlannerTokenTrace;
    const rawPhysicalDispatchCount = this.records.reduce(
      (total, record) => total + record.rawPhysicalDispatches.length,
      0,
    );
    const rawCopyCommands = Object.freeze([...readbackRecord.rawCopyCommands]);
    const rawCommands = this.publicRawCommands();
    this.finishTrace();
    return Object.freeze({
      trace,
      rawCommands,
      rawPhysicalDispatchCount,
      rawCopyCommands,
    });
  }

  endCancellationTrace(): Readonly<{
    records: readonly MutableCommandRecord[];
    submissionCount: number;
    drainCount: number;
  }> {
    if (this.activeMode !== "cancellation") {
      throw new DOMException("OPT-0010 cancellation trace is not active", "InvalidStateError");
    }
    if (this.pendingSubmission !== undefined) {
      throw new Error("OPT-0010 cancellation returned before its queue drain");
    }
    const result = Object.freeze({
      records: Object.freeze([...this.records]),
      submissionCount: this.submissionCount,
      drainCount: this.drainCount,
    });
    this.finishTrace();
    return result;
  }

  resourceSummary(): Readonly<Record<string, unknown>> & Readonly<{
    destructionTrackingSupported: boolean;
    mapTrackingSupported: boolean;
    liveTrackedBufferCount: number;
  }> {
    const records = [...this.buffers.values()];
    return Object.freeze({
      createdBufferCount: records.length,
      uniqueDestroyedBufferCount: records.filter((record) => record.destroyed).length,
      totalDestroyCallCount: records.reduce(
        (total, record) => total + record.destroyCallCount,
        0,
      ),
      liveTrackedBufferCount: records.filter((record) => !record.destroyed).length,
      destructionTrackingSupported: this.destructionTrackingSupported,
      mapTrackingSupported: this.mapTrackingSupported,
      records: Object.freeze(records.map((record) => Object.freeze({ ...record }))),
    });
  }

  private beginTrace(
    mode: "timed" | "cancellation",
    attribution: AceOpt0010PlannerTokenAttribution,
  ): void {
    if (this.activeMode !== "off" || this.pendingSubmission !== undefined) {
      throw new DOMException("OPT-0010 command tracing overlaps", "InvalidStateError");
    }
    this.records.length = 0;
    this.submissionCount = 0;
    this.drainCount = 0;
    this.attribution = attribution;
    this.activeMode = mode;
  }

  private finishTrace(): void {
    this.activeMode = "off";
    this.attribution = undefined;
  }

  private requireDrainedCompleteTrace(): void {
    if (
      this.pendingSubmission !== undefined ||
      this.records.length !== OPT_0010_COMMAND_BUFFER_COUNT ||
      this.submissionCount !== OPT_0010_COMMAND_BUFFER_COUNT ||
      this.drainCount !== OPT_0010_COMMAND_BUFFER_COUNT
    ) throw new Error("OPT-0010 trace lost a command, submission, or drain");
  }

  private publicRawCommands(): readonly Readonly<Record<string, unknown>>[] {
    return Object.freeze(this.records.map((record) => Object.freeze({
      kind: record.kind,
      commandLabel: record.commandLabel,
      quantumId: record.quantumId,
      passLabels: Object.freeze([...record.passLabels]),
      physicalDispatches: Object.freeze([...record.rawPhysicalDispatches]),
      copyCommands: Object.freeze([...record.rawCopyCommands]),
      encodeStartedAt: record.encodeStartedAt,
      encodeEndedAt: record.encodeEndedAt,
      submitStartedAt: record.submitStartedAt,
      submitReturnedAt: record.submitReturnedAt,
      drainStartedAt: record.drainStartedAt,
      drainEndedAt: record.drainEndedAt,
      idleStartedAt: record.idleStartedAt,
      progressReportedAt: record.progressReportedAt,
      idleEndedAt: record.idleEndedAt,
      nextEncodeStartedAt: record.nextEncodeStartedAt,
      mapStartedAt: record.mapStartedAt,
      mapEndedAt: record.mapEndedAt,
      reconstructStartedAt: record.reconstructStartedAt,
      reconstructEndedAt: record.reconstructEndedAt,
      invocationResolvedAt: record.invocationResolvedAt,
    })));
  }

  private createTrackedBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.target.createBuffer(descriptor);
    const record: BufferRecord = {
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      destroyCallCount: 0,
      destroyed: false,
    };
    this.buffers.set(buffer, record);
    const destroy = buffer.destroy.bind(buffer);
    try {
      Object.defineProperty(buffer, "destroy", {
        configurable: true,
        value: (): void => {
          record.destroyCallCount += 1;
          record.destroyed = true;
          destroy();
        },
      });
    } catch {
      this.destructionTrackingSupported = false;
    }
    if (record.label === "ace-planner-logit-readback") {
      this.installReadbackMapObserver(buffer);
    }
    return buffer;
  }

  private installReadbackMapObserver(buffer: GPUBuffer): void {
    const mapAsync = buffer.mapAsync.bind(buffer);
    const getMappedRange = buffer.getMappedRange.bind(buffer);
    try {
      Object.defineProperty(buffer, "mapAsync", {
        configurable: true,
        value: async (...args: Parameters<GPUBuffer["mapAsync"]>): Promise<void> => {
          const record = this.activeReadbackRecord();
          if (record !== undefined) record.mapStartedAt = performance.now();
          await mapAsync(...args);
          if (record !== undefined) record.mapEndedAt = performance.now();
        },
      });
      Object.defineProperty(buffer, "getMappedRange", {
        configurable: true,
        value: (...args: Parameters<GPUBuffer["getMappedRange"]>): ArrayBuffer => {
          const record = this.activeReadbackRecord();
          if (record !== undefined) record.reconstructStartedAt = performance.now();
          return getMappedRange(...args);
        },
      });
    } catch {
      this.mapTrackingSupported = false;
    }
  }

  private activeReadbackRecord(): MutableCommandRecord | undefined {
    if (this.activeMode === "off") return undefined;
    const record = this.records.at(-1);
    return record?.kind === "readback" ? record : undefined;
  }

  private createObservedCommandEncoder(
    descriptor?: GPUCommandEncoderDescriptor,
  ): GPUCommandEncoder {
    const encoder = this.target.createCommandEncoder(descriptor);
    if (this.activeMode === "off") return encoder;
    const commandLabel = descriptor?.label ?? "";
    const readback = commandLabel === "ace-planner-logit-readback";
    const modelMatch = /^(.*)-command$/.exec(commandLabel);
    if (!readback && modelMatch === null) {
      throw new Error(`OPT-0010 traced unexpected command ${commandLabel}`);
    }
    const now = performance.now();
    const prior = this.records.at(-1);
    if (prior !== undefined) {
      prior.nextEncodeStartedAt = now;
      prior.idleEndedAt = now;
    }
    const record: MutableCommandRecord = {
      kind: readback ? "readback" : "model",
      commandLabel,
      quantumId: readback ? null : modelMatch![1]!,
      passLabels: [],
      rawPhysicalDispatches: [],
      rawCopyCommands: [],
      encodeStartedAt: now,
    };
    this.records.push(record);
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (passDescriptor?: GPUComputePassDescriptor): GPUComputePassEncoder => {
            const passLabel = passDescriptor?.label ?? "";
            record.passLabels.push(passLabel);
            const pass = target.beginComputePass(passDescriptor);
            let pipelineLabel = "";
            let bindGroupLabel = "";
            return new Proxy(pass, {
              get: (passTarget, passProperty) => {
                if (passProperty === "setPipeline") {
                  return (pipeline: GPUComputePipeline): void => {
                    pipelineLabel = pipeline.label;
                    passTarget.setPipeline(pipeline);
                  };
                }
                if (passProperty === "setBindGroup") {
                  return (
                    index: GPUIndex32,
                    bindGroup: GPUBindGroup | null,
                    dynamicOffsets?: Iterable<GPUBufferDynamicOffset>,
                  ): void => {
                    if (index === 0 && bindGroup !== null) {
                      bindGroupLabel = bindGroup.label;
                    }
                    if (dynamicOffsets === undefined) {
                      passTarget.setBindGroup(index, bindGroup);
                    } else {
                      passTarget.setBindGroup(index, bindGroup, dynamicOffsets);
                    }
                  };
                }
                if (passProperty === "dispatchWorkgroups") {
                  return (
                    workgroupCountX: GPUSize32,
                    workgroupCountY: GPUSize32 = 1,
                    workgroupCountZ: GPUSize32 = 1,
                  ): void => {
                    if (pipelineLabel === "" || bindGroupLabel === "") {
                      throw new Error("OPT-0010 dispatch omitted pipeline or bind group");
                    }
                    record.rawPhysicalDispatches.push(Object.freeze({
                      indexInCommand: record.rawPhysicalDispatches.length,
                      pipelineLabel,
                      bindGroupLabel,
                      workgroups: Object.freeze([
                        Number(workgroupCountX),
                        Number(workgroupCountY),
                        Number(workgroupCountZ),
                      ] as const),
                    }));
                    passTarget.dispatchWorkgroups(
                      workgroupCountX,
                      workgroupCountY,
                      workgroupCountZ,
                    );
                  };
                }
                if (passProperty === "dispatchWorkgroupsIndirect") {
                  return (): never => {
                    throw new Error("OPT-0010 production unexpectedly used indirect dispatch");
                  };
                }
                const value = Reflect.get(passTarget, passProperty, passTarget) as unknown;
                return typeof value === "function"
                  ? (value as (...args: unknown[]) => unknown).bind(passTarget)
                  : value;
              },
            }) as GPUComputePassEncoder;
          };
        }
        if (property === "copyBufferToBuffer") {
          return (
            source: GPUBuffer,
            sourceOffset: GPUSize64,
            destination: GPUBuffer,
            destinationOffset: GPUSize64,
            size: GPUSize64,
          ): void => {
            const sourceRecord = this.buffers.get(source);
            const destinationRecord = this.buffers.get(destination);
            if (sourceRecord === undefined || destinationRecord === undefined) {
              throw new Error("OPT-0010 readback copied an untracked buffer");
            }
            record.rawCopyCommands.push(Object.freeze({
              index: record.rawCopyCommands.length,
              sourceBufferLabel: sourceRecord.label,
              sourceOffset: Number(sourceOffset),
              destinationBufferLabel: destinationRecord.label,
              destinationOffset: Number(destinationOffset),
              copiedBytes: Number(size),
            }));
            target.copyBufferToBuffer(
              source,
              sourceOffset,
              destination,
              destinationOffset,
              size,
            );
          };
        }
        if (property === "finish") {
          return (finishDescriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer => {
            if (record.encodeEndedAt !== undefined) {
              throw new Error("OPT-0010 command encoder finished more than once");
            }
            const commandBuffer = target.finish(finishDescriptor);
            record.encodeEndedAt = performance.now();
            this.commandRecords.set(commandBuffer, record);
            return commandBuffer;
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    }) as GPUCommandEncoder;
  }

  private createQueueProxy(target: GPUQueue): GPUQueue {
    return new Proxy(target, {
      get: (queue, property) => {
        if (property === "submit") {
          return (commandBuffers: Iterable<GPUCommandBuffer>): void => {
            const retained = [...commandBuffers];
            if (this.activeMode === "off") {
              queue.submit(retained);
              return;
            }
            if (retained.length !== 1 || this.pendingSubmission !== undefined) {
              throw new Error(
                "OPT-0010 production must retain one outstanding command buffer",
              );
            }
            const record = this.commandRecords.get(retained[0]!);
            if (record === undefined) {
              throw new Error("OPT-0010 submitted an unobserved command buffer");
            }
            record.submitStartedAt = performance.now();
            queue.submit(retained);
            record.submitReturnedAt = performance.now();
            this.submissionCount += 1;
            this.pendingSubmission = record;
          };
        }
        if (property === "onSubmittedWorkDone") {
          return async (): Promise<void> => {
            if (this.activeMode === "off") {
              await queue.onSubmittedWorkDone();
              return;
            }
            const record = this.pendingSubmission;
            if (record === undefined) {
              throw new Error("OPT-0010 drained without a pending submission");
            }
            record.drainStartedAt = performance.now();
            await queue.onSubmittedWorkDone();
            record.drainEndedAt = performance.now();
            this.drainCount += 1;
            this.pendingSubmission = undefined;
          };
        }
        const value = Reflect.get(queue, property, queue) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(queue)
          : value;
      },
    }) as GPUQueue;
  }
}

function normalizePhysicalDispatches(
  record: MutableCommandRecord,
  expected: AceOpt0010PlannerTokenAttribution["quanta"][number],
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  if (
    record.kind !== "model" ||
    record.quantumId !== expected.productionQuantum.id ||
    record.passLabels.length !== 1 ||
    record.passLabels[0] !== expected.productionQuantum.id ||
    record.rawPhysicalDispatches.length !== expected.physicalDispatches.length
  ) throw new Error(`OPT-0010 actual quantum ${expected.index} tags changed`);
  return Object.freeze(record.rawPhysicalDispatches.map((raw, indexInQuantum) => {
    const expectedDispatch = expected.physicalDispatches[indexInQuantum]!;
    const dispatchIdentity = authenticateDispatchIdentity(raw, expectedDispatch);
    authenticatePipelineLabel(raw.pipelineLabel, expectedDispatch);
    if (!sameNumbers(raw.workgroups, expectedDispatch.workgroups)) {
      throw new Error(
        `OPT-0010 actual quantum ${expected.index} workgroups changed at ` +
          indexInQuantum,
      );
    }
    return Object.freeze({
      indexInQuantum,
      pipelineIdentity: expectedDispatch.pipelineIdentity,
      dispatchIdentity,
      workgroups: Object.freeze([...raw.workgroups]) as readonly [number, number, number],
    });
  }));
}

function authenticateDispatchIdentity(
  raw: RawPhysicalDispatch,
  expected: AceOpt0010PlannerPhysicalDispatch,
): string {
  let actual = raw.bindGroupLabel.replace(/-bindings$/, "");
  actual = actual.replace(/-source-row-major-range-(\d+)$/, "-range-$1");
  if (expected.pipelineIdentity.startsWith("embedding/")) {
    const start = /-rows-(\d+)-bindings$/.exec(raw.bindGroupLabel)?.[1];
    if (
      start === undefined ||
      !expected.dispatchIdentity.startsWith(actual) ||
      !raw.pipelineLabel.endsWith(
        expected.dispatchIdentity.slice(expected.dispatchIdentity.indexOf("-rows-")),
      )
    ) throw new Error("OPT-0010 actual embedding dispatch identity changed");
    return expected.dispatchIdentity;
  }
  if (actual !== expected.dispatchIdentity) {
    throw new Error(
      `OPT-0010 actual dispatch identity ${actual} != ${expected.dispatchIdentity}`,
    );
  }
  return actual;
}

function authenticatePipelineLabel(
  actual: string,
  expected: AceOpt0010PlannerPhysicalDispatch,
): void {
  const [family] = expected.pipelineIdentity.split("/");
  let accepted = false;
  if (family === "embedding") {
    accepted = actual.includes("-embed-tokens-rows-");
  } else if (family === "gemm") {
    accepted = actual === "ace-correctness-" +
      expected.pipelineIdentity.replaceAll("/", "-");
  } else if (family === "rmsnorm") {
    const shape = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual === `ace-correctness-rmsnorm-raw-fp16-${shape}`;
  } else if (family === "batched-rope") {
    const shape = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual === `ace-correctness-batched-rope-raw-fp16-${shape}`;
  } else if (family === "kv-cache-write") {
    const shape = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual === `ace-correctness-kv-cache-write-raw-fp16-${shape}`;
  } else if (family === "attention") {
    const shape = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual ===
      `ace-correctness-attention-raw-fp16-${shape}xcausalxnonexnonexcausal-per-key`;
  } else if (family === "head-transform") {
    const operation = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual.endsWith(`-${operation}`);
  } else if (family === "transformer") {
    const operation = expected.pipelineIdentity.split("/")[2]!;
    accepted = actual.endsWith(`-${operation}`);
  } else if (family === "gather-rows") {
    accepted = actual.endsWith("-last-physical-rows-gather-rows");
  }
  if (!accepted) {
    throw new Error(
      `OPT-0010 actual pipeline ${actual} does not authenticate ` +
        expected.pipelineIdentity,
    );
  }
}

function normalizeCopyCommands(
  record: MutableCommandRecord,
): readonly AceOpt0010PlannerCopyCommand[] {
  if (
    record.kind !== "readback" ||
    record.commandLabel !== "ace-planner-logit-readback" ||
    record.passLabels.length !== 0 ||
    record.rawPhysicalDispatches.length !== 0 ||
    record.rawCopyCommands.length !== OPT_0010_READBACK_COPY_COUNT
  ) throw new Error("OPT-0010 actual readback topology changed");
  return Object.freeze(record.rawCopyCommands.map((copy) => {
    const shard = /^logits-(\d+)$/.exec(copy.sourceBufferLabel);
    if (
      (shard === null && copy.sourceBufferLabel !== "write-status") ||
      copy.sourceOffset !== 0 ||
      copy.destinationBufferLabel !== "ace-planner-logit-readback"
    ) throw new Error(`OPT-0010 actual readback copy ${copy.index} changed`);
    return Object.freeze({
      index: copy.index,
      sourceBufferLabel: copy.sourceBufferLabel,
      shardIndex: shard === null ? null : Number(shard[1]),
      sourceOffset: copy.sourceOffset,
      destinationBufferLabel: "ace-planner-logit-readback" as const,
      destinationOffset: copy.destinationOffset,
      copiedBytes: copy.copiedBytes,
    });
  }));
}

function progressPayload(
  event: AcePlannerGpuExecutorProgress,
): AceOpt0010PlannerModelProgressPayload |
  AceOpt0010PlannerReadbackProgressPayload {
  if (event.phaseKind !== "decode") {
    throw new Error("OPT-0010 traced a non-decode progress payload");
  }
  const common = {
    phaseKind: "decode" as const,
    completedCommandBuffers: event.completedCommandBuffers,
    totalCommandBuffers: event.totalCommandBuffers,
    queueDrains: event.queueDrains,
    cooperativeIdleMs: event.cooperativeIdleMs,
    peakAccountedGpuBytes: event.peakAccountedGpuBytes,
    cumulativeQueueDrains: event.cumulativeQueueDrains,
    cumulativeCooperativeIdleMs: event.cumulativeCooperativeIdleMs,
  } as const;
  return event.stage === "model"
    ? Object.freeze({
        ...common,
        stage: "model" as const,
        quantum: productionTag(event.quantum!),
      })
    : Object.freeze({
        ...common,
        stage: "readback" as const,
        quantum: null,
      });
}

function productionTag(
  quantum: Readonly<{
    readonly id: string;
    readonly logicalId?: string;
    readonly kind: AceOpt0010PlannerProductionQuantumTag["kind"];
    readonly layer: number | null;
    readonly primitiveCount: number;
  }>,
): AceOpt0010PlannerProductionQuantumTag {
  return Object.freeze({
    id: quantum.id,
    logicalId: quantum.logicalId ?? quantum.id,
    kind: quantum.kind,
    layer: quantum.layer,
    primitiveCount: quantum.primitiveCount,
  });
}

function requireModelProgress(
  progress: MutableCommandRecord["progress"],
  index: number,
): AceOpt0010PlannerModelProgressPayload {
  if (progress?.stage !== "model") {
    throw new Error(`OPT-0010 quantum ${index} omitted model progress`);
  }
  return progress;
}

function requireReadbackProgress(
  progress: MutableCommandRecord["progress"],
): AceOpt0010PlannerReadbackProgressPayload {
  if (progress?.stage !== "readback") {
    throw new Error("OPT-0010 omitted readback progress");
  }
  return progress;
}

async function preparePackage(): Promise<PreparedPackage> {
  const manifestUrl = new URL(OPT_0010_FP16_MANIFEST_PATH, self.location.href).href;
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: OPT_0010_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16",
  });
  if (
    loaded.manifestSha256 !== OPT_0010_FP16_MANIFEST_SHA256 ||
    loaded.manifest.provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION ||
    ACE_PACKAGE_CONVERTER_REVISION !== 4
  ) throw new Error("OPT-0010 manifest trust root changed");
  const inventory = validatePlannerInventory(loaded.manifest);
  const acquisitionManifest = Object.freeze({
    ...loaded.manifest,
    files: inventory.files,
  });
  const cache = await AceOpfsModelCache.open();
  let lastStatusAt = 0;
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      const now = performance.now();
      if (
        now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
        progress.fileIndex + 1 === progress.fileCount &&
          progress.fileReceivedBytes === progress.fileBytes
      ) {
        lastStatusAt = now;
        postProgress(
          `acquiring planner file ${progress.fileIndex + 1}/${progress.fileCount} ` +
            `(${formatBytes(progress.completedBytes)}/` +
            `${formatBytes(progress.totalBytes)}, ${progress.source})`,
        );
      }
    },
  });
  if (
    acquired.files.size !== OPT_0010_ACQUIRED_FILE_COUNT ||
    acquired.plan.files.length !== OPT_0010_ACQUIRED_FILE_COUNT
  ) throw new Error("OPT-0010 bounded planner acquisition changed");
  return Object.freeze({
    manifest: loaded.manifest,
    manifestUrl: loaded.manifestUrl,
    acquiredFiles: acquired.files,
    summary: Object.freeze({
      manifestSha256: loaded.manifestSha256,
      manifestByteLength: loaded.manifestByteLength,
      converterRevision: loaded.manifest.provenance.converterRevision,
      plannerTensorCount: inventory.tensorCount,
      plannerWeightFileCount: inventory.weightFileCount,
      tokenizerFileCount: OPT_0010_TOKENIZER_FILE_COUNT,
      plannerResidentBytes: inventory.residentBytes,
      acquiredFileCount: acquired.files.size,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
    }),
  });
}

function validatePlannerInventory(manifest: AcePackageManifest): Readonly<{
  files: readonly AcePackageFileRecord[];
  tensorCount: number;
  weightFileCount: number;
  residentBytes: number;
}> {
  const tensors = Object.values(manifest.tensors).filter(
    (tensor) => tensor.phase === "planner",
  );
  const weightNames = new Set(tensors.map((tensor) => tensor.shard));
  const tokenizerNames = new Set([
    "assets/planner/tokenizer.json",
    "assets/planner/tokenizer_config.json",
    "assets/planner/chat_template.jinja",
  ]);
  const files = manifest.files.filter((file) =>
    weightNames.has(file.name) || tokenizerNames.has(file.name));
  const weightFiles = files.filter((file) => weightNames.has(file.name));
  const residentBytes = sumSafe(
    weightFiles.map((file) => file.byteLength),
    "OPT-0010 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0010 planner tensor bytes",
  );
  if (
    tensors.length !== OPT_0010_PLANNER_TENSOR_COUNT ||
    weightNames.size !== OPT_0010_PLANNER_WEIGHT_FILE_COUNT ||
    weightFiles.length !== OPT_0010_PLANNER_WEIGHT_FILE_COUNT ||
    files.length !== OPT_0010_ACQUIRED_FILE_COUNT ||
    residentBytes !== OPT_0010_PLANNER_RESIDENT_BYTES ||
    tensorBytes !== OPT_0010_PLANNER_RESIDENT_BYTES
  ) throw new Error("OPT-0010 planner inventory changed from revision 4");
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    weightFileCount: weightFiles.length,
    residentBytes,
  });
}

function createCaseFixtures(
  tokenizer: AceQwenBpeTokenizer,
): readonly PlannerCaseFixture[] {
  const cotPrompt = createAcePlannerCotPrompt(ACCEPTED_PROMPT, ACCEPTED_LYRICS);
  const codePrompts = createAcePlannerCodePrompts(
    ACCEPTED_RESOLVED_CAPTION,
    ACCEPTED_LYRICS,
    ACCEPTED_COT_TEXT,
  );
  const cotBase = tokenizer.encode(cotPrompt);
  const cotContinuation = createReceiptTextDerivedCotTeacherTokens(
    tokenizer,
    cotBase,
  );
  const semanticBases = Object.freeze([
    tokenizer.encode(codePrompts.conditional),
    tokenizer.encode(codePrompts.unconditional),
  ]);
  const semanticContinuation = OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS.map(
    (code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code,
  );
  if (
    cotBase.length !== 105 ||
    cotContinuation.length !== 109 ||
    semanticContinuation.length !== 150 ||
    semanticBases[0]!.length !== 253 ||
    semanticBases[1]!.length !== 33
  ) throw new Error("OPT-0010 prompt or trajectory geometry changed");
  if (
    sha256U32Le(cotContinuation) !==
      OPT_0010_COT_DIAGNOSTIC_TRAJECTORY_SHA256 ||
    sha256U32Le(OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS) !==
      OPT_0010_ACCEPTED_SEMANTIC_CODE_SHA256
  ) throw new Error("OPT-0010 pinned diagnostic trajectory hash changed");
  return Object.freeze(OPT_0010_CASE_SPECS.map((spec) => {
    const rows = spec.mode === "cot-m1" ? 1 as const : 2 as const;
    const baseRows = rows === 1 ? [cotBase] : semanticBases;
    const continuation = rows === 1 ? cotContinuation : semanticContinuation;
    const prefill = createPaddedAcceptedPrefill(
      baseRows,
      continuation,
      spec.cachedTokensBeforeAppend,
      spec.cacheCapacity,
    );
    const decodeTokenId = continuation[
      spec.cachedTokensBeforeAppend - Math.max(...baseRows.map((row) => row.length))
    ]!;
    const acceptedContinuationCount =
      spec.cachedTokensBeforeAppend - Math.max(...baseRows.map((row) => row.length));
    const attribution = createAceOpt0010PlannerTokenAttribution(
      spec.mode,
      spec.cachedTokensBeforeAppend,
      spec.cacheCapacity,
    );
    if (
      attribution.rows !== rows ||
      attribution.totals.modelQuantumCount !== OPT_0010_MODEL_QUANTUM_COUNT ||
      attribution.totals.modelDispatchPrimitiveCount !==
        OPT_0010_MODEL_DISPATCH_PRIMITIVE_COUNT ||
      attribution.totals.modelPhysicalPrimitiveDispatchCount !==
        OPT_0010_PHYSICAL_DISPATCH_COUNT ||
      attribution.totals.commandBufferCount !== OPT_0010_COMMAND_BUFFER_COUNT
    ) throw new Error(`OPT-0010 ${spec.id} attribution topology changed`);
    return Object.freeze({
      spec,
      attribution,
      prefill,
      decodeTokenId,
      seenTokenIds: Object.freeze([...prefill.inputIds.slice(
        0,
        spec.cachedTokensBeforeAppend,
      )]),
      acceptedPromptTokenIds: Object.freeze([...(baseRows[0] ?? [])]),
      acceptedEmittedTokenIdsIncludingDecode: Object.freeze([
        ...continuation.slice(0, acceptedContinuationCount + 1),
      ]),
      promptLengths: Object.freeze(baseRows.map((row) => row.length)),
      tokenizer,
    });
  }));
}

/**
 * The accepted receipt preserves normalized CoT text and its 109 sampling
 * draws, but not the emitted CoT token IDs. This pinned longest-admitted-token
 * replay is therefore a representative diagnostic trajectory, not a claim
 * that its segmentation reproduces the historical BF16 receipt trajectory.
 */
function createReceiptTextDerivedCotTeacherTokens(
  tokenizer: AceQwenBpeTokenizer,
  promptTokenIds: readonly number[],
): readonly number[] {
  const closingTag = "</think>";
  if (!ACCEPTED_COT_TEXT.endsWith(closingTag)) {
    throw new Error("OPT-0010 accepted normalized CoT closing tag changed");
  }
  let remaining = ACCEPTED_COT_TEXT.slice(0, -closingTag.length);
  const controller = new AcePlannerMetadataConstraintController({ tokenizer });
  const accepted: number[] = [];
  const logits = new Float32Array(ACE_PLANNER_QWEN3_CONFIG.vocabularySize);
  logits.fill(-1);
  while (remaining.length > 0) {
    const candidates: Array<Readonly<{ id: number; text: string }>> = [];
    for (
      let length = 1;
      length <= Math.min(96, remaining.length);
      length += 1
    ) {
      const text = remaining.slice(0, length);
      const tokenIds = tokenizer.encode(text);
      if (tokenIds.length === 1 && tokenizer.decode(tokenIds) === text) {
        candidates.push(Object.freeze({ id: tokenIds[0]!, text }));
      }
    }
    for (const candidate of candidates) {
      logits[candidate.id] = candidate.text.length;
    }
    const step = accepted.length;
    const allowed = controller.allowedTokens({
      step,
      promptTokenIds,
      emittedTokenIds: accepted,
      logits,
    });
    const chosen = candidates
      .filter((candidate) => allowedTokenIncludes(allowed, candidate.id))
      .sort((left, right) => right.text.length - left.text.length)[0];
    if (chosen === undefined) {
      throw new Error(
        `OPT-0010 accepted normalized CoT cannot advance at step ${step}`,
      );
    }
    accepted.push(chosen.id);
    controller.acceptToken({
      step,
      tokenId: chosen.id,
      tokenText: chosen.text,
      emittedTokenIds: accepted,
    });
    for (const candidate of candidates) logits[candidate.id] = -1;
    remaining = remaining.slice(chosen.text.length);
  }
  const terminal = controller.allowedTokens({
    step: accepted.length,
    promptTokenIds,
    emittedTokenIds: accepted,
    logits,
  });
  if (
    terminal.kind !== "ids" ||
    terminal.tokenIds.length !== 1 ||
    terminal.tokenIds[0] !== ACE_QWEN_IM_END_TOKEN_ID
  ) throw new Error("OPT-0010 accepted normalized CoT did not reach terminal EOS");
  accepted.push(ACE_QWEN_IM_END_TOKEN_ID);
  return Object.freeze(accepted);
}

function allowedTokenIncludes(
  allowed: AcePlannerAllowedTokens,
  tokenId: number,
): boolean {
  if (allowed.kind === "all") return true;
  if (allowed.kind === "ids") return allowed.tokenIds.includes(tokenId);
  return tokenId >= allowed.firstTokenId &&
      tokenId < allowed.firstTokenId + allowed.tokenCount ||
    (allowed.additionalTokenIds ?? []).includes(tokenId);
}

function createPaddedAcceptedPrefill(
  baseRows: readonly (readonly number[])[],
  continuation: readonly number[],
  tokens: number,
  cacheCapacity: number,
): AcePlannerPrefillBatch {
  const rows = baseRows.length;
  if (
    (rows !== 1 && rows !== 2) ||
    continuation.length === 0 ||
    tokens >= cacheCapacity
  ) throw new Error("OPT-0010 accepted prefill geometry is invalid");
  const baseWidth = Math.max(...baseRows.map((row) => row.length));
  if (
    baseWidth >= tokens ||
    tokens - baseWidth >= continuation.length
  ) {
    throw new Error(
      "OPT-0010 case leaves the accepted continuation trajectory",
    );
  }
  const inputIds = new Uint32Array(rows * tokens);
  const keyValidity = new Uint32Array(rows * tokens);
  for (let row = 0; row < rows; row += 1) {
    const base = baseRows[row]!;
    const rowOffset = row * tokens;
    const leftPadding = baseWidth - base.length;
    inputIds.fill(ACE_QWEN_PAD_TOKEN_ID, rowOffset, rowOffset + leftPadding);
    inputIds.set(base, rowOffset + leftPadding);
    keyValidity.fill(1, rowOffset + leftPadding, rowOffset + baseWidth);
    for (let position = baseWidth; position < tokens; position += 1) {
      inputIds[rowOffset + position] =
        continuation[position - baseWidth]!;
      keyValidity[rowOffset + position] = 1;
    }
  }
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens,
    cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(0),
    validKeyLengths: Array<number>(rows).fill(tokens),
    sourceValidity: [...keyValidity],
  });
  return Object.freeze({
    kind: "prefill" as const,
    rows,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: rows === 2 ? 1 as const : null,
  });
}

function createDecodeBatch(fixture: PlannerCaseFixture) {
  const rows = fixture.attribution.rows;
  const cachedTokensBeforeAppend = fixture.spec.cachedTokensBeforeAppend;
  const inputIds = new Uint32Array(rows);
  inputIds.fill(fixture.decodeTokenId);
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens: 1,
    cacheCapacity: fixture.spec.cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(cachedTokensBeforeAppend),
    validKeyLengths: Array<number>(rows).fill(cachedTokensBeforeAppend + 1),
    sourceValidity: Array<number>(rows).fill(1),
  });
  return Object.freeze({
    kind: "decode" as const,
    rows,
    tokens: 1 as const,
    cacheCapacity: fixture.spec.cacheCapacity,
    cachedTokensBeforeAppend,
    inputIds,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: rows === 2 ? 1 as const : null,
  });
}

function sampleCase(
  fixture: PlannerCaseFixture,
  logits: readonly Float32Array[],
  allowedTokens: AcePlannerAllowedTokens,
): SampleSummary {
  const cursor = new AcePlannerSamplingCursor(
    ACCEPTED_SEED,
    fixture.spec.drawIndex,
  );
  const semantic = fixture.spec.mode === "semantic-m2";
  const sample = cursor.sample({
    conditionalLogits: logits[0]!,
    ...(semantic ? { unconditionalLogits: logits[1]! } : {}),
    seenTokenIds: [...fixture.seenTokenIds, fixture.decodeTokenId],
    ...(semantic
      ? {
          preCfgAllowedTokens: {
            kind: "range" as const,
            firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
            tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
            additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
          },
        }
      : {}),
    allowedTokens,
    parameters: {
      temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
      guidanceScale: semantic
        ? DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale
        : 1,
      topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
      topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
      repetitionPenalty: 1,
    },
  });
  return Object.freeze({
    tokenId: sample.tokenId,
    word: sample.word,
    drawIndex: sample.drawIndex.toString(),
    drawEnd: cursor.consumed.toString(),
    positiveCandidateCount: sample.positiveCandidateCount,
  });
}

function validateSampleCursor(
  fixture: PlannerCaseFixture,
  sample: SampleSummary,
  label: "reference" | "timed",
): void {
  const expectedDrawIndex = fixture.spec.drawIndex.toString();
  const expectedDrawEnd = (fixture.spec.drawIndex + 1).toString();
  if (
    sample.drawIndex !== expectedDrawIndex ||
    sample.drawEnd !== expectedDrawEnd
  ) {
    throw new Error(
      `OPT-0010 ${fixture.spec.id} ${label} sampling cursor changed`,
    );
  }
}

function createSamplingConstraintState(
  fixture: PlannerCaseFixture,
): SamplingConstraintState {
  if (fixture.spec.mode === "semantic-m2") {
    return Object.freeze({ cot: null });
  }
  const controller = new AcePlannerMetadataConstraintController({
    tokenizer: fixture.tokenizer,
  });
  const emitted: number[] = [];
  const teacherLogits = new Float32Array(
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
  );
  teacherLogits.fill(-1);
  for (
    let step = 0;
    step < fixture.acceptedEmittedTokenIdsIncludingDecode.length;
    step += 1
  ) {
    const tokenId = fixture.acceptedEmittedTokenIdsIncludingDecode[step]!;
    teacherLogits[tokenId] = 1;
    controller.allowedTokens({
      step,
      promptTokenIds: fixture.acceptedPromptTokenIds,
      emittedTokenIds: emitted,
      logits: teacherLogits,
    });
    emitted.push(tokenId);
    controller.acceptToken({
      step,
      tokenId,
      tokenText: fixture.tokenizer.decode([tokenId]),
      emittedTokenIds: emitted,
    });
    teacherLogits[tokenId] = -1;
  }
  return Object.freeze({ cot: controller });
}

function resolveCaseAllowedTokens(
  fixture: PlannerCaseFixture,
  logits: readonly Float32Array[],
  state: SamplingConstraintState,
): AcePlannerAllowedTokens {
  validateSamplingInputShape(fixture, logits);
  if (fixture.spec.mode === "semantic-m2") {
    if (state.cot !== null) {
      throw new Error("OPT-0010 semantic case retained a CoT constraint");
    }
    return Object.freeze({
      kind: "range" as const,
      firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
    });
  }
  if (state.cot === null) {
    throw new Error("OPT-0010 CoT case omitted its production constraint state");
  }
  const emitted = fixture.acceptedEmittedTokenIdsIncludingDecode;
  return state.cot.allowedTokens({
    step: emitted.length,
    promptTokenIds: fixture.acceptedPromptTokenIds,
    emittedTokenIds: emitted,
    logits: logits[0]!,
  });
}

function validateSamplingInputShape(
  fixture: PlannerCaseFixture,
  logits: readonly Float32Array[],
): void {
  if (logits.length !== fixture.attribution.rows) {
    throw new Error(`OPT-0010 ${fixture.spec.id} logit row count changed`);
  }
  for (let row = 0; row < logits.length; row += 1) {
    if (logits[row]!.length !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize) {
      throw new Error(`OPT-0010 ${fixture.spec.id} logit extent changed`);
    }
  }
}

async function summarizeLogits(
  logits: readonly Float32Array[],
  rows: 1 | 2,
): Promise<LogitSummary> {
  if (logits.length !== rows) throw new Error("OPT-0010 logit rows changed");
  const rowElements = ACE_PLANNER_QWEN3_CONFIG.vocabularySize;
  const bytes = new Uint8Array(rows * rowElements * Float32Array.BYTES_PER_ELEMENT);
  let cursor = 0;
  let finiteCount = 0;
  let nonzeroCount = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const row of logits) {
    if (row.length !== rowElements) throw new Error("OPT-0010 logit width changed");
    bytes.set(new Uint8Array(row.buffer, row.byteOffset, row.byteLength), cursor);
    cursor += row.byteLength;
    for (const value of row) {
      if (Number.isFinite(value)) finiteCount += 1;
      if (value !== 0) nonzeroCount += 1;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  return Object.freeze({
    rows,
    rowElements,
    totalElements: rows * rowElements,
    finiteCount,
    nonzeroCount,
    minimum,
    maximum,
    sha256: await sha256Hex(bytes),
  });
}

function validateCompleteLogits(summary: LogitSummary, label: string): void {
  if (
    summary.totalElements !== summary.rows * ACE_PLANNER_QWEN3_CONFIG.vocabularySize ||
    summary.finiteCount !== summary.totalElements ||
    summary.nonzeroCount === 0 ||
    !Number.isFinite(summary.minimum) ||
    !Number.isFinite(summary.maximum) ||
    !/^[0-9a-f]{64}$/.test(summary.sha256)
  ) throw new Error(`OPT-0010 ${label} failed full-logit completeness`);
}

function compareLogits(
  reference: readonly Float32Array[],
  actual: readonly Float32Array[],
): Readonly<{
  comparedU32WordCount: number;
  bitMismatchCount: number;
  firstMismatchRow: number | null;
  firstMismatchToken: number | null;
  bitExact: boolean;
}> {
  if (reference.length !== actual.length) {
    throw new Error("OPT-0010 comparison row counts differ");
  }
  let comparedU32WordCount = 0;
  let bitMismatchCount = 0;
  let firstMismatchRow: number | null = null;
  let firstMismatchToken: number | null = null;
  for (let row = 0; row < reference.length; row += 1) {
    const expectedBits = new Uint32Array(
      reference[row]!.buffer,
      reference[row]!.byteOffset,
      reference[row]!.length,
    );
    const actualBits = new Uint32Array(
      actual[row]!.buffer,
      actual[row]!.byteOffset,
      actual[row]!.length,
    );
    if (expectedBits.length !== actualBits.length) {
      throw new Error("OPT-0010 comparison logit widths differ");
    }
    comparedU32WordCount += expectedBits.length;
    for (let token = 0; token < expectedBits.length; token += 1) {
      if (expectedBits[token] !== actualBits[token]) {
        bitMismatchCount += 1;
        firstMismatchRow ??= row;
        firstMismatchToken ??= token;
      }
    }
  }
  return Object.freeze({
    comparedU32WordCount,
    bitMismatchCount,
    firstMismatchRow,
    firstMismatchToken,
    bitExact: bitMismatchCount === 0,
  });
}

function requireFloat32Logits(
  logits: readonly ArrayLike<number>[],
): readonly Float32Array[] {
  if (logits.some((row) => !(row instanceof Float32Array))) {
    throw new TypeError("OPT-0010 package-native logits are not Float32Array rows");
  }
  return Object.freeze([...logits] as Float32Array[]);
}

function sameSample(left: SampleSummary, right: SampleSummary): boolean {
  return left.tokenId === right.tokenId &&
    left.word === right.word &&
    left.drawIndex === right.drawIndex &&
    left.drawEnd === right.drawEnd &&
    left.positiveCandidateCount === right.positiveCandidateCount;
}

function validateProgressSequence(
  attribution: AceOpt0010PlannerTokenAttribution,
  progress: readonly AcePlannerGpuExecutorProgress[],
): void {
  if (progress.length !== OPT_0010_COMMAND_BUFFER_COUNT) {
    throw new Error("OPT-0010 full progress event count changed");
  }
  for (let index = 0; index < attribution.quanta.length; index += 1) {
    const event = progress[index]!;
    const expected = attribution.quanta[index]!.productionQuantum;
    if (
      event.phaseKind !== "decode" ||
      event.stage !== "model" ||
      event.completedCommandBuffers !== index + 1 ||
      event.totalCommandBuffers !== OPT_0010_COMMAND_BUFFER_COUNT ||
      event.queueDrains !== index + 1 ||
      event.cooperativeIdleMs !== index + 1 ||
      event.quantum?.id !== expected.id ||
      (event.quantum.logicalId ?? event.quantum.id) !== expected.logicalId ||
      event.quantum.kind !== expected.kind ||
      event.quantum.layer !== expected.layer ||
      event.quantum.primitiveCount !== expected.primitiveCount
    ) throw new Error(`OPT-0010 full progress payload ${index} changed`);
  }
  const readback = progress.at(-1)!;
  if (
    readback.phaseKind !== "decode" ||
    readback.stage !== "readback" ||
    readback.quantum !== null ||
    readback.completedCommandBuffers !== OPT_0010_COMMAND_BUFFER_COUNT ||
    readback.totalCommandBuffers !== OPT_0010_COMMAND_BUFFER_COUNT ||
    readback.queueDrains !== OPT_0010_COMMAND_BUFFER_COUNT ||
    readback.cooperativeIdleMs !== OPT_0010_COMMAND_BUFFER_COUNT
  ) throw new Error("OPT-0010 readback progress payload changed");
}

async function runCancellationProof(
  prepared: PreparedSession,
  fixture: PlannerCaseFixture,
): Promise<CancellationSummary> {
  let abortRequested = false;
  prepared.observer.beginCancellationTrace(fixture.attribution);
  prepared.progressRouter.begin("cancellation", (event) => {
    if (
      !abortRequested &&
      event.stage === "model" &&
      event.completedCommandBuffers === 1
    ) {
      abortRequested = true;
      prepared.abortController.abort(new DOMException(
        "OPT-0010 post-drain cancellation probe",
        "AbortError",
      ));
    }
  });
  let rejection: unknown;
  try {
    await prepared.executor.decode(createDecodeBatch(fixture));
  } catch (error) {
    rejection = error;
  }
  const rejectedAt = performance.now();
  const progress = prepared.progressRouter.end("cancellation");
  const observed = prepared.observer.endCancellationTrace();
  const first = observed.records[0];
  if (
    !(rejection instanceof DOMException) ||
    rejection.name !== "AbortError" ||
    !abortRequested ||
    progress.length !== 1 ||
    observed.records.length !== 1 ||
    observed.submissionCount !== 1 ||
    observed.drainCount !== 1 ||
    first?.kind !== "model" ||
    first.rawPhysicalDispatches.length !== 5 ||
    first.drainEndedAt === undefined ||
    first.progressReportedAt === undefined ||
    first.progressReportedAt < first.drainEndedAt ||
    rejectedAt <= first.progressReportedAt ||
    first.nextEncodeStartedAt !== undefined
  ) throw new Error("OPT-0010 post-drain cancellation accounting changed");
  return Object.freeze({
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
    encodedCommandBufferCount: observed.records.length,
    submissionCount: observed.submissionCount,
    queueDrainCount: observed.drainCount,
    completedIdleCount: 1,
    laterEncodingPrevented: observed.records.length === 1,
    laterSubmissionPrevented: observed.submissionCount === 1,
    firstQuantumFullyDrained: observed.drainCount === 1,
    realIdleCompletedBeforeRejection: rejectedAt > first.progressReportedAt,
  });
}

function validateThermalGate(
  thermal: Opt0010ThermalGateMetadata,
  warmupCompletedAtEpochMilliseconds: number,
): void {
  if (
    thermal.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    thermal.startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds < thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds !==
      thermal.completedAtEpochMilliseconds - thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds < 30_000 ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount < Math.floor(thermal.durationMilliseconds / 1_000) + 1 ||
    thermal.pollMilliseconds !== 1_000 ||
    !Number.isFinite(thermal.maximumPollGapMilliseconds) ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds > 1_250 ||
    thermal.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0010 worker rejected the external thermal gate");
}

function validateRunIdentity(identity: unknown): Opt0010RunIdentity {
  if (typeof identity !== "object" || identity === null) {
    throw new Error("OPT-0010 worker requires a frozen run identity");
  }
  const candidate = identity as Readonly<Record<string, unknown>>;
  const requiredString = (name: string): string => {
    const value = candidate[name];
    if (typeof value !== "string" || value.trim() === "" || value !== value.trim()) {
      throw new Error(`OPT-0010 worker rejected run identity ${name}`);
    }
    return value;
  };
  const requiredPositiveInteger = (name: string): number => {
    const value = candidate[name];
    if (!Number.isSafeInteger(value) || (value as number) <= 0) {
      throw new Error(`OPT-0010 worker rejected run identity ${name}`);
    }
    return value as number;
  };
  const harnessCommit = requiredString("harnessCommit");
  const productionCommit = requiredString("productionCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0010 worker rejected harnessCommit");
  }
  if (productionCommit !== OPT_0010_EXPECTED_PRODUCTION_COMMIT) {
    throw new Error("OPT-0010 worker rejected productionCommit");
  }
  return Object.freeze({
    harnessCommit,
    productionCommit: OPT_0010_EXPECTED_PRODUCTION_COMMIT,
    machineModel: requiredString("machineModel"),
    osVersion: requiredString("osVersion"),
    osBuild: requiredString("osBuild"),
    browserVersion: requiredString("browserVersion"),
    gpuCoreCount: requiredPositiveInteger("gpuCoreCount"),
    memoryBytes: requiredPositiveInteger("memoryBytes"),
  });
}

function publicPreparationSummary(
  prepared: PreparedSession,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    packageAcquisitionWallMilliseconds:
      prepared.packageAcquisitionWallMilliseconds,
    phaseUploadWallMilliseconds: prepared.phaseUploadWallMilliseconds,
    executorCompileAndReferenceWallMilliseconds:
      prepared.executorCompileWallMilliseconds,
    package: prepared.preparedPackage.summary,
    cases: Object.freeze(prepared.references.map((reference) => Object.freeze({
      spec: reference.fixture.spec,
      rows: reference.fixture.attribution.rows,
      promptLengths: reference.fixture.promptLengths,
      decodeTokenId: reference.fixture.decodeTokenId,
      prefillWallMilliseconds: reference.prefillWallMilliseconds,
      decodeWallMilliseconds: reference.decodeWallMilliseconds,
      logits: reference.logitsSummary,
      sample: reference.sample,
    }))),
    warmupCompletedAtEpochMilliseconds:
      prepared.warmupCompletedAtEpochMilliseconds,
  });
}

function startWorkerHeartbeat(): { stop(): WorkerHeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  let timerTickCount = 0;
  let maximumTimerGapMilliseconds = 0;
  let stopped = false;
  let last = performance.now();
  const handle = setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - last,
    );
    last = now;
    timerTickCount += 1;
  }, WORKER_HEARTBEAT_INTERVAL_MILLISECONDS);
  return {
    stop(): WorkerHeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        clearInterval(handle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        timerTickCount,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

function validateWorkerHeartbeat(heartbeat: WorkerHeartbeatSnapshot): void {
  if (
    heartbeat.timerTickCount < 1 ||
    !Number.isFinite(heartbeat.maximumTimerGapMilliseconds) ||
    heartbeat.maximumTimerGapMilliseconds < 0 ||
    heartbeat.completedAtEpochMilliseconds < heartbeat.startedAtEpochMilliseconds
  ) throw new Error("OPT-0010 worker heartbeat telemetry is invalid");
}

async function failAndCleanup(
  error: unknown,
  active: PreparedSession | undefined = session,
): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  session = undefined;
  active?.progressRouter.abandon();
  let cleanupError: unknown;
  if (active !== undefined) {
    try {
      await active.executor.destroy(error);
    } catch (caught) {
      cleanupError = caught;
    } finally {
      active.context.destroy();
    }
  }
  const heartbeat = workerHeartbeat?.stop();
  workerHeartbeat = undefined;
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      primary: errorValue(error),
      ...(cleanupError === undefined ? {} : { cleanup: errorValue(cleanupError) }),
      ...(heartbeat === undefined ? {} : { workerHeartbeat: heartbeat }),
    }),
  });
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0010 package omitted ${name}`);
  return file;
}

function requiredTimestamp(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value) || value < 0) {
    throw new Error(`OPT-0010 trace omitted ${label}`);
  }
  return value;
}

function sameNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0")).join("");
}

function sha256U32Le(values: readonly number[]): string {
  const bytes = new Uint8Array(
    values.length * Uint32Array.BYTES_PER_ELEMENT,
  );
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(
      index * Uint32Array.BYTES_PER_ELEMENT,
      values[index]!,
      true,
    );
  }
  return aceSha256Hex(bytes);
}

function sumSafe(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total)) throw new RangeError(`${label} overflowed`);
  }
  return total;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function errorValue(error: unknown): unknown {
  return error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
    : Object.freeze({ error: String(error) });
}
