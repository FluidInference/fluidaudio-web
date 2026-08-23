import {
  compareAceRawAudioSnapshots,
  type AceRawAudioSnapshotComparison,
} from "../../src/runtime/audio-output.js";
import type { AceVaeSchedulingReceipt } from "../../src/api.js";
import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_PRODUCT_ARM_ORDER,
  OPT_0080_PRODUCT_AUDIO_FRAMES,
  OPT_0080_PRODUCT_LATENT_FRAMES,
  OPT_0080_PRODUCT_RAW_BYTES,
  OPT_0080_PRODUCT_REQUEST_BYTES,
  OPT_0080_PRODUCT_REQUEST_SHA256,
  OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES,
  OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME,
  OPT_0080_PRODUCT_STITCH_SEAM_LATENT_FRAME,
  OPT_0080_PRODUCT_WAV_BYTES,
  type Opt0080ProductArmId,
} from "./opt-0080-product-integration-contract.js";
import {
  OPT_0080_VAE_PRODUCT_ARM_ORDER,
  OPT_0080_VAE_PRODUCT_GATE_KIND,
  requireOpt0080VaeProductSchedulingReceipt,
} from "./opt-0080-vae-product-selector-contract.js";

declare global {
  interface Window {
    __ACE_OPT0080_PRODUCT_RESULT__?: Readonly<Record<string, unknown>>;
    __ACE_OPT0080_VAE_PRODUCT_SELECTOR_RESULT__?:
      Readonly<Record<string, unknown>>;
  }
}

type RecordValue = Readonly<Record<string, unknown>>;
type RunKind = Opt0080ProductArmId | "cancellation";
type ProductGateKind = "dit" | typeof OPT_0080_VAE_PRODUCT_GATE_KIND;

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ArmReadyMessage {
  readonly type: "arm-ready";
  readonly armId: Opt0080ProductArmId;
  readonly summary: RecordValue;
  readonly finalLatentRawU32?: Uint32Array;
  readonly rawSnapshot?: Blob;
  readonly wav: Blob;
}

interface ArmReleasedMessage {
  readonly type: "arm-released";
  readonly armId: Opt0080ProductArmId;
  readonly lifecycle: RecordValue;
}

interface CancellationCompleteMessage {
  readonly type: "cancellation-complete";
  readonly summary: RecordValue;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: RecordValue;
}

type WorkerMessage = ProgressMessage | ArmReadyMessage | ArmReleasedMessage |
  CancellationCompleteMessage | FailedMessage;

interface HeartbeatSnapshot extends RecordValue {
  readonly schema: "ace-opt-0080-product-page-heartbeat-v1";
  readonly runKind: RunKind;
  readonly intervalMilliseconds: 50;
  readonly timerTickCount: number;
  readonly observedGapCount: number;
  readonly maximumGapMilliseconds: number;
  readonly p99GapMilliseconds: number;
  readonly startedAtEpochMilliseconds: number;
  readonly workerTerminatedAtEpochMilliseconds: number;
  readonly passed: true;
}

interface ReleasedArm {
  readonly lifecycle: RecordValue;
  readonly heartbeat: HeartbeatSnapshot;
}

interface RetainedArm {
  readonly armId: Opt0080ProductArmId;
  readonly summary: RecordValue;
  readonly finalLatentRawU32?: Uint32Array;
  readonly rawSnapshot?: Blob;
  readonly wav: Blob;
  readonly release: () => Promise<ReleasedArm>;
}

interface CompletedCancellation {
  readonly summary: RecordValue;
  readonly heartbeat: HeartbeatSnapshot;
}

const HEARTBEAT_INTERVAL_MILLISECONDS = 50 as const;
const HEARTBEAT_MAXIMUM_GAP_MILLISECONDS = 500 as const;
const COMPARISON_BLOCK_BYTES = 1_048_576 as const;
const FINAL_LATENT_WORDS = OPT_0080_PRODUCT_LATENT_FRAMES * 64;
const FINAL_LATENT_BYTES = FINAL_LATENT_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const AUDIO_FRAME_BYTES = 2 * Float32Array.BYTES_PER_ELEMENT;
const SEAM_RADIUS_AUDIO_FRAMES =
  OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES * 1_920;
const SEAM_START_AUDIO_FRAME =
  OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME - SEAM_RADIUS_AUDIO_FRAMES;
const SEAM_END_AUDIO_FRAME =
  OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME + SEAM_RADIUS_AUDIO_FRAMES;
const SEAM_BYTES =
  (SEAM_END_AUDIO_FRAME - SEAM_START_AUDIO_FRAME) * AUDIO_FRAME_BYTES;
const FIXED_SUCCESS_ORDER = ["control", "candidate", "production"] as const;
const productGateAttribute = document.body.dataset.productGate;
const productGate: ProductGateKind =
  productGateAttribute === OPT_0080_VAE_PRODUCT_GATE_KIND
    ? OPT_0080_VAE_PRODUCT_GATE_KIND
    : "dit";
const vaeSelectorGate = productGate === OPT_0080_VAE_PRODUCT_GATE_KIND;

const runButton = element<HTMLButtonElement>("#run");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0018RunIdentity | undefined;
let executingWorker: Worker | undefined;
let downloadUrl: string | undefined;
let settled = false;
let workersCreated = 0;
let workersTerminated = 0;
const liveWorkers = new Set<Worker>();
const retainedArms = new Set<RetainedArm>();

try {
  identity = parseOpt0018RunIdentity(new URL(location.href).searchParams);
  requireArmContract();
  progress.textContent = vaeSelectorGate
    ? "ready — fixed VAE A → B → production → cancellation; no timing claim"
    : "ready — fixed A → B → production → cancellation gate; no timing claim";
  runButton.disabled = false;
} catch (error) {
  runButton.disabled = true;
  publishFailure(error);
}

runButton.addEventListener("click", () => {
  if (identity === undefined || settled) return;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  void runGate(identity).catch(async (error) => {
    await releaseAllRetainedArms();
    publishFailure(error);
  });
}, { once: true });

window.addEventListener("beforeunload", () => {
  for (const worker of liveWorkers) worker.terminate();
  liveWorkers.clear();
  executingWorker = undefined;
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
}, { once: true });

async function runGate(runIdentity: Opt0018RunIdentity): Promise<void> {
  progress.textContent = vaeSelectorGate
    ? "control: starting forced VAE depth-one product arm"
    : "control: starting forced depth-one product arm";
  let control: RetainedArm | undefined = await runSuccessArm(
    "control",
    runIdentity,
  );
  const controlRunSummary = control.summary;

  progress.textContent = vaeSelectorGate
    ? "candidate: starting forced exact-C2314 VAE product arm"
    : "candidate: starting forced depth-two product arm";
  let candidate: RetainedArm | undefined = await runSuccessArm(
    "candidate",
    runIdentity,
  );
  const candidateRunSummary = candidate.summary;

  progress.textContent =
    "control/candidate: bounded final-latent, raw, seam, and WAV comparison";
  const controlCandidate = await compareArmPair(
    "control-candidate",
    control,
    candidate,
  );
  const controlReleased = await releaseArm(control);
  const controlSummary = Object.freeze({
    ...controlRunSummary,
    lifecycle: controlReleased.lifecycle,
  });
  const controlHeartbeat = controlReleased.heartbeat;
  control = undefined;
  await nextTask();

  progress.textContent =
    "production: starting ordinary selector with no policy override";
  let production: RetainedArm | undefined = await runSuccessArm(
    "production",
    runIdentity,
  );
  const productionRunSummary = production.summary;

  progress.textContent =
    "candidate/production: proving the ordinary selector is byte-exact";
  const candidateProduction = vaeSelectorGate
    ? await compareVaeOrdinaryArm(candidate, production)
    : await compareArmPair(
        "candidate-production",
        candidate,
        production,
      );
  if (vaeSelectorGate) {
    requireVaeQueueDrainTopology(
      controlRunSummary,
      candidateRunSummary,
      productionRunSummary,
    );
  } else {
    requireQueueDrainTopology(
      controlRunSummary,
      candidateRunSummary,
      productionRunSummary,
    );
  }
  const candidateReleased = await releaseArm(candidate);
  const candidateSummary = Object.freeze({
    ...candidateRunSummary,
    lifecycle: candidateReleased.lifecycle,
  });
  const candidateHeartbeat = candidateReleased.heartbeat;
  candidate = undefined;
  const productionReleased = await releaseArm(production);
  const productionSummary = Object.freeze({
    ...productionRunSummary,
    lifecycle: productionReleased.lifecycle,
  });
  const productionHeartbeat = productionReleased.heartbeat;
  production = undefined;
  await nextTask();

  progress.textContent =
    "cancellation: fresh ordinary seam-free run, aborting at public denoise 0/8";
  const cancellation = await runCancellation(runIdentity);
  requireCancellationSummary(cancellation.summary, runIdentity);

  if (workersCreated !== 4 || workersTerminated !== 4) {
    throw new Error("OPT-0080 product gate did not use exactly four fresh workers");
  }
  const receipt = Object.freeze({
    schema: vaeSelectorGate
      ? "ace-opt-0080-vae-product-selector-page-v1"
      : "ace-opt-0080-product-integration-page-v1",
    experimentId: "OPT-0080",
    status: "passed",
    authority: Object.freeze({
      scope: vaeSelectorGate
        ? "post-integration direct 96-second VAE selector exactness/topology plus ordinary product cancellation"
        : "post-integration direct 96-second product correctness and cancellation",
      timingClaim: false,
      thermalClaim: false,
      plannerClaim: false,
      ...(vaeSelectorGate
        ? {
            selectorNegativeCasesCoveredByRuntimeUnitTests: true,
            ordinaryProductionArmWasSeamFree: true,
          }
        : {}),
      unchangedRetryPerformed: false,
    }),
    identity: runIdentity,
    fixedOneWayOrder: Object.freeze([
      ...FIXED_SUCCESS_ORDER,
      "cancellation",
    ]),
    arms: Object.freeze({
      control: controlSummary,
      candidate: candidateSummary,
      production: productionSummary,
    }),
    comparisons: Object.freeze({
      controlCandidate,
      candidateProduction,
    }),
    cancellation: cancellation.summary,
    heartbeat: Object.freeze({
      control: controlHeartbeat,
      candidate: candidateHeartbeat,
      production: productionHeartbeat,
      cancellation: cancellation.heartbeat,
      intervalMilliseconds: HEARTBEAT_INTERVAL_MILLISECONDS,
      maximumAllowedGapMilliseconds: HEARTBEAT_MAXIMUM_GAP_MILLISECONDS,
      allPassed: true,
    }),
    lifecycle: Object.freeze({
      freshWorkerCount: workersCreated,
      terminatedWorkerCount: workersTerminated,
      maximumLargeArmPayloadsRetained: 2,
      explicitReleaseAfterComparison: true,
      largePayloadsReleasedBeforeCancellation: true,
      automaticRetryCount: 0,
      workerOrderWasOneWay: true,
    }),
  });
  publish(receipt);
}

async function runSuccessArm(
  armId: Opt0080ProductArmId,
  runIdentity: Opt0018RunIdentity,
): Promise<RetainedArm> {
  if (executingWorker !== undefined) {
    throw new Error("OPT-0080 product workers executed concurrently");
  }
  const worker = createWorker(armId);
  const heartbeat = startHeartbeat(armId);
  executingWorker = worker;
  let ready = false;
  let terminal = false;
  let releaseRequested = false;
  let retained: RetainedArm | undefined;
  let resolveRelease!: (value: ReleasedArm) => void;
  let rejectRelease!: (error: unknown) => void;
  const releaseCompletion = new Promise<ReleasedArm>((resolve, reject) => {
    resolveRelease = resolve;
    rejectRelease = reject;
  });
  void releaseCompletion.catch(() => undefined);

  return await new Promise<RetainedArm>((resolveReady, rejectReady) => {
    const terminate = (): HeartbeatSnapshot => {
      if (terminal) throw new Error(`OPT-0080 ${armId} worker terminated twice`);
      terminal = true;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.terminate();
      workersTerminated += 1;
      liveWorkers.delete(worker);
      if (executingWorker === worker) executingWorker = undefined;
      if (retained !== undefined) retainedArms.delete(retained);
      return heartbeat.stopAfterWorkerTermination();
    };
    const rejectTerminal = (error: unknown): void => {
      let failure = error;
      try {
        terminate();
      } catch (heartbeatError) {
        failure = heartbeatError;
      }
      if (ready) rejectRelease(failure);
      else rejectReady(failure);
    };
    const onMessage = (event: MessageEvent<WorkerMessage>): void => {
      const message = event.data;
      if (message.type === "progress") {
        progress.textContent = message.message;
        return;
      }
      if (message.type === "failed") {
        rejectTerminal(new Error(
          `OPT-0080 ${armId} worker failed: ${JSON.stringify(message.error)}`,
        ));
        return;
      }
      if (message.type === "arm-ready") {
        if (ready) {
          rejectTerminal(new Error("OPT-0080 product arm became ready twice"));
          return;
        }
        retained = Object.freeze({
          armId,
          summary: message.summary,
          ...(message.finalLatentRawU32 === undefined
            ? {}
            : { finalLatentRawU32: message.finalLatentRawU32 }),
          ...(message.rawSnapshot === undefined
            ? {}
            : { rawSnapshot: message.rawSnapshot }),
          wav: message.wav,
          release() {
            if (!releaseRequested) {
              releaseRequested = true;
              try {
                worker.postMessage({ type: "release", armId });
              } catch (error) {
                rejectTerminal(error);
              }
            }
            return releaseCompletion;
          },
        });
        ready = true;
        executingWorker = undefined;
        retainedArms.add(retained);
        try {
          if (message.armId !== armId) {
            throw new Error("OPT-0080 product arm ready order changed");
          }
          requireArmPayload(armId, retained, runIdentity);
          resolveReady(retained);
        } catch (error) {
          // The committed OPFS output still needs the ordinary release command.
          // The outer gate failure handler releases every registered arm before
          // publishing failure or terminating its worker.
          rejectReady(error);
        }
        return;
      }
      if (message.type === "arm-released") {
        if (
          !ready || !releaseRequested || message.armId !== armId ||
          retained === undefined
        ) {
          rejectTerminal(new Error("OPT-0080 product release order changed"));
          return;
        }
        try {
          const workerHeartbeat = terminate();
          resolveRelease(Object.freeze({
            lifecycle: message.lifecycle,
            heartbeat: workerHeartbeat,
          }));
        } catch (error) {
          rejectRelease(error);
        }
        return;
      }
      rejectTerminal(new Error(
        `OPT-0080 ${armId} emitted an unexpected ${message.type} message`,
      ));
    };
    const onError = (event: ErrorEvent): void => {
      event.preventDefault();
      rejectTerminal(event.error ?? new Error(event.message));
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    try {
      worker.postMessage({
        type: "run",
        runKind: armId,
        identity: runIdentity,
        productGate,
      });
    } catch (error) {
      rejectTerminal(error);
    }
  });
}

async function runCancellation(
  runIdentity: Opt0018RunIdentity,
): Promise<CompletedCancellation> {
  if (executingWorker !== undefined || retainedArms.size !== 0) {
    throw new Error("OPT-0080 cancellation requires every product output released");
  }
  const worker = createWorker("cancellation");
  const heartbeat = startHeartbeat("cancellation");
  executingWorker = worker;
  return await new Promise<CompletedCancellation>((resolve, reject) => {
      let terminal = false;
      const terminate = <Value>(continuation: (
        heartbeat: HeartbeatSnapshot,
      ) => Value): Value | undefined => {
        if (terminal) return undefined;
        terminal = true;
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
        worker.terminate();
        workersTerminated += 1;
        liveWorkers.delete(worker);
        executingWorker = undefined;
        return continuation(heartbeat.stopAfterWorkerTermination());
      };
      const rejectTerminal = (error: unknown): void => {
        try {
          terminate(() => reject(error));
        } catch (heartbeatError) {
          reject(heartbeatError);
        }
      };
      const onMessage = (event: MessageEvent<WorkerMessage>): void => {
        const message = event.data;
        if (message.type === "progress") {
          progress.textContent = message.message;
          return;
        }
        if (message.type === "failed") {
          rejectTerminal(new Error(
            `OPT-0080 cancellation worker failed: ${JSON.stringify(message.error)}`,
          ));
          return;
        }
        if (message.type !== "cancellation-complete") {
          rejectTerminal(new Error(
            `OPT-0080 cancellation emitted an unexpected ${message.type} message`,
          ));
          return;
        }
        try {
          terminate((workerHeartbeat) => resolve(Object.freeze({
            summary: message.summary,
            heartbeat: workerHeartbeat,
          })));
        } catch (error) {
          reject(error);
        }
      };
      const onError = (event: ErrorEvent): void => {
        event.preventDefault();
        rejectTerminal(event.error ?? new Error(event.message));
      };
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      try {
        worker.postMessage({
          type: "run",
          runKind: "cancellation",
          identity: runIdentity,
          productGate,
        });
      } catch (error) {
        rejectTerminal(error);
      }
    });
}

function createWorker(runKind: RunKind): Worker {
  const worker = new Worker(
    new URL("./opt-0080-product-integration-worker.ts", import.meta.url),
    { type: "module", name: `ace-opt-0080-product-${runKind}` },
  );
  workersCreated += 1;
  liveWorkers.add(worker);
  return worker;
}

async function releaseArm(arm: RetainedArm): Promise<ReleasedArm> {
  const released = await arm.release();
  requireReleasedLifecycle(arm.armId, released.lifecycle);
  retainedArms.delete(arm);
  return released;
}

async function releaseAllRetainedArms(): Promise<void> {
  for (const arm of [...retainedArms]) {
    try {
      await releaseArm(arm);
    } catch {
      // Preserve the primary gate failure; the worker's own finally ran cleanup.
    }
  }
}

function requireReleasedLifecycle(
  armId: Opt0080ProductArmId,
  lifecycle: RecordValue,
): void {
  if (
    lifecycle.releaseBeforeDispose !== true ||
    lifecycle.unhandledRejectionCount !== 0 ||
    lifecycle.workerReadyToTerminate !== true ||
    !Number.isSafeInteger(lifecycle.releaseCompletedAtEpochMilliseconds) ||
    !Number.isSafeInteger(lifecycle.disposeCompletedAtEpochMilliseconds)
  ) throw new Error(`OPT-0080 ${armId} release lifecycle changed`);
}

async function compareArmPair(
  pairId: "control-candidate" | "candidate-production",
  left: RetainedArm,
  right: RetainedArm,
): Promise<RecordValue> {
  if (
    left.finalLatentRawU32 === undefined ||
    right.finalLatentRawU32 === undefined ||
    left.rawSnapshot === undefined || right.rawSnapshot === undefined
  ) throw new Error(`OPT-0080 ${pairId} retained comparison input changed`);
  const summaryIdentity = requireStableSummaryFields(left.summary, right.summary);
  const leftSeamGeometry = deriveSeamGeometry(left.summary);
  const rightSeamGeometry = deriveSeamGeometry(right.summary);
  if (!sameJson(leftSeamGeometry, rightSeamGeometry)) {
    throw new Error(`OPT-0080 ${pairId} VAE seam geometry changed`);
  }
  const finalLatent = compareU32(
    left.finalLatentRawU32,
    right.finalLatentRawU32,
  );
  if (finalLatent.exactU32MismatchCount !== 0) {
    throw new Error(`OPT-0080 ${pairId} final latent was not raw-U32 exact`);
  }

  const fullRaw = await compareAceRawAudioSnapshots(
    left.rawSnapshot,
    right.rawSnapshot,
    { blockBytes: COMPARISON_BLOCK_BYTES },
  );
  requireExactRawComparison(fullRaw, OPT_0080_PRODUCT_RAW_BYTES, pairId);

  const seamStartByte = leftSeamGeometry.startAudioFrame * AUDIO_FRAME_BYTES;
  const seamEndByte = leftSeamGeometry.endAudioFrame * AUDIO_FRAME_BYTES;
  const seamRaw = await compareAceRawAudioSnapshots(
    left.rawSnapshot.slice(seamStartByte, seamEndByte),
    right.rawSnapshot.slice(seamStartByte, seamEndByte),
    { blockBytes: COMPARISON_BLOCK_BYTES },
  );
  requireExactRawComparison(seamRaw, SEAM_BYTES, `${pairId} seam`);

  const wav = await compareBlobBytes(
    left.wav,
    right.wav,
    COMPARISON_BLOCK_BYTES,
  );
  if (
    wav.byteLength !== OPT_0080_PRODUCT_WAV_BYTES ||
    wav.exactByteMismatchCount !== 0
  ) throw new Error(`OPT-0080 ${pairId} WAV was not byte-exact`);

  const suppliedHashes = requireSuppliedHashesEqual(
    left.summary,
    right.summary,
  );
  return Object.freeze({
    schema: "ace-opt-0080-product-pair-comparison-v1",
    pairId,
    finalLatent,
    fullRaw,
    seamRaw: Object.freeze({
      ...seamRaw,
      ...leftSeamGeometry,
    }),
    wav,
    suppliedHashes,
    stableSummaryIdentity: summaryIdentity,
    boundedComparisonBlockBytes: COMPARISON_BLOCK_BYTES,
    passed: true,
  });
}

async function compareVaeOrdinaryArm(
  candidate: RetainedArm,
  production: RetainedArm,
): Promise<RecordValue> {
  const summaryIdentity = requireStableSummaryFields(
    candidate.summary,
    production.summary,
  );
  const wav = await compareBlobBytes(
    candidate.wav,
    production.wav,
    COMPARISON_BLOCK_BYTES,
  );
  if (
    wav.byteLength !== OPT_0080_PRODUCT_WAV_BYTES ||
    wav.exactByteMismatchCount !== 0 ||
    nestedSha256(candidate.summary, "wav") !==
      nestedSha256(production.summary, "wav")
  ) throw new Error("OPT-0080 VAE candidate/ordinary WAV was not byte-exact");
  const candidateTopology = schedulingTopology(candidate.summary);
  const productionTopology = schedulingTopology(production.summary);
  if (!sameJson(candidateTopology, productionTopology)) {
    throw new Error("OPT-0080 VAE forced/ordinary window topology changed");
  }
  return Object.freeze({
    schema: "ace-opt-0080-vae-product-ordinary-comparison-v1",
    pairId: "candidate-production",
    wav,
    wavSha256: nestedSha256(candidate.summary, "wav"),
    perWindowTopology: candidateTopology,
    stableSummaryIdentity: summaryIdentity,
    boundedComparisonBlockBytes: COMPARISON_BLOCK_BYTES,
    ordinaryRetainedFinalLatentOrRaw: false,
    passed: true,
  });
}

function schedulingTopology(summary: RecordValue): readonly RecordValue[] {
  const receipt = requireRecord(summary.vaeScheduling, "VAE scheduling");
  if (!Array.isArray(receipt.windows)) {
    throw new Error("OPT-0080 VAE scheduling windows changed");
  }
  return Object.freeze(receipt.windows.map((value, index) => {
    const window = requireRecord(value, `VAE scheduling window ${index}`);
    return Object.freeze({
      windowIndex: window.windowIndex,
      latentWindowFrames: window.latentWindowFrames,
      schedulingProfile: window.schedulingProfile,
      decoderQuantumCount: window.decoderQuantumCount,
      quantaPerCommandBuffer: window.quantaPerCommandBuffer,
      decoderCommandBufferCount: window.decoderCommandBufferCount,
      readbackCommandBufferCount: window.readbackCommandBufferCount,
      totalCommandBufferCount: window.totalCommandBufferCount,
      commandBuffersSubmitted: window.commandBuffersSubmitted,
      queueDrains: window.queueDrains,
      cooperativeIdleTurns: window.cooperativeIdleTurns,
      maximumOutstandingCommandBuffers:
        window.maximumOutstandingCommandBuffers,
    });
  }));
}

function compareU32(left: Uint32Array, right: Uint32Array): RecordValue {
  if (
    left.length !== FINAL_LATENT_WORDS || right.length !== FINAL_LATENT_WORDS ||
    left.byteLength !== FINAL_LATENT_BYTES || right.byteLength !== FINAL_LATENT_BYTES
  ) throw new Error("OPT-0080 final latent shape changed");
  let exactU32MismatchCount = 0;
  let firstMismatch: Readonly<Record<string, number>> | undefined;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] === right[index]) continue;
    exactU32MismatchCount += 1;
    firstMismatch ??= Object.freeze({
      index,
      left: left[index]!,
      right: right[index]!,
    });
  }
  return Object.freeze({
    schema: "ace-opt-0080-final-latent-u32-comparison-v1",
    wordCount: left.length,
    byteLength: left.byteLength,
    exactU32MismatchCount,
    firstMismatch: firstMismatch ?? null,
  });
}

async function compareBlobBytes(
  left: Blob,
  right: Blob,
  blockBytes: number,
): Promise<RecordValue> {
  if (left.size !== right.size) {
    throw new Error("OPT-0080 WAV lengths differ");
  }
  let exactByteMismatchCount = 0;
  let firstMismatch: Readonly<Record<string, number>> | undefined;
  for (let offset = 0; offset < left.size; offset += blockBytes) {
    const end = Math.min(left.size, offset + blockBytes);
    const [leftBytes, rightBytes] = await Promise.all([
      left.slice(offset, end).arrayBuffer().then((value) => new Uint8Array(value)),
      right.slice(offset, end).arrayBuffer().then((value) => new Uint8Array(value)),
    ]);
    for (let index = 0; index < leftBytes.length; index += 1) {
      if (leftBytes[index] === rightBytes[index]) continue;
      exactByteMismatchCount += 1;
      firstMismatch ??= Object.freeze({
        byteOffset: offset + index,
        left: leftBytes[index]!,
        right: rightBytes[index]!,
      });
    }
  }
  return Object.freeze({
    schema: "ace-opt-0080-wav-byte-comparison-v1",
    byteLength: left.size,
    exactByteMismatchCount,
    firstMismatch: firstMismatch ?? null,
  });
}

function requireArmPayload(
  armId: Opt0080ProductArmId,
  completed: RetainedArm,
  runIdentity: Opt0018RunIdentity,
): void {
  if (vaeSelectorGate) {
    requireVaeArmPayload(armId, completed, runIdentity);
    return;
  }
  if (
    !(completed.finalLatentRawU32 instanceof Uint32Array) ||
    !(completed.rawSnapshot instanceof Blob) || !(completed.wav instanceof Blob) ||
    completed.finalLatentRawU32.length !== FINAL_LATENT_WORDS ||
    completed.rawSnapshot.size !== OPT_0080_PRODUCT_RAW_BYTES ||
    completed.wav.size !== OPT_0080_PRODUCT_WAV_BYTES
  ) throw new Error(`OPT-0080 ${armId} transferred payload changed`);
  const summary = completed.summary;
  const expectedPolicy = armId === "control"
    ? "depth1-epoch1"
    : "depth2-phase-epoch4";
  if (
    summary.schema !== "ace-opt-0080-product-arm-v1" ||
    summary.armId !== armId ||
    summary.selectedProductionPolicy !== "depth2-phase-epoch4" ||
    summary.effectiveSubmissionPolicy !== expectedPolicy ||
    !sameJson(summary.identity, runIdentity)
  ) throw new Error(`OPT-0080 ${armId} policy or run identity changed`);
  const request = requireRecord(summary.request, `${armId} request`);
  const finalLatent = requireRecord(
    summary.finalLatent,
    `${armId} final latent`,
  );
  const raw = requireRecord(summary.raw, `${armId} raw`);
  const seam = requireRecord(summary.seam, `${armId} seam`);
  const wav = requireRecord(summary.wav, `${armId} wav`);
  const finalLatentScan = requireRecord(
    finalLatent.scan,
    `${armId} final latent scan`,
  );
  const rawStats = requireRecord(raw.stats, `${armId} raw stats`);
  const seamScan = requireRecord(seam.scan, `${armId} seam scan`);
  const wavScan = requireRecord(wav.scan, `${armId} WAV scan`);
  if (
    request.byteLength !== OPT_0080_PRODUCT_REQUEST_BYTES ||
    request.sha256 !== OPT_0080_PRODUCT_REQUEST_SHA256 ||
    finalLatent.elementCount !== FINAL_LATENT_WORDS ||
    finalLatent.byteLength !== FINAL_LATENT_BYTES ||
    raw.byteLength !== OPT_0080_PRODUCT_RAW_BYTES ||
    seam.byteLength !== SEAM_BYTES || wav.byteLength !== OPT_0080_PRODUCT_WAV_BYTES ||
    !validSha256(finalLatent.sha256) || !validSha256(raw.sha256) ||
    !validSha256(seam.sha256) || !validSha256(wav.sha256) ||
    finalLatentScan.nonFiniteCount !== 0 ||
    typeof finalLatentScan.nonzeroCount !== "number" ||
    finalLatentScan.nonzeroCount <= 0 ||
    rawStats.finiteSamples !== OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    rawStats.outputInterleavedElements !== OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    rawStats.windowsDecoded !== 2 ||
    typeof rawStats.peak !== "number" || !Number.isFinite(rawStats.peak) ||
    rawStats.peak <= 0 || seamScan.nonFiniteCount !== 0 ||
    wavScan.nonFiniteCount !== 0
  ) throw new Error(`OPT-0080 ${armId} summary contract changed`);
  const metadata = requireRecord(summary.metadata, `${armId} metadata`);
  if (
    metadata.frameCount !== OPT_0080_PRODUCT_AUDIO_FRAMES ||
    metadata.audioStorageIdWasSafe !== true
  ) throw new Error(`OPT-0080 ${armId} public metadata changed`);
}

function requireVaeArmPayload(
  armId: Opt0080ProductArmId,
  completed: RetainedArm,
  runIdentity: Opt0018RunIdentity,
): void {
  if (!(completed.wav instanceof Blob) ||
    completed.wav.size !== OPT_0080_PRODUCT_WAV_BYTES) {
    throw new Error(`OPT-0080 VAE ${armId} WAV payload changed`);
  }
  const forced = armId !== "production";
  if (
    forced !== (completed.finalLatentRawU32 instanceof Uint32Array) ||
    forced !== (completed.rawSnapshot instanceof Blob) ||
    (forced && (
      completed.finalLatentRawU32!.length !== FINAL_LATENT_WORDS ||
      completed.rawSnapshot!.size !== OPT_0080_PRODUCT_RAW_BYTES
    ))
  ) throw new Error(`OPT-0080 VAE ${armId} retained payload changed`);
  const summary = completed.summary;
  if (
    summary.schema !== "ace-opt-0080-vae-product-selector-arm-v1" ||
    summary.armId !== armId || !sameJson(summary.identity, runIdentity) ||
    summary.evidenceMode !== (forced
      ? "forced-retained-output"
      : "seam-free-ordinary")
  ) throw new Error(`OPT-0080 VAE ${armId} identity or evidence mode changed`);
  requireOpt0080VaeProductSchedulingReceipt(
    summary.vaeScheduling as AceVaeSchedulingReceipt,
    armId,
  );
  const metrics = requireRecord(summary.metrics, `${armId} metrics`);
  if (!sameJson(metrics.vaeScheduling, summary.vaeScheduling)) {
    throw new Error(`OPT-0080 VAE ${armId} public scheduling receipt changed`);
  }
  const request = requireRecord(summary.request, `${armId} request`);
  const wav = requireRecord(summary.wav, `${armId} WAV`);
  const wavScan = requireRecord(wav.scan, `${armId} WAV scan`);
  const metadata = requireRecord(summary.metadata, `${armId} metadata`);
  if (
    request.byteLength !== OPT_0080_PRODUCT_REQUEST_BYTES ||
    request.sha256 !== OPT_0080_PRODUCT_REQUEST_SHA256 ||
    wav.byteLength !== OPT_0080_PRODUCT_WAV_BYTES ||
    !validSha256(wav.sha256) || wavScan.nonFiniteCount !== 0 ||
    metadata.frameCount !== OPT_0080_PRODUCT_AUDIO_FRAMES ||
    metadata.audioStorageIdWasSafe !== true
  ) throw new Error(`OPT-0080 VAE ${armId} public summary changed`);
  if (!forced) {
    if (
      summary.finalLatent !== undefined || summary.raw !== undefined ||
      summary.seam !== undefined || summary.vaeSchedulingEvidence !== undefined
    ) throw new Error("OPT-0080 ordinary VAE arm retained diagnostic evidence");
    return;
  }
  const finalLatent = requireRecord(
    summary.finalLatent,
    `${armId} final latent`,
  );
  const raw = requireRecord(summary.raw, `${armId} raw`);
  const rawStats = requireRecord(raw.stats, `${armId} raw stats`);
  const seam = requireRecord(summary.seam, `${armId} seam`);
  const seamScan = requireRecord(seam.scan, `${armId} seam scan`);
  if (
    summary.selectedProductionDitPolicy !== "depth2-phase-epoch4" ||
    summary.effectiveDitSubmissionPolicy !== "depth2-phase-epoch4" ||
    !Array.isArray(summary.vaeSchedulingEvidence) ||
    summary.vaeSchedulingEvidence.length !== 2 ||
    finalLatent.elementCount !== FINAL_LATENT_WORDS ||
    finalLatent.byteLength !== FINAL_LATENT_BYTES ||
    raw.byteLength !== OPT_0080_PRODUCT_RAW_BYTES ||
    seam.byteLength !== SEAM_BYTES ||
    !validSha256(finalLatent.sha256) || !validSha256(raw.sha256) ||
    !validSha256(seam.sha256) ||
    rawStats.finiteSamples !== OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    rawStats.outputInterleavedElements !== OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    rawStats.windowsDecoded !== 2 || seamScan.nonFiniteCount !== 0
  ) throw new Error(`OPT-0080 VAE ${armId} retained summary changed`);
}

function requireCancellationSummary(
  summary: RecordValue,
  runIdentity: Opt0018RunIdentity,
): void {
  const elapsed = summary.abortObservationThroughRejectionAndCleanupMs;
  if (
    summary.schema !== "ace-opt-0080-product-cancellation-v1" ||
    !sameJson(summary.identity, runIdentity) ||
    summary.ordinaryProductionSelector !== true ||
    summary.opt0080ProductRunOmitted !== true ||
    summary.triggerCount !== 1 ||
    !Number.isSafeInteger(summary.progressCountAtAbort) ||
    (summary.progressCountAtAbort as number) < 1 ||
    summary.publicProgressAfterAbortCount !== 0 ||
    summary.abortReasonIdentityPreserved !== true ||
    typeof elapsed !== "number" || !Number.isFinite(elapsed) ||
    elapsed < 0 || elapsed > 1_000 || summary.postDitStageCount !== 0 ||
    summary.outputCount !== 0 || summary.evidenceCount !== 0 ||
    summary.fatalDiagnosticCount !== 0 ||
    summary.unhandledRejectionCount !== 0 ||
    summary.workerReadyToTerminate !== true
  ) throw new Error("OPT-0080 ordinary cancellation contract changed");
}

function requireStableSummaryFields(
  left: RecordValue,
  right: RecordValue,
): RecordValue {
  const fields = [
    "identity",
    "request",
    "metadata",
    "runtime",
    "vaePlan",
    "diagnosticCodes",
  ] as const;
  for (const field of fields) {
    if (!sameJson(left[field], right[field])) {
      throw new Error(`OPT-0080 cross-arm ${field} changed`);
    }
  }
  const leftProgress = requireRecord(left.progress, "left progress");
  const rightProgress = requireRecord(right.progress, "right progress");
  const progressStageFields = [
    "initializationStages",
    "generationStages",
  ] as const;
  for (const field of progressStageFields) {
    if (!sameJson(leftProgress[field], rightProgress[field])) {
      throw new Error(`OPT-0080 cross-arm progress ${field} changed`);
    }
  }
  return Object.freeze({
    fields: Object.freeze([...fields]),
    progressStageFields: Object.freeze([...progressStageFields]),
    policyDependentProgressEventCountsExcluded: true,
    allExact: true,
  });
}

function deriveSeamGeometry(summary: RecordValue): RecordValue & {
  readonly startAudioFrame: number;
  readonly endAudioFrame: number;
} {
  const plan = requireRecord(summary.vaePlan, "VAE plan");
  if (!Array.isArray(plan.windows) || plan.windows.length !== 2) {
    throw new Error("OPT-0080 VAE seam plan changed");
  }
  const stitchWindows = plan.windows.slice(1).map((value, index) =>
    requireRecord(value, `VAE stitch window ${index}`)
  );
  const stitch = stitchWindows[0]!;
  const seamLatentFrame = stitch.coreStartLatentFrame;
  const seamAudioFrame = stitch.outputStartAudioFrame;
  if (
    stitchWindows.length !== 1 ||
    seamLatentFrame !== OPT_0080_PRODUCT_STITCH_SEAM_LATENT_FRAME ||
    seamAudioFrame !== OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME
  ) throw new Error("OPT-0080 derived VAE seam identity changed");
  const radiusAudioFrames =
    OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES * 1_920;
  return Object.freeze({
    seamLatentFrame,
    seamAudioFrame,
    radiusLatentFrames: OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES,
    startAudioFrame: seamAudioFrame - radiusAudioFrames,
    endAudioFrame: seamAudioFrame + radiusAudioFrames,
  });
}

function requireSuppliedHashesEqual(
  left: RecordValue,
  right: RecordValue,
): RecordValue {
  const hashes = Object.freeze({
    finalLatentSha256: nestedSha256(left, "finalLatent"),
    rawSha256: nestedSha256(left, "raw"),
    seamSha256: nestedSha256(left, "seam"),
    wavSha256: nestedSha256(left, "wav"),
  });
  if (
    hashes.finalLatentSha256 !== nestedSha256(right, "finalLatent") ||
    hashes.rawSha256 !== nestedSha256(right, "raw") ||
    hashes.seamSha256 !== nestedSha256(right, "seam") ||
    hashes.wavSha256 !== nestedSha256(right, "wav")
  ) throw new Error("OPT-0080 cross-arm supplied SHA-256 identity changed");
  return Object.freeze({ ...hashes, allExact: true });
}

function requireQueueDrainTopology(
  control: RecordValue,
  candidate: RecordValue,
  production: RecordValue,
): void {
  const a = queueDrains(control);
  const b = queueDrains(candidate);
  const p = queueDrains(production);
  if (a <= b || b !== p) {
    throw new Error("OPT-0080 product queue-drain topology changed");
  }
}

function requireVaeQueueDrainTopology(
  control: RecordValue,
  candidate: RecordValue,
  production: RecordValue,
): void {
  const controlWindows = schedulingTopology(control);
  const candidateWindows = schedulingTopology(candidate);
  const productionWindows = schedulingTopology(production);
  const windowDrains = (windows: readonly RecordValue[]): number =>
    windows.reduce((sum, window) => {
      const value = window.queueDrains;
      if (!Number.isSafeInteger(value) || (value as number) < 1) {
        throw new Error("OPT-0080 VAE window drain count changed");
      }
      return sum + (value as number);
    }, 0);
  if (
    windowDrains(controlWindows) !== 610 ||
    windowDrains(candidateWindows) !== 193 ||
    windowDrains(productionWindows) !== 193 ||
    !sameJson(candidateWindows, productionWindows)
  ) throw new Error("OPT-0080 VAE per-window queue topology changed");
  const a = queueDrains(control);
  const b = queueDrains(candidate);
  const p = queueDrains(production);
  if (a - b !== 417 || b !== p) {
    throw new Error("OPT-0080 VAE aggregate queue-drain topology changed");
  }
}

function queueDrains(summary: RecordValue): number {
  const metrics = requireRecord(summary.metrics, "generation metrics");
  const value = metrics.cooperativeGpuQueueDrains;
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new Error("OPT-0080 queue-drain metric changed");
  }
  return value as number;
}

function requireExactRawComparison(
  comparison: AceRawAudioSnapshotComparison,
  expectedBytes: number,
  label: string,
): void {
  if (
    comparison.byteLength !== expectedBytes ||
    comparison.sampleCount !== expectedBytes / Float32Array.BYTES_PER_ELEMENT ||
    comparison.exactU32MismatchCount !== 0 ||
    comparison.leftNonFiniteCount !== 0 ||
    comparison.rightNonFiniteCount !== 0 ||
    comparison.maximumAbsoluteDifference !== 0 ||
    comparison.meanAbsoluteDifference !== 0 ||
    comparison.rootMeanSquareDifference !== 0
  ) throw new Error(`OPT-0080 ${label} raw snapshot was not exact and finite`);
}

function startHeartbeat(runKind: RunKind) {
  const startedAtEpochMilliseconds = Date.now();
  const gapsMilliseconds: number[] = [];
  let lastAt = performance.now();
  let timerTickCount = 0;
  let stopped = false;
  const observe = (): void => {
    const now = performance.now();
    gapsMilliseconds.push(now - lastAt);
    lastAt = now;
  };
  const timer = window.setInterval(() => {
    timerTickCount += 1;
    observe();
  }, HEARTBEAT_INTERVAL_MILLISECONDS);
  return Object.freeze({
    stopAfterWorkerTermination(): HeartbeatSnapshot {
      if (stopped) throw new Error("OPT-0080 heartbeat stopped twice");
      stopped = true;
      observe();
      window.clearInterval(timer);
      const sorted = [...gapsMilliseconds].sort((left, right) => left - right);
      const maximumGapMilliseconds = sorted.at(-1) ?? Number.POSITIVE_INFINITY;
      const p99GapMilliseconds = sorted[
        Math.max(0, Math.ceil(sorted.length * 0.99) - 1)
      ] ?? Number.POSITIVE_INFINITY;
      if (
        timerTickCount < 1 || !Number.isFinite(maximumGapMilliseconds) ||
        maximumGapMilliseconds < 0 ||
        maximumGapMilliseconds > HEARTBEAT_MAXIMUM_GAP_MILLISECONDS
      ) throw new Error(`OPT-0080 ${runKind} page heartbeat gate failed`);
      return Object.freeze({
        schema: "ace-opt-0080-product-page-heartbeat-v1",
        runKind,
        intervalMilliseconds: HEARTBEAT_INTERVAL_MILLISECONDS,
        timerTickCount,
        observedGapCount: gapsMilliseconds.length,
        maximumGapMilliseconds,
        p99GapMilliseconds,
        startedAtEpochMilliseconds,
        workerTerminatedAtEpochMilliseconds: Date.now(),
        passed: true,
      });
    },
  });
}

function requireArmContract(): void {
  if (productGateAttribute !== undefined &&
    productGateAttribute !== OPT_0080_VAE_PRODUCT_GATE_KIND) {
    throw new Error("OPT-0080 product gate kind changed");
  }
  const arms = vaeSelectorGate
    ? OPT_0080_VAE_PRODUCT_ARM_ORDER
    : OPT_0080_PRODUCT_ARM_ORDER;
  if (
    arms.length !== 3 ||
    arms.some((arm, index) =>
      arm.id !== FIXED_SUCCESS_ORDER[index]
    ) ||
    (vaeSelectorGate
      ? OPT_0080_VAE_PRODUCT_ARM_ORDER[2]?.vaeSchedulingPolicyOverride !==
        undefined
      : OPT_0080_PRODUCT_ARM_ORDER[2]?.submissionPolicyOverride !== undefined)
  ) throw new Error("OPT-0080 product arm contract changed");
}

function nestedSha256(summary: RecordValue, field: string): string {
  const value = requireRecord(summary[field], field).sha256;
  if (!validSha256(value)) throw new Error(`OPT-0080 ${field} SHA-256 changed`);
  return value;
}

function validSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function requireRecord(value: unknown, label: string): RecordValue {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`OPT-0080 ${label} must be a record`);
  }
  return value as RecordValue;
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function publish(receipt: RecordValue): void {
  settled = true;
  for (const worker of liveWorkers) worker.terminate();
  liveWorkers.clear();
  executingWorker = undefined;
  if (vaeSelectorGate) {
    window.__ACE_OPT0080_VAE_PRODUCT_SELECTOR_RESULT__ = receipt;
  } else {
    window.__ACE_OPT0080_PRODUCT_RESULT__ = receipt;
  }
  const passed = receipt.status === "passed";
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? vaeSelectorGate
      ? "PASSED — C2314-only VAE production selection is exact and cancellable"
      : "PASSED — depth-two production selection is byte-exact and cancellable"
    : "FAILED — no retry was attempted; inspect the receipt";
  const json = JSON.stringify(receipt, null, 2);
  result.textContent = json;
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([json], {
    type: "application/json",
  }));
  download.href = downloadUrl;
  download.hidden = false;
}

function publishFailure(error: unknown): void {
  if (settled) return;
  publish(Object.freeze({
    schema: vaeSelectorGate
      ? "ace-opt-0080-vae-product-selector-page-failure-v1"
      : "ace-opt-0080-product-integration-page-failure-v1",
    experimentId: "OPT-0080",
    status: "failed",
    identity: identity ?? null,
    automaticRetryCount: 0,
    error: serializeOpt0018Failure(error),
  }));
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0080 product element ${selector}`);
  return value;
}
