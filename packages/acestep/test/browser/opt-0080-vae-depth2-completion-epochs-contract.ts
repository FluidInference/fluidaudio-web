import type {
  AceOpt0080VaeSchedulingEvidence,
  AceOpt0080VaeSchedulingProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  planAceVaeChunkedDecode,
} from "../../src/webgpu/vae-chunks.js";
import {
  OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS,
  OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS,
  requireOpt0080Heartbeat,
  requireOpt0080ThermalGate,
  requireOpt0080ThermalTrace,
  type Opt0080HeartbeatCapture,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
} from "./opt-0080-dit-depth2-completion-epochs-contract.js";
import { planOpt0059Shape } from
  "./opt-0059-vae-c2378-short-projection-contract.js";

export const OPT_0080_VAE_SCHEMA =
  "ace-opt-0080-vae-depth2-completion-epochs-gate-v1" as const;
export const OPT_0080_VAE_EXPERIMENT_ID = "OPT-0080" as const;
export const OPT_0080_VAE_LATENT_FRAMES = 2_314 as const;
export const OPT_0080_VAE_MAXIMUM_WINDOW_FRAMES = 2_378 as const;
export const OPT_0080_VAE_OVERLAP_FRAMES = 64 as const;
export const OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER = 64 as const;
export const OPT_0080_VAE_FIXTURE_SHA256 =
  "01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d" as const;
export const OPT_0080_VAE_WAVEFORM_SHA256 =
  "2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c" as const;
export const OPT_0080_VAE_OUTPUT_ELEMENTS = 8_885_760 as const;
export const OPT_0080_VAE_OUTPUT_BYTES = 35_543_040 as const;
export const OPT_0080_VAE_DECODER_QUANTA = 35_498 as const;
export const OPT_0080_VAE_DECODER_COMMAND_BUFFERS = 555 as const;
export const OPT_0080_VAE_TOTAL_COMMAND_BUFFERS = 556 as const;
export const OPT_0080_VAE_CANDIDATE_TRUE_DRAINS = 139 as const;
export const OPT_0080_VAE_CANDIDATE_IDLE_TURNS = 138 as const;
export const OPT_0080_VAE_PROJECTED_TWO_WINDOW_SAVING_GATE_MS = 800 as const;
export const OPT_0080_VAE_HEARTBEAT_INTERVAL_MILLISECONDS =
  OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS;

export type Opt0080VaeArmId = "A1" | "B1" | "B2" | "A2";
export type Opt0080VaeSchedulingProfile = AceOpt0080VaeSchedulingProfile;

export const OPT_0080_VAE_ARM_ORDER = Object.freeze([
  Object.freeze({
    armId: "A1" as const,
    order: 0 as const,
    schedulingProfile: "depth1-epoch1" as const,
  }),
  Object.freeze({
    armId: "B1" as const,
    order: 1 as const,
    schedulingProfile: "depth2-phase-epoch4" as const,
  }),
  Object.freeze({
    armId: "B2" as const,
    order: 2 as const,
    schedulingProfile: "depth2-phase-epoch4" as const,
  }),
  Object.freeze({
    armId: "A2" as const,
    order: 3 as const,
    schedulingProfile: "depth1-epoch1" as const,
  }),
]);

export interface Opt0080VaeTimingSample {
  readonly armId: Opt0080VaeArmId;
  readonly order: 0 | 1 | 2 | 3;
  readonly schedulingProfile: Opt0080VaeSchedulingProfile;
  /** Backend scheduler entry through its final true drain; excludes map. */
  readonly schedulingWallMs: number;
  /** Decode setup plus benchmark evidence/callback work before that boundary. */
  readonly preSchedulingWallMs: number;
  /** Final scheduling callback through detached readback completion. */
  readonly mapAndDetachWallMs: number;
  /** Complete decodeWindow boundary, including scheduling and readback. */
  readonly decodeWindowWallMs: number;
  readonly epochWallSumMs: number;
  readonly outputSha256: string;
  readonly rawU32MismatchCount: 0;
  readonly topology: AceOpt0080VaeSchedulingEvidence;
  readonly heartbeat: Opt0080HeartbeatCapture;
}

export interface Opt0080VaePerformanceSummary {
  readonly fixedOrder: readonly [
    "A1-depth1-epoch1",
    "B1-depth2-phase-epoch4",
    "B2-depth2-phase-epoch4",
    "A2-depth1-epoch1",
  ];
  readonly samples: readonly Opt0080VaeTimingSample[];
  readonly forwardSchedulingImproved: boolean;
  readonly reverseSchedulingImproved: boolean;
  readonly forwardDecodeImproved: boolean;
  readonly reverseDecodeImproved: boolean;
  readonly forwardDecodeSavingMs: number;
  readonly reverseDecodeSavingMs: number;
  readonly aggregateDecodeSpeedup: number;
  readonly projectedTwoWindowSavingMs: number;
  readonly heartbeatAbsolutePassed: boolean;
  readonly heartbeatRelativePassed: boolean;
  readonly wallBoundaryConsistent: boolean;
  readonly classification: "passed" | "failed" | "inconclusive";
  readonly passed: boolean;
}

export interface Opt0080VaeCancellationEvidence {
  readonly scope: "actual-c2314-vae-window";
  readonly schedulingProfile: "depth2-phase-epoch4";
  readonly abortedFromFirstProgressCallback: true;
  readonly progressEventCountAtAbort: 1;
  readonly progressEventCountAfterAbort: 0;
  readonly schedulingEvidenceCallbackCount: 0;
  readonly rejectedWithExactAbortReason: true;
  readonly queueDrainedBeforeRejection: true;
  readonly outputPublished: false;
  readonly unhandledRejectionCount: 0;
  readonly abortThroughRejectionMs: number;
  readonly postCancellationExactProbePassed: true;
}

export function planOpt0080VaeGate(): Readonly<Record<string, unknown>> {
  const plan = planAceVaeChunkedDecode(OPT_0080_VAE_LATENT_FRAMES, {
    chunkFrames: OPT_0080_VAE_MAXIMUM_WINDOW_FRAMES,
    overlapFrames: OPT_0080_VAE_OVERLAP_FRAMES,
  });
  const shape = planOpt0059Shape(OPT_0080_VAE_LATENT_FRAMES);
  const expectedWindow = plan.windows[0];
  if (
    !plan.direct || plan.windows.length !== 1 || expectedWindow === undefined ||
    expectedWindow.latentWindowFrames !== OPT_0080_VAE_LATENT_FRAMES ||
    plan.outputInterleavedElements !== OPT_0080_VAE_OUTPUT_ELEMENTS ||
    plan.outputFloat32Bytes !== OPT_0080_VAE_OUTPUT_BYTES ||
    shape.sequenceQuantumCount !== OPT_0080_VAE_DECODER_QUANTA ||
    shape.decoderCommandBufferCount !==
      OPT_0080_VAE_DECODER_COMMAND_BUFFERS ||
    shape.totalCommandBufferCount !== OPT_0080_VAE_TOTAL_COMMAND_BUFFERS
  ) throw new Error("OPT-0080 C2314 VAE gate geometry changed");
  return Object.freeze({
    latentFrames: OPT_0080_VAE_LATENT_FRAMES,
    chunkFrames: OPT_0080_VAE_MAXIMUM_WINDOW_FRAMES,
    overlapFrames: OPT_0080_VAE_OVERLAP_FRAMES,
    outputElements: plan.outputInterleavedElements,
    outputBytes: plan.outputFloat32Bytes,
    decoderQuantumCount: shape.sequenceQuantumCount,
    quantaPerCommandBuffer: OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER,
    decoderCommandBufferCount: shape.decoderCommandBufferCount,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: shape.totalCommandBufferCount,
    control: Object.freeze({
      completionFences: OPT_0080_VAE_TOTAL_COMMAND_BUFFERS,
      trueQueueDrains: OPT_0080_VAE_TOTAL_COMMAND_BUFFERS,
      cooperativeIdleTurns: OPT_0080_VAE_TOTAL_COMMAND_BUFFERS - 1,
      maximumOutstandingCommandBuffers: 1,
    }),
    candidate: Object.freeze({
      completionFences: OPT_0080_VAE_TOTAL_COMMAND_BUFFERS,
      trueQueueDrains: OPT_0080_VAE_CANDIDATE_TRUE_DRAINS,
      cooperativeIdleTurns: OPT_0080_VAE_CANDIDATE_IDLE_TURNS,
      maximumOutstandingCommandBuffers: 2,
    }),
  });
}

export function requireOpt0080VaeTopology(
  value: AceOpt0080VaeSchedulingEvidence,
  profile: Opt0080VaeSchedulingProfile = value.schedulingProfile,
): AceOpt0080VaeSchedulingEvidence {
  const candidate = profile === "depth2-phase-epoch4";
  const expectedDrains = candidate
    ? OPT_0080_VAE_CANDIDATE_TRUE_DRAINS
    : OPT_0080_VAE_TOTAL_COMMAND_BUFFERS;
  const expectedIdles = candidate
    ? OPT_0080_VAE_CANDIDATE_IDLE_TURNS
    : OPT_0080_VAE_TOTAL_COMMAND_BUFFERS - 1;
  const expectedOutstanding = candidate ? 2 : 1;
  const expectedEpochSize = candidate ? 4 : 1;
  if (
    value.schema !== "ace-opt-0080-vae-window-scheduling-v1" ||
    value.windowIndex !== 0 || value.schedulingProfile !== profile ||
    value.decoderQuantumCount !== OPT_0080_VAE_DECODER_QUANTA ||
    value.quantaPerCommandBuffer !==
      OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER ||
    value.decoderCommandBufferCount !==
      OPT_0080_VAE_DECODER_COMMAND_BUFFERS ||
    value.readbackCommandBufferCount !== 1 ||
    value.totalCommandBufferCount !== OPT_0080_VAE_TOTAL_COMMAND_BUFFERS ||
    value.commandBuffersSubmitted !== OPT_0080_VAE_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceRequestedCount !==
      OPT_0080_VAE_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceSettledCount !==
      OPT_0080_VAE_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceRejectedCount !== 0 ||
    value.trueQueueDrainCount !== expectedDrains ||
    value.completionEpochCount !== expectedDrains ||
    value.cooperativeIdleTurns !== expectedIdles ||
    value.requestedCooperativeIdleMs !== expectedIdles ||
    value.maximumOutstandingCommandBuffers !== expectedOutstanding ||
    value.commandCompletions.length !== OPT_0080_VAE_TOTAL_COMMAND_BUFFERS ||
    value.completionEpochs.length !== expectedDrains ||
    value.commandCompletions.some((completion, index) =>
      completion.commandBufferIndex !== index ||
      completion.commandKind !==
        (index < OPT_0080_VAE_DECODER_COMMAND_BUFFERS
          ? "decoder"
          : "readback") ||
      !validDuration(completion.submitThroughCompletionFenceMs) ||
      completion.trueQueueDrain !==
        (candidate ? (index + 1) % 4 === 0 ||
          index + 1 === OPT_0080_VAE_TOTAL_COMMAND_BUFFERS : true) ||
      completion.completionEpochIndex !== Math.floor(index / expectedEpochSize)
    ) ||
    value.completionEpochs.some((epoch, index) => {
      const first = index * expectedEpochSize;
      const count = Math.min(
        expectedEpochSize,
        OPT_0080_VAE_TOTAL_COMMAND_BUFFERS - first,
      );
      return epoch.completionEpochIndex !== index || epoch.phaseIndex !== 0 ||
        epoch.firstCommandBufferIndex !== first ||
        epoch.lastCommandBufferIndex !== first + count - 1 ||
        epoch.commandBufferCount !== count ||
        !validDuration(epoch.submitThroughTrueDrainMs);
    })
  ) throw new Error("OPT-0080 C2314 VAE command/fence/epoch topology changed");
  return value;
}

export function requireOpt0080VaeCancellation(
  value: Opt0080VaeCancellationEvidence,
): Opt0080VaeCancellationEvidence {
  if (
    value.scope !== "actual-c2314-vae-window" ||
    value.schedulingProfile !== "depth2-phase-epoch4" ||
    value.abortedFromFirstProgressCallback !== true ||
    value.progressEventCountAtAbort !== 1 ||
    value.progressEventCountAfterAbort !== 0 ||
    value.schedulingEvidenceCallbackCount !== 0 ||
    value.rejectedWithExactAbortReason !== true ||
    value.queueDrainedBeforeRejection !== true ||
    value.outputPublished !== false ||
    value.unhandledRejectionCount !== 0 ||
    !Number.isFinite(value.abortThroughRejectionMs) ||
    value.abortThroughRejectionMs < 0 || value.abortThroughRejectionMs > 1_000 ||
    value.postCancellationExactProbePassed !== true
  ) throw new Error("OPT-0080 C2314 VAE cancellation preflight failed");
  return value;
}

export function summarizeOpt0080VaePerformance(
  samples: readonly Opt0080VaeTimingSample[],
): Opt0080VaePerformanceSummary {
  if (
    samples.length !== OPT_0080_VAE_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0080_VAE_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.schedulingProfile !== expected.schedulingProfile ||
        !validDuration(sample.schedulingWallMs) ||
        !validDuration(sample.preSchedulingWallMs) ||
        !validDuration(sample.mapAndDetachWallMs) ||
        !validDuration(sample.decodeWindowWallMs) ||
        !validDuration(sample.epochWallSumMs) ||
        sample.decodeWindowWallMs + 1e-6 < sample.schedulingWallMs ||
        !/^[0-9a-f]{64}$/u.test(sample.outputSha256) ||
        sample.rawU32MismatchCount !== 0 ||
        !withoutThrow(() => requireOpt0080VaeTopology(
          sample.topology,
          sample.schedulingProfile,
        )) || !withoutThrow(() => requireOpt0080Heartbeat(sample.heartbeat));
    })
  ) throw new Error("OPT-0080 C2314 VAE ABBA sample inventory changed");
  const [a1, b1, b2, a2] = samples as readonly [
    Opt0080VaeTimingSample,
    Opt0080VaeTimingSample,
    Opt0080VaeTimingSample,
    Opt0080VaeTimingSample,
  ];
  const forwardSchedulingImproved =
    b1.schedulingWallMs < a1.schedulingWallMs;
  const reverseSchedulingImproved = b2.schedulingWallMs < a2.schedulingWallMs;
  const forwardDecodeImproved = b1.decodeWindowWallMs < a1.decodeWindowWallMs;
  const reverseDecodeImproved = b2.decodeWindowWallMs < a2.decodeWindowWallMs;
  const forwardDecodeSavingMs = a1.decodeWindowWallMs - b1.decodeWindowWallMs;
  const reverseDecodeSavingMs = a2.decodeWindowWallMs - b2.decodeWindowWallMs;
  const aggregateDecodeSpeedup = ratio(
    a1.decodeWindowWallMs + a2.decodeWindowWallMs,
    b1.decodeWindowWallMs + b2.decodeWindowWallMs,
  );
  const projectedTwoWindowSavingMs =
    forwardDecodeSavingMs + reverseDecodeSavingMs;
  const heartbeatAbsolutePassed = samples.every((sample) =>
    sample.heartbeat.maximumGapMilliseconds <=
      OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS
  );
  const heartbeatRelativePassed = heartbeatPairPassed(a1, b1) &&
    heartbeatPairPassed(a2, b2);
  const wallBoundaryConsistent = samples.every((sample) =>
    Math.abs(
      sample.preSchedulingWallMs + sample.schedulingWallMs +
        sample.mapAndDetachWallMs -
        sample.decodeWindowWallMs,
    ) <= 2 && sample.epochWallSumMs <= sample.schedulingWallMs + 1e-6
  );
  const passed = forwardSchedulingImproved && reverseSchedulingImproved &&
    forwardDecodeImproved && reverseDecodeImproved &&
    projectedTwoWindowSavingMs >=
      OPT_0080_VAE_PROJECTED_TWO_WINDOW_SAVING_GATE_MS &&
    heartbeatAbsolutePassed && heartbeatRelativePassed &&
    wallBoundaryConsistent;
  const directions = [
    forwardSchedulingImproved,
    reverseSchedulingImproved,
    forwardDecodeImproved,
    reverseDecodeImproved,
  ];
  const classification = passed
    ? "passed" as const
    : directions.every((value) => value === directions[0]) &&
        wallBoundaryConsistent
      ? "failed" as const
      : "inconclusive" as const;
  return Object.freeze({
    fixedOrder: Object.freeze([
      "A1-depth1-epoch1",
      "B1-depth2-phase-epoch4",
      "B2-depth2-phase-epoch4",
      "A2-depth1-epoch1",
    ]) as Opt0080VaePerformanceSummary["fixedOrder"],
    samples: Object.freeze([...samples]),
    forwardSchedulingImproved,
    reverseSchedulingImproved,
    forwardDecodeImproved,
    reverseDecodeImproved,
    forwardDecodeSavingMs,
    reverseDecodeSavingMs,
    aggregateDecodeSpeedup,
    projectedTwoWindowSavingMs,
    heartbeatAbsolutePassed,
    heartbeatRelativePassed,
    wallBoundaryConsistent,
    classification,
    passed,
  });
}

export {
  requireOpt0080Heartbeat as requireOpt0080VaeHeartbeat,
  requireOpt0080ThermalGate as requireOpt0080VaeThermalGate,
  requireOpt0080ThermalTrace as requireOpt0080VaeThermalTrace,
};
export type {
  Opt0080HeartbeatCapture as Opt0080VaeHeartbeatCapture,
  Opt0080ThermalGate as Opt0080VaeThermalGate,
  Opt0080ThermalTrace as Opt0080VaeThermalTrace,
};

function heartbeatPairPassed(
  control: Opt0080VaeTimingSample,
  candidate: Opt0080VaeTimingSample,
): boolean {
  return candidate.heartbeat.p99GapMilliseconds <= Math.max(
    100,
    1.25 * control.heartbeat.p99GapMilliseconds,
  ) && candidate.heartbeat.maximumGapMilliseconds <= Math.max(
    500,
    1.25 * control.heartbeat.maximumGapMilliseconds,
  );
}

function validDuration(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function ratio(control: number, candidate: number): number {
  return candidate === 0
    ? control === 0 ? 1 : Number.POSITIVE_INFINITY
    : control / candidate;
}

function withoutThrow(run: () => unknown): boolean {
  try {
    run();
    return true;
  } catch {
    return false;
  }
}
