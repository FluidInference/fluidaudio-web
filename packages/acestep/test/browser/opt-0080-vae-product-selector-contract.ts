import type {
  AceVaeSchedulingReceipt,
  AceVaeWindowSchedulingReceipt,
} from "../../src/api.js";
import {
  ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  type AceOpt0080VaeSchedulingEvidence,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  OPT_0080_PRODUCT_REQUEST,
} from "./opt-0080-product-integration-contract.js";

export const OPT_0080_VAE_PRODUCT_GATE_KIND = "vae-selector" as const;

export const OPT_0080_VAE_PRODUCT_ARM_ORDER = Object.freeze([
  Object.freeze({
    id: "control" as const,
    vaeSchedulingPolicyOverride: "depth1-epoch1" as const,
  }),
  Object.freeze({
    id: "candidate" as const,
    vaeSchedulingPolicyOverride:
      ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  }),
  Object.freeze({
    id: "production" as const,
    vaeSchedulingPolicyOverride: undefined,
  }),
]);

export const OPT_0080_VAE_PRODUCT_REQUEST = OPT_0080_PRODUCT_REQUEST;
export const OPT_0080_VAE_PRODUCT_FINAL_LATENT_SHA256 =
  "527cdc7e560691f21383f3b06a4a85f7f41ba92e93e6357b7be75f115a5c9e07" as const;
export const OPT_0080_VAE_PRODUCT_RAW_SHA256 =
  "c4152aa56bcf81236b60cb2dbea3976b4a7f4d800af001b9bfbbbd52dda6e82b" as const;
export const OPT_0080_VAE_PRODUCT_SEAM_SHA256 =
  "17540647f319a947860bf7402721e3e942f089cfc90a8d8aa20d8191ff889830" as const;
export const OPT_0080_VAE_PRODUCT_WAV_SHA256 =
  "c088385a6b4dabc30215d122b3a4da8406611f1a7d6d1255eba2846aa7e24e4a" as const;

export type Opt0080VaeProductArm =
  (typeof OPT_0080_VAE_PRODUCT_ARM_ORDER)[number];
export type Opt0080VaeProductArmId = Opt0080VaeProductArm["id"];

export function requireOpt0080VaeProductSchedulingReceipt(
  value: AceVaeSchedulingReceipt,
  armId: Opt0080VaeProductArmId,
): AceVaeSchedulingReceipt {
  const expectedPolicy = armId === "production"
    ? ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY
    : null;
  const expectedOverride = armId === "control"
    ? "depth1-epoch1"
    : armId === "candidate"
    ? ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY
    : null;
  const expectedSelection = armId === "production"
    ? "production"
    : "benchmark-override";
  if (
    value.schema !== "ace-vae-window-scheduling-receipt-v1" ||
    value.selectedProductionPolicy !== expectedPolicy ||
    value.benchmarkPolicyOverride !== expectedOverride ||
    value.windows.length !== 2
  ) throw new Error(`OPT-0080 VAE ${armId} scheduling receipt changed`);

  const first = value.windows[0];
  const remainder = value.windows[1];
  if (
    first === undefined || remainder === undefined ||
    !validWindow(first, 0, 2_314, 35_498, expectedSelection,
      armId === "control" ? "depth1-epoch1" : "depth2-phase-epoch4") ||
    !validWindow(remainder, 1, 214, 3_342, expectedSelection,
      "depth1-epoch1")
  ) throw new Error(`OPT-0080 VAE ${armId} per-window selector changed`);
  return value;
}

export function summarizeOpt0080VaeProductSchedulingEvidence(
  values: readonly AceOpt0080VaeSchedulingEvidence[],
  receipt: AceVaeSchedulingReceipt,
): readonly Readonly<Record<string, unknown>>[] {
  if (values.length !== receipt.windows.length) {
    throw new Error("OPT-0080 VAE forced evidence inventory changed");
  }
  return Object.freeze(values.map((value, index) => {
    const expected = receipt.windows[index];
    if (expected === undefined) {
      throw new Error("OPT-0080 VAE forced evidence lost its window");
    }
    requireSchedulingEvidence(value, expected);
    return Object.freeze({
      schema: value.schema,
      windowIndex: value.windowIndex,
      schedulingProfile: value.schedulingProfile,
      decoderQuantumCount: value.decoderQuantumCount,
      quantaPerCommandBuffer: value.quantaPerCommandBuffer,
      decoderCommandBufferCount: value.decoderCommandBufferCount,
      readbackCommandBufferCount: value.readbackCommandBufferCount,
      totalCommandBufferCount: value.totalCommandBufferCount,
      commandBuffersSubmitted: value.commandBuffersSubmitted,
      completionFenceRequestedCount: value.completionFenceRequestedCount,
      completionFenceSettledCount: value.completionFenceSettledCount,
      completionFenceRejectedCount: value.completionFenceRejectedCount,
      trueQueueDrainCount: value.trueQueueDrainCount,
      completionEpochCount: value.completionEpochCount,
      cooperativeIdleTurns: value.cooperativeIdleTurns,
      requestedCooperativeIdleMs: value.requestedCooperativeIdleMs,
      maximumOutstandingCommandBuffers:
        value.maximumOutstandingCommandBuffers,
      commandCompletionOrderExact: true,
      completionEpochTopologyExact: true,
    });
  }));
}

function requireSchedulingEvidence(
  value: AceOpt0080VaeSchedulingEvidence,
  expected: AceVaeWindowSchedulingReceipt,
): void {
  if (
    value.schema !== "ace-opt-0080-vae-window-scheduling-v1" ||
    value.windowIndex !== expected.windowIndex ||
    value.schedulingProfile !== expected.schedulingProfile ||
    value.decoderQuantumCount !== expected.decoderQuantumCount ||
    value.quantaPerCommandBuffer !== expected.quantaPerCommandBuffer ||
    value.decoderCommandBufferCount !== expected.decoderCommandBufferCount ||
    value.readbackCommandBufferCount !== 1 ||
    value.totalCommandBufferCount !== expected.totalCommandBufferCount ||
    !Number.isFinite(value.schedulingWallMs) || value.schedulingWallMs < 0 ||
    value.commandBuffersSubmitted !== expected.commandBuffersSubmitted ||
    value.completionFenceRequestedCount !== expected.totalCommandBufferCount ||
    value.completionFenceSettledCount !== expected.totalCommandBufferCount ||
    value.completionFenceRejectedCount !== 0 ||
    value.trueQueueDrainCount !== expected.queueDrains ||
    value.completionEpochCount !== expected.queueDrains ||
    value.cooperativeIdleTurns !== expected.cooperativeIdleTurns ||
    value.requestedCooperativeIdleMs !== expected.cooperativeIdleTurns ||
    value.maximumOutstandingCommandBuffers !==
      expected.maximumOutstandingCommandBuffers ||
    value.commandCompletions.length !== expected.totalCommandBufferCount ||
    value.completionEpochs.length !== expected.queueDrains
  ) throw new Error("OPT-0080 VAE forced scheduling topology changed");

  const depthTwo = value.schedulingProfile === "depth2-phase-epoch4";
  for (let index = 0; index < value.commandCompletions.length; index += 1) {
    const completion = value.commandCompletions[index];
    const trueQueueDrain = depthTwo
      ? (index + 1) % 4 === 0 || index === value.totalCommandBufferCount - 1
      : true;
    const completionEpochIndex = depthTwo ? Math.floor(index / 4) : index;
    if (
      completion === undefined || completion.commandBufferIndex !== index ||
      completion.commandKind !==
        (index < value.decoderCommandBufferCount ? "decoder" : "readback") ||
      !Number.isFinite(completion.submitThroughCompletionFenceMs) ||
      completion.submitThroughCompletionFenceMs < 0 ||
      completion.trueQueueDrain !== trueQueueDrain ||
      completion.completionEpochIndex !== completionEpochIndex
    ) throw new Error("OPT-0080 VAE completion-fence order changed");
  }
  for (let index = 0; index < value.completionEpochs.length; index += 1) {
    const epoch = value.completionEpochs[index];
    const first = depthTwo ? index * 4 : index;
    const last = depthTwo
      ? Math.min(first + 3, value.totalCommandBufferCount - 1)
      : index;
    if (
      epoch === undefined || epoch.completionEpochIndex !== index ||
      epoch.phaseIndex !== 0 || epoch.firstCommandBufferIndex !== first ||
      epoch.lastCommandBufferIndex !== last ||
      epoch.commandBufferCount !== last - first + 1 ||
      !Number.isFinite(epoch.submitThroughTrueDrainMs) ||
      epoch.submitThroughTrueDrainMs < 0
    ) throw new Error("OPT-0080 VAE completion epoch changed");
  }
}

function validWindow(
  value: AceVaeWindowSchedulingReceipt,
  windowIndex: number,
  latentWindowFrames: number,
  decoderQuantumCount: number,
  selection: AceVaeWindowSchedulingReceipt["selection"],
  schedulingProfile: AceVaeWindowSchedulingReceipt["schedulingProfile"],
): boolean {
  const depthTwo = schedulingProfile === "depth2-phase-epoch4";
  return value.windowIndex === windowIndex &&
    value.latentWindowFrames === latentWindowFrames &&
    value.decoderQuantumCount === decoderQuantumCount &&
    value.selection === selection &&
    value.schedulingProfile === schedulingProfile &&
    value.quantaPerCommandBuffer === 64 &&
    value.readbackCommandBufferCount === 1 &&
    value.decoderCommandBufferCount ===
      Math.ceil(value.decoderQuantumCount / value.quantaPerCommandBuffer) &&
    value.totalCommandBufferCount === value.decoderCommandBufferCount + 1 &&
    value.commandBuffersSubmitted === value.totalCommandBufferCount &&
    value.queueDrains === (depthTwo
      ? Math.ceil(value.totalCommandBufferCount / 4)
      : value.totalCommandBufferCount) &&
    value.cooperativeIdleTurns === value.queueDrains - 1 &&
    value.maximumOutstandingCommandBuffers === (depthTwo ? 2 : 1);
}
