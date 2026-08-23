import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";
import type {
  AceOpt0080VaeSchedulingEvidence,
  AceOpt0080VaeSchedulingProfile,
} from "../src/webgpu/vae-fp16-backend.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  OPT_0080_VAE_ARM_ORDER,
  OPT_0080_VAE_CANDIDATE_IDLE_TURNS,
  OPT_0080_VAE_CANDIDATE_TRUE_DRAINS,
  OPT_0080_VAE_DECODER_COMMAND_BUFFERS,
  OPT_0080_VAE_DECODER_QUANTA,
  OPT_0080_VAE_FIXTURE_SHA256,
  OPT_0080_VAE_OUTPUT_BYTES,
  OPT_0080_VAE_OUTPUT_ELEMENTS,
  OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER,
  OPT_0080_VAE_TOTAL_COMMAND_BUFFERS,
  OPT_0080_VAE_WAVEFORM_SHA256,
  planOpt0080VaeGate,
  requireOpt0080VaeCancellation,
  requireOpt0080VaeTopology,
  summarizeOpt0080VaePerformance,
  type Opt0080VaeHeartbeatCapture,
  type Opt0080VaeTimingSample,
} from "./browser/opt-0080-vae-depth2-completion-epochs-contract.js";

const WORKER_SOURCE = source(
  "./browser/opt-0080-vae-depth2-completion-epochs-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0080-vae-depth2-completion-epochs.ts",
);
const RESULT_SOURCE = source(
  "./browser/opt-0080-vae-depth2-completion-epochs-result.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0080-vae-depth2-completion-epochs.html",
);

describe("OPT-0080 C2314 VAE depth-two completion epochs", () => {
  it("freezes the exact C2314/qpc64 one-window topology", () => {
    expect(OPT_0080_VAE_FIXTURE_SHA256).toBe(
      "01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d",
    );
    expect(OPT_0080_VAE_WAVEFORM_SHA256).toBe(
      "2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c",
    );
    expect(OPT_0080_VAE_OUTPUT_ELEMENTS).toBe(8_885_760);
    expect(OPT_0080_VAE_OUTPUT_BYTES).toBe(35_543_040);
    expect(OPT_0080_VAE_DECODER_QUANTA).toBe(35_498);
    expect(OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER).toBe(64);
    expect(OPT_0080_VAE_DECODER_COMMAND_BUFFERS).toBe(555);
    expect(OPT_0080_VAE_TOTAL_COMMAND_BUFFERS).toBe(556);
    expect(planOpt0080VaeGate()).toEqual({
      latentFrames: 2_314,
      chunkFrames: 2_378,
      overlapFrames: 64,
      outputElements: 8_885_760,
      outputBytes: 35_543_040,
      decoderQuantumCount: 35_498,
      quantaPerCommandBuffer: 64,
      decoderCommandBufferCount: 555,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 556,
      control: {
        completionFences: 556,
        trueQueueDrains: 556,
        cooperativeIdleTurns: 555,
        maximumOutstandingCommandBuffers: 1,
      },
      candidate: {
        completionFences: 556,
        trueQueueDrains: 139,
        cooperativeIdleTurns: 138,
        maximumOutstandingCommandBuffers: 2,
      },
    });
  });

  it("requires 556 fences while reducing only true drains/idles", () => {
    const control = topology("depth1-epoch1");
    const candidate = topology("depth2-phase-epoch4");
    expect(requireOpt0080VaeTopology(control)).toBe(control);
    expect(requireOpt0080VaeTopology(candidate)).toBe(candidate);
    expect(candidate).toMatchObject({
      commandBuffersSubmitted: 556,
      completionFenceRequestedCount: 556,
      completionFenceSettledCount: 556,
      completionFenceRejectedCount: 0,
      trueQueueDrainCount: OPT_0080_VAE_CANDIDATE_TRUE_DRAINS,
      completionEpochCount: 139,
      cooperativeIdleTurns: OPT_0080_VAE_CANDIDATE_IDLE_TURNS,
      maximumOutstandingCommandBuffers: 2,
    });
    expect(candidate.commandCompletions).toHaveLength(556);
    expect(candidate.completionEpochs).toHaveLength(139);
    expect(() => requireOpt0080VaeTopology({
      ...candidate,
      trueQueueDrainCount: 140,
    })).toThrow(/topology/);
  });

  it("places guards at both active C2314 and allocation C2378 ends", () => {
    const active = planAceVaeDecoder(2_314);
    const allocation = planAceVaeDecoder(2_378);
    const activeEnds = [
      active.inputElements * Float32Array.BYTES_PER_ELEMENT,
      active.inputElements * Uint16Array.BYTES_PER_ELEMENT,
      ...Array(3).fill(
        active.maximumActivationElements * Uint16Array.BYTES_PER_ELEMENT,
      ),
      active.outputElements * Float32Array.BYTES_PER_ELEMENT,
    ];
    const allocationEnds = [
      allocation.inputElements * Float32Array.BYTES_PER_ELEMENT,
      allocation.inputElements * Uint16Array.BYTES_PER_ELEMENT,
      ...Array(3).fill(
        allocation.maximumActivationElements * Uint16Array.BYTES_PER_ELEMENT,
      ),
      allocation.outputElements * Float32Array.BYTES_PER_ELEMENT,
    ];
    expect(activeEnds).toHaveLength(6);
    expect(activeEnds.every((end, index) => end + 256 <= allocationEnds[index]!))
      .toBe(true);
    expect(WORKER_SOURCE).toContain(
      "scheme: \"c2314-active-end-and-c2378-allocation-end-guards-v1\"",
    );
    expect(WORKER_SOURCE).toContain("guardedRegionCount: regions.length");
    expect(WORKER_SOURCE).toContain(
      "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d",
    );
  });

  it("gates both ABBA directions and at least 800 ms across two windows", () => {
    const samples = [
      sample("A1", 0, "depth1-epoch1", 15_000, 16_000),
      sample("B1", 1, "depth2-phase-epoch4", 14_000, 15_000),
      sample("B2", 2, "depth2-phase-epoch4", 14_200, 15_200),
      sample("A2", 3, "depth1-epoch1", 15_100, 16_100),
    ] as const;
    const summary = summarizeOpt0080VaePerformance(samples);
    expect(summary).toMatchObject({
      fixedOrder: [
        "A1-depth1-epoch1",
        "B1-depth2-phase-epoch4",
        "B2-depth2-phase-epoch4",
        "A2-depth1-epoch1",
      ],
      forwardSchedulingImproved: true,
      reverseSchedulingImproved: true,
      forwardDecodeImproved: true,
      reverseDecodeImproved: true,
      forwardDecodeSavingMs: 1_000,
      reverseDecodeSavingMs: 900,
      projectedTwoWindowSavingMs: 1_900,
      heartbeatAbsolutePassed: true,
      heartbeatRelativePassed: true,
      classification: "passed",
      passed: true,
    });
    const thresholdMiss = summarizeOpt0080VaePerformance([
      sample("A1", 0, "depth1-epoch1", 15_000, 16_000),
      sample("B1", 1, "depth2-phase-epoch4", 14_800, 15_700),
      sample("B2", 2, "depth2-phase-epoch4", 14_800, 15_700),
      sample("A2", 3, "depth1-epoch1", 15_000, 16_000),
    ]);
    expect(thresholdMiss.projectedTwoWindowSavingMs).toBe(600);
    expect(thresholdMiss.passed).toBe(false);
    expect(thresholdMiss.classification).toBe("failed");
  });

  it("requires terminal candidate cancellation and exact owner reuse", () => {
    const valid = {
      scope: "actual-c2314-vae-window" as const,
      schedulingProfile: "depth2-phase-epoch4" as const,
      abortedFromFirstProgressCallback: true as const,
      progressEventCountAtAbort: 1 as const,
      progressEventCountAfterAbort: 0 as const,
      schedulingEvidenceCallbackCount: 0 as const,
      rejectedWithExactAbortReason: true as const,
      queueDrainedBeforeRejection: true as const,
      outputPublished: false as const,
      unhandledRejectionCount: 0 as const,
      abortThroughRejectionMs: 4.2,
      postCancellationExactProbePassed: true as const,
    };
    expect(requireOpt0080VaeCancellation(valid)).toBe(valid);
    expect(() => requireOpt0080VaeCancellation({
      ...valid,
      progressEventCountAfterAbort: 1 as 0,
    })).toThrow(/cancellation/);
  });

  it("binds the persistent actual backend, manual thermal/heartbeat, and ignored artifact", () => {
    expect(WORKER_SOURCE).toContain(
      "AceOpt0011Fp16VaeChunkGpuBackend.create({",
    );
    expect(WORKER_SOURCE).toContain(
      "quantaPerCommandBuffer: OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER",
    );
    expect(WORKER_SOURCE).toContain(
      "submissionPolicy: \"depth1-epoch1\"",
    );
    expect(WORKER_SOURCE).toContain(
      "backend.decodeWindow(window, signal, {",
    );
    expect(WORKER_SOURCE).toContain(
      "schedulingProfile: \"depth2-phase-epoch4\"",
    );
    expect(WORKER_SOURCE).toContain(
      "persistentDeviceBackendAndPackageOwnerCount: 1",
    );
    expect(WORKER_SOURCE).toContain(
      "size: Number(descriptor.size) + GPU_GUARD_BYTES",
    );
    expect(WORKER_SOURCE).toContain(
      "encoder.copyBufferToBuffer(\n        region.record.buffer,\n        region.offset",
    );
    expect(WORKER_SOURCE).toContain(
      "await audit.initializeGpuGuards(signal);",
    );
    expect(WORKER_SOURCE).toContain(
      "terminalFenceResolvedAtRejection = state.terminalFenceResolved",
    );
    expect(WORKER_SOURCE).toContain(
      "outputSha256 !== OPT_0080_VAE_WAVEFORM_SHA256",
    );
    expect(WORKER_SOURCE).not.toContain("gpuGuardCanariesAvailable");
    expect(PAGE_SOURCE).toContain(
      "OPT_0080_VAE_HEARTBEAT_INTERVAL_MILLISECONDS",
    );
    expect(PAGE_SOURCE).toContain("type: \"complete-thermal\"");
    expect(RESULT_SOURCE).toContain(
      "value.controlOutputSha256 !== OPT_0080_VAE_WAVEFORM_SHA256",
    );
    expect(RESULT_SOURCE).toContain("arm.guardEvidence[\"passed\"] !== true");
    expect(HTML_SOURCE).toContain("A1 depth1, B1 depth2/epoch4, B2 depth2/epoch4, A2");
    expect(HTML_SOURCE).toContain(
      "optimization/artifacts/OPT-0080/vae-scheduler/",
    );
    expect(HTML_SOURCE).toContain("all twelve regions are checked");
    expect(HTML_SOURCE).toContain("does not make a product, listening, or under-one-minute");
    expect(OPT_0080_VAE_ARM_ORDER.map((arm) => arm.armId)).toEqual([
      "A1", "B1", "B2", "A2",
    ]);
  });
});

function topology(
  schedulingProfile: AceOpt0080VaeSchedulingProfile,
): AceOpt0080VaeSchedulingEvidence {
  const candidate = schedulingProfile === "depth2-phase-epoch4";
  const epochSize = candidate ? 4 : 1;
  const epochCount = candidate ? 139 : 556;
  return Object.freeze({
    schema: "ace-opt-0080-vae-window-scheduling-v1",
    windowIndex: 0,
    schedulingProfile,
    decoderQuantumCount: 35_498,
    quantaPerCommandBuffer: 64,
    decoderCommandBufferCount: 555,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 556,
    schedulingWallMs: candidate ? 14_000 : 15_000,
    commandBuffersSubmitted: 556,
    completionFenceRequestedCount: 556,
    completionFenceSettledCount: 556,
    completionFenceRejectedCount: 0,
    trueQueueDrainCount: epochCount,
    completionEpochCount: epochCount,
    cooperativeIdleTurns: epochCount - 1,
    requestedCooperativeIdleMs: epochCount - 1,
    maximumOutstandingCommandBuffers: candidate ? 2 : 1,
    commandCompletions: Object.freeze(Array.from({ length: 556 }, (_, index) =>
      Object.freeze({
        commandBufferIndex: index,
        commandKind: index < 555 ? "decoder" as const : "readback" as const,
        submitThroughCompletionFenceMs: 10,
        trueQueueDrain: candidate
          ? (index + 1) % 4 === 0 || index === 555
          : true,
        completionEpochIndex: Math.floor(index / epochSize),
      })
    )),
    completionEpochs: Object.freeze(Array.from(
      { length: epochCount },
      (_, index) => {
        const first = index * epochSize;
        const count = Math.min(epochSize, 556 - first);
        return Object.freeze({
          completionEpochIndex: index,
          phaseIndex: 0 as const,
          firstCommandBufferIndex: first,
          lastCommandBufferIndex: first + count - 1,
          commandBufferCount: count,
          submitThroughTrueDrainMs: 10,
        });
      },
    )),
  });
}

function sample(
  armId: "A1" | "B1" | "B2" | "A2",
  order: 0 | 1 | 2 | 3,
  profile: AceOpt0080VaeSchedulingProfile,
  schedulingWallMs: number,
  decodeWindowWallMs: number,
): Opt0080VaeTimingSample {
  return Object.freeze({
    armId,
    order,
    schedulingProfile: profile,
    schedulingWallMs,
    preSchedulingWallMs: 0,
    mapAndDetachWallMs: decodeWindowWallMs - schedulingWallMs,
    decodeWindowWallMs,
    epochWallSumMs: schedulingWallMs - 100,
    outputSha256: "a".repeat(64),
    rawU32MismatchCount: 0,
    topology: topology(profile),
    heartbeat: heartbeat(),
  });
}

function heartbeat(): Opt0080VaeHeartbeatCapture {
  return Object.freeze({
    intervalMilliseconds: 50,
    startedAtEpochMilliseconds: 1_000_000,
    completedAtEpochMilliseconds: 1_001_000,
    gapsMilliseconds: Object.freeze([50, 49, 51, 50]),
    maximumGapMilliseconds: 51,
    p99GapMilliseconds: 51,
  });
}

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}
