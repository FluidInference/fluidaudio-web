import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import type {
  AceVaeSchedulingReceipt,
  AceVaeSchedulingSelection,
  AceVaeWindowSchedulingReceipt,
} from "../src/api.js";
import {
  ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  type AceOpt0080VaeSchedulingEvidence,
} from "../src/webgpu/vae-fp16-backend.js";
import { planAceOpt0011Fp16VaeChunkDispatches } from
  "../src/webgpu/vae-fp16-decoder.js";
import {
  OPT_0080_PRODUCT_REQUEST,
} from "./browser/opt-0080-product-integration-contract.js";
import {
  OPT_0080_VAE_PRODUCT_ARM_ORDER,
  OPT_0080_VAE_PRODUCT_FINAL_LATENT_SHA256,
  OPT_0080_VAE_PRODUCT_GATE_KIND,
  OPT_0080_VAE_PRODUCT_RAW_SHA256,
  OPT_0080_VAE_PRODUCT_REQUEST,
  OPT_0080_VAE_PRODUCT_SEAM_SHA256,
  OPT_0080_VAE_PRODUCT_WAV_SHA256,
  requireOpt0080VaeProductSchedulingReceipt,
  summarizeOpt0080VaeProductSchedulingEvidence,
  type Opt0080VaeProductArmId,
} from "./browser/opt-0080-vae-product-selector-contract.js";

const PAGE_SOURCE = source("./browser/opt-0080-product-integration.ts");
const WORKER_SOURCE = source(
  "./browser/opt-0080-product-integration-worker.ts",
);
const HTML_SOURCE = source("./browser/opt-0080-vae-product-selector.html");

describe("OPT-0080 VAE post-integration product selector gate", () => {
  it("pins the canonical product request and C2314-plus-C214 topology", () => {
    expect(OPT_0080_VAE_PRODUCT_REQUEST).toBe(OPT_0080_PRODUCT_REQUEST);
    expect(OPT_0080_VAE_PRODUCT_GATE_KIND).toBe("vae-selector");
    expect([
      OPT_0080_VAE_PRODUCT_FINAL_LATENT_SHA256,
      OPT_0080_VAE_PRODUCT_RAW_SHA256,
      OPT_0080_VAE_PRODUCT_SEAM_SHA256,
      OPT_0080_VAE_PRODUCT_WAV_SHA256,
    ]).toEqual([
      "527cdc7e560691f21383f3b06a4a85f7f41ba92e93e6357b7be75f115a5c9e07",
      "c4152aa56bcf81236b60cb2dbea3976b4a7f4d800af001b9bfbbbd52dda6e82b",
      "17540647f319a947860bf7402721e3e942f089cfc90a8d8aa20d8191ff889830",
      "c088385a6b4dabc30215d122b3a4da8406611f1a7d6d1255eba2846aa7e24e4a",
    ]);
    expect(OPT_0080_VAE_PRODUCT_ARM_ORDER).toEqual([
      {
        id: "control",
        vaeSchedulingPolicyOverride: "depth1-epoch1",
      },
      {
        id: "candidate",
        vaeSchedulingPolicyOverride:
          ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
      },
      { id: "production", vaeSchedulingPolicyOverride: undefined },
    ]);
    const plan = planAceOpt0011Fp16VaeChunkDispatches(2_400, 2_378, 256);
    expect(plan.topologies.map((topology) => ({
      inputFrames: topology.inputFrames,
      decoderQuantumCount: topology.sequenceQuantumCount,
      decoderCommandBufferCount: Math.ceil(topology.sequenceQuantumCount / 64),
      totalCommandBufferCount:
        Math.ceil(topology.sequenceQuantumCount / 64) + 1,
    }))).toEqual([
      {
        inputFrames: 214,
        decoderQuantumCount: 3_342,
        decoderCommandBufferCount: 53,
        totalCommandBufferCount: 54,
      },
      {
        inputFrames: 2_314,
        decoderQuantumCount: 35_498,
        decoderCommandBufferCount: 555,
        totalCommandBufferCount: 556,
      },
    ]);
  });

  it("accepts only the exact forced and seam-free per-window receipts", () => {
    for (const armId of ["control", "candidate", "production"] as const) {
      const value = receipt(armId);
      expect(requireOpt0080VaeProductSchedulingReceipt(value, armId)).toBe(
        value,
      );
    }
    const production = receipt("production");
    expect(() => requireOpt0080VaeProductSchedulingReceipt({
      ...production,
      windows: [
        production.windows[0]!,
        {
          ...production.windows[1]!,
          schedulingProfile: "depth2-phase-epoch4",
        },
      ],
    }, "production")).toThrow(/per-window selector changed/u);
    const candidate = receipt("candidate");
    expect(() => requireOpt0080VaeProductSchedulingReceipt({
      ...candidate,
      selectedProductionPolicy:
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
    }, "candidate")).toThrow(/scheduling receipt changed/u);
  });

  it("validates all singleton fences and completion epochs, then compacts them", () => {
    for (const armId of ["control", "candidate"] as const) {
      const scheduling = receipt(armId);
      const evidence = scheduling.windows.map(evidenceForWindow);
      const compact = summarizeOpt0080VaeProductSchedulingEvidence(
        evidence,
        scheduling,
      );
      expect(compact).toHaveLength(2);
      expect(compact.map((window) => ({
        profile: window.schedulingProfile,
        total: window.totalCommandBufferCount,
        drains: window.trueQueueDrainCount,
        maximumOutstanding: window.maximumOutstandingCommandBuffers,
      }))).toEqual(armId === "control"
        ? [
            {
              profile: "depth1-epoch1",
              total: 556,
              drains: 556,
              maximumOutstanding: 1,
            },
            {
              profile: "depth1-epoch1",
              total: 54,
              drains: 54,
              maximumOutstanding: 1,
            },
          ]
        : [
            {
              profile: "depth2-phase-epoch4",
              total: 556,
              drains: 139,
              maximumOutstanding: 2,
            },
            {
              profile: "depth1-epoch1",
              total: 54,
              drains: 54,
              maximumOutstanding: 1,
            },
          ]);
      expect(compact.every((window) =>
        window.commandCompletionOrderExact === true &&
        window.completionEpochTopologyExact === true
      )).toBe(true);
    }
    const scheduling = receipt("candidate");
    const evidence = scheduling.windows.map(evidenceForWindow);
    expect(() => summarizeOpt0080VaeProductSchedulingEvidence([
      {
        ...evidence[0]!,
        completionFenceSettledCount:
          evidence[0]!.completionFenceSettledCount - 1,
      },
      evidence[1]!,
    ], scheduling)).toThrow(/forced scheduling topology changed/u);
  });

  it("reuses the retained-output product gate without weakening seam-free selection", () => {
    expect(HTML_SOURCE).toContain('data-product-gate="vae-selector"');
    expect(HTML_SOURCE).toContain("not a timing or thermal comparison");
    expect(HTML_SOURCE).toContain("forced exact-C2314 candidate selection");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0080-product-integration.ts"',
    );
    expect(PAGE_SOURCE).toContain(
      "__ACE_OPT0080_VAE_PRODUCT_SELECTOR_RESULT__",
    );
    expect(PAGE_SOURCE).toContain("compareArmPair(");
    expect(PAGE_SOURCE).toContain("compareVaeOrdinaryArm(");
    expect(PAGE_SOURCE).toContain("ordinaryRetainedFinalLatentOrRaw: false");
    expect(PAGE_SOURCE).toContain("a - b !== 417 || b !== p");
    expect(WORKER_SOURCE).toContain(
      'submissionPolicyOverride: "depth2-phase-epoch4" as const',
    );
    expect(WORKER_SOURCE).toContain(
      "vaeSchedulingPolicyOverride: forcedPolicy!",
    );
    expect(WORKER_SOURCE).toContain('arm.id === "production"\n        ? {}');
    expect(WORKER_SOURCE).toContain(
      'evidenceMode: captured === undefined\n        ? "seam-free-ordinary"',
    );
    expect(WORKER_SOURCE).toContain(
      "summarizeOpt0080VaeProductSchedulingEvidence(",
    );
    expect(WORKER_SOURCE).toContain(
      "wav.sha256 !== OPT_0080_VAE_PRODUCT_WAV_SHA256",
    );
    expect(WORKER_SOURCE).toContain(
      "raw.sha256 !== OPT_0080_VAE_PRODUCT_RAW_SHA256",
    );
  });

  it("retains the existing fresh-worker release, cancellation, and heartbeat gate", () => {
    expect(PAGE_SOURCE).toContain("workersCreated !== 4");
    expect(PAGE_SOURCE).toContain("workersTerminated !== 4");
    expect(PAGE_SOURCE).toContain("const HEARTBEAT_INTERVAL_MILLISECONDS = 50");
    expect(PAGE_SOURCE).toContain(
      "const HEARTBEAT_MAXIMUM_GAP_MILLISECONDS = 500",
    );
    expect(PAGE_SOURCE).toContain("await releaseAllRetainedArms()");
    expect(PAGE_SOURCE).toContain("runCancellation(runIdentity)");
    expect(WORKER_SOURCE).toContain("opt0080ProductRunOmitted: true");
    expect(WORKER_SOURCE).toContain("await backend.releaseResult(result)");
    expect(WORKER_SOURCE).toContain("await backend.dispose()");
    expect(PAGE_SOURCE).not.toMatch(/\bretry\s*\(/iu);
  });
});

function receipt(armId: Opt0080VaeProductArmId): AceVaeSchedulingReceipt {
  const candidate = armId !== "control";
  const selection: AceVaeSchedulingSelection = armId === "production"
    ? "production"
    : "benchmark-override";
  return Object.freeze({
    schema: "ace-vae-window-scheduling-receipt-v1",
    selectedProductionPolicy: armId === "production"
      ? ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY
      : null,
    benchmarkPolicyOverride: armId === "control"
      ? "depth1-epoch1"
      : armId === "candidate"
      ? ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY
      : null,
    windows: Object.freeze([
      windowReceipt(0, 2_314, 35_498, selection,
        candidate ? "depth2-phase-epoch4" : "depth1-epoch1"),
      windowReceipt(1, 214, 3_342, selection, "depth1-epoch1"),
    ]),
  });
}

function windowReceipt(
  windowIndex: number,
  latentWindowFrames: number,
  decoderQuantumCount: number,
  selection: AceVaeSchedulingSelection,
  schedulingProfile: AceVaeWindowSchedulingReceipt["schedulingProfile"],
): AceVaeWindowSchedulingReceipt {
  const decoderCommandBufferCount = Math.ceil(decoderQuantumCount / 64);
  const totalCommandBufferCount = decoderCommandBufferCount + 1;
  const depthTwo = schedulingProfile === "depth2-phase-epoch4";
  const queueDrains = depthTwo
    ? Math.ceil(totalCommandBufferCount / 4)
    : totalCommandBufferCount;
  return Object.freeze({
    windowIndex,
    latentWindowFrames,
    selection,
    schedulingProfile,
    decoderQuantumCount,
    quantaPerCommandBuffer: 64,
    decoderCommandBufferCount,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount,
    commandBuffersSubmitted: totalCommandBufferCount,
    queueDrains,
    cooperativeIdleTurns: queueDrains - 1,
    maximumOutstandingCommandBuffers: depthTwo ? 2 : 1,
  });
}

function evidenceForWindow(
  window: AceVaeWindowSchedulingReceipt,
): AceOpt0080VaeSchedulingEvidence {
  const depthTwo = window.schedulingProfile === "depth2-phase-epoch4";
  const commandCompletions = Object.freeze(Array.from(
    { length: window.totalCommandBufferCount },
    (_, index) => Object.freeze({
      commandBufferIndex: index,
      commandKind: index < window.decoderCommandBufferCount
        ? "decoder" as const
        : "readback" as const,
      submitThroughCompletionFenceMs: 1,
      trueQueueDrain: depthTwo
        ? (index + 1) % 4 === 0 ||
          index === window.totalCommandBufferCount - 1
        : true,
      completionEpochIndex: depthTwo ? Math.floor(index / 4) : index,
    }),
  ));
  const completionEpochs = Object.freeze(Array.from(
    { length: window.queueDrains },
    (_, index) => {
      const first = depthTwo ? index * 4 : index;
      const last = depthTwo
        ? Math.min(first + 3, window.totalCommandBufferCount - 1)
        : index;
      return Object.freeze({
        completionEpochIndex: index,
        phaseIndex: 0 as const,
        firstCommandBufferIndex: first,
        lastCommandBufferIndex: last,
        commandBufferCount: last - first + 1,
        submitThroughTrueDrainMs: 1,
      });
    },
  ));
  return Object.freeze({
    schema: "ace-opt-0080-vae-window-scheduling-v1",
    windowIndex: window.windowIndex,
    schedulingProfile: window.schedulingProfile,
    decoderQuantumCount: window.decoderQuantumCount,
    quantaPerCommandBuffer: window.quantaPerCommandBuffer,
    decoderCommandBufferCount: window.decoderCommandBufferCount,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: window.totalCommandBufferCount,
    schedulingWallMs: 1,
    commandBuffersSubmitted: window.commandBuffersSubmitted,
    completionFenceRequestedCount: window.totalCommandBufferCount,
    completionFenceSettledCount: window.totalCommandBufferCount,
    completionFenceRejectedCount: 0,
    trueQueueDrainCount: window.queueDrains,
    completionEpochCount: window.queueDrains,
    cooperativeIdleTurns: window.cooperativeIdleTurns,
    requestedCooperativeIdleMs: window.cooperativeIdleTurns,
    maximumOutstandingCommandBuffers:
      window.maximumOutstandingCommandBuffers,
    commandCompletions,
    completionEpochs,
  });
}

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}
