import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";
import {
  OPT_0080_PRODUCT_ARM_ORDER,
  OPT_0080_PRODUCT_AUDIO_FRAMES,
  OPT_0080_PRODUCT_LATENT_FRAMES,
  OPT_0080_PRODUCT_RAW_BYTES,
  OPT_0080_PRODUCT_REQUEST,
  OPT_0080_PRODUCT_REQUEST_BYTES,
  OPT_0080_PRODUCT_REQUEST_SHA256,
  OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME,
  OPT_0080_PRODUCT_WAV_BYTES,
} from "./browser/opt-0080-product-integration-contract.js";

const PAGE_SOURCE = source("./browser/opt-0080-product-integration.ts");
const WORKER_SOURCE = source(
  "./browser/opt-0080-product-integration-worker.ts",
);
const HTML_SOURCE = source("./browser/opt-0080-product-integration.html");
const COMMITTED_RESULT: unknown = JSON.parse(source(
  "../optimization/results/OPT-0080/product-integration.json",
));

describe("OPT-0080 post-integration product browser gate", () => {
  it("pins the canonical 96-second request and real C2378 seam", () => {
    expect(OPT_0080_PRODUCT_LATENT_FRAMES).toBe(2_400);
    expect(OPT_0080_PRODUCT_AUDIO_FRAMES).toBe(4_608_000);
    expect(OPT_0080_PRODUCT_RAW_BYTES).toBe(36_864_000);
    expect(OPT_0080_PRODUCT_WAV_BYTES).toBe(36_864_044);
    const requestJson = JSON.stringify(OPT_0080_PRODUCT_REQUEST);
    expect(new TextEncoder().encode(requestJson)).toHaveLength(
      OPT_0080_PRODUCT_REQUEST_BYTES,
    );
    expect(createHash("sha256").update(requestJson).digest("hex")).toBe(
      OPT_0080_PRODUCT_REQUEST_SHA256,
    );
    expect(OPT_0080_PRODUCT_REQUEST_SHA256).toBe(
      "ecc1d8d0fd7a87e14d0cf827563280fe35853526368becf883d98f4d42cb1ad4",
    );

    const plan = planAceVaeChunkedDecode(OPT_0080_PRODUCT_LATENT_FRAMES, {
      chunkFrames: 2_378,
      overlapFrames: 64,
    });
    expect(plan.direct).toBe(false);
    expect(plan.windows.map((window) => ({
      window: [
        window.windowStartLatentFrame,
        window.windowEndLatentFrame,
      ],
      core: [window.coreStartLatentFrame, window.coreEndLatentFrame],
      prefix: window.discardPrefixLatentFrames,
      suffix: window.discardSuffixLatentFrames,
      outputStart: window.outputStartAudioFrame,
    }))).toEqual([
      {
        window: [0, 2_314],
        core: [0, 2_250],
        prefix: 0,
        suffix: 64,
        outputStart: 0,
      },
      {
        window: [2_186, 2_400],
        core: [2_250, 2_400],
        prefix: 64,
        suffix: 0,
        outputStart: OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME,
      },
    ]);
  });

  it("uses forced A/B then an unforced production selector", () => {
    expect(OPT_0080_PRODUCT_ARM_ORDER).toEqual([
      {
        id: "control",
        submissionPolicyOverride: "depth1-epoch1",
        effectiveSubmissionPolicy: "depth1-epoch1",
      },
      {
        id: "candidate",
        submissionPolicyOverride: "depth2-phase-epoch4",
        effectiveSubmissionPolicy: "depth2-phase-epoch4",
      },
      {
        id: "production",
        submissionPolicyOverride: undefined,
        effectiveSubmissionPolicy: "depth2-phase-epoch4",
      },
    ]);
    expect(WORKER_SOURCE).toContain(
      "arm.submissionPolicyOverride === undefined",
    );
    expect(WORKER_SOURCE).toContain("? {}\n          : { submissionPolicyOverride");
    expect(PAGE_SOURCE).toContain(
      'const FIXED_SUCCESS_ORDER = ["control", "candidate", "production"]',
    );
    expect(PAGE_SOURCE.indexOf('runSuccessArm(\n    "control"'))
      .toBeLessThan(PAGE_SOURCE.indexOf('runSuccessArm(\n    "candidate"'));
    expect(PAGE_SOURCE.indexOf('runSuccessArm(\n    "candidate"'))
      .toBeLessThan(PAGE_SOURCE.indexOf('runSuccessArm(\n    "production"'));
    expect(PAGE_SOURCE.indexOf('runSuccessArm(\n    "production"'))
      .toBeLessThan(PAGE_SOURCE.indexOf("runCancellation(runIdentity)"));
  });

  it("compares complete outputs boundedly and publishes no Blob payload", () => {
    expect(PAGE_SOURCE).toContain("compareU32(");
    expect(PAGE_SOURCE).toContain("compareAceRawAudioSnapshots(");
    expect(PAGE_SOURCE).toContain("left.rawSnapshot.slice(seamStartByte");
    expect(PAGE_SOURCE).toContain("compareBlobBytes(");
    expect(PAGE_SOURCE).toContain(
      "const COMPARISON_BLOCK_BYTES = 1_048_576",
    );
    expect(PAGE_SOURCE).toContain("control = undefined");
    expect(PAGE_SOURCE).toContain("candidate = undefined");
    expect(PAGE_SOURCE).toContain("production = undefined");
    expect(PAGE_SOURCE).toContain(
      "window.__ACE_OPT0080_PRODUCT_RESULT__ = receipt",
    );
    expect(PAGE_SOURCE).not.toContain("rawSnapshot: control");
    expect(PAGE_SOURCE).not.toContain("wav: control");
    expect(PAGE_SOURCE).toContain(
      'const progressStageFields = [\n    "initializationStages",\n    "generationStages"',
    );
    expect(PAGE_SOURCE).toContain(
      "policyDependentProgressEventCountsExcluded: true",
    );
    expect(WORKER_SOURCE.indexOf('type: "arm-ready"')).toBeLessThan(
      WORKER_SOURCE.indexOf("await backend.releaseResult(result)"),
    );
    expect(PAGE_SOURCE.indexOf('compareArmPair(\n    "control-candidate"'))
      .toBeLessThan(PAGE_SOURCE.indexOf("releaseArm(control)"));
    expect(PAGE_SOURCE.indexOf('compareArmPair(\n    "candidate-production"'))
      .toBeLessThan(PAGE_SOURCE.indexOf("releaseArm(candidate)"));
    expect(PAGE_SOURCE.indexOf("retainedArms.add(retained)")).toBeLessThan(
      PAGE_SOURCE.indexOf("requireArmPayload(armId, retained, runIdentity)"),
    );
    expect(PAGE_SOURCE).toContain(
      "The outer gate failure handler releases every registered arm",
    );
  });

  it("uses four fresh workers, a through-termination heartbeat, and no retry", () => {
    expect(PAGE_SOURCE).toContain("function createWorker(");
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain("worker.terminate();\n        workersTerminated += 1");
    expect(PAGE_SOURCE).toContain("heartbeat.stopAfterWorkerTermination()");
    expect(PAGE_SOURCE).toContain(
      "const HEARTBEAT_INTERVAL_MILLISECONDS = 50",
    );
    expect(PAGE_SOURCE).toContain(
      "const HEARTBEAT_MAXIMUM_GAP_MILLISECONDS = 500",
    );
    expect(PAGE_SOURCE).toContain("workersCreated !== 4");
    expect(PAGE_SOURCE).toContain("automaticRetryCount: 0");
    expect(PAGE_SOURCE).toContain('worker.postMessage({ type: "release", armId })');
    expect(WORKER_SOURCE).toContain('type: "arm-released"');
    expect(PAGE_SOURCE).not.toMatch(/\bretry\s*\(/i);
    expect(WORKER_SOURCE).toContain("opt0080ProductRunOmitted: true");
    expect(WORKER_SOURCE).toContain("ordinaryProductionSelector: true");
    expect(WORKER_SOURCE).not.toContain("captureTrace");
    expect(PAGE_SOURCE).not.toContain("captureTrace");
  });

  it("exposes a single explicit correctness run in the HTML", () => {
    expect(HTML_SOURCE).toContain('id="run"');
    expect(HTML_SOURCE).toContain("no automatic\n      retry");
    expect(HTML_SOURCE).toContain("not a timing or\n      thermal comparison");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0080-product-integration.ts"',
    );
  });

  it("binds the passing retained-output browser receipt", () => {
    const committed = record(COMMITTED_RESULT);
    const identity = record(committed.identity);
    const selection = record(committed.selection);
    const control = record(selection.control);
    const candidate = record(selection.candidate);
    const production = record(selection.production);
    const comparisons = record(committed.comparisons);
    const passingReceipt = record(record(committed.artifacts).passingReceipt);
    for (const pair of ["controlCandidate", "candidateProduction"] as const) {
      const comparison = record(comparisons[pair]);
      expect(record(comparison.finalLatent).exactU32MismatchCount).toBe(0);
      expect(record(comparison.fullRaw).exactU32MismatchCount).toBe(0);
      expect(record(comparison.seamRaw).exactU32MismatchCount).toBe(0);
      expect(record(comparison.wav).exactByteMismatchCount).toBe(0);
    }
    expect(committed.status).toBe("passed");
    expect(committed.disposition).toBe("integrated");
    expect(identity.coreCommit).toBe(
      "dfb2a24c979f840f13909b6baee0742bd7ee4f40",
    );
    expect(identity.directOnlySelectorCommit).toBe(
      "023bdecbf670b9309db37b6ac3030293ffe3b463",
    );
    expect(control.effectiveSubmissionPolicy).toBe("depth1-epoch1");
    expect(candidate.effectiveSubmissionPolicy).toBe("depth2-phase-epoch4");
    expect(production.effectiveSubmissionPolicy).toBe("depth2-phase-epoch4");
    expect(control.cooperativeGpuQueueDrains).toBeGreaterThan(
      candidate.cooperativeGpuQueueDrains as number,
    );
    expect(passingReceipt).toEqual({
      path: "optimization/artifacts/OPT-0080/product/attempt-004-passed.json",
      bytes: 51_285,
      sha256: "d91dc34fc1a1bb14102121dff9ea8d7e66ad666e0cf4911f4accb8afca67f9e5",
    });
    expect((committed.attempts as unknown[])).toHaveLength(4);
  });
});

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Expected an OPT-0080 result record");
  }
  return value as Readonly<Record<string, unknown>>;
}
