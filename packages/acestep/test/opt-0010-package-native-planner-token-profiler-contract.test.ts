import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  OPT_0010_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0010_PRODUCTION_COMMIT,
  OPT_0010_THERMAL_POLL_MILLISECONDS,
  OPT_0010_THERMAL_SOURCE,
  parseOpt0010RunIdentity,
  parseOpt0010ThermalGateMetadata,
} from "./browser/opt-0010-package-native-planner-token-profiler.js";
import {
  OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS,
  OPT_0010_CASE_SPECS,
  OPT_0010_COMMAND_BUFFER_COUNT,
  OPT_0010_COT_DIAGNOSTIC_TRAJECTORY_SHA256,
  OPT_0010_MODEL_DISPATCH_PRIMITIVE_COUNT,
  OPT_0010_PHYSICAL_DISPATCH_COUNT,
  OPT_0010_READBACK_COPY_COUNT,
} from "./browser/opt-0010-package-native-planner-token-profiler-worker.js";
import { aceSha256Hex } from "../src/model/sha256.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0010-package-native-planner-token-profiler-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0010-package-native-planner-token-profiler.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0010-package-native-planner-token-profiler.html",
  import.meta.url,
), "utf8");

describe("OPT-0010 package-native planner token browser profiler", () => {
  it("retains structured worker failures for actionable Chrome diagnostics", () => {
    expect(PAGE_SOURCE).toContain(": error,");
    expect(PAGE_SOURCE).not.toContain(": String(error),");
  });

  it("fails closed on the frozen core and complete machine identity", () => {
    const valid = new URLSearchParams({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      productionCommit: OPT_0010_PRODUCTION_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    expect(parseOpt0010RunIdentity(valid)).toEqual({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      productionCommit: OPT_0010_PRODUCTION_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    for (const [name, value] of [
      ["harnessCommit", "badc0de"],
      ["productionCommit", "0123456789abcdef0123456789abcdef01234567"],
      ["machineModel", "   "],
      ["gpuCoreCount", "0"],
      ["memoryBytes", "9007199254740992"],
    ] as const) {
      const invalid = new URLSearchParams(valid);
      invalid.set(name, value);
      expect(() => parseOpt0010RunIdentity(invalid)).toThrow(/OPT-0010/);
    }
    expect(WORKER_SOURCE).toContain(OPT_0010_PRODUCTION_COMMIT);
    expect(WORKER_SOURCE.indexOf("validateRunIdentity(identity)")).toBeLessThan(
      WORKER_SOURCE.indexOf("await preparePackage()"),
    );
  });

  it("requires a post-reference 30-second continuous nominal gate", () => {
    const warmupCompletedAt = 1_000_000;
    const startedAt = warmupCompletedAt + 1;
    const completedAt = startedAt + OPT_0010_MINIMUM_NOMINAL_MILLISECONDS;
    const valid = new URLSearchParams({
      thermalSource: OPT_0010_THERMAL_SOURCE,
      thermalStartedAtEpochMilliseconds: String(startedAt),
      thermalCompletedAtEpochMilliseconds: String(completedAt),
      thermalObservations: "31",
      thermalPollMilliseconds: String(OPT_0010_THERMAL_POLL_MILLISECONDS),
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0010ThermalGateMetadata(
      valid,
      warmupCompletedAt,
      completedAt + 1,
    )).toEqual({
      source: OPT_0010_THERMAL_SOURCE,
      startedAtEpochMilliseconds: startedAt,
      completedAtEpochMilliseconds: completedAt,
      durationMilliseconds: OPT_0010_MINIMUM_NOMINAL_MILLISECONDS,
      observationCount: 31,
      pollMilliseconds: OPT_0010_THERMAL_POLL_MILLISECONDS,
      maximumPollGapMilliseconds: 1010,
      nonNominalObservationCount: 0,
    });
    const beforeWarmup = new URLSearchParams(valid);
    beforeWarmup.set("thermalStartedAtEpochMilliseconds", "999999");
    expect(() => parseOpt0010ThermalGateMetadata(
      beforeWarmup,
      warmupCompletedAt,
      completedAt + 1,
    )).toThrow(/after warmup/);
    const pressured = new URLSearchParams(valid);
    pressured.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0010ThermalGateMetadata(
      pressured,
      warmupCompletedAt,
      completedAt + 1,
    )).toThrow(/non-nominal/);
  });

  it("pins six explicit short/mid/long M1 and M2 package-native cases", () => {
    expect(OPT_0010_CASE_SPECS).toEqual([
      expect.objectContaining({
        id: "cot-m1-short", mode: "cot-m1", position: "short",
        cachedTokensBeforeAppend: 120, cacheCapacity: 512, drawIndex: 16,
      }),
      expect.objectContaining({
        id: "cot-m1-mid", mode: "cot-m1", position: "mid",
        cachedTokensBeforeAppend: 160, cacheCapacity: 1_024, drawIndex: 56,
      }),
      expect.objectContaining({
        id: "cot-m1-long", mode: "cot-m1", position: "long",
        cachedTokensBeforeAppend: 212, cacheCapacity: 2_048, drawIndex: 108,
      }),
      expect.objectContaining({
        id: "semantic-m2-short", mode: "semantic-m2", position: "short",
        cachedTokensBeforeAppend: 268, cacheCapacity: 768, drawIndex: 125,
      }),
      expect.objectContaining({
        id: "semantic-m2-mid", mode: "semantic-m2", position: "mid",
        cachedTokensBeforeAppend: 328, cacheCapacity: 1_280, drawIndex: 185,
      }),
      expect.objectContaining({
        id: "semantic-m2-long", mode: "semantic-m2", position: "long",
        cachedTokensBeforeAppend: 401, cacheCapacity: 2_048, drawIndex: 258,
      }),
    ]);
    expect(new Set(OPT_0010_CASE_SPECS.map((entry) => entry.drawIndex)).size)
      .toBe(6);
    const codeBytes = new Uint8Array(
      OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS.length * 4,
    );
    const codeView = new DataView(codeBytes.buffer);
    OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS.forEach((value, index) => {
      codeView.setUint32(index * 4, value, true);
    });
    expect(OPT_0010_ACCEPTED_SEMANTIC_CODE_IDS).toHaveLength(150);
    expect(aceSha256Hex(codeBytes)).toBe(
      "42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be",
    );
    expect(WORKER_SOURCE).toContain("createAcePlannerCotPrompt(");
    expect(WORKER_SOURCE).toContain("createAcePlannerCodePrompts(");
    expect(WORKER_SOURCE).toContain("OPT_0010_ACCEPTED_RECEIPT_SHA256");
    expect(WORKER_SOURCE).toContain("OPT_0010_ACCEPTED_SEMANTIC_CODE_SHA256");
    expect(OPT_0010_COT_DIAGNOSTIC_TRAJECTORY_SHA256).toBe(
      "476515e1db6ebc30e1622eb30ac02a8ef4289d89ca12e34c64b5f911bc960da2",
    );
    expect(WORKER_SOURCE).toContain(
      "createReceiptTextDerivedCotTeacherTokens(",
    );
    expect(WORKER_SOURCE).toContain(
      "representative diagnostic trajectory, not a claim",
    );
    expect(WORKER_SOURCE).toContain(
      "accepted-receipt-text-derived-fsm-admitted-longest-token-diagnostic",
    );
    expect(WORKER_SOURCE).toContain("semanticContinuation.length !== 150");
    expect(WORKER_SOURCE).toContain("cotBase.length !== 105");
    expect(WORKER_SOURCE).toContain("semanticBases[0]!.length !== 253");
    expect(WORKER_SOURCE).toContain(
      "OPT-0010 case leaves the accepted continuation trajectory",
    );
  });

  it("authenticates and loads only the FP16 planner phase and tokenizer", () => {
    expect(HTML_SOURCE).toContain("raw-FP16 planner phase");
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain("{ type: \"module\" }");
    expect(WORKER_SOURCE).toContain(
      "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b",
    );
    expect(WORKER_SOURCE).toContain("expectedProfile: \"fp16\"");
    expect(WORKER_SOURCE).toContain("files: inventory.files");
    expect(WORKER_SOURCE).toContain("acquireAceModelFiles({");
    expect(WORKER_SOURCE).toContain("AceGpuTensorPhase.load(");
    expect(WORKER_SOURCE).toContain("[\"planner\"]");
    expect(WORKER_SOURCE).toContain("loadPinnedAceTokenizer(\"planner\"");
    expect(WORKER_SOURCE).toContain("AcePlannerGpuExecutor.create({");
    expect(WORKER_SOURCE).toContain("device: observer.device");
    expect(WORKER_SOURCE).not.toContain("fromPreparedResources");
  });

  it("reconciles the production 624 aggregate with 628 observed dispatches", () => {
    expect(OPT_0010_MODEL_DISPATCH_PRIMITIVE_COUNT).toBe(624);
    expect(OPT_0010_PHYSICAL_DISPATCH_COUNT).toBe(628);
    expect(OPT_0010_COMMAND_BUFFER_COUNT).toBe(34);
    expect(OPT_0010_READBACK_COPY_COUNT).toBe(6);
    expect(WORKER_SOURCE).toContain("property === \"beginComputePass\"");
    expect(WORKER_SOURCE).toContain("passProperty === \"setPipeline\"");
    expect(WORKER_SOURCE).toContain("passProperty === \"setBindGroup\"");
    expect(WORKER_SOURCE).toContain("passProperty === \"dispatchWorkgroups\"");
    expect(WORKER_SOURCE).toContain("pipelineLabel = pipeline.label");
    expect(WORKER_SOURCE).toContain("bindGroupLabel = bindGroup.label");
    expect(WORKER_SOURCE).toContain("rawPhysicalDispatches.push(");
    expect(WORKER_SOURCE).toContain("authenticateDispatchIdentity(");
    expect(WORKER_SOURCE).toContain("authenticatePipelineLabel(");
    expect(WORKER_SOURCE).not.toContain(
      "physicalDispatches: expected.physicalDispatches",
    );
    expect(WORKER_SOURCE).toContain("property === \"copyBufferToBuffer\"");
    expect(WORKER_SOURCE).toContain("sourceBufferLabel: sourceRecord.label");
    expect(WORKER_SOURCE).toContain("destinationBufferLabel: destinationRecord.label");
    expect(WORKER_SOURCE).toContain("sourceOffset: Number(sourceOffset)");
    expect(WORKER_SOURCE).toContain("destinationOffset: Number(destinationOffset)");
    expect(WORKER_SOURCE).toContain("copiedBytes: Number(size)");
  });

  it("retains full progress, queue, idle, map, and reconstruction timing", () => {
    expect(WORKER_SOURCE).toContain("property === \"submit\"");
    expect(WORKER_SOURCE).toContain("property === \"onSubmittedWorkDone\"");
    expect(WORKER_SOURCE).toContain("prior.nextEncodeStartedAt = now");
    expect(WORKER_SOURCE).toContain("prior.idleEndedAt = now");
    expect(WORKER_SOURCE).toContain("record.idleStartedAt = record.drainEndedAt");
    expect(WORKER_SOURCE).toContain("record.progress = progressPayload(event)");
    expect(WORKER_SOURCE).toContain("Object.defineProperty(buffer, \"mapAsync\"");
    expect(WORKER_SOURCE).toContain("Object.defineProperty(buffer, \"getMappedRange\"");
    expect(WORKER_SOURCE).toContain("reconstructEndedAt = invocationResolvedAt");
    expect(WORKER_SOURCE).toContain("validateProgressSequence(");
    expect(WORKER_SOURCE).toContain("validateAceOpt0010PlannerTokenTrace(");
    expect(WORKER_SOURCE).toContain("summarizeAceOpt0010PlannerTokenTrace(");
    expect(WORKER_SOURCE).toContain(
      "performanceTimeOriginEpochMilliseconds: performance.timeOrigin",
    );
  });

  it("uses independent package-native full-logit and production-sampler references", () => {
    const initialize = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function initializeSession"),
      WORKER_SOURCE.indexOf("async function runTimedAndCleanup"),
    );
    const timed = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function runTimedAndCleanup"),
      WORKER_SOURCE.indexOf("class ProgressRouter"),
    );
    expect(initialize).toContain("await executor.prefill(fixture.prefill)");
    expect(initialize).toContain("await executor.decode(createDecodeBatch(fixture))");
    expect(initialize).toContain(
      "const sample = sampleCase(fixture, logits, allowedTokens)",
    );
    expect(initialize).toContain(
      'validateSampleCursor(fixture, sample, "reference")',
    );
    expect(timed).toContain("await prepared.executor.prefill(fixture.prefill)");
    expect(timed).toContain("prepared.observer.beginTimedTrace(fixture.attribution)");
    expect(timed).toContain("compareLogits(reference.logits, logits)");
    expect(timed).toContain("sameSample(reference.sample, sample)");
    expect(timed).toContain(
      'validateSampleCursor(fixture, sample, "timed")',
    );
    expect(WORKER_SOURCE).toContain(
      "const expectedDrawIndex = fixture.spec.drawIndex.toString()",
    );
    expect(WORKER_SOURCE).toContain(
      "const expectedDrawEnd = (fixture.spec.drawIndex + 1).toString()",
    );
    expect(WORKER_SOURCE).toContain("new AcePlannerSamplingCursor(");
    expect(WORKER_SOURCE).toContain("DEFAULT_ACE_PLANNER_CONFIGURATION");
    expect(WORKER_SOURCE).toContain(
      "new AcePlannerMetadataConstraintController({",
    );
    expect(WORKER_SOURCE).toContain("controller.allowedTokens({");
    expect(WORKER_SOURCE).toContain("controller.acceptToken({");
    expect(WORKER_SOURCE).toContain("resolveCaseAllowedTokens(");
    expect(WORKER_SOURCE).toContain("bitMismatchCount");
    expect(WORKER_SOURCE).toContain("finiteCount !== summary.totalElements");
    expect(WORKER_SOURCE).toContain("sha256: await sha256Hex(bytes)");
  });

  it("proves post-drain cancellation, heartbeat, and complete cleanup", () => {
    const cancellation = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function runCancellationProof"),
      WORKER_SOURCE.indexOf("function validateThermalGate"),
    );
    expect(cancellation).toContain("event.completedCommandBuffers === 1");
    expect(cancellation).toContain("prepared.abortController.abort(new DOMException(");
    expect(cancellation).toContain("observed.records.length !== 1");
    expect(cancellation).toContain("observed.submissionCount !== 1");
    expect(cancellation).toContain("observed.drainCount !== 1");
    expect(cancellation).toContain("first.progressReportedAt < first.drainEndedAt");
    expect(cancellation).toContain("laterEncodingPrevented");
    expect(cancellation).toContain("realIdleCompletedBeforeRejection");
    expect(WORKER_SOURCE).toContain("await prepared.executor.destroy()");
    expect(WORKER_SOURCE).toContain(
      "if (!resources.destructionTrackingSupported)",
    );
    expect(WORKER_SOURCE).toContain("liveTrackedBufferCount !== 0");
    expect(WORKER_SOURCE).toContain("prepared.runtimeEvents.length !== 0");
    expect(WORKER_SOURCE).toContain("startWorkerHeartbeat()");
    expect(PAGE_SOURCE).toContain("startHeartbeat()");
    expect(HTML_SOURCE).toContain("Keep the external thermal logger running");
    expect(WORKER_SOURCE).toContain('status: "pending-external-artifact-join"');
  });
});
