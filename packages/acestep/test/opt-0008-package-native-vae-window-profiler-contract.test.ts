import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  OPT_0008_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0008_PRODUCTION_COMMIT,
  OPT_0008_THERMAL_POLL_MILLISECONDS,
  OPT_0008_THERMAL_SOURCE,
  parseOpt0008RunIdentity,
  parseOpt0008ThermalGateMetadata,
} from "./browser/opt-0008-package-native-vae-window-profiler.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0008-package-native-vae-window-profiler-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0008-package-native-vae-window-profiler.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0008-package-native-vae-window-profiler.html",
  import.meta.url,
), "utf8");

describe("OPT-0008 package-native VAE browser profiler contract", () => {
  it("fails closed on the complete frozen run identity before preparation", () => {
    const valid = new URLSearchParams({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      productionCommit: OPT_0008_PRODUCTION_COMMIT,
      machineModel: "Mac15,9",
      osVersion: "15.6.1",
      osBuild: "24G90",
      browserVersion: "Chrome 140.0.7339.80",
      gpuCoreCount: "30",
      memoryBytes: "38654705664",
    });
    expect(parseOpt0008RunIdentity(valid)).toEqual({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      productionCommit: OPT_0008_PRODUCTION_COMMIT,
      machineModel: "Mac15,9",
      osVersion: "15.6.1",
      osBuild: "24G90",
      browserVersion: "Chrome 140.0.7339.80",
      gpuCoreCount: 30,
      memoryBytes: 38_654_705_664,
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
      expect(() => parseOpt0008RunIdentity(invalid)).toThrow(/OPT-0008/);
    }
    const missing = new URLSearchParams(valid);
    missing.delete("osBuild");
    expect(() => parseOpt0008RunIdentity(missing)).toThrow(/osBuild/);

    const parseIdentity = PAGE_SOURCE.indexOf(
      "runIdentity = parseOpt0008RunIdentity(",
    );
    const createWorker = PAGE_SOURCE.indexOf("const worker = new Worker(");
    expect(parseIdentity).toBeGreaterThan(0);
    expect(createWorker).toBeGreaterThan(parseIdentity);
    expect(PAGE_SOURCE).toContain(
      'worker.postMessage({ type: "initialize", identity: runIdentity })',
    );
    expect(WORKER_SOURCE).toContain("validateRunIdentity(identity)");
    expect(WORKER_SOURCE).toContain(`identity: prepared.runIdentity`);
    expect(WORKER_SOURCE).toContain(OPT_0008_PRODUCTION_COMMIT);
    expect(WORKER_SOURCE.indexOf(
      "const runIdentity = validateRunIdentity(identity)",
    )).toBeLessThan(WORKER_SOURCE.indexOf(
      'postProgress("authenticating the converter-revision-4 reference manifest")',
    ));
  });

  it("fails closed on a post-warmup 30-second external thermal gate", () => {
    const warmupCompletedAt = 1_000_000;
    const startedAt = warmupCompletedAt + 10;
    const completedAt = startedAt + OPT_0008_MINIMUM_NOMINAL_MILLISECONDS;
    const valid = new URLSearchParams({
      thermalSource: OPT_0008_THERMAL_SOURCE,
      thermalStartedAtEpochMilliseconds: String(startedAt),
      thermalCompletedAtEpochMilliseconds: String(completedAt),
      thermalObservations: "31",
      thermalPollMilliseconds: String(OPT_0008_THERMAL_POLL_MILLISECONDS),
      thermalMaximumPollGapMilliseconds: "1012.5",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0008ThermalGateMetadata(
      valid,
      warmupCompletedAt,
      completedAt + 1,
    )).toEqual({
      source: OPT_0008_THERMAL_SOURCE,
      startedAtEpochMilliseconds: startedAt,
      completedAtEpochMilliseconds: completedAt,
      durationMilliseconds: OPT_0008_MINIMUM_NOMINAL_MILLISECONDS,
      observationCount: 31,
      pollMilliseconds: OPT_0008_THERMAL_POLL_MILLISECONDS,
      maximumPollGapMilliseconds: 1012.5,
      nonNominalObservationCount: 0,
    });

    const beforeWarmup = new URLSearchParams(valid);
    beforeWarmup.set(
      "thermalStartedAtEpochMilliseconds",
      String(warmupCompletedAt - 1),
    );
    expect(() => parseOpt0008ThermalGateMetadata(
      beforeWarmup,
      warmupCompletedAt,
      completedAt + 1,
    )).toThrow(/after warmup/);
    const short = new URLSearchParams(valid);
    short.set(
      "thermalCompletedAtEpochMilliseconds",
      String(completedAt - 1),
    );
    expect(() => parseOpt0008ThermalGateMetadata(
      short,
      warmupCompletedAt,
      completedAt + 1,
    )).toThrow(/30 continuous/);
    const pressured = new URLSearchParams(valid);
    pressured.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0008ThermalGateMetadata(
      pressured,
      warmupCompletedAt,
      completedAt + 1,
    )).toThrow(/non-nominal/);
    const sparse = new URLSearchParams(valid);
    sparse.set(
      "thermalCompletedAtEpochMilliseconds",
      String(completedAt + 1_000),
    );
    expect(() => parseOpt0008ThermalGateMetadata(
      sparse,
      warmupCompletedAt,
      completedAt + 1_001,
    )).toThrow(/continuous nominal/);
  });

  it("keeps package authentication, phase loading, and inference in one worker", () => {
    expect(HTML_SOURCE).toContain("Prepare and warm up VAE");
    expect(HTML_SOURCE).toContain("Run one timed production window");
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain("{ type: \"module\" }");
    expect(PAGE_SOURCE).toContain("type: \"initialize\"");
    expect(PAGE_SOURCE).toContain("type: \"run-timed\"");
    expect(PAGE_SOURCE).toContain("startHeartbeat()");
    expect(WORKER_SOURCE).toContain(
      "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
    );
    expect(WORKER_SOURCE).toContain("expectedProfile: \"reference\"");
    expect(WORKER_SOURCE).toContain("files: inventory.files");
    expect(WORKER_SOURCE).toContain("acquireAceModelFiles({");
    expect(WORKER_SOURCE).toContain("phases.length !== 1");
    expect(WORKER_SOURCE).toContain("phase.phases[0] !== \"vae\"");
    expect(WORKER_SOURCE.match(/AceGpuTensorPhase\.load\(/g)).toHaveLength(1);
    expect(WORKER_SOURCE.match(/AceVaeChunkGpuBackend\.create\(/g)).toHaveLength(1);
    const observerCreation = WORKER_SOURCE.indexOf(
      "const observer = new ProductionDeviceObserver(context.device)",
    );
    const phaseLoad = WORKER_SOURCE.indexOf("phase = await AceGpuTensorPhase.load(");
    expect(observerCreation).toBeGreaterThan(0);
    expect(phaseLoad).toBeGreaterThan(observerCreation);
    expect(WORKER_SOURCE.slice(phaseLoad, phaseLoad + 180)).toContain(
      "observer.device",
    );
  });

  it("uses the unchanged shipped 256-frame batch-eight backend topology", () => {
    expect(WORKER_SOURCE).toContain("planAceVaeChunkedDecode(");
    expect(WORKER_SOURCE).toContain("OPT_0008_LATENT_FRAMES = 256");
    expect(WORKER_SOURCE).toContain("OPT_0008_LOGICAL_QUANTUM_COUNT = 3_942");
    expect(WORKER_SOURCE).toContain("OPT_0008_PRIMITIVE_DISPATCH_COUNT = 3_988");
    expect(WORKER_SOURCE).toContain("OPT_0008_DECODER_BATCH_COUNT = 493");
    expect(WORKER_SOURCE).toContain("OPT_0008_TOTAL_COMMAND_BUFFER_COUNT = 494");
    expect(WORKER_SOURCE).toContain(
      "quantaPerCommandBuffer: ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER",
    );
    const backendCreate = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("backend = await AceVaeChunkGpuBackend.create({"),
      WORKER_SOURCE.indexOf("const backendCompileWallMilliseconds"),
    );
    expect(backendCreate).not.toContain("decoderQuantaPerCommandBuffer");
    expect(backendCreate).not.toContain("quantumWorkPolicy");
    expect(backendCreate).toContain("device: observer.device");
    expect(WORKER_SOURCE).not.toContain("fromPreparedResources");
    expect(WORKER_SOURCE).toContain("createAceOpt0008VaeWindowAttribution({");
    expect(WORKER_SOURCE).toContain("validateAceOpt0008VaeWindowTrace(");
    expect(WORKER_SOURCE).toContain("summarizeAceOpt0008VaeWindowTrace(");
  });

  it("prefills and drains the complete output outside both measured decodes", () => {
    expect(WORKER_SOURCE).toContain(
      "OPT_0008_OUTPUT_SENTINEL_BITS = 0x7fc0_0000",
    );
    expect(WORKER_SOURCE.match(/await prefillCompleteOutput\(/g)).toHaveLength(2);
    const prefill = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function prefillCompleteOutput"),
      WORKER_SOURCE.indexOf("async function summarizeOutput"),
    );
    expect(prefill).toContain("queue.writeBuffer(output, 0, sentinel)");
    expect(prefill).toContain("await queue.onSubmittedWorkDone()");
    const timed = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function runTimedAndCleanup"),
      WORKER_SOURCE.indexOf("class ProgressRouter"),
    );
    expect(timed.indexOf("await prefillCompleteOutput(")).toBeLessThan(
      timed.indexOf("prepared.observer.beginTimedTrace()"),
    );
    expect(WORKER_SOURCE).toContain("sentinelBitCount !== 0");
    expect(WORKER_SOURCE).toContain("finiteCount !== OPT_0008_OUTPUT_ELEMENTS");
    expect(WORKER_SOURCE).toContain("compareOutputs(");
    expect(WORKER_SOURCE).toContain("bitMismatchCount !== 0");
    expect(WORKER_SOURCE).toContain("sha256: await sha256Hex(bytesOf(output))");
  });

  it("observes every existing pass, dispatch, submit, drain, and fenced idle", () => {
    expect(WORKER_SOURCE).toContain("new Proxy(target");
    expect(WORKER_SOURCE).toContain("property === \"createCommandEncoder\"");
    expect(WORKER_SOURCE).toContain("property === \"beginComputePass\"");
    expect(WORKER_SOURCE).toContain("passProperty === \"dispatchWorkgroups\"");
    expect(WORKER_SOURCE).toContain("property === \"submit\"");
    expect(WORKER_SOURCE).toContain("property === \"onSubmittedWorkDone\"");
    expect(WORKER_SOURCE).toContain("prior.nextCommandEncodeStartedAt = now");
    expect(WORKER_SOURCE).toContain("actualDispatchCount !== OPT_0008_PRIMITIVE_DISPATCH_COUNT");
    expect(WORKER_SOURCE).toContain("ace-vae-window-0-readback");
    expect(WORKER_SOURCE).toContain("performance.timeOrigin +");
    expect(WORKER_SOURCE).toContain("fenced drain-end-to-next-encode interval");
    expect(WORKER_SOURCE).toContain("attributionPlan: prepared.attribution");
    expect(WORKER_SOURCE).toContain("actualCommandTags");
    expect(WORKER_SOURCE).not.toContain("postMessage({ type: \"batch\"");
  });

  it("leaves final thermal attribution to the externally hashed continuous log", () => {
    expect(HTML_SOURCE).toContain("external thermal");
    expect(HTML_SOURCE).toContain("through cleanup");
    expect(HTML_SOURCE).toContain("joined to");
    expect(WORKER_SOURCE).toContain('status: "pending-external-artifact-join"');
    expect(WORKER_SOURCE).toContain("preGateOnly: true");
    expect(WORKER_SOURCE).toContain(
      "browserReceiptClaimsPlanValidThermalCoverage: false",
    );
    expect(WORKER_SOURCE).toContain(
      "continuousLoggerRequiredThroughEpochMilliseconds",
    );
    expect(WORKER_SOURCE).toContain("timedStartedAtEpochMilliseconds");
    expect(WORKER_SOURCE).toContain("timedCompletedAtEpochMilliseconds");
    expect(WORKER_SOURCE).toContain("cleanupCompletedAtEpochMilliseconds");
    expect(WORKER_SOURCE).toContain(
      "completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds",
    );
  });

  it("cancels only after the first production batch drain and proves cleanup", () => {
    const cancellation = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("async function runCancellationProof"),
      WORKER_SOURCE.indexOf("async function verifyPostDestroy"),
    );
    expect(cancellation).toContain("event.completedCommandBuffers === 1");
    expect(cancellation).toContain("controller.abort(new DOMException(");
    expect(cancellation).toContain("progress.length !== 8");
    expect(cancellation).toContain("observed.records.length !== 1");
    expect(cancellation).toContain("observed.submissionCount !== 1");
    expect(cancellation).toContain("observed.drainCount !== 1");
    expect(cancellation).toContain("first.progressReportedAt < first.drainEndedAt");
    expect(cancellation).toContain("laterBatchEncodingPrevented");
    expect(cancellation).toContain("realIdleCompletedBeforeRejection: true");
    expect(WORKER_SOURCE).toContain("uniqueDestroyedBufferCount");
    expect(WORKER_SOURCE).toContain("liveTrackedBufferCount");
    expect(WORKER_SOURCE).toContain("OPT_0008_TRACKED_BUFFER_COUNT = 15");
    expect(WORKER_SOURCE).toContain("!resources.destructionTrackingSupported");
    expect(WORKER_SOURCE).toContain("decodeRejected: true");
    expect(WORKER_SOURCE).toContain("rejection.name !== \"InvalidStateError\"");
    expect(WORKER_SOURCE).toContain("prepared.context.destroy()");
  });
});
