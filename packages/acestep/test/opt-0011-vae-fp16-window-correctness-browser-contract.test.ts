import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { createAceOpt0011LatentFixture } from
  "../benchmark/opt-0011-vae-fp16-storage-window.js";
import {
  OPT_0011_FP16_WINDOW_BOUNDS,
  OPT_0011_FP16_WINDOW_FIXTURE_SHA256,
  OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS,
  OPT_0011_FP16_WINDOW_RUNTIME_COMMIT,
  OPT_0011_FP16_WINDOW_RUNTIME_SOURCE_SHA256,
  type Opt0011ExecutionTraceArm,
  type Opt0011ExecutionTraceTopology,
  compareOpt0011Fp16WindowU32,
  compareOpt0011Fp16WindowWaveforms,
  opt0011Fp16WindowGeneratedShaderSourceCount,
  parseOpt0011Fp16WindowRawResultChunkOffset,
  parseOpt0011Fp16WindowRunIdentity,
  sliceOpt0011Fp16WindowRawResultChunk,
  validateOpt0011ExecutionTraceTopology,
} from "./browser/opt-0011-vae-fp16-window-correctness.js";

const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-fp16-window-correctness.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-fp16-window-correctness.html",
  import.meta.url,
), "utf8");
const RUNTIME_SOURCE = readFileSync(new URL(
  "../src/webgpu/vae-fp16-decoder.ts",
  import.meta.url,
));

describe("OPT-0011 complete FP16 B-256 window browser contract", () => {
  it("pins the independently audited decoder runtime and frozen fixture", () => {
    expect(sha256(RUNTIME_SOURCE)).toBe(
      OPT_0011_FP16_WINDOW_RUNTIME_SOURCE_SHA256,
    );
    expect(OPT_0011_FP16_WINDOW_RUNTIME_COMMIT).toBe(
      "d5178ed84e3144e609c461af44e0c71d75d565ba",
    );
    const fixture = createAceOpt0011LatentFixture(256);
    expect(fixture).toHaveLength(65_536);
    expect(sha256(fixture)).toBe(OPT_0011_FP16_WINDOW_FIXTURE_SHA256);
    expect(HARNESS_SOURCE).toContain(
      "createAceOpt0011LatentFixture(FRAMES)",
    );
    expect(HARNESS_SOURCE).toContain(
      "OPT_0011_FP16_WINDOW_FIXTURE_SHA256",
    );
  });

  it("fails closed on the complete run identity before worker execution", () => {
    const valid = new URLSearchParams({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      runtimeCommit: OPT_0011_FP16_WINDOW_RUNTIME_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    expect(parseOpt0011Fp16WindowRunIdentity(valid)).toEqual({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      runtimeCommit: OPT_0011_FP16_WINDOW_RUNTIME_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "151.0.7922.138",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    for (const [name, value] of [
      ["harnessCommit", "short"],
      ["runtimeCommit", "0123456789abcdef0123456789abcdef01234567"],
      ["gpuCoreCount", "0"],
      ["memoryBytes", "9007199254740992"],
    ] as const) {
      const invalid = new URLSearchParams(valid);
      invalid.set(name, value);
      expect(() => parseOpt0011Fp16WindowRunIdentity(invalid)).toThrow(
        /OPT-0011/,
      );
    }
    const duplicate = new URLSearchParams(valid);
    duplicate.append("osBuild", "duplicate");
    expect(() => parseOpt0011Fp16WindowRunIdentity(duplicate)).toThrow(
      /osBuild/,
    );
  });

  it("keeps both heavyweight arms sequential in one dedicated worker", () => {
    expect(HARNESS_SOURCE).toContain("new Worker(workerUrl, { type: \"module\" })");
    expect(HARNESS_SOURCE).toContain(
      'searchParams.get("dedicatedWorker") === "1"',
    );
    expect(HARNESS_SOURCE).toContain("async function runWorker(");
    expect(HARNESS_SOURCE).toContain("async function runOracleArm(");
    expect(HARNESS_SOURCE).toContain("async function runCandidateArm(");
    const oracleRun = HARNESS_SOURCE.indexOf("const oracle = await runOracleArm(");
    const oracleDestroyed = HARNESS_SOURCE.indexOf('audit.liveCount("A") !== 0');
    const candidateRun = HARNESS_SOURCE.indexOf(
      "const candidate = await runCandidateArm(",
    );
    expect(oracleRun).toBeGreaterThan(0);
    expect(oracleDestroyed).toBeGreaterThan(oracleRun);
    expect(candidateRun).toBeGreaterThan(oracleDestroyed);
    expect(HARNESS_SOURCE).toContain(
      "allOracleResourcesDestroyedBeforeCandidateAcquisition: true",
    );
    expect(HARNESS_SOURCE).toContain("simultaneousHeavyweightPackageCount: 1");
    expect(HTML_SOURCE).toContain("dedicated worker");
    expect(HTML_SOURCE).toContain("destroys every oracle GPU resource");
  });

  it("authenticates only each exact VAE phase and both frozen package identities", () => {
    expect(HARNESS_SOURCE).toContain(
      '"/model/files-reference/manifest.json"',
    );
    expect(HARNESS_SOURCE).toContain(
      '"/model/files-fp16-vae-experimental/manifest.json"',
    );
    expect(HARNESS_SOURCE).toContain("ACE_VAE_FP32_ORACLE_MANIFEST_SHA256");
    expect(HARNESS_SOURCE).toContain("ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256");
    expect(HARNESS_SOURCE).toContain("ORACLE_VAE_TENSORS = 146");
    expect(HARNESS_SOURCE).toContain("ORACLE_VAE_FILES = 8");
    expect(HARNESS_SOURCE).toContain("ORACLE_VAE_BYTES = 337_583_104");
    expect(HARNESS_SOURCE).toContain("ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT");
    expect(HARNESS_SOURCE).toContain("ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES");
    expect(HARNESS_SOURCE.match(/AceGpuTensorPhase\.load\(/gu)).toHaveLength(2);
    expect(HARNESS_SOURCE).toContain('["vae"]');
    expect(HARNESS_SOURCE).toContain("files: prepared.files");
    expect(HARNESS_SOURCE).toContain("acquired.plan.runtimeBytes");
  });

  it("executes A twice and B twice through complete batch-eight windows", () => {
    expect(HARNESS_SOURCE.match(/backend\.decodeWindow\(/gu)).toHaveLength(2);
    expect(HARNESS_SOURCE.match(/executeCandidateWindow\(/gu)).toHaveLength(3);
    expect(HARNESS_SOURCE).toContain("B_SEQUENCE_QUANTA = 3_943");
    expect(HARNESS_SOURCE).toContain("B_GRAPH_QUANTA = 3_942");
    expect(HARNESS_SOURCE).toContain("B_COMPUTE_COMMAND_BUFFERS = 493");
    expect(HARNESS_SOURCE).toContain("TOTAL_COMMAND_BUFFERS = 494");
    expect(HARNESS_SOURCE).toContain("COMPLETED_IDLE_TURNS = 493");
    expect(HARNESS_SOURCE).toContain(
      "ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER",
    );
    expect(HARNESS_SOURCE).toContain("scheduler.runLazy({");
    expect(HARNESS_SOURCE).toContain("queueDrains !== TOTAL_COMMAND_BUFFERS");
    expect(HARNESS_SOURCE).toContain("cooperativeIdleMs !== COMPLETED_IDLE_TURNS");
    expect(HARNESS_SOURCE).toContain("A_PRIMITIVE_DISPATCHES = 3_988");
    expect(HARNESS_SOURCE).toContain(
      'validateExecutionTrace(firstTrace, "oracle", "A first")',
    );
    expect(HARNESS_SOURCE).toContain(
      'validateExecutionTrace(mutableTrace, "candidate", id)',
    );
    expect(HARNESS_SOURCE).toContain("A_READBACK_COPIES = 1");
    expect(HARNESS_SOURCE).toContain("B_READBACK_COPIES = 13");
  });

  it("pins distinct production-A and candidate-B command topologies", () => {
    const oracle = validExecutionTopology("oracle", "A-first");
    const candidate = validExecutionTopology("candidate", "B-first");
    expect(() => validateOpt0011ExecutionTraceTopology(
      oracle,
      "oracle",
      "A first",
    )).not.toThrow();
    expect(() => validateOpt0011ExecutionTraceTopology(
      candidate,
      "candidate",
      "B first",
    )).not.toThrow();

    const incorrectOnePassOracle = Object.freeze({
      ...oracle,
      computePassCounts: Object.freeze([
        ...Array.from({ length: 493 }, () => 1),
        0,
      ]),
    });
    expect(() => validateOpt0011ExecutionTraceTopology(
      incorrectOnePassOracle,
      "oracle",
      "A diagnostic",
    )).toThrow(
      /compute-pass topology changed: .*"computeBatchPassHistogram":\{"1":493\}/u,
    );

    const incorrectCandidateReadback = Object.freeze({
      ...candidate,
      copyCounts: Object.freeze([
        ...Array.from({ length: 493 }, () => 0),
        1,
      ]),
    });
    expect(() => validateOpt0011ExecutionTraceTopology(
      incorrectCandidateReadback,
      "candidate",
      "B diagnostic",
    )).toThrow(
      /compute\/readback command topology changed: .*"readback":\{.*"copyCount":1/u,
    );
    const rejectMutation = (
      topology: Opt0011ExecutionTraceTopology,
      arm: Opt0011ExecutionTraceArm,
    ): void => {
      expect(() => validateOpt0011ExecutionTraceTopology(
        topology,
        arm,
        "mutation diagnostic",
      )).toThrow(/changed: .*"computeBatchPassHistogram"/u);
    };
    rejectMutation(Object.freeze({
      ...oracle,
      computePassCounts: candidate.computePassCounts,
      dispatchCounts: candidate.dispatchCounts,
    }), "oracle");
    const incorrectFinalOraclePass = [...oracle.computePassCounts];
    incorrectFinalOraclePass[492] = 8;
    rejectMutation(Object.freeze({
      ...oracle,
      computePassCounts: Object.freeze(incorrectFinalOraclePass),
    }), "oracle");
    const computeCopy = [...oracle.copyCounts];
    computeCopy[0] = 1;
    rejectMutation(Object.freeze({
      ...oracle,
      copyCounts: Object.freeze(computeCopy),
    }), "oracle");
    const computeClear = [...candidate.clearCounts];
    computeClear[7] = 1;
    rejectMutation(Object.freeze({
      ...candidate,
      clearCounts: Object.freeze(computeClear),
    }), "candidate");
    rejectMutation(Object.freeze({
      ...candidate,
      writeBufferCount: 1,
    }), "candidate");
    expect(HARNESS_SOURCE).toContain("computeBatchPassHistogram");
    expect(HARNESS_SOURCE).toContain("computeBatchDispatchHistogram");
    expect(HARNESS_SOURCE).toContain("computeBatchCopyHistogram");
    expect(HARNESS_SOURCE).toContain("computeBatchClearHistogram");
  });

  it("binds the complete B graph and all 39 portable shader modules", () => {
    expect(opt0011Fp16WindowGeneratedShaderSourceCount()).toBe(34);
    expect(HARNESS_SOURCE).toContain("resolveAceOpt0011Fp16VaePackageBindings(");
    expect(HARNESS_SOURCE).toContain("AceOpt0011Fp16VaeDecoderRuntime.create(");
    expect(HARNESS_SOURCE).toContain("operationCount !== B_OPERATION_COUNT");
    expect(HARNESS_SOURCE).toContain('kinds["ingress-cast"] !== 1');
    expect(HARNESS_SOURCE).toContain("kinds.conv1d !== 2_459");
    expect(HARNESS_SOURCE).toContain('kinds["conv-transpose1d"] !== 322');
    expect(HARNESS_SOURCE).toContain("kinds.snake !== 813");
    expect(HARNESS_SOURCE).toContain("kinds.add !== 348");
    expect(HARNESS_SOURCE).toContain(
      'shaderIdentity["shaderModuleCreateCount"] !== 39',
    );
    expect(HARNESS_SOURCE).toContain(
      'shaderIdentity["uniqueShaderCount"] !== 34',
    );
    expect(HARNESS_SOURCE).toContain("topologySha256");
    expect(HARNESS_SOURCE).toContain("operationBindingSha256");
    expect(HARNESS_SOURCE).toContain("precisionMapSha256");
  });

  it("requires complete qNaN-prefilled raw output, guards, and U32 reruns", () => {
    expect(HARNESS_SOURCE).toContain("OUTPUT_ELEMENTS = 983_040");
    expect(HARNESS_SOURCE).toContain("OUTPUT_BYTES = 3_932_160");
    expect(HARNESS_SOURCE).toContain("OUTPUT_QNAN_WORD = 0x7fc5_0011");
    expect(HARNESS_SOURCE).toContain("words.fill(OUTPUT_QNAN_WORD)");
    expect(HARNESS_SOURCE).toContain("qNaNSentinelCount !== 0");
    expect(HARNESS_SOURCE).toContain("finiteCount !== OUTPUT_ELEMENTS");
    expect(HARNESS_SOURCE).toContain("stereoDifferenceFrameCount === 0");
    expect(HARNESS_SOURCE).toContain("GUARD_BYTES = 256");
    expect(HARNESS_SOURCE).toContain("validateGuards(");
    expect(HARNESS_SOURCE).toContain("compareU32(first, rerun)");

    const positiveZero = new Float32Array([0]);
    const negativeZero = new Float32Array([-0]);
    expect(compareOpt0011Fp16WindowU32(positiveZero, negativeZero)).toEqual({
      comparedWordCount: 1,
      mismatchCount: 1,
      firstMismatchIndex: 0,
      bitExact: false,
    });
  });

  it("freezes and applies every declared complete-waveform A/B bound", () => {
    expect(OPT_0011_FP16_WINDOW_BOUNDS).toEqual({
      maximumNormalizedRmsError: 0.003,
      minimumSnrDecibels: 50,
      minimumCorrelation: 0.9999,
      maximumRelativeRmsDrift: 0.005,
      maximumRelativeEnergyDrift: 0.005,
      maximumRelativePeakDrift: 0.01,
      maximumDcDriftScale: 0.001,
      maximumNormalizedAbsoluteError: 0.02,
    });
    for (const token of [
      "maximumNormalizedRmsError",
      "minimumSnrDecibels",
      "minimumCorrelation",
      "maximumRelativeRmsDrift",
      "maximumRelativeEnergyDrift",
      "maximumRelativePeakDrift",
      "maximumDcDriftScale",
      "maximumNormalizedAbsoluteError",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(HARNESS_SOURCE).toContain(
      "!(joint.correlation >= bounds.minimumCorrelation)",
    );
    expect(HARNESS_SOURCE).toContain(
      "!(normalizedMaximumAbsoluteError <= bounds.maximumNormalizedAbsoluteError)",
    );

    const oracle = new Float32Array(983_040);
    for (let index = 0; index < oracle.length; index += 1) {
      oracle[index] = Math.fround(Math.sin(index * 0.001) * 0.5);
    }
    const metrics = compareOpt0011Fp16WindowWaveforms(oracle, oracle);
    expect(metrics["maximumAbsoluteError"]).toBe(0);
    expect((metrics["joint"] as { normalizedRmsError: number })
      .normalizedRmsError).toBe(0);
  });

  it("cancels after exactly one drained and idled eight-quantum batch", () => {
    const cancellation = HARNESS_SOURCE.slice(
      HARNESS_SOURCE.indexOf("async function runCandidateCancellation"),
      HARNESS_SOURCE.indexOf("function validateCandidateDispatch"),
    );
    expect(cancellation).toContain("progress.completedCommandBuffers === 1");
    expect(cancellation).toContain("trace.commands.length !== 1");
    expect(cancellation).toContain("trace.submissionCount !== 1");
    expect(cancellation).toContain("trace.drainCount !== 1");
    expect(cancellation).toContain("trace.commands[0]!.computePassCount !== 1");
    expect(cancellation).toContain("trace.commands[0]!.dispatchCount !== 8");
    expect(cancellation).toContain("trace.commands[0]!.copyCount !== 0");
    expect(cancellation).toContain("trace.commands[0]!.clearCount !== 0");
    expect(cancellation).toContain("readbackPrevented: true");
    expect(cancellation).toContain("metricsPublicationPrevented: true");
    expect(cancellation).toContain(
      "realQueueEmptyIdleDeliveredBeforeRejection: true",
    );
  });

  it("keeps a compact receipt, lifecycle heartbeats, and bounded retrieval", () => {
    expect(HARNESS_SOURCE).not.toContain("rawWaveform: Array.from");
    expect(HARNESS_SOURCE).not.toContain("quanta: dispatch.quanta");
    expect(HARNESS_SOURCE).toContain("completeCompactTrace: true");
    expect(HARNESS_SOURCE).toContain("maximumAnimationFrameGapMilliseconds");
    expect(HARNESS_SOURCE).toContain("maximumTimerGapMilliseconds");
    expect(HARNESS_SOURCE).toContain("audit.destroyAll();\n    audit.destroyAll();");
    expect(HARNESS_SOURCE).toContain('loss.reason !== "destroyed"');
    expect(HARNESS_SOURCE).toContain("runtimeEvents.length !== 0");
    expect(HARNESS_SOURCE).toContain(
      "preDestroyRuntimeObservationMacrotaskTurnCount: 2",
    );
    expect(HARNESS_SOURCE).toContain("postLossMacrotaskTurnCount: 2");
    expect(HARNESS_SOURCE.match(/await yieldToBrowser\(\);/gu)).toHaveLength(4);
    const preDestroySettlement = HARNESS_SOURCE.indexOf(
      "OPT-0011 observed a queued runtime event before destroy",
    );
    const contextDestroy = HARNESS_SOURCE.indexOf("context.destroy();");
    const postLossSettlement = HARNESS_SOURCE.indexOf(
      "OPT-0011 observed a runtime event through post-loss cleanup",
    );
    const heartbeatStop = HARNESS_SOURCE.indexOf(
      "const workerHeartbeat = heartbeat.stop();",
    );
    expect(preDestroySettlement).toBeGreaterThan(0);
    expect(contextDestroy).toBeGreaterThan(preDestroySettlement);
    expect(postLossSettlement).toBeGreaterThan(contextDestroy);
    expect(heartbeatStop).toBeGreaterThan(postLossSettlement);
    expect(HTML_SOURCE).toContain("raw-result-retrieval");
    expect(HTML_SOURCE).toContain("raw-result-chunk");

    expect(parseOpt0011Fp16WindowRawResultChunkOffset("0")).toBe(0);
    expect(parseOpt0011Fp16WindowRawResultChunkOffset("32768")).toBe(32_768);
    expect(() => parseOpt0011Fp16WindowRawResultChunkOffset("01"))
      .toThrow(/canonical/);
    const raw = "a".repeat(
      OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS + 7,
    );
    const first = sliceOpt0011Fp16WindowRawResultChunk(raw, 0);
    expect(first.chunk).toHaveLength(
      OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS,
    );
    expect(first.complete).toBe(false);
    expect(sliceOpt0011Fp16WindowRawResultChunk(raw, first.nextOffset))
      .toMatchObject({ chunk: "a".repeat(7), complete: true });
  });

  it("makes no timing, thermal, listening, selector, or FP16-512 claim", () => {
    expect(HTML_SOURCE).toContain("records no performance or thermal result");
    expect(HARNESS_SOURCE).toContain("performanceClaim: null");
    expect(HARNESS_SOURCE).toContain("thermalClaim: null");
    expect(HARNESS_SOURCE).toContain("listeningClaim: null");
    expect(HARNESS_SOURCE).toContain("selectorClaim: null");
    expect(HARNESS_SOURCE).toContain("productionIntegrationClaim: null");
    expect(HARNESS_SOURCE).toContain("fp16512Claim: null");
    expect(HARNESS_SOURCE).not.toContain("run-timed");
    expect(HARNESS_SOURCE).not.toContain("timestamp-query");
  });
});

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function validExecutionTopology(
  arm: Opt0011ExecutionTraceArm,
  id: string,
): Opt0011ExecutionTraceTopology {
  const computePassCounts = arm === "oracle"
    ? [
      ...Array.from({ length: 492 }, () => 8),
      6,
    ]
    : Array.from({ length: 493 }, () => 1);
  const dispatchCounts = arm === "oracle"
    ? computePassCounts.map((count, index) => {
      if (index === 0) return 14;
      if (index <= 5) return 16;
      return count;
    })
    : [
      ...Array.from({ length: 492 }, () => 8),
      7,
    ];
  const commandLabels = Array.from(
    { length: 493 },
    (_, index) => arm === "oracle"
      ? `ace-vae-window-0-batch-${index}`
      : `${id}-batch-${index}`,
  );
  commandLabels.push(
    arm === "oracle" ? "ace-vae-window-0-readback" : `${id}-readback`,
  );
  return Object.freeze({
    id,
    commandLabels: Object.freeze(commandLabels),
    computePassCounts: Object.freeze([...computePassCounts, 0]),
    dispatchCounts: Object.freeze([...dispatchCounts, 0]),
    copyCounts: Object.freeze([
      ...Array.from({ length: 493 }, () => 0),
      arm === "oracle" ? 1 : 13,
    ]),
    clearCounts: Object.freeze(Array.from({ length: 494 }, () => 0)),
    submissionCount: 494,
    drainCount: 494,
    writeBufferCount: arm === "oracle" ? 1 : 0,
    incompleteCommandCount: 0,
  });
}
