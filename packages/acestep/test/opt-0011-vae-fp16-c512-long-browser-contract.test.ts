import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { createAceOpt0011LatentFixture } from
  "../benchmark/opt-0011-vae-fp16-storage-window.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8,
  ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES,
  ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  planAceOpt0011Fp16VaeChunkDispatches,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  OPT_0011_C512_DIRECT_FIXTURE_SHA256,
  OPT_0011_C512_LONG_B256_ARTIFACT_BYTES,
  OPT_0011_C512_LONG_B256_ARTIFACT_SHA256,
  OPT_0011_C512_LONG_B256_BINDING_SHA256,
  OPT_0011_C512_LONG_B256_OUTPUT_SHA256,
  OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256,
  OPT_0011_C512_LONG_FIXTURE_SHA256,
  OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS,
  OPT_0011_C512_LONG_RUNTIME_COMMIT,
  OPT_0011_C512_LONG_RUNTIME_SOURCE_SHA256,
  compareOpt0011C512LongU32,
  opt0011C512LongExpectedTopology,
  parseOpt0011C512LongRawResultChunkOffset,
  parseOpt0011C512LongRunIdentity,
  sliceOpt0011C512LongRawResultChunk,
} from "./browser/opt-0011-vae-fp16-c512-long-correctness.js";

const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-fp16-c512-long-correctness.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-fp16-c512-long-correctness.html",
  import.meta.url,
), "utf8");
const RUNTIME_SOURCE = readFileSync(new URL(
  "../src/webgpu/vae-fp16-decoder.ts",
  import.meta.url,
));
const B256_AUTHORITY_BYTES = readFileSync(new URL(
  "../optimization/artifacts/OPT-0011/raw/fp16-b256-window-correctness.json",
  import.meta.url,
));

describe("OPT-0011 FP16 C-512 and long-latent browser contract", () => {
  it("pins the generalized runtime, accepted B-256 authority, and both fixtures", () => {
    expect(OPT_0011_C512_LONG_RUNTIME_COMMIT).toBe(
      "d5178ed84e3144e609c461af44e0c71d75d565ba",
    );
    expect(OPT_0011_C512_LONG_RUNTIME_SOURCE_SHA256).toBe(
      "dd83ec341e7d27d2c4f3cc1f673c1d7d7a05818212de12f76146ae546bb508d7",
    );
    expect(sha256(RUNTIME_SOURCE)).toBe(
      OPT_0011_C512_LONG_RUNTIME_SOURCE_SHA256,
    );

    expect(OPT_0011_C512_LONG_B256_ARTIFACT_BYTES).toBe(28_763);
    expect(B256_AUTHORITY_BYTES).toHaveLength(
      OPT_0011_C512_LONG_B256_ARTIFACT_BYTES,
    );
    expect(sha256(B256_AUTHORITY_BYTES)).toBe(
      OPT_0011_C512_LONG_B256_ARTIFACT_SHA256,
    );
    expect(OPT_0011_C512_LONG_B256_ARTIFACT_SHA256).toBe(
      "827d6d46feeac13ad45e78487d4e19e9fca4c10baacf139499206a5c497f1f54",
    );
    expect(OPT_0011_C512_LONG_B256_OUTPUT_SHA256).toBe(
      "782ac4036233045b7facbc583369f31c5c74dd83ff4d3197daaeccc829886327",
    );
    expect(OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256).toBe(
      "9b950a596069ee381e178d1ccc4cc5d5c4bc77d2ba37c6727bd9fdb3216ed9f9",
    );
    expect(OPT_0011_C512_LONG_B256_BINDING_SHA256).toBe(
      "5664970bd6dc29c4168dbf75cef7e17c4340a2057d4ee34591b607e8ae62e89a",
    );

    const authority = JSON.parse(B256_AUTHORITY_BYTES.toString("utf8")) as {
      schema: string;
      status: string;
      candidate: {
        first: { elementCount: number; byteLength: number; sha256: string };
        graphIdentity: {
          sequenceQuantumCount: number;
          topologySha256: string;
          operationBindingSha256: string;
          precisionMapSha256: string;
        };
      };
    };
    expect(authority).toMatchObject({
      schema: "ace-opt-0011-fp16-vae-b256-window-correctness-v1",
      status: "passed",
      candidate: {
        first: {
          elementCount: 983_040,
          byteLength: 3_932_160,
          sha256: OPT_0011_C512_LONG_B256_OUTPUT_SHA256,
        },
        graphIdentity: {
          sequenceQuantumCount: 3_943,
          topologySha256: OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256,
          operationBindingSha256:
            OPT_0011_C512_LONG_B256_BINDING_SHA256,
          precisionMapSha256:
            "465d40d9150912abc7ddfaa3ccf2b80c7639fd304722a495cdd37c16119d64ee",
        },
      },
    });

    const directB = createAceOpt0011LatentFixture(256);
    const directC = createAceOpt0011LatentFixture(512);
    const long = createAceOpt0011LatentFixture(1_024);
    expect(directB).toHaveLength(65_536);
    expect(directC).toHaveLength(131_072);
    expect(long).toHaveLength(262_144);
    expect(sha256(directB)).toBe(
      "55333d3ae4a0aca83dc1509b837c577f54646924e658e01e53889dc8a5a44875",
    );
    expect(sha256(directC)).toBe(OPT_0011_C512_DIRECT_FIXTURE_SHA256);
    expect(sha256(long)).toBe(OPT_0011_C512_LONG_FIXTURE_SHA256);
    expect(OPT_0011_C512_DIRECT_FIXTURE_SHA256).toBe(
      "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899",
    );
    expect(OPT_0011_C512_LONG_FIXTURE_SHA256).toBe(
      "e8919adc02d83f2efcd60bcb6dec4f104628d2ed66742d0eddbffc6b0a481a14",
    );
  });

  it("fails closed on complete run identity before worker execution", () => {
    const valid = new URLSearchParams({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      runtimeCommit: OPT_0011_C512_LONG_RUNTIME_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    expect(parseOpt0011C512LongRunIdentity(valid)).toEqual({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      runtimeCommit: OPT_0011_C512_LONG_RUNTIME_COMMIT,
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
      expect(() => parseOpt0011C512LongRunIdentity(invalid)).toThrow(
        /OPT-0011/,
      );
    }
    const duplicate = new URLSearchParams(valid);
    duplicate.append("osBuild", "duplicate");
    expect(() => parseOpt0011C512LongRunIdentity(duplicate)).toThrow(
      /osBuild/,
    );
  });

  it("uses one dedicated worker and loads only the revision-5 candidate VAE", () => {
    expect(HARNESS_SOURCE).toContain("new Worker(workerUrl, { type: \"module\" })");
    expect(HARNESS_SOURCE).toContain(
      'searchParams.get("dedicatedWorker") === "1"',
    );
    expect(HARNESS_SOURCE).toContain("async function runWorker(");
    expect(HARNESS_SOURCE).toContain(
      '"/model/files-fp16-vae-experimental/manifest.json"',
    );
    expect(HARNESS_SOURCE).toContain("ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256");
    expect(HARNESS_SOURCE.match(/AceGpuTensorPhase\.load\(/gu)).toHaveLength(1);
    expect(HARNESS_SOURCE).toContain('["vae"]');
    expect(HARNESS_SOURCE).not.toContain("/model/files-reference/");
    expect(HARNESS_SOURCE).not.toContain("AceVaeChunkGpuBackend");
    expect(HARNESS_SOURCE).not.toContain("runOracleArm");
    const sourceAuthentication = HARNESS_SOURCE.indexOf(
      "const sourceAuthority = await authenticateSources()",
    );
    const b256Authentication = HARNESS_SOURCE.indexOf(
      "const b256Authority = await authenticateB256Artifact()",
    );
    const deviceRequest = HARNESS_SOURCE.indexOf(
      "const context = await requestAceWebGpuDevice(",
    );
    expect(sourceAuthentication).toBeGreaterThan(0);
    expect(b256Authentication).toBeGreaterThan(sourceAuthentication);
    expect(deviceRequest).toBeGreaterThan(b256Authentication);
    expect(HTML_SOURCE).toContain("one dedicated worker");
    expect(HTML_SOURCE).toMatch(/No FP32 oracle package is\s+loaded/u);
  });

  it("keeps one package resident while B and C arm resources remain sequential", () => {
    const bRun = HARNESS_SOURCE.indexOf("const b = await runBPhase(");
    const bDestroyed = HARNESS_SOURCE.indexOf(
      'audit.liveCount("B") !== 0 || audit.liveCount("package") !== 7',
    );
    const cRun = HARNESS_SOURCE.indexOf("const c = await runCPhase(");
    const cDestroyed = HARNESS_SOURCE.indexOf(
      'audit.liveCount("C") !== 0 || audit.liveCount("package") !== 7',
    );
    const packageDestroyed = HARNESS_SOURCE.indexOf("phase.destroy();");
    expect(bRun).toBeGreaterThan(0);
    expect(bDestroyed).toBeGreaterThan(bRun);
    expect(cRun).toBeGreaterThan(bDestroyed);
    expect(cDestroyed).toBeGreaterThan(cRun);
    expect(packageDestroyed).toBeGreaterThan(cDestroyed);
    expect(HARNESS_SOURCE).toContain("packageLoadCount: 1");
    expect(HARNESS_SOURCE).toContain("simultaneousArmResourceSetCount: 1");
    expect(HARNESS_SOURCE).toContain("allBResourcesDestroyedBeforeC: true");
    expect(HARNESS_SOURCE).toContain("packageResidentAcrossSequentialArms: true");
  });

  it("pins exact C-512 and unpadded B/C long topology", () => {
    const c512 = planAceOpt0011Fp16VaeChunkDispatches(512, 512, 256);
    expect(c512).toMatchObject({
      maximumWindowFramesProfile: 512,
      uniqueWindowFrames: [512],
      windowTopologyIndices: [0],
      maximumFp16WorkspaceBytes: ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      aggregateGraphQuantumCount:
        ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
      aggregateSequenceQuantumCount:
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
      aggregateCommandBufferCountAtBatch8:
        ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8,
    });
    expect(c512.topologies[0]).toMatchObject({
      inputFrames: 512,
      operationCount: 88,
      graphQuantumCount: 7_854,
      sequenceQuantumCount: 7_855,
      quantumFamilyCounts: {
        conv1d: 4_909,
        "conv-transpose1d": 644,
        snake: 1_611,
        add: 690,
      },
      activeStagingInputBytes: 131_072,
      activeDecoderInputBytes: 65_536,
      activeOutputBytes: 7_864_320,
      fp16WorkspaceBytes: 251_658_240,
      decoderCommandBufferCountAtBatch8: 982,
      commandBufferCountAtBatch8: 983,
    });
    expect(ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES).toBe(2_010_640);

    const b = planAceOpt0011Fp16VaeChunkDispatches(1_024, 256, 256);
    expect(b.uniqueWindowFrames).toEqual([192, 256]);
    expect(b.chunkPlan.windows.map((window) => window.latentWindowFrames))
      .toEqual([192, 256, 256, 256, 256, 256, 256, 192]);
    expect(b.windowTopologyIndices).toEqual([0, 1, 1, 1, 1, 1, 1, 0]);
    expect(b).toMatchObject({
      aggregateGraphQuantumCount: 29_586,
      aggregateSequenceQuantumCount: 29_594,
      aggregateCommandBufferCountAtBatch8: 3_708,
      maximumFp16WorkspaceBytes: 125_829_120,
    });

    const c = planAceOpt0011Fp16VaeChunkDispatches(1_024, 512, 256);
    expect(c.uniqueWindowFrames).toEqual([320, 448, 512]);
    expect(c.chunkPlan.windows.map((window) => window.latentWindowFrames))
      .toEqual([448, 512, 320]);
    expect(c.windowTopologyIndices).toEqual([1, 2, 0]);
    expect(c).toMatchObject({
      aggregateGraphQuantumCount: 19_684,
      aggregateSequenceQuantumCount: 19_687,
      aggregateCommandBufferCountAtBatch8: 2_465,
      maximumFp16WorkspaceBytes: 251_658_240,
    });

    expect(opt0011C512LongExpectedTopology()).toEqual({
      C512: {
        operationCount: 88,
        graphQuantumCount: 7_854,
        sequenceQuantumCount: 7_855,
        familyCounts: {
          conv1d: 4_909,
          transpose: 644,
          snake: 1_611,
          add: 690,
        },
        controlBytes: 2_010_640,
        workspaceBytes: 251_658_240,
        logicalGpuBytes: 941_702_160,
        computeCommandBufferCount: 982,
        commandBufferCount: 983,
      },
      B: {
        uniqueWindowFrames: [192, 256],
        windowFrames: [192, 256, 256, 256, 256, 256, 256, 192],
        windowCount: 8,
        graphQuantumCount: 29_586,
        sequenceQuantumCount: 29_594,
        commandBufferCount: 3_708,
        controlBytes: 1_768_736,
        maximumWorkspaceBytes: 125_829_120,
        uniqueShapeFamilyCounts: {
          conv1d: 4_302,
          transpose: 565,
          snake: 1_430,
          add: 612,
        },
        aggregateWindowFamilyCounts: {
          conv1d: 18_440,
          transpose: 2_418,
          snake: 6_112,
          add: 2_616,
        },
        decodedLatentFrames: 1_920,
        scheduledDecodedFloat32Bytes: 29_491_200,
        retainedOutputFloat32Bytes: 15_728_640,
        logicalGpuBytes: 556_010_272,
      },
      C: {
        uniqueWindowFrames: [320, 448, 512],
        windowFrames: [448, 512, 320],
        windowCount: 3,
        graphQuantumCount: 19_684,
        sequenceQuantumCount: 19_687,
        commandBufferCount: 2_465,
        controlBytes: 5_039_152,
        maximumWorkspaceBytes: 251_658_240,
        uniqueShapeFamilyCounts: {
          conv1d: 12_275,
          transpose: 1_606,
          snake: 4_063,
          add: 1_740,
        },
        aggregateWindowFamilyCounts: {
          conv1d: 12_275,
          transpose: 1_606,
          snake: 4_063,
          add: 1_740,
        },
        decodedLatentFrames: 1_280,
        scheduledDecodedFloat32Bytes: 19_660_800,
        retainedOutputFloat32Bytes: 15_728_640,
        logicalGpuBytes: 944_730_672,
      },
      fullGate: {
        fullWindowExecutionCount: 26,
        partialWindowExecutionCount: 1,
        dispatchCount: 121_191,
        commandBufferCount: 15_179,
        readbackCommandBufferCount: 26,
        readbackCopyCount: 338,
        completedRealIdleCount: 15_172,
        rawU32ComparisonCount: 26_050_560,
        createdBufferCount: 28,
        maximumLiveBufferCount: 18,
        lifetimeCreatedBufferBytes: 1_331_961_680,
      },
    });
  });

  it("authenticates the B-256 authority and reproduces its output bridge", () => {
    expect(HARNESS_SOURCE).toContain(
      '"/optimization/artifacts/OPT-0011/raw/fp16-b256-window-correctness.json"',
    );
    expect(HARNESS_SOURCE).toContain(
      "OPT_0011_C512_LONG_B256_ARTIFACT_SHA256",
    );
    expect(HARNESS_SOURCE).toContain("OPT_0011_C512_LONG_B256_OUTPUT_SHA256");
    expect(HARNESS_SOURCE).toContain("OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256");
    expect(HARNESS_SOURCE).toContain("OPT_0011_C512_LONG_B256_BINDING_SHA256");
    expect(HARNESS_SOURCE).toContain(
      "const bridgeExecution = await executeWindow(",
    );
    expect(HARNESS_SOURCE).toContain(
      'classification: "one-candidate-execution-bridge-no-A-load-no-rerun"',
    );
    expect(HARNESS_SOURCE).toContain("generic B256 output bridge failed");
    expect(HARNESS_SOURCE).toContain("983_040");
    expect(HTML_SOURCE).toContain("reproduces the accepted B-256 output");
  });

  it("validates and emits aggregate actual gate accounting", () => {
    const aggregation = HARNESS_SOURCE.indexOf(
      "const actualGateAccounting = audit.aggregateActualExecutionAccounting()",
    );
    const validation = HARNESS_SOURCE.indexOf(
      "validateActualGateAccounting(actualGateAccounting)",
      aggregation,
    );
    const aggregateGateReceipt = HARNESS_SOURCE.indexOf(
      "aggregateGate: opt0011C512LongExpectedTopology().fullGate",
      validation,
    );
    const actualReceipt = HARNESS_SOURCE.indexOf(
      "actualGateAccounting,",
      aggregateGateReceipt,
    );
    expect(aggregation).toBeGreaterThan(0);
    expect(validation).toBeGreaterThan(aggregation);
    expect(aggregateGateReceipt).toBeGreaterThan(validation);
    expect(actualReceipt).toBeGreaterThan(aggregateGateReceipt);

    expect(HARNESS_SOURCE).toContain(
      "aggregateActualExecutionAccounting(): ActualGateAccounting",
    );
    expect(HARNESS_SOURCE).toContain(
      "function validateActualGateAccounting(actual: ActualGateAccounting)",
    );
    expect(HARNESS_SOURCE).toContain(
      "for (const [name, value] of Object.entries(expected))",
    );
    expect(HARNESS_SOURCE).toContain(
      "actual.executionTraceCount !==\n      actual.fullWindowExecutionCount + actual.partialWindowExecutionCount",
    );
    expect(HARNESS_SOURCE).toContain(
      "actual.queueDrainCount !== actual.commandBufferCount",
    );
    expect(HARNESS_SOURCE).toContain(
      "actual.readbackCommandBufferCount * READBACK_COPY_COUNT",
    );
    expect(HARNESS_SOURCE).toContain(
      "everyCompletedExecutionAggregatedExactlyOnce: true",
    );
    expect(HARNESS_SOURCE).toContain(
      "this.rawU32ComparisonCount += result.comparedWordCount",
    );
  });

  it("poisons and verifies only the inactive maximum-output tail", () => {
    expect(HARNESS_SOURCE).toContain("OUTPUT_QNAN_WORD");
    expect(HARNESS_SOURCE).toContain("outputPoison.fill(OUTPUT_QNAN_WORD)");
    expect(HARNESS_SOURCE).toContain("validateInactiveOutputTail(");
    expect(HARNESS_SOURCE).toContain("comparedInactiveTailU32Count");
    expect(HARNESS_SOURCE).toContain("wrote its poisoned inactive output tail");
    expect(HARNESS_SOURCE).toContain("activeStagingInputBytes");
    expect(HARNESS_SOURCE).toContain("activeDecoderInputBytes");
    expect(HARNESS_SOURCE).toContain("activeOutputBytes");
    expect(HARNESS_SOURCE).toContain(
      "latent.byteLength !== dispatch.activeStagingInputBytes",
    );
    expect(HARNESS_SOURCE).toContain(
      "queue.writeBuffer(resources.stagingInput.buffer, GUARD_BYTES, latent)",
    );
    expect(HARNESS_SOURCE).toContain(
      "maximumOutputBindingWithPoisonedInactiveTail: true",
    );
    expect(HTML_SOURCE).toMatch(
      /poisoned inactive output-tail\s+preservation/u,
    );
  });

  it("executes C-512 twice and requires exact deterministic U32 output", () => {
    expect(HARNESS_SOURCE).toContain('"C512-first"');
    expect(HARNESS_SOURCE).toContain('"C512-rerun"');
    expect(HARNESS_SOURCE).toMatch(
      /const deterministic = audit\.compareU32\(\s*first\.activeOutput,\s*rerun\.activeOutput,?\s*\)/u,
    );
    expect(HARNESS_SOURCE).toContain("complete C512 deterministic rerun failed");
    expect(HARNESS_SOURCE).toContain("completeActiveOutputAndPhysicalTailChecked");

    const positiveZero = new Float32Array([0]);
    const negativeZero = new Float32Array([-0]);
    expect(compareOpt0011C512LongU32(positiveZero, negativeZero)).toEqual({
      comparedWordCount: 1,
      mismatchCount: 1,
      firstMismatchIndex: 0,
      worstMismatchIndex: 0,
      bitExact: false,
    });
  });

  it("traces every batch and the separate 13-copy readback command", () => {
    expect(HARNESS_SOURCE).toContain(
      "validateExecutionTrace(trace, dispatch, id)",
    );
    expect(HARNESS_SOURCE).toContain(
      "trace.commands.length !== totalCount",
    );
    expect(HARNESS_SOURCE).toContain(
      "trace.submissionCount !== totalCount || trace.drainCount !== totalCount",
    );
    expect(HARNESS_SOURCE).toContain("command.computePassCount !== 1");
    expect(HARNESS_SOURCE).toContain(
      "command.dispatchCount !== expectedDispatches",
    );
    expect(HARNESS_SOURCE).toContain(
      "readback.copyCount !== READBACK_COPY_COUNT",
    );
    expect(HARNESS_SOURCE).toContain("READBACK_COPY_COUNT = 13");
    expect(HARNESS_SOURCE).toContain(
      "everyCommandFinishedSubmittedAndDrained: true",
    );
    expect(HARNESS_SOURCE).toContain("firstExecutionTraces");
    expect(HARNESS_SOURCE).toContain("rerunExecutionTraces");
  });

  it("requires complete B/C long U32 identity, coverage, and seam receipts", () => {
    expect(HARNESS_SOURCE).toContain("bResult");
    expect(HARNESS_SOURCE).toContain("cResult");
    expect(HARNESS_SOURCE).toContain(
      "comparison = audit.compareU32(bResult.output, cResult.output)",
    );
    expect(HARNESS_SOURCE).toContain("requireExactCoverage(coverage");
    expect(HARNESS_SOURCE).toContain("everyAudioFrameCoveredExactlyOnce: true");
    expect(HARNESS_SOURCE).toContain("coverageGapCount");
    expect(HARNESS_SOURCE).toContain("coverageDuplicationCount");
    expect(HARNESS_SOURCE).toContain(
      "const bLatentSeams = Object.freeze([128, 256, 384, 512, 640, 768, 896])",
    );
    expect(HARNESS_SOURCE).toContain(
      "const cLatentSeams = Object.freeze([384, 768])",
    );
    expect(HARNESS_SOURCE).toContain("firstDifferenceJump");
    expect(HARNESS_SOURCE).toContain("localRms");
    expect(HARNESS_SOURCE).toContain("localP999AbsoluteFirstDifference");
    expect(HARNESS_SOURCE).toContain("matchedInterior");
    expect(HTML_SOURCE).toContain("exact B/C U32 waveform and hash equality");
    expect(HTML_SOURCE).toMatch(/exact once-only output\s+coverage/u);
  });

  it("cancels between windows and prevents all later work and publication", () => {
    expect(HARNESS_SOURCE).toContain("runBetweenWindowCancellation");
    expect(HARNESS_SOURCE).toContain("runBetweenBatchCancellation");
    expect(HARNESS_SOURCE).toContain("AbortError");
    expect(HARNESS_SOURCE).toContain("laterEncodingPrevented");
    expect(HARNESS_SOURCE).toContain("laterSubmissionPrevented");
    expect(HARNESS_SOURCE).toContain("readbackPrevented");
    expect(HARNESS_SOURCE).toContain("sinkWritePrevented");
    expect(HARNESS_SOURCE).toContain("normalizationPrevented");
    expect(HARNESS_SOURCE).toContain("outputFinalizationPrevented");
    expect(HARNESS_SOURCE).toContain("metricsPublicationPrevented");
    expect(HARNESS_SOURCE).toContain("drainBeforeRelease");
  });

  it("keeps lifecycle heartbeats, cleanup, and bounded DOM retrieval", () => {
    expect(HARNESS_SOURCE).not.toContain("rawWaveform: Array.from");
    expect(HARNESS_SOURCE).not.toContain("quanta: dispatch.quanta");
    expect(HARNESS_SOURCE).toContain("maximumAnimationFrameGapMilliseconds");
    expect(HARNESS_SOURCE).toContain("maximumTimerGapMilliseconds");
    expect(HARNESS_SOURCE).toContain("audit.destroyAll();\n    audit.destroyAll();");
    expect(HARNESS_SOURCE).toContain('loss.reason !== "destroyed"');
    expect(HARNESS_SOURCE).toContain("runtimeEvents.length !== 0");
    expect(HTML_SOURCE).toContain("raw-result-retrieval");
    expect(HTML_SOURCE).toContain("raw-result-chunk");

    expect(parseOpt0011C512LongRawResultChunkOffset("0")).toBe(0);
    expect(parseOpt0011C512LongRawResultChunkOffset("32768")).toBe(32_768);
    expect(() => parseOpt0011C512LongRawResultChunkOffset("01"))
      .toThrow(/canonical/);
    const raw = "a".repeat(
      OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS + 7,
    );
    const first = sliceOpt0011C512LongRawResultChunk(raw, 0);
    expect(first.chunk).toHaveLength(
      OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS,
    );
    expect(first.complete).toBe(false);
    expect(sliceOpt0011C512LongRawResultChunk(raw, first.nextOffset))
      .toMatchObject({ chunk: "a".repeat(7), complete: true });
  });

  it("makes no timing, thermal, listening, selector, or integration claim", () => {
    expect(HTML_SOURCE).toContain("records no performance or thermal result");
    expect(HARNESS_SOURCE).toContain("performanceClaim: null");
    expect(HARNESS_SOURCE).toContain("thermalClaim: null");
    expect(HARNESS_SOURCE).toContain("listeningClaim: null");
    expect(HARNESS_SOURCE).toContain("selectorClaim: null");
    expect(HARNESS_SOURCE).toContain("productionIntegrationClaim: null");
    expect(HARNESS_SOURCE).toContain("responsivenessClaim: null");
    expect(HARNESS_SOURCE).not.toContain("run-timed");
    expect(HARNESS_SOURCE).not.toContain("timestamp-query");
  });
});

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
