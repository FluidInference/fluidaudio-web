import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  OPT_0059_EDGE_ORDER,
  OPT_0059_FIXTURE_SHA256,
  OPT_0059_MAIN_ORDER,
  OPT_0059_MAXIMUM_LIVE_GPU_BYTES,
  OPT_0059_PROFILE_FAMILIES,
  OPT_0059_QUANTA_PER_COMMAND_BUFFER,
  evaluateOpt0059Timing,
  parseOpt0059ThermalGate,
  planOpt0059Gate,
  planOpt0059Shape,
  type Opt0059Shape,
  type Opt0059TimingSample,
} from "./browser/opt-0059-vae-c2378-short-projection-contract.js";

const WORKER_SOURCE = source(
  "./browser/opt-0059-vae-c2378-short-projection-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0059-vae-c2378-short-projection.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0059-vae-c2378-short-projection.html",
);

describe("OPT-0059 authenticated C2378 short projected-wall gate", () => {
  it("pins all four exact pure revision-7 shapes and batch-64 topology", () => {
    expect(OPT_0059_QUANTA_PER_COMMAND_BUFFER).toBe(64);
    expect(planOpt0059Shape(340)).toMatchObject({
      inputFrames: 340,
      operationCount: 88,
      k1RouteCount: 15,
      graphQuantumCount: 5_241,
      sequenceQuantumCount: 5_242,
      decoderCommandBufferCount: 82,
      totalCommandBufferCount: 83,
      requestedCooperativeIdleMilliseconds: 82,
      familyQuantumCounts: {
        "k7-conv1d": 2_725,
        "k1-conv1d": 546,
        "conv-transpose1d": 429,
        snake: 1_079,
        add: 462,
      },
      dynamicControlBytes: 1_341_712,
      workspaceBytes: 167_116_800,
      outputBytes: 5_222_400,
    });
    expect(planOpt0059Shape(448)).toMatchObject({
      graphQuantumCount: 6_894,
      sequenceQuantumCount: 6_895,
      decoderCommandBufferCount: 108,
      totalCommandBufferCount: 109,
      dynamicControlBytes: 1_764_880,
      workspaceBytes: 220_200_960,
      outputBytes: 6_881_280,
    });
    expect(planOpt0059Shape(512)).toMatchObject({
      graphQuantumCount: 7_854,
      sequenceQuantumCount: 7_855,
      decoderCommandBufferCount: 123,
      totalCommandBufferCount: 124,
      familyQuantumCounts: {
        "k7-conv1d": 4_090,
        "k1-conv1d": 819,
        "conv-transpose1d": 644,
        snake: 1_611,
        add: 690,
      },
      dynamicControlBytes: 2_010_640,
      workspaceBytes: 251_658_240,
      outputBytes: 7_864_320,
    });
    expect(planOpt0059Shape(2_314)).toMatchObject({
      graphQuantumCount: 35_497,
      sequenceQuantumCount: 35_498,
      decoderCommandBufferCount: 555,
      totalCommandBufferCount: 556,
      familyQuantumCounts: {
        "k7-conv1d": 18_491,
        "k1-conv1d": 3_687,
        "conv-transpose1d": 2_894,
        snake: 7_299,
        add: 3_126,
      },
      dynamicControlBytes: 9_087_248,
      workspaceBytes: 1_137_377_280,
      outputBytes: 35_543_040,
    });
  });

  it("pins the exact projections, one maximum allocation, and six mapping changes", () => {
    const gate = planOpt0059Gate();
    expect(gate.mainOrder).toEqual([512, 2_314, 2_314, 512]);
    expect(gate.edgeOrder).toEqual([340, 448, 448, 340]);
    expect(gate.controlC4500).toEqual({
      windowCounts: { "340": 1, "448": 1, "512": 10 },
      decodedLatentFrames: 5_908,
      graphQuantumCount: 90_675,
      totalCommandBufferCount: 1_432,
      requestedCooperativeIdleMilliseconds: 1_431,
    });
    expect(gate.candidateC4500).toEqual({
      windowCounts: { "2314": 2 },
      decodedLatentFrames: 4_628,
      overlapFrames: 64,
      windows: [
        {
          latent: [0, 2_314],
          core: [0, 2_250],
          discardPrefixFrames: 0,
          discardSuffixFrames: 64,
        },
        {
          latent: [2_186, 4_500],
          core: [2_250, 4_500],
          discardPrefixFrames: 64,
          discardSuffixFrames: 0,
        },
      ],
      graphQuantumCount: 70_994,
      totalCommandBufferCount: 1_112,
      requestedCooperativeIdleMilliseconds: 1_111,
    });
    expect(gate.sixC128K1MappingChanges).toEqual([
      "block-3-res-1-conv2",
      "block-3-res-2-conv2",
      "block-3-res-3-conv2",
      "block-4-res-1-conv2",
      "block-4-res-2-conv2",
      "block-4-res-3-conv2",
    ]);
    expect(gate.maximumAllocation).toMatchObject({
      maximumActualWindowFrames: 2_314,
      residentWeightBytes: 168_791_552,
      guardedWorkspaceBytesEach: 1_137_377_792,
      workspaceBufferCount: 3,
      plannedLiveGpuBytes: 3_667_109_696,
      belowFourGigabytes: true,
    });
    expect(gate.maximumAllocation.plannedLiveGpuBytes)
      .toBeLessThan(OPT_0059_MAXIMUM_LIVE_GPU_BYTES);
    expect(Object.keys(OPT_0059_FIXTURE_SHA256)).toEqual([
      "340", "448", "512", "2314",
    ]);
    expect(OPT_0059_FIXTURE_SHA256).toEqual({
      "340": "286edbeba4ab49407f50564e7e84eef3218316b408cb7cef753d1225c3c50347",
      "448": "a1b3d1bebcfbdce1c4665a76534fd67a888a78072adb66e84d29fa7d05e26653",
      "512": "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899",
      "2314": "01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d",
    });
  });

  it("pins the six bounded C128 K1 route changes without changing operations", () => {
    const c512 = planOpt0059Shape(512).k1Routes.find((route) =>
      route.k1Label === "block-4-res-1-conv2"
    )!;
    const c2314 = planOpt0059Shape(2_314).k1Routes.find((route) =>
      route.k1Label === "block-4-res-1-conv2"
    )!;
    expect(c512.mapping).toBe("flat-x");
    expect(c2314.mapping).toBe("column-x-row-y");
    expect(planOpt0059Gate().sixC128K1MappingChanges).toHaveLength(6);
  });

  it("proves exact K1/Add/Snake slots and rejects the stale Add redirection", () => {
    let futureSkipCount = 0;
    let crossWorkgroupAliasHazardCount = 0;
    for (const inputFrames of [340, 448, 512, 2_314] as const) {
      const graph = planAceVaeDecoder(inputFrames);
      const routes = planOpt0059Shape(inputFrames).k1Routes;
      expect(routes).toHaveLength(15);
      for (const route of routes) {
        const k1 = graph.operations[route.k1OperationIndex]!;
        const add = graph.operations[route.addOperationIndex]!;
        const successor = graph.operations[route.successorSnakeOperationIndex]!;
        expect(k1.kind).toBe("conv1d");
        expect(add.kind).toBe("add");
        expect(successor.kind).toBe("snake");
        if (k1.kind !== "conv1d" || add.kind !== "add" ||
          successor.kind !== "snake") throw new Error("unreachable narrowing");
        expect(add.output).toBe(k1.input);
        expect(successor.output).toBe(add.input);
        expect(k1.output).toBe(add.right);
        expect(route).toMatchObject({
          k1InputSlot: k1.input,
          k1OutputSlot: k1.output,
          addLeftSlot: add.input,
          addOutputSlot: add.output,
          successorOutputSlot: successor.output,
          addOutputAliasesK1Input: true,
          successorOutputAliasesAddLeft: true,
        });
        // The abandoned mapping stored Add into the K1 scratch. That leaves
        // the true Add destination (and next residual skip) stale.
        expect(k1.output).not.toBe(add.output);
        expect(k1.output).not.toBe(k1.input);
        expect(k1.output).not.toBe(add.input);
        if (/res-[12]-conv2$/.test(k1.label)) {
          const nextResidualAdd = graph.operations[route.k1OperationIndex + 6];
          expect(nextResidualAdd?.kind).toBe("add");
          if (nextResidualAdd?.kind !== "add") {
            throw new Error("unreachable next residual narrowing");
          }
          expect(nextResidualAdd.input).toBe(add.output);
          futureSkipCount += 1;
        }
        if (k1.shape.outputChannels > 128) {
          // Multiple output-column workgroups all read the complete K1 input
          // row, so writing Add back to that aliased input has a data race.
          expect(add.output).toBe(k1.input);
          crossWorkgroupAliasHazardCount += 1;
        }
      }
    }
    expect(futureSkipCount).toBe(4 * 10);
    expect(crossWorkgroupAliasHazardCount).toBe(4 * 9);
  });

  it("requires the one-check 30-second/5-second thermal protocol", () => {
    const ready = 1_000_000;
    const valid = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: String(ready),
      thermalCheckedAtEpochMilliseconds: String(ready + 30_000),
      thermalObservations: "1",
      thermalObservedLevel: "0",
    });
    expect(parseOpt0059ThermalGate(valid, ready, ready + 34_999)).toMatchObject({
      durationMilliseconds: 30_000,
      launchDelayMilliseconds: 4_999,
      observationCount: 1,
      observedLevel: 0,
    });
    expect(() => parseOpt0059ThermalGate(valid, ready, ready + 35_001))
      .toThrow(/within 5 seconds/);
    valid.set("thermalObservedLevel", "1");
    expect(() => parseOpt0059ThermalGate(valid, ready, ready + 30_001))
      .toThrow(/level-0/);
  });

  it("gates both ABBA directions and exact projection formulas", () => {
    const positiveMain = OPT_0059_MAIN_ORDER.map((shape) => sample(shape, {
      512: 140,
      2314: 500,
    }[shape as 512 | 2314])) as readonly Opt0059TimingSample[];
    const positiveEdge = OPT_0059_EDGE_ORDER.map((shape) => sample(shape, {
      340: 100,
      448: 120,
    }[shape as 340 | 448])) as readonly Opt0059TimingSample[];
    const positive = evaluateOpt0059Timing(positiveMain, positiveEdge);
    expect(positive.passed).toBe(true);
    expect(positive["forward"]).toMatchObject({
      controlProjection: { formula: "10*C512 + C448 + C340 + 11ms" },
      candidateProjection: { formula: "2*C2314 + 1ms" },
      requiredSpeedup: 1.15,
      maximumNormalizedDecoderRatio: 1.10,
      passed: true,
    });
    const negativeMain = OPT_0059_MAIN_ORDER.map((shape) =>
      sample(shape, shape === 512 ? 140 : 900)
    );
    expect(evaluateOpt0059Timing(negativeMain, positiveEdge).passed).toBe(false);
    const wrongOrder = [...positiveMain];
    [wrongOrder[0], wrongOrder[1]] = [wrongOrder[1]!, wrongOrder[0]!];
    expect(() => evaluateOpt0059Timing(
      wrongOrder,
      positiveEdge,
    )).toThrow(/balanced ABBA/);
  });

  it("keeps the benchmark explicit, self-contained, and out of production routing", () => {
    expect(WORKER_SOURCE.indexOf("authenticateFixtures()"))
      .toBeLessThan(WORKER_SOURCE.indexOf("requestAceWebGpuDevice({"));
    expect(WORKER_SOURCE.indexOf("authenticateAndAcquirePackage()"))
      .toBeLessThan(WORKER_SOURCE.indexOf("requestAceWebGpuDevice({"));
    expect(WORKER_SOURCE.match(/AceGpuTensorPhase\.load\(/g)).toHaveLength(1);
    expect(WORKER_SOURCE).toContain("c2314CapableActivationAllocationCount: 1");
    expect(WORKER_SOURCE).toContain("for (const inputFrames of OPT_0059_SHAPES)");
    expect(WORKER_SOURCE).toContain("createProfiledQuanta(");
    expect(WORKER_SOURCE).toContain(
      "const quanta = Object.freeze(base.quanta.map",
    );
    expect(WORKER_SOURCE).toContain(
      "base.graphQuantumCount !== shapePlan.graphQuantumCount",
    );
    expect(WORKER_SOURCE).not.toContain("createInPlaceRangeDispatch(");
    expect(WORKER_SOURCE).not.toContain("AceOpt0063VaeK1AddSnakeExactFusedKernel");
    expect(WORKER_SOURCE).toContain("audit.requireLiveBuffer(");
    expect(WORKER_SOURCE).toContain("retainedRangeControlBufferBytes");
    expect(WORKER_SOURCE).toContain("opt0063FusionIntegrated: false");
    expect(WORKER_SOURCE).toContain(
      "standaloneK1AddAndSuccessorSnakeDispatchesRetained: true",
    );
    expect(WORKER_SOURCE).toContain('"dual-k4-profile-repeat"');
    expect(WORKER_SOURCE).toContain("matchesPreparedSameProfileOracle: true");
    expect(WORKER_SOURCE).toContain(
      "disposeAbortsActiveWorkBeforeSerializedCleanup: true",
    );
    expect(WORKER_SOURCE).toContain(
      "progressReportedEvery16CommandBuffersAndAtCompletion: true",
    );
    expect(WORKER_SOURCE).toContain("preparation: prepared.preparationReceipt");
    expect(WORKER_SOURCE).toContain("preparation: retained.preparationReceipt");
    expect(WORKER_SOURCE).toContain("correctnessAuthorityRetainedInFinalReceipt: true");
    expect(WORKER_SOURCE).toContain("maximumLiveGpuBytesBelowFourGigabytes: true");
    expect(WORKER_SOURCE).toContain("expectedCanaryReadbackSha256");
    expect(WORKER_SOURCE).not.toContain("vae-fp16-backend");
    expect(WORKER_SOURCE).not.toContain("webgpu-pipeline");
    expect(PAGE_SOURCE.match(/new Worker\(/g)).toHaveLength(1);
    expect(PAGE_SOURCE.indexOf('addEventListener("click"'))
      .toBeLessThan(PAGE_SOURCE.indexOf("new Worker("));
    expect(HTML_SOURCE).toContain("No worker, WebGPU device, or GPU resource");
    expect(HTML_SOURCE).toContain("512/C2314/C2314/C512");
    expect(HTML_SOURCE).toContain("340/C448/C448/C340");
    expect(HTML_SOURCE).toContain("1 ms queue-empty interval");
    expect(HTML_SOURCE).toContain("K1, Add, and successor Snake remain");
    expect(HTML_SOURCE).not.toContain("exact OPT-0063");
  });

  it("separates canonical B256 package authentication from exact dispatch geometry", () => {
    expect(WORKER_SOURCE).toMatch(
      /resolveAceOpt0054Fp16VaePackageBindings\(\s*[\s\S]*?planAceVaeDecoder\(256\),/,
    );
    expect(WORKER_SOURCE).not.toMatch(
      /resolveAceOpt0054Fp16VaePackageBindings\(\s*[\s\S]*?planAceVaeDecoder\(2_314\),/,
    );
    expect(WORKER_SOURCE).toContain("packageBindingAuthenticationFrames: 256");
    expect(WORKER_SOURCE).toContain("packageBindingsAreFrameNeutral: true");
    expect(WORKER_SOURCE).toContain(
      "for (const inputFrames of OPT_0059_SHAPES)",
    );
    expect(WORKER_SOURCE).toContain(
      "runtime.createChunkDispatchSet(",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES",
    );
  });

  it("separates the guarded physical buffer boundary from its logical storage binding", () => {
    const gate = planOpt0059Gate();
    const logicalBytes = gate.shapes["2314"].workspaceBytes;
    const physicalBytes = gate.maximumAllocation.guardedWorkspaceBytesEach;
    expect(logicalBytes).toBe(1_137_377_280);
    expect(physicalBytes).toBe(logicalBytes + 2 * 256);
    expect(WORKER_SOURCE).toContain(
      "maximumPhysicalBufferBytes:\n" +
        "      gatePlan.maximumAllocation.guardedWorkspaceBytesEach",
    );
    expect(WORKER_SOURCE).toContain(
      "maximumLogicalStorageBindingBytes:\n" +
        '      gatePlan.shapes["2314"].workspaceBytes',
    );
    expect(WORKER_SOURCE).toContain(
      "maxBufferSize: storageLimitPlan.maximumPhysicalBufferBytes",
    );
    expect(WORKER_SOURCE).toContain(
      "storageLimitPlan.maximumLogicalStorageBindingBytes",
    );
    expect(WORKER_SOURCE).toContain(
      "size: logicalBytes + 2 * GUARD_BYTES",
    );
    expect(WORKER_SOURCE).toContain(
      "binding: Object.freeze({ buffer, offset: GUARD_BYTES, size: logicalBytes })",
    );
    expect(WORKER_SOURCE).toContain("requireDevice(context, storageLimitPlan)");
    expect(WORKER_SOURCE).toContain("storageLimitPlan,");
  });
});

function sample(inputFrames: Opt0059Shape, wall: number): Opt0059TimingSample {
  const family = Object.freeze(Object.fromEntries(
    OPT_0059_PROFILE_FAMILIES.map((name) => [name, Object.freeze({
      batchCount: 2,
      quantumCount: 128,
      submitThroughDrainMs: wall * 0.4,
    })]),
  )) as Opt0059TimingSample["families"];
  return Object.freeze({
    inputFrames,
    decoderSubmitThroughDrainMs: wall * 0.8,
    decoderWallMs: wall,
    readbackSubmitThroughDrainMs: 1,
    readbackMapWallMs: 1,
    outerWindowWallMs: wall,
    families: family,
    mixed: Object.freeze({
      batchCount: 1,
      quantumCount: 64,
      submitThroughDrainMs: wall * 0.2,
    }),
  });
}

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}
