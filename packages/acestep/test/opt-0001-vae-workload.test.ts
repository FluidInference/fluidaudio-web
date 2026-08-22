import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import {
  createAceOpt0001VaeWorkloadReport,
  serializeAceOpt0001VaeWorkloadReport,
  summarizeAceOpt0001VaeRepresentativeTimings,
} from "../benchmark/opt-0001-vae-workload.js";

describe("OPT-0001 authenticated VAE workload profiler", () => {
  it("accounts one canonical 256-latent-frame production window", () => {
    const report = createAceOpt0001VaeWorkloadReport();

    expect(report.identity).toMatchObject({
      fixtureId: "ace-opt-0001-vae-window-256-v1",
      experimentId: "OPT-0001",
      aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      aceMainModelSnapshot: "19671f406d603126926c1b7e2adc169acbcade22",
      modelManifestSha256:
        "d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55",
      executionProfile: "reference-bf16-subgroups",
      vaeStorageDtype: "float32",
      latentWindowFrames: 256,
      quantumOutputElementCap: 32_768,
    });
    expect(report.identity.transposeParts[
      "vae.decoder.block.0.conv_t1.weight"
    ]).toEqual([
      { partStart: 0, partEnd: 614 },
      { partStart: 614, partEnd: 1_024 },
    ]);
    expect(report.graph).toMatchObject({
      batch: 1,
      inputFrames: 256,
      outputFrames: 491_520,
      hopLength: 1_920,
      parameterBytes: 337_583_104,
      maximumActivationElements: 62_914_560,
      workspaceBytes: 251_658_240,
      allWorkspaceBytes: 754_974_720,
    });
    expect(report.operations).toHaveLength(88);
    expect(report.operationFamilies.map((family) => ({
      family: family.family,
      operations: family.operationCount,
    }))).toEqual([
      { family: "conv1d", operations: 32 },
      { family: "conv-transpose1d", operations: 5 },
      { family: "snake", operations: 36 },
      { family: "add", operations: 15 },
    ]);

    expect(report.totals.operationCount).toBe(88);
    expect(report.totals.decoderCommandBufferCount).toBe(
      report.totals.decoderQuantumCount,
    );
    expect(report.totals.totalCommandBufferCount).toBe(
      report.totals.decoderQuantumCount + 1,
    );
    expect(report.totals.queueDrainCount).toBe(
      report.totals.totalCommandBufferCount,
    );
    expect(report.totals.configuredCooperativeIdleMilliseconds).toBe(
      report.totals.decoderQuantumCount,
    );
    expect(report.totals.decoderPrimitiveDispatchCount).toBe(
      report.totals.rangeControlRecordCount,
    );
    expect(report.totals.decoderPrimitiveDispatchCount).toBeGreaterThan(
      report.totals.decoderQuantumCount,
    );
    expect(report.totals.maximumOutstandingCommandBuffers).toBe(1);
    expect(report.totals.decodedAudioFrames).toBe(491_520);
    expect(report.totals.decodedInterleavedElements).toBe(983_040);
    expect(report.totals.decodedFloat32Bytes).toBe(3_932_160);

    expect(report.totals.outputElements).toBe(sum(
      report.operationFamilies.map((family) => family.outputElements),
    ));
    expect(report.totals.validMultiplyAccumulates).toBe(sum(
      report.operationFamilies.map((family) =>
        family.validMultiplyAccumulates
      ),
    ));
    expect(report.totals.convolutionFlops).toBe(
      2 * report.totals.validMultiplyAccumulates,
    );
    expect(report.operationFamilies.find((family) => family.family === "snake"))
      .toMatchObject({ validMultiplyAccumulates: 0, biasElements: 0 });
    expect(report.operationFamilies.find((family) => family.family === "add"))
      .toMatchObject({ validMultiplyAccumulates: 0, biasElements: 0 });

    expect(sum(report.quantumClasses.map((quantumClass) =>
      quantumClass.quantumCount))).toBe(report.totals.decoderQuantumCount);
    expect(sum(report.quantumClasses.map((quantumClass) =>
      quantumClass.primitiveDispatchCount))).toBe(
      report.totals.decoderPrimitiveDispatchCount,
    );
    expect(sum(report.quantumClasses.map((quantumClass) =>
      quantumClass.logicalOutputElements))).toBe(report.totals.outputElements);
    expect(new Set(report.quantumClasses.map((quantumClass) => quantumClass.id))
      .size).toBe(report.quantumClasses.length);
    expect(report.quantumClasses.filter((quantumClass) =>
      quantumClass.physicalOutputChannels.length > 1
    )).toEqual([
      expect.objectContaining({
        family: "conv-transpose1d",
        operationLabels: ["block-0-conv-t1"],
        physicalOutputChannels: [614, 410],
        quantumCount: 80,
        primitiveDispatchCount: 160,
      }),
    ]);
    expect(report.quantumClasses).toHaveLength(38);
    expect(report.operationFamilies).toEqual([
      {
        family: "conv1d",
        operationCount: 32,
        outputElements: 725_024_768,
        validMultiplyAccumulates: 548_502_631_424,
        denseKernelMultiplyAccumulates: 548_724_015_104,
        convolutionFlops: 1_097_005_262_848,
        biasElements: 724_041_728,
        quantumCount: 22_126,
        primitiveDispatchCount: 22_126,
        configuredCooperativeIdleMilliseconds: 22_126,
      },
      {
        family: "conv-transpose1d",
        operationCount: 5,
        outputElements: 120_586_240,
        validMultiplyAccumulates: 75_137_122_304,
        denseKernelMultiplyAccumulates: 365_072_220_160,
        convolutionFlops: 150_274_244_608,
        biasElements: 120_586_240,
        quantumCount: 3_680,
        primitiveDispatchCount: 3_760,
        configuredCooperativeIdleMilliseconds: 3_680,
      },
      {
        family: "snake",
        operationCount: 36,
        outputElements: 844_627_968,
        validMultiplyAccumulates: 0,
        denseKernelMultiplyAccumulates: 0,
        convolutionFlops: 0,
        biasElements: 0,
        quantumCount: 25_776,
        primitiveDispatchCount: 25_776,
        configuredCooperativeIdleMilliseconds: 25_776,
      },
      {
        family: "add",
        operationCount: 15,
        outputElements: 361_758_720,
        validMultiplyAccumulates: 0,
        denseKernelMultiplyAccumulates: 0,
        convolutionFlops: 0,
        biasElements: 0,
        quantumCount: 11_040,
        primitiveDispatchCount: 11_040,
        configuredCooperativeIdleMilliseconds: 11_040,
      },
    ]);
    expect(report.totals).toEqual({
      operationCount: 88,
      outputElements: 2_051_997_696,
      validMultiplyAccumulates: 623_639_753_728,
      denseKernelMultiplyAccumulates: 913_796_235_264,
      convolutionFlops: 1_247_279_507_456,
      biasElements: 844_627_968,
      decoderQuantumCount: 62_622,
      decoderPrimitiveDispatchCount: 62_702,
      decoderCommandBufferCount: 62_622,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 62_623,
      queueDrainCount: 62_623,
      maximumOutstandingCommandBuffers: 1,
      rangeControlRecordCount: 62_702,
      configuredCooperativeIdleMilliseconds: 62_622,
      decodedAudioFrames: 491_520,
      decodedInterleavedElements: 983_040,
      decodedFloat32Bytes: 3_932_160,
    });
  });

  it("serializes byte-identically and remains a small machine-readable artifact", () => {
    const first = serializeAceOpt0001VaeWorkloadReport();
    const second = serializeAceOpt0001VaeWorkloadReport();
    expect(second).toBe(first);
    expect(first.endsWith("\n")).toBe(true);
    const report = createAceOpt0001VaeWorkloadReport();
    expect(JSON.parse(first)).toEqual(report);
    const reordered = {
      totals: report.totals,
      quantumClasses: report.quantumClasses,
      operations: report.operations,
      operationFamilies: report.operationFamilies,
      graph: report.graph,
      identity: report.identity,
      reportKind: report.reportKind,
      schemaVersion: report.schemaVersion,
    } as const;
    expect(serializeAceOpt0001VaeWorkloadReport(reordered)).toBe(first);
    expect(new TextEncoder().encode(first).byteLength).toBe(90_570);
    expect(createHash("sha256").update(first).digest("hex")).toBe(
      "4310c4a97ab3e17749b431c119e2e8b55772ce1efa01b1a6eeb287065d25827c",
    );
  });

  it("weights representative timing classes without claiming partial coverage", () => {
    const report = createAceOpt0001VaeWorkloadReport();
    const complete = summarizeAceOpt0001VaeRepresentativeTimings(
      report,
      report.quantumClasses.map((quantumClass, index) => ({
        quantumClassId: quantumClass.id,
        encodeMilliseconds: [index + 0.1, index + 0.3, index + 0.2],
        submitThroughDrainMilliseconds: [index + 1.2, index + 1.0],
      })),
    );
    expect(complete.measuredClassCount).toBe(report.quantumClasses.length);
    expect(complete.measuredQuantumCount).toBe(
      report.totals.decoderQuantumCount,
    );
    expect(complete.quantumCoverageRatio).toBe(1);
    expect(complete.missingQuantumClassIds).toEqual([]);
    expect(complete.completeWeightedMedianEncodeMilliseconds).not.toBeNull();
    expect(complete.completeWeightedMedianSubmitThroughDrainMilliseconds)
      .not.toBeNull();
    expect(complete.completeWeightedComponentSumMilliseconds).toBe(
      complete.completeWeightedMedianEncodeMilliseconds! +
        complete.completeWeightedMedianSubmitThroughDrainMilliseconds! +
        report.totals.configuredCooperativeIdleMilliseconds,
    );
    expect(complete.componentSumExcludes).toEqual([
      "scheduler-and-owner-orchestration",
      "idle-timer-overshoot",
      "readback-and-map",
      "window-postprocessing",
    ]);

    const firstClass = report.quantumClasses[0]!;
    const partial = summarizeAceOpt0001VaeRepresentativeTimings(report, [{
      quantumClassId: firstClass.id,
      encodeMilliseconds: [0.3, 0.1],
      submitThroughDrainMilliseconds: [1.5, 1.1, 1.3],
    }]);
    expect(partial.classes[0]).toMatchObject({
      encode: {
        samples: [0.3, 0.1],
        sampleCount: 2,
        min: 0.1,
        median: 0.2,
        max: 0.3,
      },
      submitThroughDrain: {
        samples: [1.5, 1.1, 1.3],
        sampleCount: 3,
        min: 1.1,
        median: 1.3,
        max: 1.5,
      },
    });
    expect(partial.measuredQuantumCount).toBe(firstClass.quantumCount);
    expect(partial.quantumCoverageRatio).toBeLessThan(1);
    expect(partial.completeWeightedMedianEncodeMilliseconds).toBeNull();
    expect(partial.completeWeightedMedianSubmitThroughDrainMilliseconds)
      .toBeNull();
    expect(partial.completeWeightedComponentSumMilliseconds).toBeNull();
  });

  it("rejects unknown, duplicate, empty, and invalid representative timings", () => {
    const report = createAceOpt0001VaeWorkloadReport();
    const firstClass = report.quantumClasses[0]!;
    expect(() => summarizeAceOpt0001VaeRepresentativeTimings(report, [{
      quantumClassId: "absent",
      encodeMilliseconds: [1],
      submitThroughDrainMilliseconds: [1],
    }])).toThrow(/unknown class absent/);
    const valid = {
      quantumClassId: firstClass.id,
      encodeMilliseconds: [1],
      submitThroughDrainMilliseconds: [1],
    } as const;
    expect(() => summarizeAceOpt0001VaeRepresentativeTimings(
      report,
      [valid, valid],
    )).toThrow(/repeats class/);
    expect(() => summarizeAceOpt0001VaeRepresentativeTimings(report, [{
      ...valid,
      encodeMilliseconds: [],
    }])).toThrow(/samples must not be empty/);
    expect(() => summarizeAceOpt0001VaeRepresentativeTimings(report, [{
      ...valid,
      submitThroughDrainMilliseconds: [Number.NaN],
    }])).toThrow(/must be a finite number/);
  });
});

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
