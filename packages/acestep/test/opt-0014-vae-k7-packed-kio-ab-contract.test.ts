import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceOpt0014VaeConv1dPackedKioWeight } from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import {
  OPT_0014_C300_EXPECTED_TOPOLOGY,
  buildOpt0014C300Topology,
  parseOpt0014ThermalGate,
  summarizeOpt0014WeightedTiming,
} from "./browser/opt-0014-vae-k7-packed-kio-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0014-vae-k7-packed-kio-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0014-vae-k7-packed-kio-ab.html",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const PACKED_KIO_CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.ts",
  import.meta.url,
));

describe("OPT-0014 target-browser packed-KIO K7 gate", () => {
  it("freezes all 17 C300 K7 operations and all 2,404 graph ranges", () => {
    const topology = buildOpt0014C300Topology();
    expect(topology).toHaveLength(17);
    expect(topology.reduce(
      (sum, operation) => sum + operation.ranges.length,
      0,
    )).toBe(2_404);
    expect(topology.reduce(
      (sum, operation) => sum + operation.correctnessProbes.length,
      0,
    )).toBe(51);
    expect(topology.reduce(
      (sum, operation) => sum + operation.timingStrata.length,
      0,
    )).toBe(50);
    expect(topology.flatMap((operation) => operation.timingStrata).reduce(
      (sum, stratum) => sum + stratum.weight!,
      0,
    )).toBe(2_404);
    expect(topology.flatMap((operation) =>
      operation.correctnessProbes
    ).reduce((sum, probe) => sum + probe.count, 0) * 2).toBe(13_854_720);
    expect(topology.filter((operation) =>
      operation.outputStorage === "float16"
    ).flatMap((operation) => operation.correctnessProbes).reduce(
      (sum, probe) => sum + probe.count,
      0,
    ) * 2).toBe(12_599_296);
    expect(topology.filter((operation) =>
      operation.outputStorage === "float32"
    ).flatMap((operation) => operation.correctnessProbes).reduce(
      (sum, probe) => sum + probe.count,
      0,
    ) * 2).toBe(1_255_424);

    expect(topology.map((operation) => {
      const [first, interior, tail] = operation.correctnessProbes;
      return {
        label: operation.label,
        inputFrames: operation.shape.inputFrames,
        inputChannels: operation.shape.inputChannels,
        outputChannels: operation.shape.outputChannels,
        dilation: operation.shape.dilation,
        padding: operation.shape.padding,
        outputStorage: operation.outputStorage,
        rangeCount: operation.ranges.length,
        firstCount: first!.count,
        interiorRangeIndex: interior!.rangeIndex,
        interiorBase: interior!.base,
        interiorCount: interior!.count,
        interiorWeight: interior!.weight,
        tailBase: tail!.base,
        tailCount: tail!.count,
      };
    })).toEqual(OPT_0014_C300_EXPECTED_TOPOLOGY);

    expect(topology[0]!.correctnessProbes[1]).toMatchObject({
      stratum: "interior",
      source: "synthetic-centered",
      rangeIndex: null,
      base: 290_816,
      count: 32_768,
      firstOutputRow: 142,
      outputRowCount: 16,
      weight: null,
    });
    expect(topology[0]!.timingStrata.map(({ stratum, weight }) =>
      ({ stratum, weight })
    )).toEqual([
      { stratum: "first", weight: 1 },
      { stratum: "tail", weight: 1 },
    ]);
  });

  it("pins the complete twice-verified repack footprint", () => {
    const topology = buildOpt0014C300Topology();
    const plans = topology.map((operation) =>
      planAceOpt0014VaeConv1dPackedKioWeight(operation.shape)
    );
    expect(plans.reduce(
      (sum, plan) => sum + plan.packedWordCount * 2,
      0,
    )).toBe(30_508_800);
    expect(plans.reduce(
      (sum, plan) => sum + plan.packedBindingBytes,
      0,
    )).toBe(61_017_600);
    expect(plans.reduce(
      (sum, plan) => sum + plan.repackWorkgroups,
      0,
    )).toBe(59_588);
    expect(Math.max(...plans.map((plan) => plan.packedBindingBytes)))
      .toBe(14_680_064);
    expect(30_508_800 * 2).toBe(61_017_600);
  });

  it("accepts exactly one fresh 30-second nominal pre-gate", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0014ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0014ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete or non-nominal/);
  });

  it("weights 50 real strata by their exact 2,404-range multiplicity", () => {
    const topology = buildOpt0014C300Topology();
    const result = summarizeOpt0014WeightedTiming(topology.flatMap(
      (operation) => operation.timingStrata.map((stratum) => ({
        operationLabel: operation.label,
        stratum: stratum.stratum,
        weight: stratum.weight!,
        fixed32Samples: [10, 12],
        packedKioSamples: [4, 6],
      })),
    ));
    expect(result).toMatchObject({
      exactRangeCount: 2_404,
      fixed32ProjectedMilliseconds: 26_444,
      packedKioProjectedMilliseconds: 12_020,
      projectedSpeedup: 2.2,
    });
  });

  it("pins frozen identity, raw guards/reruns, balanced timing, and no retry", () => {
    expect(createHash("sha256").update(readFileSync(PACKED_KIO_CORE_PATH))
      .digest("hex")).toBe(
        "802cb0ad1d2c57c0cc51cbd4a7c88632e00d543b526f2ed0b94e9fc393a3d8d8",
      );
    expect(HARNESS_SOURCE).toContain(
      "12e128ab323c0024ed683313b4d06c07041213e7",
    );
    expect(HARNESS_SOURCE).toContain(
      "802cb0ad1d2c57c0cc51cbd4a7c88632e00d543b526f2ed0b94e9fc393a3d8d8",
    );
    expect(HARNESS_SOURCE).toContain(
      "AceFp16VaeConv1dSubgroupKernel.create",
    );
    expect(HARNESS_SOURCE).toContain(
      "AceOpt0014VaeConv1dPackedKioSubgroupKernel.create",
    );
    expect(HARNESS_SOURCE).toContain("executeAndVerifyRepack");
    expect(HARNESS_SOURCE).toContain("qNaNPrefillCompleteWrites: true");
    expect(HARNESS_SOURCE).toContain("redzonesUntouched: true");
    expect(HARNESS_SOURCE).toContain(
      "guardsAndAdjacentCanariesUntouched: true",
    );
    expect(HARNESS_SOURCE).toContain('["fixed32", "packedKio"]');
    expect(HARNESS_SOURCE).toContain('["packedKio", "fixed32"]');
    expect(HARNESS_SOURCE).toContain("measuredAfterAllConvTiming: true");
    expect(HARNESS_SOURCE).toContain("commandBufferCount: 1");
    expect(HARNESS_SOURCE).toContain("workgroupCount: 59_588");
    expect(HARNESS_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("continuous 30-second nominal interval");
    expect(HTML_SOURCE).toContain("Repack timing is measured late");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0014-vae-k7-packed-kio-ab.ts"',
    );
  });
});
