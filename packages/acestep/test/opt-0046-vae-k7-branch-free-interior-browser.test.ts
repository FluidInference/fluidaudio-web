import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildOpt0046Cases,
  parseOpt0046ThermalGate,
  summarizeOpt0046Timing,
} from "./browser/opt-0046-vae-k7-branch-free-interior-dot4.js";

const SOURCE = readFileSync(new URL(
  "./browser/opt-0046-vae-k7-branch-free-interior-dot4.ts",
  import.meta.url,
), "utf8");
const HTML = readFileSync(new URL(
  "./browser/opt-0046-vae-k7-branch-free-interior-dot4.html",
  import.meta.url,
), "utf8");

describe("OPT-0046 lean browser gate contract", () => {
  it("covers all twelve C512 production tier/dilation combinations", () => {
    const cases = buildOpt0046Cases();
    expect(cases).toHaveLength(12);
    expect(new Set(cases.map(({ id }) => id))).toEqual(new Set([
      "c1024-d1", "c1024-d3", "c1024-d9",
      "c512-d1", "c512-d3", "c512-d9",
      "c256-d1", "c256-d3", "c256-d9",
      "c128-d1", "c128-d3", "c128-d9",
    ]));
    expect(cases.every(({ shape }) =>
      shape.inputFrames === 512 && shape.kernelSize === 7 &&
      shape.padding === shape.dilation * 3
    )).toBe(true);
    expect(cases.filter(({ timingWeight }) => timingWeight > 0).map(
      ({ id, timingWeight }) => [id, timingWeight],
    )).toEqual([
      ["c1024-d1", 282],
      ["c512-d3", 423],
      ["c256-d1", 423],
      ["c128-d9", 1_269],
    ]);
  });

  it("requires every tier non-slower and at least 1.15x weighted", () => {
    const positive = summarizeOpt0046Timing([
      tier("c1024-d1", 282, 20, 15),
      tier("c512-d3", 423, 12, 9),
      tier("c256-d1", 423, 8, 6),
      tier("c128-d9", 1_269, 4, 3),
    ]);
    expect(positive).toMatchObject({
      weightTotal: 2_397,
      everyTierNonSlower: true,
      requiredWeightedSpeedup: 1.15,
      passed: true,
    });
    const oneRegression = summarizeOpt0046Timing([
      tier("c1024-d1", 282, 20, 10),
      tier("c512-d3", 423, 12, 6),
      tier("c256-d1", 423, 8, 4),
      tier("c128-d9", 1_269, 4, 4.01),
    ]);
    expect(oneRegression).toMatchObject({
      everyTierNonSlower: false,
      passed: false,
    });
  });

  it("accepts exactly one level-0 observation after at least 30 seconds", () => {
    const parameters = thermalParameters();
    expect(parseOpt0046ThermalGate(parameters, 900, 31_100)).toMatchObject({
      observationCount: 1,
      observedLevel: 0,
      durationMilliseconds: 30_000,
      maximumObservationGapMilliseconds: 30_000,
      launchDelayMilliseconds: 100,
    });
    const two = thermalParameters();
    two.set("thermalObservations", "2");
    expect(() => parseOpt0046ThermalGate(two, 900, 31_100))
      .toThrow(/one truthful level-0/);
    const short = thermalParameters();
    short.set("thermalCheckedAtEpochMilliseconds", "30999");
    expect(() => parseOpt0046ThermalGate(short, 900, 31_100))
      .toThrow(/one truthful level-0/);
  });

  it("keeps all timing behind one disabled button after full exact preparation", () => {
    expect(HTML).toContain('id="run" type="button" disabled');
    expect(HTML).toContain("exactly one external");
    expect(SOURCE).toContain('run.addEventListener("click"');
    expect(SOURCE).toContain("raw-U16 exact and deterministic; timing has not run");
    expect(SOURCE).toContain("OUTPUT_PREFILL_QNAN_F16");
    expect(SOURCE).toContain("prefixCanaryIntact");
    expect(SOURCE).toContain("suffixCanaryIntact");
    expect(SOURCE).toContain("allOutputsFinite: true");
    expect(SOURCE).toContain("candidateDispatches.push(segment.kind === \"interior\"");
    expect(SOURCE).toContain("boundaryOwnerUnchanged: true");
    expect(SOURCE).toContain("productionIntegrationAuthorized: false");
    expect(SOURCE).not.toContain("navigator.ml");
  });
});

function tier(
  id: string,
  weight: number,
  control: number,
  candidate: number,
) {
  return Object.freeze({
    id,
    weight,
    opt0024SamplesMilliseconds: Object.freeze([
      control,
      control,
      control,
      control,
    ]),
    branchFreeInteriorSamplesMilliseconds: Object.freeze([
      candidate,
      candidate,
      candidate,
      candidate,
    ]),
  });
}

function thermalParameters(): URLSearchParams {
  return new URLSearchParams({
    thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
    thermalStartedAtEpochMilliseconds: "1000",
    thermalCheckedAtEpochMilliseconds: "31000",
    thermalObservations: "1",
    thermalObservedLevel: "0",
    thermalMaximumObservationGapMilliseconds: "",
  });
}
