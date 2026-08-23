import {
  OPT_0002_LEGACY_UNIFORM_WORK_POLICY,
  OPT_0002_PAIRED_ORDERS,
  OPT_0002_REPRESENTATIVE_OPERATIONS,
  createOpt0002StaticProtocol,
  summarizeOpt0002Milliseconds,
} from "./browser/opt-0002-vae-quantum-ab.js";
import { ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY } from
  "../src/webgpu/vae-decoder.js";
import { describe, expect, it } from "vitest";

describe("OPT-0002 VAE quantum paired A/B browser contract", () => {
  it("pins an explicit test-only legacy oracle and the production candidate", () => {
    expect(OPT_0002_LEGACY_UNIFORM_WORK_POLICY).toEqual({
      maximumConvolutionMultiplyAccumulates: 0xffff_ffff,
      maximumOutputElements: 32_768,
    });
    expect(ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY).toEqual({
      maximumConvolutionMultiplyAccumulates: 234_881_024,
      maximumOutputElements: 1_048_576,
    });
    expect(Object.isFrozen(OPT_0002_LEGACY_UNIFORM_WORK_POLICY)).toBe(true);
  });

  it("uses a balanced AB/BA order with four retained samples per policy", () => {
    expect(OPT_0002_PAIRED_ORDERS).toEqual([
      ["legacy", "candidate"],
      ["candidate", "legacy"],
      ["candidate", "legacy"],
      ["legacy", "candidate"],
    ]);
    expect(OPT_0002_PAIRED_ORDERS.flat().filter((id) => id === "legacy"))
      .toHaveLength(4);
    expect(OPT_0002_PAIRED_ORDERS.flat().filter((id) => id === "candidate"))
      .toHaveLength(4);
  });

  it("pins exact production-window counts and four operation families", () => {
    const protocol = createOpt0002StaticProtocol();
    expect(protocol.productionWindow).toEqual({
      inputFrames: 256,
      legacyQuantumCount: 62_622,
      legacyPrimitiveDispatchCount: 62_702,
      candidateQuantumCount: 3_942,
      candidatePrimitiveDispatchCount: 3_988,
    });
    expect(protocol.representativeCases.map(({ id }) => id)).toEqual(
      OPT_0002_REPRESENTATIVE_OPERATIONS.map(({ id }) => id),
    );
    expect(protocol.representativeCases.map(({ operationKind }) =>
      operationKind
    )).toEqual(["conv1d", "conv-transpose1d", "snake", "add"]);
    for (const fixture of protocol.representativeCases) {
      expect(fixture.candidateQuantumCount).toBe(1);
      expect(fixture.legacyQuantumCount).toBeGreaterThan(1);
      expect(fixture.logicalOutputCount).toBeLessThanOrEqual(1_048_576);
      expect(fixture.estimatedMaximumMultiplyAccumulates)
        .toBeLessThanOrEqual(234_881_024);
    }
  });

  it("retains raw timing samples and reports median and range", () => {
    expect(summarizeOpt0002Milliseconds([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
      range: 8,
    });
    expect(() => summarizeOpt0002Milliseconds([])).toThrow(/must not be empty/);
    expect(() => summarizeOpt0002Milliseconds([1, Number.NaN])).toThrow(
      /finite non-negative/,
    );
  });
});
