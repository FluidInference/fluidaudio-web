import { describe, expect, it } from "vitest";

import {
  createAceOpt0002VaeWorkloadComparison,
} from "../benchmark/opt-0002-vae-workload.js";
import { ACE_OPT_0001_VAE_TRANSPOSE_PARTS } from
  "../benchmark/opt-0001-vae-workload.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";

describe("OPT-0002 work-aware VAE quantum workload", () => {
  it("pins the exact authenticated 256-frame baseline and candidate plans", () => {
    const comparison = createAceOpt0002VaeWorkloadComparison();

    expect(comparison.baseline).toMatchObject({
      quantumWorkPolicy: {
        maximumConvolutionMultiplyAccumulates: 234_881_024,
        maximumOutputElements: 32_768,
      },
      decoderQuantumCount: 62_622,
      primitiveDispatchCount: 62_702,
      commandBufferCountIncludingReadback: 62_623,
      configuredCooperativeIdleMilliseconds: 62_622,
      familyQuantumCounts: {
        conv1d: 22_126,
        "conv-transpose1d": 3_680,
        snake: 25_776,
        add: 11_040,
      },
    });
    expect(comparison.candidate).toEqual({
      quantumWorkPolicy: {
        maximumConvolutionMultiplyAccumulates: 234_881_024,
        maximumOutputElements: 1_048_576,
      },
      decoderQuantumCount: 3_942,
      primitiveDispatchCount: 3_988,
      commandBufferCountIncludingReadback: 3_943,
      configuredCooperativeIdleMilliseconds: 3_942,
      familyQuantumCounts: {
        conv1d: 2_459,
        "conv-transpose1d": 322,
        snake: 813,
        add: 348,
      },
      maximumLogicalOutputElements: 1_048_576,
      maximumEstimatedMultiplyAccumulates: 234_881_024,
      outputBudgetViolationCount: 0,
      convolutionMacBudgetViolationCount: 0,
    });
    expect(comparison.decoderQuantumReduction).toBe(58_680);
    expect(comparison.decoderQuantumReductionRatio).toBeCloseTo(15.88584475);
    expect(comparison.configuredCooperativeIdleReductionMilliseconds)
      .toBe(58_680);
  });

  it.each([
    [174, 2_712, 2_744],
    [192, 2_967, 3_002],
    [256, 3_942, 3_988],
  ] as const)(
    "pins the %i-frame candidate planner to %i quanta and %i dispatches",
    (inputFrames, quantumCount, primitiveDispatchCount) => {
      const cooperative = planAceVaeDecoderQuanta(
        planAceVaeDecoder(inputFrames),
        ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
      );
      expect(cooperative.quantumCount).toBe(quantumCount);
      expect(cooperative.primitiveDispatchCount).toBe(primitiveDispatchCount);
    },
  );
});
