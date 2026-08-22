import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

const RAW_ARTIFACTS = [
  ["native-correctness.json", "9533036f898cf772f83e57a7035e1ba71ee0cd85c41ae589f4644b65626a9b56"],
  ["native-timing.json", "cc212f5c93ea8dc1a29dfd76c60cf21464caf606461ad233b2ffdfd61bcce4bb"],
  ["native-timing-thermal.jsonl", "d1bb19ad505c13fcf8b16d61d19873bbc4f9f9b4e0f7d9b1d21085b83c276ee5"],
  ["accumulation-correctness.json", "3c0a732e03b8cfa19223a7090673415827879158a7fcde773f3098b1ce407ce4"],
  ["accumulation-timing.json", "ab89be2eb42f85673ad5a62108777c31ffdf8e58d2af72750c8438542c0dde5b"],
  ["accumulation-timing-thermal.jsonl", "d2d3c0d35000329532ac7ac89974423221dfb934a3730674987bee681eae1b3b"],
] as const;

describe("OPT-0009 optimization result", () => {
  it("commits canonical positive benchmark-only schema-v2 evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0009/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0009",
      riskClass: "approximate",
      baselineCommit: "303ab8df036df71768a56774c59c75c4cfe30aa9",
      candidateCommit: "b41108dc1be75da9ba7a72ea64faa98c9dc81ecd",
      identity: {
        nativeHarnessCommit: "30b3b76c8114d2fa55bdc020d21d57ae53be70f3",
        sourceIdentity: {
          parakeetCommit: "7ee112738262a6f5a0efd2f150748a4087432fbb",
          parakeetGemmSha256:
            "35db4fe52a2d096af347ef4f2411159895d563b5df3aecfa19d70a9fb3f47286",
          aceSubgroupGemmSha256:
            "9ba0c589f975f19b7f7990aae5199581a83f87500a3b97ff15eb6ef5a43311ea",
        },
        productionPackageLoaded: false,
        productionRuntimeChanged: false,
        webgpuAdapter: {
          info: { subgroupMinSize: 32, subgroupMaxSize: 32 },
        },
      },
      protocol: { samples: 6, thermalGateSeconds: 30 },
      metrics: {
        candidate: {
          selectedMechanism:
            "fp16-operands-and-storage-with-fp32-accumulation",
          rejectedMechanism: "native-fp16-accumulation",
          correctness: {
            adversarialFixtureCount: 5,
            fp16Fp32BitExactToIndependentCpuModel: true,
            exactShapeOutputElementsPerArmPerPass: 25_344_000,
            nativeFp16Failures: {
              signedZeroCollapsedToZeroCount: 1_409,
              cancellationClassificationMismatchCount: 1_024,
              cancellationMaximumAbsoluteError: 4.50390625,
              rangePositiveInfinityCount: 2_112,
              rangeNegativeInfinityCount: 2_112,
              longKMaximumAbsoluteError: 0.15625,
              longKRootMeanSquareError: 0.10170662264158173,
            },
          },
          protocolCaveats: {
            heartbeatClaimed: false,
            perDispatchMaximumClaimed: false,
          },
          scope: {
            productionIntegrationPerformed: false,
            finalLatentGatePerformed: false,
            listeningGatePerformed: false,
            songRunPerformed: false,
          },
        },
        delta: {
          weightedDenseDiagnostic: {
            shapeMultiplicities: [4, 2, 2, 1],
            layerCount: 24,
            evaluationCount: 8,
            oracleMilliseconds: 39_139.2,
            selectedFp16Fp32Milliseconds: 28_896,
            rejectedNativeFp16Milliseconds: 22_099.2,
            observedEndToEnd: false,
          },
        },
      },
      correctness: { passed: true, listeningRequired: false },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });

    expect(committed.metrics.candidate.exactShapeTiming).toHaveLength(4);
    expect(committed.metrics.candidate.nativeParakeetCalibration).toHaveLength(3);
    const shaderIdentity = (committed.identity as unknown as {
      shaderIdentity: { native: unknown[]; accumulation: unknown[] };
    }).shaderIdentity;
    expect(shaderIdentity.native).toHaveLength(7);
    expect(shaderIdentity.accumulation).toHaveLength(4);
    expect(committed.artifacts).toEqual(RAW_ARTIFACTS.map(([name, sha256]) => ({
      location: `optimization/artifacts/OPT-0009/raw/${name}`,
      sha256,
    })));
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("closes the record and ledger without claiming integration", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0009-fp16-gemm-calibration.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain("1,409 expected tiny nonzero values collapsed to zero");
    expect(record).toContain("does not claim a layer, package-native model");
    expect(record).not.toContain("Result JSON: pending");

    const ledgerRow = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0009 |"),
    );
    expect(ledgerRow).toContain("| positive | benchmark-only |");
    expect(ledgerRow).toContain("[result](results/OPT-0009/result.json)");
    expect(ledgerRow).toContain("no production integration occurred");
  });
});
