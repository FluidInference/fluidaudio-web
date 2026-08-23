import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0009_GENERATED_SHADER_SHA256,
  ACE_OPT_0009_NATIVE_CALIBRATION_SCOPE,
  ACE_OPT_0009_SHAPES,
  OPT_0009_PARAKEET_CAPABILITIES_SOURCE_SHA256,
  OPT_0009_PARAKEET_COMMIT,
  OPT_0009_PARAKEET_GEMM_SOURCE_SHA256,
  OPT_0009_PARAKEET_RUNTIME_PLAN_SOURCE_SHA256,
  aceOpt0009VariantWgsl,
  aceOpt0009WeightScalarIndex,
  isAceOpt0009NativeCalibrationPage,
  planAceOpt0009Variants,
  requireAceOpt0009Fixed32Device,
} from "../benchmark/opt-0009-parakeet-gemm-calibration.js";

describe("OPT-0009 Parakeet GEMM calibration contract", () => {
  it("installs its browser UI only on the native calibration page", () => {
    expect(isAceOpt0009NativeCalibrationPage(
      "/ace-step-1.5.wgsl/benchmark/opt-0009-parakeet-gemm-calibration.html",
    )).toBe(true);
    expect(isAceOpt0009NativeCalibrationPage(
      "/ace-step-1.5.wgsl/benchmark/opt-0009-ace-gemm-accumulation-ab.html",
    )).toBe(false);
  });

  it("pins the authenticated Parakeet sources", async () => {
    expect(OPT_0009_PARAKEET_COMMIT).toBe(
      "7ee112738262a6f5a0efd2f150748a4087432fbb",
    );
    const files = [
      [
        "../../parakeet.wgsl/src/webgpu/kernels/gemm.ts",
        OPT_0009_PARAKEET_GEMM_SOURCE_SHA256,
      ],
      [
        "../../parakeet.wgsl/src/model/runtime-plan.ts",
        OPT_0009_PARAKEET_RUNTIME_PLAN_SOURCE_SHA256,
      ],
      [
        "../../parakeet.wgsl/src/webgpu/capabilities.ts",
        OPT_0009_PARAKEET_CAPABILITIES_SOURCE_SHA256,
      ],
    ] as const;
    for (const [relative, expected] of files) {
      const bytes = await readFile(new URL(relative, import.meta.url));
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(expected);
    }
  });

  it("plans three native Parakeet pairs and four guarded ACE-shape pairs", () => {
    expect(ACE_OPT_0009_SHAPES.map((shape) => shape.id)).toEqual([
      "parakeet-m7520-k1024-n4096",
      "parakeet-m7520-k4096-n1024",
      "parakeet-m7520-k1024-n1024",
      "ace-m2250-k2048-n2048",
      "ace-m2250-k2048-n1024",
      "ace-m2250-k2048-n6144",
      "ace-m2250-k6144-n2048",
    ]);
    const plans = planAceOpt0009Variants();
    expect(plans).toHaveLength(14);
    expect(plans.map((plan) => [plan.variantId, plan.workgroups])).toEqual([
      ["parakeet-m7520-k1024-n4096-fp16", [16, 235, 1]],
      ["parakeet-m7520-k1024-n4096-fp32", [32, 235, 1]],
      ["parakeet-m7520-k4096-n1024-fp16", [4, 235, 1]],
      ["parakeet-m7520-k4096-n1024-fp32", [8, 235, 1]],
      ["parakeet-m7520-k1024-n1024-fp16", [4, 235, 1]],
      ["parakeet-m7520-k1024-n1024-fp32", [8, 235, 1]],
      ["ace-m2250-k2048-n2048-fp16", [8, 47, 1]],
      ["ace-m2250-k2048-n2048-fp32", [16, 47, 1]],
      ["ace-m2250-k2048-n1024-fp16", [4, 47, 1]],
      ["ace-m2250-k2048-n1024-fp32", [8, 47, 1]],
      ["ace-m2250-k2048-n6144-fp16", [24, 47, 1]],
      ["ace-m2250-k2048-n6144-fp32", [48, 47, 1]],
      ["ace-m2250-k6144-n2048-fp16", [8, 47, 1]],
      ["ace-m2250-k6144-n2048-fp32", [16, 47, 1]],
    ]);
    for (const plan of plans) {
      expect(plan.sourceClassification).toBe("unchanged-parakeet-generator");
      expect(plan.scheduledRows).toBe(plan.scope === "parakeet-production-calibration"
        ? 7_520
        : 2_256);
      expect(plan.rowUtilization).toBe(plan.rows / plan.scheduledRows);
      expect(plan.validFlops).toBe(plan.validMacs * 2);
      expect(plan.scheduledFlops).toBe(plan.scheduledMacs * 2);
    }
  });

  it("uses unchanged direct sources only on their supported native domains", () => {
    expect(ACE_OPT_0009_NATIVE_CALIBRATION_SCOPE).toEqual({
      comparison: "native-fp16-accumulation-vs-native-fp32-accumulation",
      answersFp16OperandsFp32Accumulation: false,
      closesExperiment: false,
    });
    const plans = planAceOpt0009Variants();
    for (const plan of plans) {
      const shader = aceOpt0009VariantWgsl(plan);
      expect(createHash("sha256").update(shader).digest("hex")).toBe(
        ACE_OPT_0009_GENERATED_SHADER_SHA256[
          plan.variantId as keyof typeof ACE_OPT_0009_GENERATED_SHADER_SHA256
        ],
      );
      expect(shader).toContain("enable subgroups;");
      expect(shader).not.toMatch(/subgroup.?matrix/i);
      if (plan.precision === "fp16") {
        expect(shader).toContain("enable f16;");
      } else {
        expect(shader).not.toContain("enable f16;");
      }
      if (plan.scope === "ace-180s-shape-calibration") {
        expect(plan.kernelPath).toBe("parakeet-staged-row-major");
        expect(shader).toContain("a_row < ROWS");
      } else {
        expect(plan.kernelPath).toBe("parakeet-direct-tile-major");
      }
    }
  });

  it("maps both tile-major layouts bijectively across K32/N tile boundaries", () => {
    for (const precision of ["fp16", "fp32"] as const) {
      const columns = precision === "fp16" ? 512 : 256;
      const count = 64 * columns;
      const seen = new Set<number>();
      for (let logical = 0; logical < count; logical += 1) {
        const physical = aceOpt0009WeightScalarIndex({
          precision,
          kernelPath: "parakeet-direct-tile-major",
          inner: 64,
          columns,
        }, logical);
        expect(physical).toBeGreaterThanOrEqual(0);
        expect(physical).toBeLessThan(count);
        seen.add(physical);
      }
      expect(seen.size).toBe(count);
      const tileColumns = precision === "fp16" ? 256 : 128;
      const boundaryScalars = [
        0,
        tileColumns - 1,
        tileColumns,
        columns - 1,
        31 * columns + tileColumns - 1,
        32 * columns,
        count - 1,
      ];
      expect(new Set(boundaryScalars.map((logical) =>
        aceOpt0009WeightScalarIndex({
          precision,
          kernelPath: "parakeet-direct-tile-major",
          inner: 64,
          columns,
        }, logical)
      )).size).toBe(boundaryScalars.length);
    }
    expect(aceOpt0009WeightScalarIndex({
      precision: "fp16",
      kernelPath: "parakeet-staged-row-major",
      inner: 64,
      columns: 512,
    }, 12_345)).toBe(12_345);
  });

  it("fails closed without shader-f16 and fixed 32-lane subgroups", () => {
    expect(() => requireAceOpt0009Fixed32Device(
      new Set(["shader-f16", "subgroups"]),
      32,
      32,
    )).not.toThrow();
    expect(() => requireAceOpt0009Fixed32Device(
      new Set(["subgroups"]),
      32,
      32,
    )).toThrow(/shader-f16/);
    expect(() => requireAceOpt0009Fixed32Device(
      new Set(["shader-f16", "subgroups"]),
      16,
      32,
    )).toThrow(/fixed 32-lane/);
  });
});
