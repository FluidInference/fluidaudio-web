import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { ACE_DIT_DENSE_K4_FP16_LAYOUT } from
  "../src/model/manifest.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  aceOpt0032PackedWeightIndex,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
  ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT,
  AceOpt0056DenseK4ExactKernel,
  aceOpt0056DenseK4ExactWgsl,
  aceOpt0056DenseK4LogicalCoordinates,
  aceOpt0056DenseK4PackedWeightIndex,
  packAceOpt0056DenseK4ExactWeightU16,
  planAceOpt0056DenseK4Exact,
  unpackAceOpt0056DenseK4ExactWeightU16,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-exact.js";
import {
  ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT,
  ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT,
  ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT,
  ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
  AceOpt0056SelectiveDenseKernel,
  resolveAceOpt0056DenseRoute,
  type AceOpt0056DenseOperation,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-selective-exact.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

const EXACT_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-exact.ts",
  import.meta.url,
), "utf8");
const SELECTOR_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-selective-exact.ts",
  import.meta.url,
), "utf8");

const OPERATIONS = Object.freeze([
  "self-query-projection",
  "self-key-projection",
  "self-value-projection",
  "self-output-projection",
  "cross-query-projection",
  "cross-output-projection",
  "mlp-gate-projection",
  "mlp-up-projection",
  "mlp-down-projection",
] as const satisfies readonly AceOpt0056DenseOperation[]);

describe("OPT-0056 exact revision-8 K4 owner", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("plans the registered rev8 geometry for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0056DenseK4Exact({
      rows: 2_250,
      inner,
      columns,
    });
    expect(plan).toMatchObject({
      rows: 2_250,
      inner,
      columns,
      tileRows: 32,
      tileColumns: 128,
      tileInner: 4,
      workgroupSize: 128,
      subgroupSize: 32,
      rowTiles: 71,
      columnTiles: columns / 128,
      innerK4Groups: inner / 4,
      outputRangeCount: 1,
    });
    expect(plan.packedWeightStorageShape).toEqual([
      columns / 128,
      inner / 4,
      4,
      32,
      4,
    ]);
  });

  it("exhaustively proves forward/inverse rev8 layout and raw U16 identity", () => {
    const inner = 12;
    const columns = 256;
    const logical = Uint16Array.from(
      { length: inner * columns },
      (_, index) => (index * 4051 + 17) & 0xffff,
    );
    const packed = packAceOpt0056DenseK4ExactWeightU16(
      logical,
      inner,
      columns,
    );
    const visited = new Set<number>();
    for (let k = 0; k < inner; k += 1) {
      for (let column = 0; column < columns; column += 1) {
        const physical = aceOpt0056DenseK4PackedWeightIndex(
          k,
          column,
          inner,
          columns,
        );
        const inverse = aceOpt0056DenseK4LogicalCoordinates(
          physical,
          inner,
          columns,
        );
        expect(inverse).toEqual({ innerIndex: k, column });
        expect(physical).toBe(
          aceOpt0032PackedWeightIndex(k, column, inner, columns),
        );
        expect(packed[physical]).toBe(logical[k * columns + column]);
        visited.add(physical);
      }
    }
    expect(visited.size).toBe(logical.length);
    expect(unpackAceOpt0056DenseK4ExactWeightU16(
      packed,
      inner,
      columns,
    )).toEqual(logical);
    expect(ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT).toBe(
      ACE_DIT_DENSE_K4_FP16_LAYOUT,
    );
  });

  it("emits four increasing-K FP32 multiply/add steps without FP16 partials", () => {
    const wgsl = aceOpt0056DenseK4ExactWgsl({
      rows: 2_250,
      inner: 6_144,
      columns: 2_048,
    });
    expect(ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID).toMatch(
      /exact-increasing-k-fp32-fixed32-wg128-m32-n128-v1$/u,
    );
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("weight: array<vec4<f16>>");
    expect(wgsl).toContain("var acc0 = vec4<f32>(0.0)");
    expect(wgsl.match(/acc0 = acc0 \+ vec4<f32>\(f32\(a0\.[xyzw]\)\)/gu))
      .toHaveLength(4);
    expect(wgsl.indexOf("f32(a0.x)")).toBeLessThan(wgsl.indexOf("f32(a0.y)"));
    expect(wgsl.indexOf("f32(a0.y)")).toBeLessThan(wgsl.indexOf("f32(a0.z)"));
    expect(wgsl.indexOf("f32(a0.z)")).toBeLessThan(wgsl.indexOf("f32(a0.w)"));
    expect(wgsl).not.toContain("dot(");
    expect(wgsl).not.toMatch(/partial/u);
    expect(wgsl).not.toContain("var<workgroup>");
    expect(wgsl).not.toContain("workgroupBarrier");
  });

  it("owns and destroys exact dispatches fail closed", async () => {
    const kernel = AceOpt0056DenseK4ExactKernel.create(fakeDevice(), {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 1, inner: 2_048, columns: 2_048 } as const;
    const dispatch = await kernel.createDispatch(
      "opt-0056-exact-owner",
      shape,
      bindingsFor(shape),
    );
    expect(dispatch.kernelId).toBe(ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID);
    expect(dispatch.weightLayout).toBe(ACE_DIT_DENSE_K4_FP16_LAYOUT);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "opt-0056-after-destroy",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/was destroyed/u);
  });
});

describe("OPT-0056 selective exact-down selector", () => {
  it("routes only exact MLP down and rejects every label/shape mismatch", () => {
    expect(resolveAceOpt0056DenseRoute(
      "ace-dit-eval-7-layer-23-mlp-down-projection",
      { rows: 2_250, inner: 6_144, columns: 2_048 },
    )).toMatchObject({
      evaluation: 7,
      layer: 23,
      operation: "mlp-down-projection",
      owner: "opt-0056-k4-exact-fp32",
      kernelId: ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
    });
    expect(resolveAceOpt0056DenseRoute(
      "ace-dit-eval-0-layer-0-mlp-up-projection",
      { rows: 2_250, inner: 2_048, columns: 6_144 },
    )).toMatchObject({
      owner: "opt-0032-k4-fp16-partials",
      kernelId: ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
    });
    expect(() => resolveAceOpt0056DenseRoute(
      "ace-dit-eval-0-layer-0-mlp-down-projection",
      { rows: 2_250, inner: 2_048, columns: 2_048 },
    )).toThrow(/expected M2250\/K6144\/N2048/u);
    expect(() => resolveAceOpt0056DenseRoute(
      "ace-dit-eval-8-layer-0-mlp-down-projection",
      { rows: 2_250, inner: 6_144, columns: 2_048 },
    )).toThrow(/unregistered dense label/u);
    expect(() => resolveAceOpt0056DenseRoute(
      "ace-dit-eval-0-layer-0-cross-key-projection",
      { rows: 2_250, inner: 2_048, columns: 1_024 },
    )).toThrow(/unregistered dense label/u);
  });

  it("records the actual 216-route inventory and owns lifecycle exactly once", async () => {
    const selector = AceOpt0056SelectiveDenseKernel.create(fakeDevice(), {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    for (let evaluation = 0; evaluation < 8; evaluation += 1) {
      for (let layer = 0; layer < 24; layer += 1) {
        for (const operation of OPERATIONS) {
          const shape = shapeFor(operation);
          await selector.createDispatch(
            `ace-dit-eval-${evaluation}-layer-${layer}-${operation}`,
            shape,
            bindingsFor(shape),
          );
        }
      }
    }
    const profile = selector.finalizeRoutes();
    expect(profile).toMatchObject({
      schema: "ace-opt-0056-selective-dense-routes-v1",
      kernelSetId: ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
      routeCount: ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT,
      approximateRouteCount: ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT,
      exactDownRouteCount: ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT,
    });
    expect(profile.routes).toHaveLength(216);
    expect(profile.routes.filter(({ operation }) =>
      operation === "mlp-down-projection"
    ).every(({ owner }) => owner === "opt-0056-k4-exact-fp32")).toBe(true);
    expect(profile.routes.filter(({ operation }) =>
      operation !== "mlp-down-projection"
    ).every(({ owner }) => owner === "opt-0032-k4-fp16-partials")).toBe(true);
    await expect(selector.createDispatch(
      "ace-dit-eval-0-layer-0-self-query-projection",
      shapeFor("self-query-projection"),
      bindingsFor(shapeFor("self-query-projection")),
    )).rejects.toThrow(/were finalized/u);
    expect(() => selector.finalizeRoutes()).toThrow(/already finalized/u);
    selector.destroy();
    selector.destroy();
  });

  it("does not leak the diagnostic selector into production-facing source", () => {
    expect(EXACT_SOURCE).toContain("strictly increasing logical K order");
    expect(SELECTOR_SOURCE).toContain("Benchmark-only selector");
    expect(SELECTOR_SOURCE).not.toContain("OPT-0051");
    expect(SELECTOR_SOURCE).not.toContain("vae");
  });
});

function shapeFor(operation: AceOpt0056DenseOperation): AceGemmShape {
  if (operation === "self-key-projection" || operation === "self-value-projection") {
    return Object.freeze({ rows: 2_250, inner: 2_048, columns: 1_024 });
  }
  if (operation === "mlp-gate-projection" || operation === "mlp-up-projection") {
    return Object.freeze({ rows: 2_250, inner: 2_048, columns: 6_144 });
  }
  if (operation === "mlp-down-projection") {
    return Object.freeze({ rows: 2_250, inner: 6_144, columns: 2_048 });
  }
  return Object.freeze({ rows: 2_250, inner: 2_048, columns: 2_048 });
}

function bindingsFor(shape: AceGemmShape): AceGemmBufferBindings {
  return Object.freeze({
    activation: binding(shape.rows * shape.inner * 4),
    weight: binding(shape.inner * shape.columns * 2),
    output: binding(shape.rows * shape.columns * 4),
  });
}

function binding(size: number): GPUBufferBinding {
  return Object.freeze({
    buffer: { size } as GPUBuffer,
    offset: 0,
    size,
  });
}

function fakeDevice(): GPUDevice {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
    createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: [] }),
    }),
    createComputePipelineAsync: async () => ({
      getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
}
