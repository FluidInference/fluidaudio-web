import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../src/webgpu/kernels/dit-dense-fp16.js";

describe("OPT-0009 production dense GEMM", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("accepts only the proven K%i/N%i geometry", (inner, columns) => {
    const plan = planAceOpt0009DenseGemm({ rows: 321, inner, columns });
    expect(plan).toMatchObject({
      tileRows: ACE_OPT_0009_DENSE_TILE_ROWS,
      tileColumns: ACE_OPT_0009_DENSE_TILE_COLUMNS,
      tileInner: ACE_OPT_0009_DENSE_TILE_INNER,
      rowTiles: 11,
      columnTiles: columns / ACE_OPT_0009_DENSE_TILE_COLUMNS,
      innerTiles: inner / ACE_OPT_0009_DENSE_TILE_INNER,
      outputRangeCount: 1,
    });
    expect(plan.packedWeightStorageShape).toEqual([
      columns / 256,
      inner / 32,
      32,
      256,
    ]);
  });

  it("rejects every unproven shape and preserves FP16 operands/FP32 accumulation", () => {
    expect(() => planAceOpt0009DenseGemm({
      rows: 321,
      inner: 1_024,
      columns: 2_048,
    })).toThrow(/non-production/);
    const wgsl = aceOpt0009DenseGemmWgsl({
      rows: 321,
      inner: 2_048,
      columns: 2_048,
    });
    expect(wgsl).toContain("lane_a = f16(activation[");
    expect(wgsl).toContain("var acc0_0 = vec4<f32>(0.0)");
    expect(wgsl).toContain("vec4<f32>(f32(a0)) * vec4<f32>(b0)");
    expect(wgsl).toContain("var<storage, read_write> output: array<vec4<f32>>");
  });
});
