import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID,
  ACE_OPT_0088_DENSE_PORTABLE_WEIGHT_LAYOUT,
  ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES,
  AceOpt0088DensePortableKernel,
  aceOpt0088DensePortableWgsl,
} from "../src/webgpu/kernels/dit-dense-fp16-portable.js";
import {
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
} from "../src/webgpu/dit-fp16-package.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-portable.ts",
  import.meta.url,
), "utf8");

const PRODUCTION_SHAPE = { rows: 2_250, inner: 2_048, columns: 2_048 };

/** Trimmed matching lines, in emission order, for reduction-order identity. */
function extractLines(wgsl: string, pattern: RegExp): string[] {
  return wgsl
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => pattern.test(line));
}

function portableDevice(features: readonly string[]): GPUDevice {
  return {
    features: new Set(features),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16_384,
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

function productionBindings(): {
  activation: GPUBufferBinding;
  weight: GPUBufferBinding;
  output: GPUBufferBinding;
} {
  const binding = (size: number): GPUBufferBinding => ({
    buffer: { size } as GPUBuffer,
    offset: 0,
    size,
  });
  return {
    activation: binding(2_048 * 4),
    weight: binding(2_048 * 2_048 * 2),
    output: binding(2_048 * 4),
  };
}

describe("OPT-0088 dense FP16 portable (OPT-0009 rev7-oracle port)", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("shares the OPT-0009 plan and geometry for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0009DenseGemm({
      rows: 2_250,
      inner,
      columns,
    });
    expect(plan).toMatchObject({
      tileRows: ACE_OPT_0009_DENSE_TILE_ROWS,
      tileColumns: ACE_OPT_0009_DENSE_TILE_COLUMNS,
      tileInner: ACE_OPT_0009_DENSE_TILE_INNER,
      workgroupSize: ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
      subgroupSize: ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
      rowTiles: 71,
      columnTiles: columns / 256,
      innerTiles: inner / 32,
      workgroupCount: 71 * columns / 256,
      outputRangeCount: 1,
    });
    // The portable generator accepts exactly the shapes the OPT-0009
    // generator accepts and embeds the same tile geometry.
    const wgsl = aceOpt0088DensePortableWgsl({ rows: 2_250, inner, columns });
    expect(wgsl).toContain(`const INNER = ${inner}u;`);
    expect(wgsl).toContain(`const COLUMNS = ${columns}u;`);
    expect(wgsl).toContain(`const INNER_TILES = ${inner / 32}u;`);
    expect(() => aceOpt0088DensePortableWgsl({
      rows: 2_250,
      inner: 1_024,
      columns: 2_048,
    })).toThrow(/non-production/);
  });

  it("consumes the identical OPT-0009 rev7-oracle package weight layout", () => {
    expect(ACE_OPT_0088_DENSE_PORTABLE_WEIGHT_LAYOUT).toBe(
      ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    );
    // The dit-fp16-package kernel-set id is the portable kernel identity.
    expect(ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID).toBe(
      ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID,
    );
  });

  it("emits portable WGSL with no subgroup use and the OPT-0009 arithmetic", () => {
    const wgsl = aceOpt0088DensePortableWgsl(PRODUCTION_SHAPE);
    expect(ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID).toBe(
      "opt-0088-dense-fp16-fp32-portable-v1",
    );
    // Entry point, workgroup size, and bindings match the OPT-0009 contract.
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("fn main(");
    expect(wgsl).toContain(
      "@group(0) @binding(0) var<storage, read> activation: array<f32>;",
    );
    expect(wgsl).toContain(
      "@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;",
    );
    expect(wgsl).toContain(
      "@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;",
    );
    // Portable transport: workgroup staging plus barriers, never subgroups.
    expect(wgsl).not.toContain("enable subgroups");
    expect(wgsl.toLowerCase()).not.toContain("subgroup");
    expect(wgsl).toContain(
      `var<workgroup> staged_a: array<f16, ${
        ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES / 2
      }>;`,
    );
    expect(wgsl.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(wgsl.match(/let a[0-7] = staged_a\[staged_base \+ [0-7]u\];/g))
      .toHaveLength(8);
    // Same per-output arithmetic tokens as the OPT-0009 kernel.
    expect(wgsl).toContain("var lane_a = 0.0h;");
    expect(wgsl).toContain("let b0 = unpack_f16x4(packed_b.x, packed_b.y);");
    expect(wgsl).toContain("let b1 = unpack_f16x4(packed_b.z, packed_b.w);");
    expect(
      wgsl.match(/acc\d+_[01] = acc\d+_[01] \+ vec4<f32>\(f32\(a\d+\)\) \* vec4<f32>\(b[01]\);/g),
    ).toHaveLength(16);
    expect(wgsl.match(/var acc\d+_[01] = vec4<f32>\(0\.0\);/g))
      .toHaveLength(16);
    expect(wgsl).not.toMatch(/var acc\d+_[01]\s*=\s*vec4<f16>/);
    expect(KERNEL_SOURCE).not.toContain("subgroupBroadcast");
    expect(KERNEL_SOURCE).not.toContain("enable subgroups");
    expect(ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES).toBe(64);
    expect(ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES)
      .toBeLessThanOrEqual(16_384);
  });

  it("preserves the OPT-0009 per-output reduction order token for token", () => {
    const portable = aceOpt0088DensePortableWgsl(PRODUCTION_SHAPE);
    // Normalize the only intentional renames (lane/slice builtin plumbing);
    // every arithmetic line must then be byte-identical and in the same order.
    const sibling = aceOpt0009DenseGemmWgsl(PRODUCTION_SHAPE)
      .replaceAll("subgroup_lane", "lane");
    for (const pattern of [
      /^var acc\d+_[01] = vec4<f32>\(0\.0\);$/,
      /^acc\d+_[01] = acc\d+_[01] \+ vec4<f32>\(f32\(a\d+\)\) \* vec4<f32>\(b[01]\);$/,
      /^var lane_a = 0\.0h;$/,
      /^lane_a = f16\(activation\[lane_row \* INNER \+ inner\]\);$/,
      /^if \(lane < 8u && lane_row < ROWS\) \{$/,
      /^let inner = inner_tile \* 32u \+ inner_in_tile;$/,
      /^let b[01] = unpack_f16x4\(packed_b\.[xz], packed_b\.[yw]\);$/,
      /^weight_tile_base \+$/,
      /^inner_in_tile \* 32u \+$/,
      /^let vector_base = row \* \(COLUMNS \/ 4u\) \+ column_base \/ 4u;$/,
      /^output\[vector_base( \+ 1u)?\] = acc\d+_[01];$/,
      /^lane \* 8u;$/,
    ]) {
      const portableLines = extractLines(portable, pattern);
      expect(portableLines.length).toBeGreaterThan(0);
      expect(portableLines).toEqual(extractLines(sibling, pattern));
    }
  });

  it("creates without the subgroups feature and fails closed after destroy", async () => {
    expect(() => AceOpt0088DensePortableKernel.create(
      portableDevice([]),
    )).toThrow(/shader-f16/);
    const device = portableDevice(["shader-f16"]);
    const bindings = productionBindings();
    const kernel = AceOpt0088DensePortableKernel.create(device);
    const dispatch = await kernel.createDispatch(
      "opt-0088-dense-portable",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    expect(dispatch.kernelId).toBe(ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID);
    expect(dispatch.weightLayout).toBe(ACE_DIT_DENSE_FP16_TILE_LAYOUT);
    expect(dispatch.rangeCount).toBe(1);
    expect(Object.isFrozen(dispatch)).toBe(true);
    expect(dispatch.plan).toEqual(planAceOpt0009DenseGemm({
      rows: 1,
      inner: 2_048,
      columns: 2_048,
    }));
    expect(dispatch.plan.packedWeightStorageShape).toEqual([
      8, 64, 32, 256,
    ]);
    await expect(kernel.createDispatch(
      "opt-0088-dense-portable-bias",
      { rows: 1, inner: 2_048, columns: 2_048 },
      { ...bindings, bias: bindings.activation },
    )).rejects.toThrow(/do not accept bias/);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "opt-0088-dense-portable-after-destroy",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    )).rejects.toThrow(/portable dense GEMM kernel was destroyed/);
  });
});
