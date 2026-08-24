import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { ACE_DIT_DENSE_K4_FP16_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0032_DENSE_TILE_COLUMNS,
  ACE_OPT_0032_DENSE_TILE_INNER,
  ACE_OPT_0032_DENSE_TILE_ROWS,
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
  aceOpt0032DenseK4PartialsWgsl,
  aceOpt0032PackedWeightIndex,
  packAceOpt0032DenseWeightU16,
  planAceOpt0032DenseK4Partials,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID,
  ACE_OPT_0088_DENSE_K4_PORTABLE_WEIGHT_LAYOUT,
  ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES,
  AceOpt0088DenseK4PortableKernel,
  aceOpt0088DenseK4PortableWgsl,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-portable.js";
import {
  ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_KERNEL_ID,
  ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_WEIGHT_LAYOUT,
  AceOpt0088DenseK4PortableProductionKernel,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-portable-production.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-portable.ts",
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

describe("OPT-0088 dense FP16 K4 portable", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("shares the OPT-0032 plan and geometry for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0032DenseK4Partials({
      rows: 2_250,
      inner,
      columns,
    });
    expect(plan).toMatchObject({
      tileRows: ACE_OPT_0032_DENSE_TILE_ROWS,
      tileColumns: ACE_OPT_0032_DENSE_TILE_COLUMNS,
      tileInner: ACE_OPT_0032_DENSE_TILE_INNER,
      workgroupSize: ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
      subgroupSize: ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
      rowTiles: 71,
      columnTiles: columns / 128,
      innerK4Groups: inner / 4,
      workgroupCount: 71 * columns / 128,
      outputRangeCount: 1,
    });
    // The portable generator accepts exactly the shapes the OPT-0032
    // generator accepts and embeds the same tile geometry.
    const wgsl = aceOpt0088DenseK4PortableWgsl({ rows: 2_250, inner, columns });
    expect(wgsl).toContain(`const INNER = ${inner}u;`);
    expect(wgsl).toContain(`const COLUMNS = ${columns}u;`);
    expect(wgsl).toContain(`const INNER_K4_GROUPS = ${inner / 4}u;`);
    expect(() => aceOpt0088DenseK4PortableWgsl({
      rows: 2_250,
      inner: 1_024,
      columns: 2_048,
    })).toThrow(/non-production/);
  });

  it("consumes the identical OPT-0032 packed weight layout", () => {
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_WEIGHT_LAYOUT).toBe(
      ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
    );
    // The WGSL weight_base vector index must address exactly the scalars the
    // exported OPT-0032 packer produced: vector*4 + innerInK4 == scalar index.
    const inner = 8;
    const columns = 256;
    const innerK4Groups = inner / 4;
    const logical = Uint16Array.from(
      { length: inner * columns },
      (_, index) => index + 1,
    );
    const packed = packAceOpt0032DenseWeightU16(logical, inner, columns);
    for (let k = 0; k < inner; k += 1) {
      for (let column = 0; column < columns; column += 1) {
        const columnTile = Math.floor(column / 128);
        const columnInTile = column % 128;
        const outputInLane = columnInTile % 4;
        const lane = Math.floor(columnInTile / 4);
        const innerK4 = Math.floor(k / 4);
        const innerInK4 = k % 4;
        // Mirrors the generated `weight_base` expression plus the b0..b3
        // vector offsets: ((tile*groups + k4) * 4) * 32 + lane, + 32*output.
        const wgslVectorIndex =
          ((columnTile * innerK4Groups + innerK4) * 4) * 32 +
          lane + 32 * outputInLane;
        expect(wgslVectorIndex * 4 + innerInK4).toBe(
          aceOpt0032PackedWeightIndex(k, column, inner, columns),
        );
        expect(packed[wgslVectorIndex * 4 + innerInK4]).toBe(
          logical[k * columns + column],
        );
      }
    }
  });

  it("emits portable WGSL with no subgroup use and the OPT-0032 arithmetic", () => {
    const wgsl = aceOpt0088DenseK4PortableWgsl(PRODUCTION_SHAPE);
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID).toBe(
      "opt-0088-dense-k4-fp16-portable-v1",
    );
    // Entry point, workgroup size, and bindings match the OPT-0032 contract.
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("fn main(");
    expect(wgsl).toContain(
      "@group(0) @binding(0) var<storage, read> activation: array<f32>;",
    );
    expect(wgsl).toContain(
      "@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;",
    );
    expect(wgsl).toContain(
      "@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;",
    );
    // Portable transport: workgroup staging plus barriers, never subgroups.
    expect(wgsl).not.toContain("enable subgroups");
    expect(wgsl.toLowerCase()).not.toContain("subgroup");
    expect(wgsl).toContain(
      `var<workgroup> staged_a: array<vec4<f16>, ${
        ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES / 8
      }>;`,
    );
    expect(wgsl.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(wgsl.match(/let a[0-7] = staged_a\[staged_base \+ [0-7]u\];/g))
      .toHaveLength(8);
    // Same per-output arithmetic tokens as the OPT-0032 kernel.
    expect(wgsl).toContain("var lane_a = vec4<f16>(0.0h)");
    expect(wgsl.match(/dot\(a0, b[0-3]\)/g)).toHaveLength(4);
    expect(wgsl).toContain("let partial0 = vec4<f16>(");
    expect(wgsl).toContain("acc0 = acc0 + vec4<f32>(partial0)");
    expect(wgsl).toContain("var acc0 = vec4<f32>(0.0)");
    expect(wgsl).not.toMatch(/var acc\d+\s*=\s*vec4<f16>/);
    expect(KERNEL_SOURCE).not.toContain("OPT-0020");
    expect(KERNEL_SOURCE).not.toContain("subgroupBroadcast");
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES).toBe(256);
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES)
      .toBeLessThanOrEqual(16_384);
  });

  it("preserves the OPT-0032 per-output reduction order token for token", () => {
    const portable = aceOpt0088DenseK4PortableWgsl(PRODUCTION_SHAPE);
    // Normalize the only intentional renames (lane/slice builtin plumbing);
    // every arithmetic line must then be byte-identical and in the same order.
    const sibling = aceOpt0032DenseK4PartialsWgsl(PRODUCTION_SHAPE)
      .replaceAll("subgroup_lane", "lane");
    for (const pattern of [
      /^var acc\d+ = vec4<f32>\(0\.0\);$/,
      /^dot\(a\d, b\d\),?$/,
      /^let partial\d+ = vec4<f16>\($/,
      /^acc\d+ = acc\d+ \+ vec4<f32>\(partial\d+\);$/,
      /^f16\(activation\[activation_base( \+ \du)?\]\),?$/,
      /^let b\d = weight\[weight_base( \+ \d+u)?\];$/,
      /^output\[row \* \(COLUMNS \/ 4u\) \+ column_vector\] = acc\d+;$/,
      /^if \(lane < 8u && lane_row < ROWS\) \{$/,
    ]) {
      const portableLines = extractLines(portable, pattern);
      expect(portableLines.length).toBeGreaterThan(0);
      expect(portableLines).toEqual(extractLines(sibling, pattern));
    }
  });

  it("creates without the subgroups feature and fails closed after destroy", async () => {
    expect(() => AceOpt0088DenseK4PortableKernel.create(
      portableDevice([]),
    )).toThrow(/shader-f16/);
    const device = portableDevice(["shader-f16"]);
    const bindings = productionBindings();
    const kernel = AceOpt0088DenseK4PortableKernel.create(device);
    const dispatch = await kernel.createDispatch(
      "opt-0088-portable",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    expect(dispatch.kernelId).toBe(ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID);
    expect(dispatch.weightLayout).toBe(
      ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
    );
    expect(dispatch.rangeCount).toBe(1);
    expect(Object.isFrozen(dispatch)).toBe(true);
    expect(dispatch.plan).toEqual(planAceOpt0032DenseK4Partials({
      rows: 1,
      inner: 2_048,
      columns: 2_048,
    }));
    expect(dispatch.plan.packedWeightStorageShape).toEqual([
      16, 512, 4, 32, 4,
    ]);
    await expect(kernel.createDispatch(
      "opt-0088-portable-bias",
      { rows: 1, inner: 2_048, columns: 2_048 },
      { ...bindings, bias: bindings.activation },
    )).rejects.toThrow(/do not accept bias/);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "opt-0088-after-destroy",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    )).rejects.toThrow(/portable dense K4 kernel was destroyed/);
  });

  it("binds the portable owner behind the rev8 production profile", async () => {
    const device = portableDevice(["shader-f16"]);
    const bindings = productionBindings();
    const kernel = AceOpt0088DenseK4PortableProductionKernel.create(device, {});
    const dispatch = await kernel.createDispatch(
      "opt-0088-production",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_KERNEL_ID).toBe(
      "opt-0088-dense-k4-fp16-portable-production-v1",
    );
    expect(ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_WEIGHT_LAYOUT).toBe(
      ACE_DIT_DENSE_K4_FP16_LAYOUT,
    );
    expect(dispatch.weightLayout).toBe(ACE_DIT_DENSE_K4_FP16_LAYOUT);
    const plan = dispatch.plan as ReturnType<
      typeof planAceOpt0032DenseK4Partials
    >;
    expect(plan.packedWeightStorageShape).toEqual([
      16, 512, 4, 32, 4,
    ]);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "opt-0088-production-after-destroy",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    )).rejects.toThrow(/portable production K4 kernel was destroyed/);
  });
});
