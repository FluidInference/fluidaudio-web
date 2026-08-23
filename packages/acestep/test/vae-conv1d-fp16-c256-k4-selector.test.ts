import { readdirSync, readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
} from "../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0057VaeK7ShapeSelectorKernel,
  selectAceOpt0057VaeK7,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS,
  ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT,
  ACE_OPT_0076_VAE_C256_K4_ROUTES,
  ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
  ACE_OPT_0076_VAE_NATIVE_K4_ROUTE_COUNT,
  ACE_OPT_0076_VAE_NATIVE_SCALAR_ROUTE_COUNT,
  ACE_OPT_0076_VAE_ROW_REUSE_K4_ROUTE_COUNT,
  AceOpt0076VaeC256K4SelectorKernel,
  selectAceOpt0076VaeC256K4,
} from "../src/webgpu/kernels/vae-conv1d-fp16-c256-k4-selector.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });
const SELECTOR_SOURCE_URL = new URL(
  "../src/webgpu/kernels/vae-conv1d-fp16-c256-k4-selector.ts",
  import.meta.url,
);

describe("OPT-0076 VAE C256 native-K4 selector", () => {
  it("reconciles all 17 literal OPT-0057 routes and changes only C256", () => {
    const operations = k7Operations();
    expect(operations).toHaveLength(ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT);
    expect(ACE_OPT_0076_VAE_C256_K4_ROUTES).toHaveLength(
      ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT,
    );
    expect(new Set(ACE_OPT_0076_VAE_C256_K4_ROUTES.map(
      ({ operationLabel }) => operationLabel,
    )).size).toBe(ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT);
    expect(Object.isFrozen(ACE_OPT_0076_VAE_C256_K4_ROUTES)).toBe(true);

    const targets = new Set<string>(ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS);
    const selections = operations.map((operation) => {
      const outputStorage = outputStorageFor(operation.output);
      const literal = selectAceOpt0057VaeK7(
        operation.label,
        operation.shape,
        operation.bias !== undefined,
        outputStorage,
      );
      const selected = selectAceOpt0076VaeC256K4(
        operation.label,
        operation.shape,
        operation.bias !== undefined,
        outputStorage,
      );
      expect(selected).toMatchObject({
        selectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
        literalSelectorKernelId:
          ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
        operationLabel: operation.label,
      });
      expect(selected.route).toMatchObject({
        operationLabel: literal.route.operationLabel,
        literalOwner: literal.route.owner,
        inputChannels: literal.route.inputChannels,
        outputChannels: literal.route.outputChannels,
        dilation: literal.route.dilation,
        hasBias: literal.route.hasBias,
        outputStorage: literal.route.outputStorage,
      });
      if (targets.has(operation.label)) {
        expect(selected).toMatchObject({
          owner: "native-k4",
          kernelId:
            ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
        });
        expect(selected.route).toMatchObject({
          literalOwner: "native-scalar-fp32",
          literalKernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
          inputChannels: 256,
          outputChannels: 256,
          hasBias: true,
          outputStorage: "float16",
        });
      } else {
        expect(selected.owner).toBe(literal.route.owner);
        expect(selected.kernelId).toBe(literal.route.owner === "row-reuse-k4"
          ? ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
          : ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID);
      }
      return selected;
    });
    expect(selections.filter(({ owner }) => owner === "row-reuse-k4"))
      .toHaveLength(ACE_OPT_0076_VAE_ROW_REUSE_K4_ROUTE_COUNT);
    expect(selections.filter(({ owner }) => owner === "native-k4"))
      .toHaveLength(ACE_OPT_0076_VAE_NATIVE_K4_ROUTE_COUNT);
    expect(selections.filter(({ owner }) => owner === "native-scalar-fp32"))
      .toHaveLength(ACE_OPT_0076_VAE_NATIVE_SCALAR_ROUTE_COUNT);
    expect(selections.filter(({ owner }) => owner === "native-k4").map(
      ({ operationLabel, route }) => [operationLabel, route.dilation],
    )).toEqual([
      ["block-2-res-1-conv1", 1],
      ["block-2-res-2-conv1", 3],
      ["block-2-res-3-conv1", 9],
    ]);
  });

  it("fails closed on every C256 label, shape, bias, and storage mismatch", () => {
    const operation = k7Operations().find(
      ({ label }) => label === "block-2-res-1-conv1",
    )!;
    const valid = operation.shape;
    const invalidShapes: readonly AceVaeConv1dShape[] = [
      { ...valid, batch: 2 },
      { ...valid, inputChannels: 128 },
      { ...valid, outputChannels: 128 },
      { ...valid, kernelSize: 5 },
      { ...valid, stride: 2 },
      { ...valid, dilation: 3 },
      { ...valid, padding: 9 },
    ];
    for (const shape of invalidShapes) {
      expect(() => selectAceOpt0076VaeC256K4(
        operation.label,
        shape,
        true,
        "float16",
      )).toThrow(/authenticated K7 contract/u);
    }
    expect(() => selectAceOpt0076VaeC256K4(
      operation.label,
      valid,
      false,
      "float16",
    )).toThrow(/authenticated K7 contract/u);
    expect(() => selectAceOpt0076VaeC256K4(
      operation.label,
      valid,
      true,
      "float32",
    )).toThrow(/authenticated K7 contract/u);
    expect(() => selectAceOpt0076VaeC256K4(
      "block-2-res-4-conv1",
      valid,
      true,
      "float16",
    )).toThrow(/no K7 route/u);
  });

  it("dispatches through the selected owner and owns lifecycle once", async () => {
    const literalDestroy = vi.spyOn(
      AceOpt0057VaeK7ShapeSelectorKernel.prototype,
      "destroy",
    );
    const nativeK4Destroy = vi.spyOn(
      AceOpt0024VaeConv1dDirectDot4SubgroupKernel.prototype,
      "destroy",
    );
    const device = fakeDevice();
    const selector = AceOpt0076VaeC256K4SelectorKernel.create(
      device,
      FIXED32,
    );
    const targetOperation = k7Operations().find(
      ({ label }) => label === "block-2-res-2-conv1",
    )!;
    const targetPlan = planAceFp16VaeConv1d(
      targetOperation.shape,
      "float16",
    );
    const target = await selector.createDispatch(
      "target-dispatch",
      targetOperation.label,
      targetOperation.shape,
      bindingsFor(targetOperation.shape, "float16", true),
      "float16",
      fullRange(targetPlan),
    );
    expect(target).toMatchObject({
      label: "target-dispatch",
      selectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
      literalSelectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: "block-2-res-2-conv1",
      owner: "native-k4",
      kernelId: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
      plan: targetPlan,
      route: {
        literalOwner: "native-scalar-fp32",
        dilation: 3,
      },
    });

    const delegatedOperation = k7Operations().find(
      ({ label }) => label === "block-3-res-1-conv1",
    )!;
    const delegatedPlan = planAceFp16VaeConv1d(
      delegatedOperation.shape,
      "float16",
    );
    const delegated = await selector.createDispatch(
      "delegated-dispatch",
      delegatedOperation.label,
      delegatedOperation.shape,
      bindingsFor(delegatedOperation.shape, "float16", true),
      "float16",
      fullRange(delegatedPlan),
    );
    expect(delegated).toMatchObject({
      label: "delegated-dispatch",
      selectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
      literalSelectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: "block-3-res-1-conv1",
      owner: "row-reuse-k4",
      kernelId: ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
      plan: delegatedPlan,
      route: {
        literalOwner: "row-reuse-k4",
      },
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);

    const pass = fakePass();
    target.encode(pass);
    delegated.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(2);

    selector.destroy();
    selector.destroy();
    expect(literalDestroy).toHaveBeenCalledOnce();
    expect(nativeK4Destroy).toHaveBeenCalledOnce();
    expect(() => target.encode(pass)).toThrow(/OPT-0076.*destroyed/u);
    expect(() => delegated.encode(pass)).toThrow(/OPT-0076.*destroyed/u);
    await expect(selector.createDispatch(
      "after-destroy",
      targetOperation.label,
      targetOperation.shape,
      bindingsFor(targetOperation.shape, "float16", true),
      "float16",
      fullRange(targetPlan),
    )).rejects.toThrow(/OPT-0076.*destroyed/u);
    literalDestroy.mockRestore();
    nativeK4Destroy.mockRestore();
  });

  it("is benchmark-only and absent from every other production source", () => {
    const selectorSource = readFileSync(SELECTOR_SOURCE_URL, "utf8");
    expect(selectorSource).toContain("Benchmark-only OPT-0076 selector");
    expect(selectorSource).toContain(
      "AceOpt0024VaeConv1dDirectDot4SubgroupKernel",
    );
    expect(selectorSource).toContain("AceOpt0057VaeK7ShapeSelectorKernel");
    for (const sourceUrl of sourceFiles(new URL("../src/", import.meta.url))) {
      if (sourceUrl.href === SELECTOR_SOURCE_URL.href) continue;
      const source = readFileSync(sourceUrl, "utf8");
      expect(source).not.toContain("vae-conv1d-fp16-c256-k4-selector");
      expect(source).not.toContain("AceOpt0076VaeC256K4SelectorKernel");
      expect(source).not.toContain("ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID");
    }
  });
});

function k7Operations() {
  return planAceVaeDecoder(256).operations.filter((operation) =>
    operation.kind === "conv1d" && operation.shape.kernelSize === 7
  ).map((operation) => {
    if (operation.kind !== "conv1d") throw new Error("unreachable");
    return operation;
  });
}

function outputStorageFor(
  output: string,
): AceFp16VaeConv1dOutputStorage {
  return output === "output" ? "float32" : "float16";
}

function bindingsFor(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  hasBias: boolean,
): AceFp16VaeConv1dBindings {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  const bindings = Object.freeze({
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  });
  return hasBias
    ? Object.freeze({ ...bindings, bias: fakeBinding(plan.biasBindingBytes) })
    : bindings;
}

function fullRange(
  plan: ReturnType<typeof planAceFp16VaeConv1d>,
): AceVaeOutputRangeBinding {
  return Object.freeze({
    base: 0,
    count: plan.outputElements,
    control: fakeBinding(16, fakeBuffer(256)),
  });
}

function fakeBinding(size: number, buffer = fakeBuffer(size)): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeDevice(): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 32_768,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minUniformBufferOffsetAlignment: 256,
      minStorageBufferOffsetAlignment: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({})),
    createBindGroup: vi.fn(() => ({})),
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
  };
}

function fakePass(): GPUComputePassEncoder & {
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
  };
}

function sourceFiles(directory: URL): URL[] {
  const files: URL[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const child = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directory);
    if (entry.isDirectory()) files.push(...sourceFiles(child));
    else if (entry.isFile() && entry.name.endsWith(".ts")) files.push(child);
  }
  return files;
}
