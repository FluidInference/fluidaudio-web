import { describe, expect, it, vi } from "vitest";

import {
  ACE_DETOKENIZER_PATCHES,
  AceCorrectnessDetokenizerExpandKernel,
  aceCorrectnessDetokenizerExpandWgsl,
  planAceDetokenizerExpand,
} from "../src/webgpu/kernels/detokenizer-expand.js";

describe("ACE detokenizer expansion contract", () => {
  it("pins five patches per semantic code", () => {
    expect(ACE_DETOKENIZER_PATCHES).toBe(5);
    expect(planAceDetokenizerExpand({ codeCount: 3, width: 4 })).toEqual({
      codeCount: 3,
      width: 4,
      patches: 5,
      codeElements: 12,
      specialElements: 20,
      outputElements: 60,
      workgroupsX: 1,
      workgroupsY: 1,
    });
  });

  it("decodes packed BF16 specials in the reference profile", () => {
    const source = aceCorrectnessDetokenizerExpandWgsl(
      "reference-bf16",
      { codeCount: 2, width: 4 },
    );
    expect(source).toContain("special_tokens: array<u32>");
    expect(source).toContain("bits16 << 16u");
    expect(source).toContain("patch_index * WIDTH + column");
    expect(source).not.toContain("enable f16;");
  });

  it("uses native FP16 activation and weight addition in raw mode", () => {
    const source = aceCorrectnessDetokenizerExpandWgsl(
      "raw-fp16",
      { codeCount: 2, width: 4 },
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("embedded_codes: array<f16>");
    expect(source).toContain("special_tokens: array<f16>");
  });

  it("validates packed-weight byte size and dispatches", async () => {
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
    };
    const pipeline = {
      getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    } as unknown as GPUComputePipeline;
    const device = fakeDevice(pipeline);
    const kernel = AceCorrectnessDetokenizerExpandKernel.create(
      device,
      "reference-bf16",
    );
    const dispatch = await kernel.createDispatch(
      "expand",
      { codeCount: 2, width: 4 },
      {
        embeddedCodes: fakeBinding(32),
        specialTokens: fakeBinding(40),
        output: fakeBinding(160),
      },
    );
    dispatch.encode(pass as unknown as GPUComputePassEncoder);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(1, 1, 1);

    await expect(kernel.createDispatch(
      "short-specials",
      { codeCount: 2, width: 4 },
      {
        embeddedCodes: fakeBinding(32),
        specialTokens: fakeBinding(36),
        output: fakeBinding(160),
      },
    )).rejects.toThrow(/special tokens/);
  });
});

function fakeDevice(pipeline: GPUComputePipeline): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
    createShaderModule: vi.fn(() => ({ label: "module" })),
    createComputePipelineAsync: vi.fn(async () => pipeline),
    createBindGroup: vi.fn(() => ({ label: "bindings" })),
  } as unknown as GPUDevice;
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}
