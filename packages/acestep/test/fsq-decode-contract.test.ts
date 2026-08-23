import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import oracle from "./fsq-codebook-vectors.json";
import {
  ACE_FSQ_BASES,
  ACE_FSQ_CODEBOOK_SIZE,
  ACE_FSQ_CODE_DIMENSION,
  ACE_FSQ_LEVEL_5_BFLOAT16_VALUES,
  ACE_FSQ_LEVEL_8_BFLOAT16_VALUES,
  ACE_FSQ_LEVELS,
  AceCorrectnessFsqDecodeKernel,
  aceCorrectnessFsqDecodeWgsl,
  decodeAceFsqCodes,
  planAceFsqDecode,
} from "../src/webgpu/kernels/fsq-decode.js";

describe("ACE FSQ decode contract", () => {
  it("pins the upstream mixed-radix codebook", () => {
    expect(ACE_FSQ_LEVELS).toEqual([8, 8, 8, 5, 5, 5]);
    expect(ACE_FSQ_BASES).toEqual([1, 8, 64, 512, 2_560, 12_800]);
    expect(ACE_FSQ_CODE_DIMENSION).toBe(6);
    expect(ACE_FSQ_CODEBOOK_SIZE).toBe(64_000);
    expect(ACE_FSQ_LEVEL_8_BFLOAT16_VALUES).toEqual(
      oracle.scalarValues.level8.map((record) => record.bfloat16Value),
    );
    expect(ACE_FSQ_LEVEL_5_BFLOAT16_VALUES).toEqual(
      oracle.scalarValues.level5.map((record) => record.bfloat16Value),
    );
  });

  it("decodes independent static vectors at radix boundaries exactly", () => {
    const codes = oracle.vectors.map((vector) => vector.code);
    const actual = decodeAceFsqCodes(codes);
    expect([...actual]).toEqual(
      oracle.vectors.flatMap((vector) => vector.bfloat16Values),
    );
  });

  it("matches independent full-codebook BF16 and derived FP16 hashes", () => {
    const codes = Array.from(
      { length: ACE_FSQ_CODEBOOK_SIZE },
      (_, code) => code,
    );
    const decoded = decodeAceFsqCodes(codes);
    const bfloat16Bytes = new Uint8Array(decoded.length * 2);
    const fp16Bytes = new Uint8Array(decoded.length * 2);
    const bfloat16View = new DataView(bfloat16Bytes.buffer);
    const fp16View = new DataView(fp16Bytes.buffer);
    for (let index = 0; index < decoded.length; index += 1) {
      const value = decoded[index]!;
      bfloat16View.setUint16(index * 2, numberToBfloat16Bits(value), true);
      fp16View.setUint16(index * 2, numberToFp16Bits(value), true);
    }
    expect(sha256(bfloat16Bytes)).toBe(
      oracle.fullCodebook.bfloat16LittleEndianSha256,
    );
    expect(sha256(fp16Bytes)).toBe(
      oracle.fullCodebook.fp16LittleEndianFromBfloat16Sha256,
    );
  });

  it("rejects empty, fractional, negative, and out-of-range CPU inputs", () => {
    expect(() => decodeAceFsqCodes([])).toThrow(/at least one/);
    expect(() => decodeAceFsqCodes([0.5])).toThrow(/integer/);
    expect(() => decodeAceFsqCodes([-1])).toThrow(/integer/);
    expect(() => decodeAceFsqCodes([64_000])).toThrow(/integer/);
  });

  it("plans a bounded 2D dispatch", () => {
    expect(planAceFsqDecode({ codeCount: 60 })).toEqual({
      codeCount: 60,
      outputElements: 360,
      workgroupsX: 2,
      workgroupsY: 1,
    });
    expect(() => planAceFsqDecode({ codeCount: 0 })).toThrow(/positive/);
  });

  it("emits profile-specific storage and fail-closed validation", () => {
    const reference = aceCorrectnessFsqDecodeWgsl("reference-bf16", {
      codeCount: 2,
    });
    expect(reference).toContain("array<f32>");
    expect(reference).not.toContain("enable f16;");
    expect(reference).toContain("-0.71484375");
    expect(reference).not.toContain("2.0 / f32(level - 1u)");
    expect(reference).toContain("atomicStore(&validation_status[0], 1u)");

    const fp16 = aceCorrectnessFsqDecodeWgsl("raw-fp16", { codeCount: 2 });
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("array<f16>");
    expect(fp16).toContain("f16(-0.71484375)");
  });

  it("compiles, binds, and dispatches through the kernel wrapper", async () => {
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
    };
    const pipeline = {
      getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    } as unknown as GPUComputePipeline;
    const device = fakeDevice(pipeline);
    const kernel = AceCorrectnessFsqDecodeKernel.create(
      device,
      "reference-bf16",
    );
    const codeIds = fakeBinding(8);
    const output = fakeBinding(48);
    const status = fakeBinding(4);
    const dispatch = await kernel.createDispatch(
      "fsq-test",
      { codeCount: 2 },
      { codeIds, output, validationStatus: status },
    );
    dispatch.encode(pass as unknown as GPUComputePassEncoder);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(1, 1, 1);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
  });

  it("validates binding size and output aliasing before compilation", async () => {
    const pipeline = {
      getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    } as unknown as GPUComputePipeline;
    const device = fakeDevice(pipeline);
    const kernel = AceCorrectnessFsqDecodeKernel.create(
      device,
      "reference-bf16",
    );
    await expect(kernel.createDispatch(
      "short",
      { codeCount: 2 },
      {
        codeIds: fakeBinding(4),
        output: fakeBinding(48),
        validationStatus: fakeBinding(4),
      },
    )).rejects.toThrow(/code IDs/);

    const shared = fakeBuffer(64);
    await expect(kernel.createDispatch(
      "alias",
      { codeCount: 2 },
      {
        codeIds: { buffer: shared, offset: 0, size: 8 },
        output: { buffer: shared, offset: 0, size: 48 },
        validationStatus: fakeBinding(4),
      },
    )).rejects.toThrow(/must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

function fakeDevice(pipeline: GPUComputePipeline): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
    createShaderModule: vi.fn(() => ({ label: "module" })),
    createComputePipelineAsync: vi.fn(async () => pipeline),
    createBindGroup: vi.fn(() => ({ label: "bindings" })),
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
  };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function numberToBfloat16Bits(value: number): number {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0]!;
  const upper = bits >>> 16;
  const lower = bits & 0xffff;
  return (
    upper + (lower > 0x8000 || (lower === 0x8000 && (upper & 1) !== 0) ? 1 : 0)
  ) & 0xffff;
}

function numberToFp16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    if ((mantissa & 0x1000) !== 0) mantissa += 0x2000;
    return sign | (mantissa >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  if ((mantissa & 0x1000) !== 0) {
    mantissa += 0x2000;
    if ((mantissa & 0x800000) !== 0) {
      mantissa = 0;
      exponent += 1;
      if (exponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}
