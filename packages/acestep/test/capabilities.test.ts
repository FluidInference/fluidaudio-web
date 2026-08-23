import { describe, expect, it } from "vitest";

import {
  ACE_REQUIRED_WEBGPU_LIMITS,
  ACE_WEBGPU_LIMIT_NAMES,
  AceWebGpuUnavailableError,
  findAceWebGpuLimitDeficits,
  selectAceExecutionProfile,
  snapshotAceWebGpuLimits,
} from "../src/webgpu/capabilities.js";

function limits(): Readonly<Record<string, number>> {
  return Object.fromEntries(
    ACE_WEBGPU_LIMIT_NAMES.map((name) => [name, 1024 * 1024 * 1024]),
  );
}

describe("WebGPU execution profile selection", () => {
  it("requests 256 MiB for the canonical 240 MiB VAE workspace", () => {
    expect(ACE_REQUIRED_WEBGPU_LIMITS).toMatchObject({
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 256 * 1024 * 1024,
    });
    expect(findAceWebGpuLimitDeficits({
      ...limits(),
      maxBufferSize: 128 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
    }, ACE_REQUIRED_WEBGPU_LIMITS)).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "maxBufferSize" }),
      expect.objectContaining({ name: "maxStorageBufferBindingSize" }),
    ]));
  });

  it("keeps packed BF16 as the default math profile regardless of f16", () => {
    expect(
      selectAceExecutionProfile("reference-bf16", ["shader-f16"]).id,
    ).toBe("reference-bf16-portable");
  });

  it("uses subgroup kernels only for audited packed-BF16 fixed-32 math", () => {
    expect(
      selectAceExecutionProfile(
        "reference-bf16",
        ["subgroups"],
        32,
        32,
      ).id,
    ).toBe("reference-bf16-subgroups");
    expect(
      selectAceExecutionProfile(
        "raw-fp16",
        ["shader-f16", "subgroups"],
        32,
        32,
      ).id,
    ).toBe("raw-fp16-portable");
    expect(
      selectAceExecutionProfile(
        "raw-fp16",
        ["shader-f16", "subgroups"],
        16,
        32,
      ).id,
    ).toBe("raw-fp16-portable");
  });

  it("fails closed when raw FP16 is explicitly selected without shader-f16", () => {
    expect(() => selectAceExecutionProfile("raw-fp16", [])).toThrow(
      AceWebGpuUnavailableError,
    );
  });

  it("fails closed for an unknown model profile", () => {
    expect(() =>
      selectAceExecutionProfile("future-profile" as never, ["shader-f16"]),
    ).toThrow(/Unknown ACE model profile/);
  });

  it("treats min-alignment limits as lower-is-better", () => {
    const source = {
      ...limits(),
      minStorageBufferOffsetAlignment: 512,
      minUniformBufferOffsetAlignment: 256,
    };
    const snapshot = snapshotAceWebGpuLimits(source);
    expect(
      findAceWebGpuLimitDeficits(snapshot, {
        minStorageBufferOffsetAlignment: 256,
      }),
    ).toEqual([
      {
        name: "minStorageBufferOffsetAlignment",
        requested: 256,
        available: 512,
        relation: "at-most",
      },
    ]);
  });
});
