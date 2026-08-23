import { describe, expect, it, vi } from "vitest";

import {
  composeAceSemanticSourceCpu,
  planAceSemanticSource,
} from "../src/webgpu/kernels/condition-layout.js";
import {
  AceConditioningGpuExecutor,
  AceConditioningGpuDeviceLostError,
  aceConditioningNeedsIdleAfterQuantum,
  decodeAceConditioningActivation,
  planAceConditioningGpuMemory,
  validateAceConditioningPhaseManifest,
  validateAceConditioningPhaseSet,
  snapshotAceConditioningGpuRequest,
} from "../src/webgpu/conditioning-executor.js";

describe("ACE production conditioning owner contracts", () => {
  it("pads semantic hints with silence from frame zero and records accounting", () => {
    const plan = planAceSemanticSource({
      batch: 1,
      semanticFrames: 2,
      outputFrames: 4,
    });
    expect(plan).toMatchObject({
      copiedFrames: 2,
      paddedFrames: 2,
      truncatedFrames: 0,
      outputElements: 256,
    });
    const hints = new Float32Array(2 * 64);
    for (let index = 0; index < hints.length; index += 1) hints[index] = index + 1;
    const silence = new Float32Array(64 * 15_000);
    for (let channel = 0; channel < 64; channel += 1) {
      silence[channel * 15_000] = 1_000 + channel;
      silence[channel * 15_000 + 1] = 2_000 + channel;
    }
    const output = composeAceSemanticSourceCpu(hints, silence, 1, 2, 4);
    expect([...output.subarray(0, 128)]).toEqual([...hints]);
    expect([...output.subarray(128, 132)]).toEqual([1000, 1001, 1002, 1003]);
    expect([...output.subarray(192, 196)]).toEqual([2000, 2001, 2002, 2003]);
  });

  it("crops semantic hints on the right without touching preceding frames", () => {
    expect(planAceSemanticSource({
      batch: 2,
      semanticFrames: 5,
      outputFrames: 3,
    })).toMatchObject({
      copiedFrames: 3,
      paddedFrames: 0,
      truncatedFrames: 2,
    });
  });

  it("destroys text ownership when factory validation fails", () => {
    let destroyed = 0;
    const textPhase = {
      phases: ["semantic"],
      residentBytes: 1,
      destroy: () => {
        destroyed += 1;
      },
    };
    expect(() => AceConditioningGpuExecutor.create({
      device: fakeDevice(),
      manifest: { profile: "fp16", tensors: {} } as never,
      modelProfile: "raw-fp16",
      ownedTextWeights: textPhase as never,
      loadConditionerWeights: async () => textPhase as never,
    })).toThrow();
    expect(destroyed).toBe(1);
  });

  it("names device-loss failures for worker diagnostics", () => {
    const error = new AceConditioningGpuDeviceLostError({
      reason: "unknown",
      message: "removed",
    });
    expect(error.name).toBe("AceConditioningGpuDeviceLostError");
    expect(error.message).toContain("removed");
  });

  it("cancellation transactionally destroys transferred text ownership", async () => {
    const controller = new AbortController();
    const owned = fakeOwnedTextPhase();
    const executor = AceConditioningGpuExecutor.fromPreparedOptionsForTest({
      device: fakeDevice(),
      manifest: {} as never,
      modelProfile: "raw-fp16",
      ownedTextWeights: owned.phase,
      loadConditionerWeights: async () => owned.phase,
      signal: controller.signal,
    });
    controller.abort(new DOMException("cancelled", "AbortError"));
    await executor.destroy();
    expect(owned.destroy).toHaveBeenCalledTimes(1);
  });

  it("device loss transactionally destroys transferred text ownership", async () => {
    const lost = Promise.withResolvers<GPUDeviceLostInfo>();
    const owned = fakeOwnedTextPhase();
    const executor = AceConditioningGpuExecutor.fromPreparedOptionsForTest({
      device: fakeDevice(lost.promise),
      manifest: {} as never,
      modelProfile: "raw-fp16",
      ownedTextWeights: owned.phase,
      loadConditionerWeights: async () => owned.phase,
    });
    lost.resolve({ reason: "unknown", message: "removed" } as GPUDeviceLostInfo);
    await vi.waitFor(() => expect(owned.destroy).toHaveBeenCalledTimes(1));
    await executor.destroy();
    expect(owned.destroy).toHaveBeenCalledTimes(1);
  });

  it("accounts every retained, working, readback, and CPU result allocation", () => {
    const direct = planAceConditioningGpuMemory({
      modelProfile: "raw-fp16",
      textTokens: 32,
      lyricTokens: 64,
      latentFrames: 300,
      semanticCodeTokens: 0,
      textWeightBytes: 10,
      semanticWeightBytes: 0,
      conditionerWeightBytes: 20,
    });
    const planner = planAceConditioningGpuMemory({
      modelProfile: "raw-fp16",
      textTokens: 32,
      lyricTokens: 64,
      latentFrames: 300,
      semanticCodeTokens: 60,
      textWeightBytes: 10,
      semanticWeightBytes: 30,
      conditionerWeightBytes: 20,
    });
    expect(direct.textRetainedBytes).toBeGreaterThan(0);
    expect(direct.conditionerWorkingBytes).toBeGreaterThan(0);
    expect(direct.resultReadbackBytes).toBeGreaterThan(0);
    expect(direct.returnedCpuBytes).toBeGreaterThan(direct.resultBytes);
    expect(planner.semanticRetainedBytes).toBeGreaterThan(0);
    expect(planner.semanticWorkingBytes).toBeGreaterThan(0);
    expect(planner.peakAccountedGpuBytes).toBeGreaterThan(direct.peakAccountedGpuBytes);
  });

  it("snapshots all mutable request arrays before awaiting GPU ownership", () => {
    const text = new Uint32Array([1, 2]);
    const lyric = new Uint32Array([3]);
    const textMask = new Uint32Array([1, 1]);
    const lyricMask = new Uint32Array([1]);
    const codes = new Uint32Array([4, 5]);
    const snapshot = snapshotAceConditioningGpuRequest({
      textTokenIds: text,
      lyricTokenIds: lyric,
      textMask,
      lyricMask,
      latentFrames: 10,
      mode: { kind: "planner", semanticCodeIds: codes },
    });
    text.fill(9);
    lyric.fill(9);
    textMask.fill(0);
    lyricMask.fill(0);
    codes.fill(9);
    expect([...snapshot.textTokenIds]).toEqual([1, 2]);
    expect([...snapshot.lyricTokenIds]).toEqual([3]);
    expect([...snapshot.textMask]).toEqual([1, 1]);
    expect([...snapshot.lyricMask]).toEqual([1]);
    expect(snapshot.mode.kind === "planner" && [...snapshot.mode.semanticCodeIds])
      .toEqual([4, 5]);
  });

  it("keeps a real idle gap across phase boundaries and before readback", () => {
    expect(aceConditioningNeedsIdleAfterQuantum(false, 2, 3)).toBe(true);
    expect(aceConditioningNeedsIdleAfterQuantum(true, 1, 3)).toBe(true);
    expect(aceConditioningNeedsIdleAfterQuantum(true, 2, 3)).toBe(false);
  });

  it("rejects resident phases from another authenticated manifest identity", () => {
    const expected = {} as never;
    expect(() => validateAceConditioningPhaseManifest(
      { packageManifest: {} as never },
      expected,
      "semantic",
    )).toThrow(/manifest identity/);
    expect(() => validateAceConditioningPhaseManifest(
      { packageManifest: expected },
      expected,
      "semantic",
    )).not.toThrow();
    expect(() => validateAceConditioningPhaseSet(
      { phases: ["text"] },
      ["semantic"],
      "semantic",
    )).toThrow(/requires phases semantic/);
  });

  it("decodes raw-FP16 normal and subnormal readback values without drift", () => {
    const bits = new Uint16Array([0x0001, 0x3c00, 0xc000, 0x7c00]);
    const output = decodeAceConditioningActivation(
      "raw-fp16",
      bits.buffer,
      0,
      bits.length,
    );
    expect(output[0]).toBe(2 ** -24);
    expect(output[1]).toBe(1);
    expect(output[2]).toBe(-2);
    expect(output[3]).toBe(Number.POSITIVE_INFINITY);
  });
});

function fakeDevice(
  lost: Promise<GPUDeviceLostInfo> = new Promise(() => undefined),
): GPUDevice {
  return {
    lost,
  } as GPUDevice;
}

function fakeOwnedTextPhase(): Readonly<{
  phase: import("../src/model/gpu-tensors.js").AceGpuTensorPhase;
  destroy: ReturnType<typeof vi.fn>;
}> {
  const destroy = vi.fn();
  return Object.freeze({
    destroy,
    phase: {
      phases: ["text"],
      residentBytes: 1,
      packageManifest: {},
      destroy,
    } as never,
  });
}
