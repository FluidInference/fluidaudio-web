import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  OPT_0027_C512_ACCEPTED_OUTPUT_SHA256,
  OPT_0027_C512_FIXTURE_SHA256,
  OPT_0027_QUANTA_PER_COMMAND_BUFFER,
  OPT_0027_TIMED_ORDER,
  OPT_0027_WARMUP_ORDER,
  compareOpt0027RawFp32,
  planOpt0027Scheduling,
  summarizeOpt0027Pair,
  type Opt0027ExecutionTiming,
} from "./browser/opt-0027-vae-batch64.js";

const SOURCE = readFileSync(new URL(
  "./browser/opt-0027-vae-batch64.ts",
  import.meta.url,
), "utf8");
const HTML = readFileSync(new URL(
  "./browser/opt-0027-vae-batch64.html",
  import.meta.url,
), "utf8");

describe("OPT-0027 stock-Chrome C512 batch64 harness", () => {
  it("pins the balanced scheduling topology", () => {
    expect(OPT_0027_QUANTA_PER_COMMAND_BUFFER).toEqual({
      batch8: 8,
      batch64: 64,
    });
    expect(OPT_0027_WARMUP_ORDER).toEqual(["batch8", "batch64"]);
    expect(OPT_0027_TIMED_ORDER).toEqual([
      "batch8",
      "batch64",
      "batch64",
      "batch8",
    ]);
    expect(planOpt0027Scheduling(8)).toEqual({
      quantaPerCommandBuffer: 8,
      decoderQuantumCount: 7_855,
      decoderCommandBufferCount: 982,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 983,
      queueDrainCount: 983,
      requestedCooperativeIdleMs: 982,
    });
    expect(planOpt0027Scheduling(64)).toEqual({
      quantaPerCommandBuffer: 64,
      decoderQuantumCount: 7_855,
      decoderCommandBufferCount: 123,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 124,
      queueDrainCount: 124,
      requestedCooperativeIdleMs: 123,
    });
    expect(() => planOpt0027Scheduling(32)).toThrow(/batch8 or batch64/);
  });

  it("compares FP32 output by raw U32 identity", () => {
    const control = new Float32Array([0, 1, -2, Number.NaN]);
    const exact = new Float32Array(control.buffer.slice(0));
    expect(compareOpt0027RawFp32(control, exact)).toEqual({
      comparedU32WordCount: 4,
      mismatchCount: 0,
      firstMismatchIndex: null,
      firstControlWord: null,
      firstCandidateWord: null,
      rawFp32U32Exact: true,
    });

    const signedZero = new Float32Array(control.buffer.slice(0));
    new Uint32Array(signedZero.buffer)[0] = 0x8000_0000;
    expect(compareOpt0027RawFp32(control, signedZero)).toMatchObject({
      comparedU32WordCount: 4,
      mismatchCount: 1,
      firstMismatchIndex: 0,
      firstControlWord: 0,
      firstCandidateWord: 0x8000_0000,
      rawFp32U32Exact: false,
    });
  });

  it("reports paired wall, drain, command-buffer, and idle deltas", () => {
    const batch8 = timing(12_000, 10_500, 982, 983, 983, 982);
    const batch64 = timing(10_000, 9_700, 123, 124, 124, 123);
    expect(summarizeOpt0027Pair(
      batch8,
      batch64,
      "batch8-batch64",
    )).toMatchObject({
      order: "batch8-batch64",
      outerWindowWallSpeedup: 1.2,
      outerWindowWallSavingMs: 2_000,
      decoderSubmitThroughDrainSavingMs: 800,
      decoderCommandBufferReduction: 859,
      requestedCooperativeIdleReductionMs: 859,
    });
  });

  it("statically excludes alternate K7 math and gates timing behind READY", () => {
    expect(OPT_0027_C512_FIXTURE_SHA256).toMatch(/^[0-9a-f]{64}$/);
    expect(OPT_0027_C512_ACCEPTED_OUTPUT_SHA256).toBe(
      "893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47",
    );
    expect(SOURCE).toContain(
      "ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY",
    );
    expect(SOURCE).toContain(
      '"opt-0028-mixed-fp16-fixed32-exact-packed-v1"',
    );
    expect(SOURCE).toContain("AceOpt0011Fp16VaeChunkGpuBackend.create({");
    expect(SOURCE).toContain("ownedBatch8Phase");
    expect(SOURCE).toContain("ownedBatch64Phase");
    expect(SOURCE).toContain("compareOpt0027RawFp32(");
    expect(SOURCE).toContain("OPT_0027_C512_ACCEPTED_OUTPUT_SHA256");
    expect(SOURCE).toContain("onFamilyProfile: observer.onFamilyProfile");
    expect(SOURCE).toContain("window.__ACE_OPT0027_RESULT__ = receipt");
    expect(SOURCE).toContain("await retained.destroy()");
    expect(SOURCE).not.toContain("vae-conv1d-fp16-direct-dot4-subgroup");
    expect(SOURCE).not.toContain("replaceK7Kernel");
    expect(SOURCE).not.toContain("ACE_OPT_0024");
    expect(HTML).toContain('id="run" type="button" disabled');
    expect(HTML).not.toContain('id="prepare"');
    expect(HTML).toContain("confirm nominal thermal state");
    expect(HTML).toContain("opt-0027-vae-batch64.ts");
  });
});

function timing(
  outerWindowWallMs: number,
  decoderSubmitThroughDrainMs: number,
  decoderCommandBufferCount: number,
  totalCommandBufferCount: number,
  queueDrainCount: number,
  requestedCooperativeIdleMs: number,
): Opt0027ExecutionTiming {
  return Object.freeze({
    outerWindowWallMs,
    decoderSubmitThroughDrainMs,
    decoderCommandBufferCount,
    totalCommandBufferCount,
    queueDrainCount,
    requestedCooperativeIdleMs,
  });
}
