import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
  planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory,
} from "../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";
import {
  planAceOpt0025VaeK1SubgroupGemm,
  planAceOpt0025VaeK1SubgroupGemmRange,
  planAceOpt0025VaeK1SubgroupGemmRangeDispatch,
} from "../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  planAceFp16VaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  planAceFp16VaeConv1dSubgroupRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  planAceOpt0026VaeConvTranspose1d,
  planAceOpt0026VaeConvTranspose1dRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  planAceFp16VaeSnake,
  planAceFp16VaeSnakeRange,
} from "../src/webgpu/kernels/vae-snake-fp16.js";
import {
  planAceFp16VaeAdd,
  planAceFp16VaeIngress,
  planAceFp16VaePointwiseRange,
} from "../src/webgpu/kernels/vae-pointwise-fp16.js";
import {
  OPT_0035_CANDIDATE_CHUNK_FRAMES,
  OPT_0035_CONTROL_CHUNK_FRAMES,
  OPT_0035_LATENT_BYTES,
  OPT_0035_LATENT_ELEMENTS,
  OPT_0035_LATENT_FRAMES,
  OPT_0035_LATENT_SHA256,
  OPT_0035_OUTPUT_BYTES,
  OPT_0035_OUTPUT_ELEMENTS,
  OPT_0035_SCHEMA,
  OPT_0035_SPEEDUP_GATE,
  OPT_0035_TIMED_ORDER,
  compareOpt0035Waveforms,
  planOpt0035Coverage,
  resolveOpt0035QuantaPerCommandBuffer,
} from "./browser/opt-0035-vae-c2378-two-window-worker.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0035-vae-c2378-two-window-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0035-vae-c2378-two-window.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0035-vae-c2378-two-window.html",
  import.meta.url,
), "utf8");

describe("OPT-0035 C2378 two-window browser gate", () => {
  it("pins the exact C4500 control and two-window candidate coverage", () => {
    const coverage = planOpt0035Coverage();

    expect(OPT_0035_SCHEMA).toBe(
      "ace-opt-0035-vae-c2378-two-window-abba-v1",
    );
    expect(OPT_0035_LATENT_FRAMES).toBe(4_500);
    expect(OPT_0035_LATENT_ELEMENTS).toBe(288_000);
    expect(OPT_0035_LATENT_BYTES).toBe(1_152_000);
    expect(OPT_0035_OUTPUT_ELEMENTS).toBe(17_280_000);
    expect(OPT_0035_OUTPUT_BYTES).toBe(69_120_000);
    expect(OPT_0035_LATENT_SHA256).toMatch(/^[0-9a-f]{64}$/);
    expect(OPT_0035_CONTROL_CHUNK_FRAMES).toBe(512);
    expect(OPT_0035_CANDIDATE_CHUNK_FRAMES).toBe(2_378);
    expect(coverage.control.windows).toHaveLength(12);
    expect(coverage.controlDecodedLatentFrames).toBe(5_908);
    expect(coverage.controlSeams).toEqual([
      384, 768, 1_152, 1_536, 1_920, 2_304, 2_688, 3_072, 3_456,
      3_840, 4_224,
    ]);
    expect(coverage.candidate.windows.map((window) => ({
      latentWindow: [
        window.windowStartLatentFrame,
        window.windowEndLatentFrame,
      ],
      core: [
        window.coreStartLatentFrame,
        window.coreEndLatentFrame,
      ],
      prefix: window.discardPrefixLatentFrames,
      suffix: window.discardSuffixLatentFrames,
    }))).toEqual([
      {
        latentWindow: [0, 2_314],
        core: [0, 2_250],
        prefix: 0,
        suffix: 64,
      },
      {
        latentWindow: [2_186, 4_500],
        core: [2_250, 4_500],
        prefix: 64,
        suffix: 0,
      },
    ]);
    expect(coverage.candidateDecodedLatentFrames).toBe(4_628);
    expect(coverage.decodedLatentFrameReduction).toBe(1_280);
    expect(coverage.decodedLatentFrameReductionRatio).toBeCloseTo(
      1_280 / 5_908,
      15,
    );
    expect(coverage.candidateSeams).toEqual([2_250]);
  });

  it("keeps C2378 behind a benchmark-only memory planner", () => {
    const coverage = planOpt0035Coverage();
    const candidate = planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory(
      coverage.candidate,
      256,
      64,
    );

    expect(ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES).toBe(2_378);
    expect(candidate).toMatchObject({
      workspaceBufferBytes: ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
      workspaceBufferCount: 3,
      maximumWindowFrames: 2_378,
      quantaPerCommandBuffer: 64,
    });
    expect(candidate.workspaceBufferBytes).toBe(1_168_834_560);
    expect(candidate.accountedGpuBytes).toBeLessThan(4_000_000_000);
    expect(candidate.accountedGpuBytes).toBeGreaterThan(3_700_000_000);
    expect(() => planAceOpt0011Fp16VaeChunkGpuBackendMemory(
      coverage.candidate,
      256,
      8,
    )).toThrow(/exact C-512\/64/);
  });

  it("pins balanced serial timing and validates the command-buffer batch", () => {
    expect(OPT_0035_TIMED_ORDER).toEqual([
      "c512",
      "c2378",
      "c2378",
      "c512",
    ]);
    expect(OPT_0035_SPEEDUP_GATE).toBe(1.15);
    expect(resolveOpt0035QuantaPerCommandBuffer(8)).toBe(8);
    expect(resolveOpt0035QuantaPerCommandBuffer(64)).toBe(64);
    expect(() => resolveOpt0035QuantaPerCommandBuffer(32)).toThrow(
      /must be 8 or 64/,
    );
  });

  it("plans every selected exact owner at the maximum C2378 geometry", () => {
    const decoder = planAceVaeDecoder(2_378);
    const cooperative = planAceVaeDecoderQuanta(decoder);
    const maximumDispatch = 65_535;
    const kindCounts = {
      k1: 0,
      k7: 0,
      transpose: 0,
      snake: 0,
      add: 0,
    };

    const ingress = planAceFp16VaeIngress({
      batch: decoder.batch,
      frames: decoder.inputFrames,
      channels: decoder.config.decoderInputChannels,
    });
    const ingressRange = planAceFp16VaePointwiseRange(ingress, {
      base: 0,
      count: decoder.inputElements,
    });
    expect(ingressRange.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
    expect(ingressRange.workgroupsY).toBeLessThanOrEqual(maximumDispatch);

    for (const quantum of cooperative.quanta) {
      const operation = decoder.operations[quantum.operationIndex]!;
      const range = quantum.primitives[0]!;
      const outputRange = Object.freeze({
        base: range.outputBase,
        count: range.outputCount,
      });
      switch (operation.kind) {
        case "conv1d": {
          if (operation.shape.kernelSize === 1) {
            const plan = planAceOpt0025VaeK1SubgroupGemm(operation.shape);
            const plannedRange = planAceOpt0025VaeK1SubgroupGemmRange(
              plan,
              outputRange,
            );
            const dispatch = planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
              plan,
              plannedRange,
            );
            expect(dispatch.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
            expect(dispatch.workgroupsY).toBeLessThanOrEqual(maximumDispatch);
            kindCounts.k1 += 1;
          } else {
            const plan = planAceFp16VaeConv1d(
              operation.shape,
              "float16",
            );
            expect(
              range.outputBase % plan.outputChannels,
              `${operation.label} quantum ${quantum.id} base`,
            ).toBe(0);
            expect(
              range.outputCount % plan.outputChannels,
              `${operation.label} quantum ${quantum.id} count`,
            ).toBe(0);
            expect(
              range.outputBase + range.outputCount <= plan.outputElements,
              `${operation.label} quantum ${quantum.id} end`,
            ).toBe(true);
            const firstRow = range.outputBase / plan.outputChannels;
            expect(
              firstRow % plan.outputFrames +
                  range.outputCount / plan.outputChannels <=
                plan.outputFrames,
              `${operation.label} quantum ${quantum.id} batch boundary`,
            ).toBe(true);
            const dispatch = planAceFp16VaeConv1dSubgroupRange(
              plan,
              outputRange,
            );
            expect(dispatch.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
            expect(dispatch.workgroupsY).toBeLessThanOrEqual(maximumDispatch);
            kindCounts.k7 += 1;
          }
          break;
        }
        case "conv-transpose1d": {
          const plan = planAceOpt0026VaeConvTranspose1d(operation.shape);
          const dispatch = planAceOpt0026VaeConvTranspose1dRange(
            plan,
            outputRange,
          );
          expect(dispatch.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
          expect(dispatch.workgroupsY).toBeLessThanOrEqual(maximumDispatch);
          expect(dispatch.workgroupsZ).toBeLessThanOrEqual(maximumDispatch);
          kindCounts.transpose += 1;
          break;
        }
        case "snake": {
          const plan = planAceFp16VaeSnake(operation.shape);
          const dispatch = planAceFp16VaeSnakeRange(plan, outputRange);
          expect(dispatch.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
          expect(dispatch.workgroupsY).toBeLessThanOrEqual(maximumDispatch);
          kindCounts.snake += 1;
          break;
        }
        case "add": {
          const plan = planAceFp16VaeAdd(operation.shape);
          const dispatch = planAceFp16VaePointwiseRange(plan, outputRange);
          expect(dispatch.workgroupsX).toBeLessThanOrEqual(maximumDispatch);
          expect(dispatch.workgroupsY).toBeLessThanOrEqual(maximumDispatch);
          kindCounts.add += 1;
          break;
        }
      }
    }

    expect(Object.values(kindCounts).every((count) => count > 0)).toBe(true);
    expect(kindCounts.k1).toBeGreaterThan(15);
    expect(cooperative.quanta).toHaveLength(
      kindCounts.k1 + kindCounts.k7 + kindCounts.transpose +
        kindCounts.snake + kindCounts.add,
    );
  });

  it("computes bounded stereo waveform gates without hiding non-finite data", () => {
    const control = new Float32Array([
      0.25, -0.5, 0.75, -1, 0.5, -0.25, -0.75, 1,
    ]);
    const exact = compareOpt0035Waveforms(
      control,
      new Float32Array(control),
    );
    expect(exact.joint).toMatchObject({
      count: 8,
      nrmse: 0,
      snrDb: Number.POSITIVE_INFINITY,
      pearson: 1,
      maximumAbsoluteError: 0,
      finite: true,
      passed: true,
    });
    expect(exact.left.count).toBe(4);
    expect(exact.right.count).toBe(4);

    const divergent = compareOpt0035Waveforms(
      control,
      Float32Array.from(control, (value) => -value),
    );
    expect(divergent.joint.passed).toBe(false);
    expect(divergent.joint.pearson).toBeCloseTo(-1, 12);

    const nonFinite = new Float32Array(control);
    nonFinite[3] = Number.NaN;
    expect(compareOpt0035Waveforms(control, nonFinite).joint).toMatchObject({
      count: 7,
      finite: false,
      passed: false,
    });
    expect(() => compareOpt0035Waveforms(
      new Float32Array(2),
      new Float32Array(4),
    )).toThrow(/equal stereo arrays/);
  });

  it("statically enforces worker ownership, cleanup, and untouched production", () => {
    expect(WORKER_SOURCE).toContain("class ActiveArmGuard");
    expect(WORKER_SOURCE).toContain("class Opt0035DeviceResourceAudit");
    expect(WORKER_SOURCE).toContain(
      'throw new Error("OPT-0035 attempted simultaneous VAE arm ownership")',
    );
    expect(WORKER_SOURCE).toContain("await backend.destroy();");
    expect(WORKER_SOURCE).toContain(
      "await input.resourceAudit.device.queue.onSubmittedWorkDone();",
    );
    expect(WORKER_SOURCE).toContain("release();");
    expect(WORKER_SOURCE).toContain("input.resourceAudit.finishArm()");
    expect(WORKER_SOURCE).toContain("maximumLiveBufferBytes");
    expect(WORKER_SOURCE).toContain("everyBufferDestroyedExactlyOnce");
    expect(WORKER_SOURCE).toContain("await execution.raw.remove();");
    expect(WORKER_SOURCE).toContain("productionDefaultChanged: false");
    expect(WORKER_SOURCE).toContain(
      'runtimeProfileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1"',
    );
    expect(WORKER_SOURCE).not.toContain("ACE_OPT_0024");
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain('type: "prepare"');
    expect(PAGE_SOURCE).toContain('type: "run"');
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0035_RESULT__ = receipt");
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(HTML_SOURCE).toContain("No two weight phases");
    expect(HTML_SOURCE).toContain("production C512 default is untouched");
  });
});
