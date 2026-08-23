import { describe, expect, it } from "vitest";

import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  planAceCappedFp16VaeC2176ChunkGpuBackendMemory,
} from "../src/webgpu/vae-fp16-backend.js";
import {
  ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES,
  ACE_CAPPED_VAE_FP16_C2176_WORKSPACE_BYTES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
  planAceOpt0011Fp16VaeChunkDispatches,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
  ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
  ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_FP16_WORKSPACE_BYTES_PER_LATENT_FRAME,
  requireAceVaeWindowRuntimeProfile,
  selectAceVaeWindowRuntimeProfileForLimits,
} from "../src/webgpu/vae-window-profile.js";

/** Every iOS WebGPU adapter reports exactly 2^30 for both buffer limits. */
const IOS_ADAPTER_LIMIT_BYTES = 1_073_741_824;
/** 240-second product maximum at the 25 Hz latent rate. */
const FULL_DURATION_LATENT_FRAMES = 6_000;

function bufferLimits(
  maxBufferSize: number,
  maxStorageBufferBindingSize = maxBufferSize,
) {
  return { maxBufferSize, maxStorageBufferBindingSize } as const;
}

describe("capped C2176 VAE window geometry for one-GiB adapters", () => {
  it("authenticates the capped contract and rejects a size alone", () => {
    expect(requireAceVaeWindowRuntimeProfile(
      ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
      2_176,
    )).toEqual({
      id: ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
      maximumWindowFrames: ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
      overlapFrames: 64,
      requiredWorkspaceBytes: ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
    });
    expect(() => requireAceVaeWindowRuntimeProfile(undefined, 2_176))
      .toThrow(/not authenticated/);
    expect(() => requireAceVaeWindowRuntimeProfile(
      ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
      2_378,
    )).toThrow(/not authenticated/);
  });

  it("derives the workspace requirement from the decoder graph exactly", () => {
    // The widest FP16 activation is 128 channels x 1,920 audio frames per
    // latent frame; one workspace buffer is linear in the window length.
    for (const frames of [256, 512, 2_176, 2_378]) {
      expect(planAceVaeDecoder(frames).maximumActivationElements * 2).toBe(
        frames * ACE_VAE_FP16_WORKSPACE_BYTES_PER_LATENT_FRAME,
      );
    }
    expect(ACE_CAPPED_VAE_FP16_C2176_WORKSPACE_BYTES).toBe(
      ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
    );
    expect(ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES).toBe(
      ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
    );
    // The capped geometry fits the iOS limit; the production C2378 does not.
    expect(ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES)
      .toBeLessThanOrEqual(IOS_ADAPTER_LIMIT_BYTES);
    expect(ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES)
      .toBeGreaterThan(IOS_ADAPTER_LIMIT_BYTES);
  });

  it("selects the capped geometry only when C2378 cannot bind", () => {
    const c2378 = requireAceVaeWindowRuntimeProfile(
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      2_378,
    );
    const c512 = requireAceVaeWindowRuntimeProfile(undefined, 512);
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      bufferLimits(2_000_000_000),
    )).toBe(c2378);
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      bufferLimits(IOS_ADAPTER_LIMIT_BYTES),
    ).id).toBe(ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE);
    // Either limit alone below the C2378 workspace forces the downshift.
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      bufferLimits(2_000_000_000, IOS_ADAPTER_LIMIT_BYTES),
    ).id).toBe(ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE);
    // Below even the capped workspace the configured contract fails closed.
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      bufferLimits(800_000_000),
    )).toBe(c2378);
    // A C512 configuration never changes geometry.
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c512,
      bufferLimits(IOS_ADAPTER_LIMIT_BYTES),
    )).toBe(c512);
    expect(selectAceVaeWindowRuntimeProfileForLimits(
      c512,
      bufferLimits(128 * 1024 * 1024),
    )).toBe(c512);
    expect(() => selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      bufferLimits(Number.NaN),
    )).toThrow(RangeError);
    expect(ACE_VAE_C512_WINDOW_RUNTIME_PROFILE).toBe(
      "ace-vae-c512-overlap64-v1",
    );
  });

  it("decodes the full 240-second product duration within the iOS limit", () => {
    const plan = planAceVaeChunkedDecode(FULL_DURATION_LATENT_FRAMES, {
      chunkFrames: ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
      overlapFrames: 64,
    });
    // The chunk planner already fails unless the cores cover the output
    // exactly once; pin the seam geometry the downshift produces.
    expect(plan.windows).toHaveLength(3);
    expect(plan.maximumWindowFrames).toBe(2_176);
    expect(plan.outputAudioFrames).toBe(FULL_DURATION_LATENT_FRAMES * 1_920);
    expect(plan.windows.map((window) => [
      window.windowStartLatentFrame,
      window.windowEndLatentFrame,
      window.coreStartLatentFrame,
      window.coreEndLatentFrame,
    ])).toEqual([
      [0, 2_112, 0, 2_048],
      [1_984, 4_160, 2_048, 4_096],
      [4_032, 6_000, 4_096, 6_000],
    ]);
    for (const window of plan.windows) {
      expect(window.latentWindowFrames).toBeLessThanOrEqual(2_176);
    }

    const memory = planAceCappedFp16VaeC2176ChunkGpuBackendMemory(plan);
    expect(memory.maximumWindowFrames).toBe(2_176);
    expect(memory.workspaceBufferBytes).toBe(
      ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
    );
    expect(memory.workspaceBufferCount).toBe(3);
    // No single allocation may exceed a one-GiB adapter profile.
    for (const [label, bytes] of Object.entries({
      residentWeightBytes: memory.residentWeightBytes,
      stagingInputBufferBytes: memory.stagingInputBufferBytes,
      decoderInputBufferBytes: memory.decoderInputBufferBytes,
      workspaceBufferBytes: memory.workspaceBufferBytes,
      outputBufferBytes: memory.outputBufferBytes,
      readbackBufferBytes: memory.readbackBufferBytes,
      controlBufferBytes: memory.controlBufferBytes,
    })) {
      expect(bytes, label).toBeLessThanOrEqual(IOS_ADAPTER_LIMIT_BYTES);
    }

    const dispatches = planAceOpt0011Fp16VaeChunkDispatches(
      FULL_DURATION_LATENT_FRAMES,
      ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES,
      256,
    );
    expect(dispatches.maximumFp16WorkspaceBytes).toBe(
      ACE_CAPPED_VAE_FP16_C2176_WORKSPACE_BYTES,
    );
    expect(dispatches.uniqueWindowFrames).toEqual([1_968, 2_112, 2_176]);
  });
});
