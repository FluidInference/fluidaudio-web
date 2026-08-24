import { describe, expect, it } from "vitest";

import { ACE_REQUIRED_WEBGPU_LIMITS } from "../src/webgpu/capabilities.js";
import { planAceDitBackendArena } from "../src/webgpu/dit-backend.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
} from "../src/webgpu/dit-sampler-profile.js";
import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";
import {
  planAceCappedFp16VaeC2176ChunkGpuBackendMemory,
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
} from "../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  planAceOpt0011Fp16VaeChunkDispatches,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES,
  requireAceVaeWindowRuntimeProfile,
  selectAceVaeWindowRuntimeProfileForLimits,
} from "../src/webgpu/vae-window-profile.js";

/** 240-second product maximum at the 25 Hz latent rate. */
const FULL_DURATION_LATENT_FRAMES = 6_000;

function iosLimits() {
  return {
    maxBufferSize: ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES,
    maxStorageBufferBindingSize: ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES,
  } as const;
}

describe("C512 downshift keeps the VAE phase inside an iPhone tab budget", () => {
  it("resolves the one-GiB adapter class to the C512 baseline contract", () => {
    const c2378 = requireAceVaeWindowRuntimeProfile(
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      2_378,
    );
    const effective = selectAceVaeWindowRuntimeProfileForLimits(
      c2378,
      iosLimits(),
    );
    expect(effective.id).toBe(ACE_VAE_C512_WINDOW_RUNTIME_PROFILE);
    expect(effective.maximumWindowFrames).toBe(512);
    expect(effective.overlapFrames).toBe(64);
    expect(effective.requiredWorkspaceBytes).toBe(
      ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    );
    expect(ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES).toBe(2 ** 30);
  });

  it("decodes the full 240-second product duration with ~755 MB workspaces", () => {
    const plan = planAceVaeChunkedDecode(FULL_DURATION_LATENT_FRAMES, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    expect(plan.maximumWindowFrames).toBe(512);
    expect(plan.outputAudioFrames).toBe(FULL_DURATION_LATENT_FRAMES * 1_920);
    // Overlap-64 discard cores stride by 384; 6,000 frames need 16 windows.
    expect(plan.windows).toHaveLength(16);
    for (const window of plan.windows) {
      expect(window.latentWindowFrames).toBeLessThanOrEqual(512);
    }

    const memory = planAceOpt0011Fp16VaeChunkGpuBackendMemory(plan);
    expect(memory.maximumWindowFrames).toBe(512);
    expect(memory.workspaceBufferBytes).toBe(
      ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    );
    expect(memory.workspaceBufferCount).toBe(3);
    expect(memory.workspaceBufferBytes * memory.workspaceBufferCount).toBe(
      754_974_720,
    );
    // The complete VAE-phase GPU accounting stays under one GB, where the
    // capped C2176 geometry held ~3.45 GB and iOS jetsam killed the tab.
    expect(memory.accountedGpuBytes).toBeLessThan(1_000_000_000);
    const c2176 = planAceCappedFp16VaeC2176ChunkGpuBackendMemory(
      planAceVaeChunkedDecode(FULL_DURATION_LATENT_FRAMES, {
        chunkFrames: 2_176,
        overlapFrames: 64,
      }),
    );
    expect(c2176.accountedGpuBytes - memory.accountedGpuBytes)
      .toBeGreaterThan(2_400_000_000);
    // No single allocation may exceed the one-GiB adapter limits.
    for (const [label, bytes] of Object.entries({
      residentWeightBytes: memory.residentWeightBytes,
      stagingInputBufferBytes: memory.stagingInputBufferBytes,
      decoderInputBufferBytes: memory.decoderInputBufferBytes,
      workspaceBufferBytes: memory.workspaceBufferBytes,
      outputBufferBytes: memory.outputBufferBytes,
      readbackBufferBytes: memory.readbackBufferBytes,
      controlBufferBytes: memory.controlBufferBytes,
    })) {
      expect(bytes, label).toBeLessThanOrEqual(
        ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES,
      );
    }

    const dispatches = planAceOpt0011Fp16VaeChunkDispatches(
      FULL_DURATION_LATENT_FRAMES,
      512,
      256,
    );
    expect(dispatches.maximumFp16WorkspaceBytes).toBe(
      ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
    );
  });

  it("keeps every DiT arena slot inside the C512-derived device limits", () => {
    // The C512 downshift shrinks the derived device request back to the base
    // Stage-1 contract (256 MiB). The DiT phase ran under exactly these
    // limits before OPT-0070 raised them for the C2378 workspace; pin that
    // the current production arena still fits at the 240-second maximum.
    const baseLimitBytes = ACE_REQUIRED_WEBGPU_LIMITS.maxBufferSize!;
    expect(baseLimitBytes).toBe(256 * 1024 * 1024);
    const arena = planAceDitBackendArena(
      "reference-bf16",
      {
        batch: 1,
        latentFrames: FULL_DURATION_LATENT_FRAMES,
        conditionTokens: 256,
      },
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      undefined,
    );
    for (const slot of arena.slots) {
      expect(slot.byteLength).toBeLessThanOrEqual(baseLimitBytes);
    }
  });
});
