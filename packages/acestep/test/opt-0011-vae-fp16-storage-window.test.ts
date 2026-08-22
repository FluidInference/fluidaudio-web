import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  createAceOpt0011LatentFixture,
  planAceOpt0011Arm,
  planAceOpt0011ChunkGeometry,
  planAceOpt0011TemporalSupport,
} from "../benchmark/opt-0011-vae-fp16-storage-window.js";

describe("OPT-0011 FP16 VAE storage and window contract", () => {
  it("pins the registered three-arm resource accounting", () => {
    expect(planAceOpt0011Arm("fp32-256")).toMatchObject({
      windowFrames: 256,
      storageElementBytes: 4,
      parameterElements: 84_395_776,
      parameterBytes: 337_583_104,
      maximumActivationElements: 62_914_560,
      workspaceBytes: 251_658_240,
      allWorkspaceBytes: 754_974_720,
      inputBytes: 65_536,
      outputFloat32Bytes: 3_932_160,
      readbackFloat32Bytes: 3_932_160,
      namedBufferSubtotalBytes: 1_100_487_680,
    });
    expect(planAceOpt0011Arm("fp16-256")).toMatchObject({
      windowFrames: 256,
      storageElementBytes: 2,
      parameterBytes: 168_791_552,
      workspaceBytes: 125_829_120,
      allWorkspaceBytes: 377_487_360,
      inputBytes: 32_768,
      outputFloat32Bytes: 3_932_160,
      readbackFloat32Bytes: 3_932_160,
      namedBufferSubtotalBytes: 554_176_000,
    });
    expect(planAceOpt0011Arm("fp16-512")).toMatchObject({
      windowFrames: 512,
      storageElementBytes: 2,
      parameterBytes: 168_791_552,
      maximumActivationElements: 125_829_120,
      workspaceBytes: 251_658_240,
      allWorkspaceBytes: 754_974_720,
      inputBytes: 65_536,
      outputFloat32Bytes: 7_864_320,
      readbackFloat32Bytes: 7_864_320,
      namedBufferSubtotalBytes: 939_560_448,
    });
  });

  it("pins 180-second and long-fixture chunk geometry", () => {
    expect(planAceOpt0011ChunkGeometry(4_500, 256)).toEqual({
      logicalLatentFrames: 4_500,
      windowFrames: 256,
      overlapFrames: 64,
      strideFrames: 128,
      windowCount: 36,
      decodedLatentFrames: 8_936,
      duplicatedLatentFrames: 4_436,
    });
    expect(planAceOpt0011ChunkGeometry(4_500, 512)).toEqual({
      logicalLatentFrames: 4_500,
      windowFrames: 512,
      overlapFrames: 64,
      strideFrames: 384,
      windowCount: 12,
      decodedLatentFrames: 5_908,
      duplicatedLatentFrames: 1_408,
    });
    expect(planAceOpt0011ChunkGeometry(1_024, 256).windowCount).toBe(8);
    expect(planAceOpt0011ChunkGeometry(1_024, 512).windowCount).toBe(3);
  });

  it("reproduces the frozen deterministic latent fixtures", () => {
    expect(hash(createAceOpt0011LatentFixture(256))).toBe(
      "55333d3ae4a0aca83dc1509b837c577f54646924e658e01e53889dc8a5a44875",
    );
    expect(hash(createAceOpt0011LatentFixture(1_024))).toBe(
      "e8919adc02d83f2efcd60bcb6dec4f104628d2ed66742d0eddbffc6b0a481a14",
    );
  });

  it("enumerates the exact decoder temporal-support classes", () => {
    expect(planAceOpt0011TemporalSupport()).toEqual({
      hopLength: 1_920,
      maximumPastLatentFrames: 9,
      maximumFutureLatentFrames: 9,
      maximumRadiusLatentFrames: 9,
      classes: [
        {
          firstRelativeLatentFrame: -9,
          lastRelativeLatentFrame: 8,
          outputPhaseCount: 645,
        },
        {
          firstRelativeLatentFrame: -8,
          lastRelativeLatentFrame: 8,
          outputPhaseCount: 630,
        },
        {
          firstRelativeLatentFrame: -8,
          lastRelativeLatentFrame: 9,
          outputPhaseCount: 645,
        },
      ],
    });
  });

  it("fails closed on invalid fixture and chunk geometry", () => {
    expect(() => createAceOpt0011LatentFixture(0)).toThrow(/positive/);
    expect(() => planAceOpt0011Arm("unknown" as "fp32-256")).toThrow(
      /Unknown OPT-0011 arm/,
    );
    expect(() => planAceOpt0011ChunkGeometry(1_024, 768 as 512)).toThrow(
      /exactly 256 or 512/,
    );
  });
});

function hash(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
