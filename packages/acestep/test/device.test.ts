import { describe, expect, it, vi } from "vitest";

import {
  ACE_WEBGPU_LIMIT_NAMES,
  type AceWebGpuLimits,
} from "../src/webgpu/capabilities.js";
import { requestAceWebGpuDevice } from "../src/webgpu/device.js";

describe("ACE WebGPU device context", () => {
  it("unions the production VAE feature and C512 limits with reference main", async () => {
    const fixture = fakeGpu(["shader-f16"], ["shader-f16"]);
    const context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: fixture.gpu,
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        maxBufferSize: 251_658_240,
        maxStorageBufferBindingSize: 251_658_240,
      },
    });

    expect(fixture.requestDevice.mock.calls[0]![0]).toMatchObject({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        maxBufferSize: 256 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
      },
    });
    expect(context.capabilities.executionProfile.id).toBe(
      "reference-bf16-portable",
    );
    expect(context.capabilities.requiredFeatures).toEqual(["shader-f16"]);
    expect(context.capabilities.stockFeatures["shader-f16"]).toMatchObject({
      adapterSupported: true,
      deviceEnabled: true,
      required: true,
      requested: true,
    });
  });

  it("derives adapter-aware capacities and still fails closed on true deficits", async () => {
    const fixture = fakeGpu(["shader-f16"], ["shader-f16"]);
    const derive = vi.fn((adapterLimits: AceWebGpuLimits) => ({
      maxBufferSize: Math.min(adapterLimits.maxBufferSize, 1_069_547_520),
      maxStorageBufferBindingSize: Math.min(
        adapterLimits.maxStorageBufferBindingSize,
        1_069_547_520,
      ),
    }));
    const context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: fixture.gpu,
      deriveRequiredLimits: derive,
    });
    expect(derive).toHaveBeenCalledTimes(1);
    expect(derive).toHaveBeenCalledWith(expect.objectContaining({
      maxBufferSize: 1_073_741_824,
      maxStorageBufferBindingSize: 1_073_741_824,
    }));
    expect(fixture.requestDevice.mock.calls[0]![0]).toMatchObject({
      requiredLimits: {
        maxBufferSize: 1_069_547_520,
        maxStorageBufferBindingSize: 1_069_547_520,
      },
    });
    context.destroy();

    const deficient = fakeGpu(["shader-f16"], ["shader-f16"]);
    await expect(requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: deficient.gpu,
      deriveRequiredLimits: () => ({ maxBufferSize: 1_168_834_560 }),
    })).rejects.toMatchObject({ code: "LIMIT_UNAVAILABLE" });
    expect(deficient.requestDevice).not.toHaveBeenCalled();
  });

  it("requests only selected features/capacities and reports runtime failures", async () => {
    const fixture = fakeGpu(["subgroups", "timestamp-query"], ["subgroups", "timestamp-query"]);
    const events: unknown[] = [];
    const context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: fixture.gpu,
      enableTimestampQueries: true,
      onRuntimeEvent: (event) => events.push(event),
    });
    expect(fixture.requestDevice).toHaveBeenCalledOnce();
    const descriptor = fixture.requestDevice.mock.calls[0]![0]!;
    expect(descriptor.requiredFeatures).toEqual(["subgroups", "timestamp-query"]);
    expect(descriptor.requiredLimits).not.toHaveProperty(
      "minUniformBufferOffsetAlignment",
    );
    expect(descriptor.requiredLimits).toMatchObject({
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 256 * 1024 * 1024,
    });
    expect(context.capabilities.executionProfile.id).toBe(
      "reference-bf16-subgroups",
    );
    expect(context.capabilities.stockFeatures["timestamp-query"]).toMatchObject({
      adapterSupported: true,
      deviceEnabled: true,
      requested: true,
    });

    fixture.emitUncaptured({ message: "bad bind group" } as GPUError);
    expect(events).toContainEqual({
      type: "uncaptured-error",
      errorType: "Object",
      message: "bad bind group",
    });
    fixture.resolveLost({
      reason: "unknown",
      message: "watchdog",
    } as GPUDeviceLostInfo);
    await expect(context.lost).resolves.toEqual({
      type: "device-lost",
      reason: "unknown",
      message: "watchdog",
    });
    context.destroy();
    context.destroy();
    expect(fixture.destroy).toHaveBeenCalledOnce();
    expect(fixture.removeEventListener).toHaveBeenCalledOnce();
  });

  it("fails before requestDevice when a profile feature or limit is absent", async () => {
    const featureFixture = fakeGpu([], []);
    await expect(
      requestAceWebGpuDevice({
        modelProfile: "raw-fp16",
        gpu: featureFixture.gpu,
      }),
    ).rejects.toMatchObject({ code: "FEATURE_UNAVAILABLE" });
    expect(featureFixture.requestDevice).not.toHaveBeenCalled();

    const limitFixture = fakeGpu([], [], { maxStorageBufferBindingSize: 1 });
    await expect(
      requestAceWebGpuDevice({
        modelProfile: "reference-bf16",
        gpu: limitFixture.gpu,
      }),
    ).rejects.toMatchObject({ code: "LIMIT_UNAVAILABLE" });
    expect(limitFixture.requestDevice).not.toHaveBeenCalled();
  });

  it("destroys a device obtained after cancellation", async () => {
    const fixture = fakeGpu([], []);
    const controller = new AbortController();
    fixture.requestDevice.mockImplementationOnce(async () => {
      controller.abort();
      return fixture.device;
    });
    await expect(
      requestAceWebGpuDevice({
        modelProfile: "reference-bf16",
        gpu: fixture.gpu,
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fixture.destroy).toHaveBeenCalledOnce();
  });

  it("isolates diagnostic callback failures from events and the loss promise", async () => {
    const fixture = fakeGpu([], []);
    const context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: fixture.gpu,
      onRuntimeEvent: () => {
        throw new Error("diagnostic consumer failed");
      },
    });
    expect(() =>
      fixture.emitUncaptured({ message: "validation" } as GPUError),
    ).not.toThrow();
    fixture.resolveLost({ reason: "unknown", message: "lost" } as GPUDeviceLostInfo);
    await expect(context.lost).resolves.toMatchObject({
      type: "device-lost",
      reason: "unknown",
    });
  });

  it("does not report owner-initiated destruction as an unexpected loss", async () => {
    const fixture = fakeGpu([], []);
    const events: unknown[] = [];
    const context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      gpu: fixture.gpu,
      onRuntimeEvent: (event) => events.push(event),
    });
    context.destroy();
    fixture.resolveLost({
      reason: "destroyed",
      message: "owner destroyed device",
    } as GPUDeviceLostInfo);
    await expect(context.lost).resolves.toMatchObject({
      type: "device-lost",
      reason: "destroyed",
    });
    expect(events).toEqual([]);
  });
});

function fakeGpu(
  adapterFeatureNames: readonly string[],
  deviceFeatureNames: readonly string[],
  limitOverrides: Readonly<Record<string, number>> = {},
) {
  const adapterFeatures = new Set(adapterFeatureNames) as unknown as GPUSupportedFeatures;
  const deviceFeatures = new Set(deviceFeatureNames) as unknown as GPUSupportedFeatures;
  const limits = Object.fromEntries(
    ACE_WEBGPU_LIMIT_NAMES.map((name) => [
      name,
      name.startsWith("min") ? 256 : 1024 * 1024 * 1024,
    ]),
  );
  Object.assign(limits, limitOverrides);
  let uncaptured: ((event: GPUUncapturedErrorEvent) => void) | undefined;
  let resolveLost = (_info: GPUDeviceLostInfo) => {};
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
    resolveLost = resolve;
  });
  const destroy = vi.fn();
  const removeEventListener = vi.fn();
  const device = {
    features: deviceFeatures,
    limits,
    lost,
    destroy,
    addEventListener: vi.fn(
      (_type: "uncapturederror", callback: (event: GPUUncapturedErrorEvent) => void) => {
        uncaptured = callback;
      },
    ),
    removeEventListener,
  } as unknown as GPUDevice;
  const requestDevice = vi.fn(async (_descriptor?: GPUDeviceDescriptor) => device);
  const adapter = {
    features: adapterFeatures,
    limits,
    info: {
      vendor: "apple",
      architecture: "apple-silicon",
      device: "test",
      description: "test adapter",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    },
    requestDevice,
  } as unknown as GPUAdapter;
  const gpu = {
    requestAdapter: vi.fn(async () => adapter),
  } as unknown as GPU;
  return {
    gpu,
    device,
    requestDevice,
    destroy,
    removeEventListener,
    resolveLost,
    emitUncaptured(error: GPUError) {
      uncaptured?.({ error } as GPUUncapturedErrorEvent);
    },
  };
}
