import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  isAceClientMessage,
  isAceRuntimeDiagnosticsValue,
} from "../src/runtime/protocol.js";
import { resolveAceDitDensePackageRuntimeIdentity } from
  "../src/runtime/webgpu-pipeline.js";
import {
  classifyAceOpt0018DitCommandMember,
  type AceDitGraphQuantum,
} from "../src/webgpu/dit-graph.js";
import {
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
  ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
} from "../src/webgpu/dit-fp16-package.js";
import { ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID } from
  "../src/webgpu/kernels/dit-dense-fp16-k4-exact.js";
import {
  OPT_0056_EVALUATIONS,
  OPT_0056_LATENT_ELEMENTS,
  compareOpt0056Trajectory,
  exactOpt0056TrajectoryIdentity,
} from "./browser/opt-0056-dit-selective-k4-contract.js";
import { testDiagnostics, testInitializeMessage } from "./runtime-fixtures.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0056-dit-selective-k4-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0056-dit-selective-k4.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0056-dit-selective-k4.html",
  import.meta.url,
), "utf8");
const PIPELINE_SOURCE = readFileSync(new URL(
  "../src/runtime/webgpu-pipeline.ts",
  import.meta.url,
), "utf8");
const DEMO_SOURCE = readFileSync(new URL("../demo/main.ts", import.meta.url),
  "utf8");

describe("OPT-0056 authenticated benchmark-only runtime identity", () => {
  it("resolves only the rev8/selective pair and reports its distinct owner", () => {
    expect(resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/rev8/manifest.json",
      manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
      runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    })).toEqual({
      role: "opt-0056-rev8-benchmark",
      manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
    });
    expect(() => resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/crossed/manifest.json",
      manifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
      runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    } as never)).toThrow(/exact authenticated manifest\/profile pair/u);
  });

  it("accepts the exact protocol pair and rejects a crossed trust root", () => {
    const base = testInitializeMessage();
    const selective = {
      ...base,
      configuration: {
        ...base.configuration,
        schedulingProfile: "cooperative",
        ditDensePackage: {
          manifestUrl: "https://example.test/rev8/manifest.json",
          manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
          runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
        },
      },
    };
    expect(isAceClientMessage(selective)).toBe(true);
    expect(isAceClientMessage({
      ...selective,
      configuration: {
        ...selective.configuration,
        ditDensePackage: {
          ...selective.configuration.ditDensePackage,
          manifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
        },
      },
    })).toBe(false);

    const diagnostics = testDiagnostics({
      ditDenseManifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
      ditDenseManifestByteLength: ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
      ditDenseRuntimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      ditDenseKernelSetId: ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
      ditDenseLayerBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      ditResidentWeightBytes: ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
    });
    expect(isAceRuntimeDiagnosticsValue(diagnostics)).toBe(true);
    expect(isAceRuntimeDiagnosticsValue({
      ...diagnostics,
      ditDenseManifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
    })).toBe(false);
  });

  it("cannot reach product output without its private checkpoint seam", () => {
    expect(PIPELINE_SOURCE).toContain(
      "OPT-0056 selective dense runtime is benchmark-only and requires its checkpoint seam",
    );
    expect(PIPELINE_SOURCE).toContain(
      "OPT-0056 selective dense runtime cannot continue beyond its checkpoint",
    );
    expect(DEMO_SOURCE).toContain(
      'runtimeProfile: "opt-0009-fp16-fp32-dense-v1"',
    );
    expect(DEMO_SOURCE).toContain(
      "/model/files-fp16-dit-rev7-oracle/manifest.json",
    );
    expect(DEMO_SOURCE).not.toContain("opt-0056-selective-k4-exact-down-v1");
  });

  it("attributes exact-down and approximate projection commands distinctly", () => {
    const layer = {
      index: 27,
      kind: "layer",
      evaluation: 0,
      layer: 0,
      label: "ace-dit-eval-0-layer-0",
    } as const satisfies AceDitGraphQuantum;
    expect(classifyAceOpt0018DitCommandMember(
      layer,
      `${layer.label}-mlp-down-projection`,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    )).toEqual({
      family: "feed-forward",
      backend: "opt-0056-k4-exact-fp32",
      kernel: ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
    });
    expect(classifyAceOpt0018DitCommandMember(
      layer,
      `${layer.label}-mlp-up-projection`,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    )).toEqual({
      family: "feed-forward",
      backend: "opt-0032-k4-fp16-partials",
      kernel: ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
    });
  });
});

describe("OPT-0056 eight-tap numerical contract", () => {
  it("localizes the first difference while applying the unchanged final gate", () => {
    const exact = deterministicLatent();
    const control = Object.freeze(Array.from(
      { length: OPT_0056_EVALUATIONS },
      () => exact,
    ));
    expect(exactOpt0056TrajectoryIdentity(control, control)).toBe(true);
    expect(compareOpt0056Trajectory(control, control)).toMatchObject({
      evaluationCount: 8,
      firstDifferingEvaluation: null,
      passedFinalEnvelope: true,
      final: { differingU32Count: 0, passed: true },
    });

    const early = exact.slice();
    early[17] = Math.fround(early[17]! + 0.001);
    const localized = [...control];
    localized[2] = early;
    expect(compareOpt0056Trajectory(control, localized)).toMatchObject({
      firstDifferingEvaluation: 2,
      passedFinalEnvelope: true,
      final: { differingU32Count: 0, passed: true },
    });

    const failedFinal = exact.slice();
    failedFinal[23] = Math.fround(failedFinal[23]! + 0.5);
    localized[7] = failedFinal;
    expect(compareOpt0056Trajectory(control, localized)).toMatchObject({
      firstDifferingEvaluation: 2,
      passedFinalEnvelope: false,
      final: { maximumAbsoluteError: 0.5, passed: false },
    });
    expect(exactOpt0056TrajectoryIdentity(control, localized)).toBe(false);
    expect(() => compareOpt0056Trajectory(control.slice(0, 7), control))
      .toThrow(/exactly eight taps/u);
  });
});

describe("OPT-0056 button-gated four-shape and sequential browser gate", () => {
  it("constructs no worker or device until the explicit one-shot click", () => {
    expect(PAGE_SOURCE).toContain('runButton.addEventListener("click"');
    expect(PAGE_SOURCE.indexOf("new Worker(")).toBeGreaterThan(
      PAGE_SOURCE.indexOf('runButton.addEventListener("click"'),
    );
    expect(HTML_SOURCE).toContain('id="run"');
    expect(HTML_SOURCE).toContain("idle — no GPU work");
    expect(HTML_SOURCE).toContain("No worker,\n      WebGPU device, or model load");
  });

  it("runs four full primitive shapes and all M2250 arms without overlap", () => {
    const inventory = WORKER_SOURCE.match(
      /const shapes = Object\.freeze\(\[([\s\S]*?)\]\s+as const\);/u,
    )?.[1];
    expect(inventory).toBeDefined();
    expect([...(inventory ?? "").matchAll(/\{ id: "([^"]+)"/gu)].map(
      (match) => match[1],
    )).toEqual(["h-h", "h-1024", "h-6144", "6144-h"]);
    expect(WORKER_SOURCE).toContain("shapes.length !== 4");
    expect(WORKER_SOURCE).toContain("new Set(shapeIds).size !== 4");
    expect(WORKER_SOURCE).toContain(
      'shapeIds.join(",") !== "h-h,h-1024,h-6144,6144-h"',
    );
    expect(WORKER_SOURCE).toContain(
      'weightFixture: "deterministic-synthetic-logical-fp16"',
    );
    expect(WORKER_SOURCE).toContain(
      "actualPackageWeightIsolatedU32IdentityMeasured: false",
    );
    expect(WORKER_SOURCE).toContain(
      "[CONTROL, ALL_K4, SELECTIVE, SELECTIVE_REPEAT]",
    );
    expect(WORKER_SOURCE).toContain("await runArm(definition, request)");
    expect(WORKER_SOURCE).toContain("previous.lifecycle[\"disposeCompletedOrdinal\"]");
    expect(WORKER_SOURCE).toContain("await backend.dispose()");
    expect(WORKER_SOURCE).toContain("sequentialNonOverlappingArms: true");
    expect(WORKER_SOURCE).toContain("gpuPackageRuntimeCoResidency: false");
  });

  it("requires truthful eight-tap topology and actual selective owner counts", () => {
    expect(WORKER_SOURCE).toContain("checkpoint.evaluations.length !== OPT_0056_EVALUATIONS");
    expect(WORKER_SOURCE).toContain("checkpoint.snapshotCopyCount !== 8");
    expect(WORKER_SOURCE).toContain("checkpoint.snapshotExtraCommandBufferCount !== 0");
    expect(WORKER_SOURCE).toContain("checkpoint.snapshotExtraQueueDrainCount !== 0");
    expect(WORKER_SOURCE).toContain("profile.routeCount !== 216");
    expect(WORKER_SOURCE).toContain("profile.dispatchCount !== 1_728");
    expect(WORKER_SOURCE).toContain("profile.approximateRouteCount !== 192");
    expect(WORKER_SOURCE).toContain("profile.exactDownRouteCount !== 24");
    expect(WORKER_SOURCE).toContain('owner === "opt-0056-k4-exact-fp32"');
    expect(WORKER_SOURCE).toContain("controller.abort(privateStop)");
    expect(WORKER_SOURCE).toContain("vaeWeightAcquireStarted: false");
    expect(WORKER_SOURCE).not.toContain("decodeWindow(");
    expect(WORKER_SOURCE).toContain('schedulingProfile: "cooperative"');
    expect(WORKER_SOURCE).not.toContain('schedulingProfile: "benchmark"');
  });
});

function deterministicLatent(): Float32Array<ArrayBuffer> {
  const latent = new Float32Array(OPT_0056_LATENT_ELEMENTS);
  for (let index = 0; index < latent.length; index += 1) {
    latent[index] = Math.fround(((index % 257) - 128) / 31);
  }
  return latent;
}
