import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { parseAcePackageManifest } from "../src/model/manifest.js";
import { loadAcePackageManifest } from "../src/model/package.js";
import {
  isAceClientMessage,
  isAceRuntimeDiagnosticsValue,
} from "../src/runtime/protocol.js";
import {
  createAceOpt0009DitDenseAcquisitionManifest,
  createAceOpt0037DitK4AcquisitionManifest,
  resolveAceDitDensePackageRuntimeIdentity,
} from "../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  requireAceOpt0009DitDensePackageIdentity,
  requireAceOpt0037DitK4PackageIdentity,
} from "../src/webgpu/dit-fp16-package.js";
import {
  OPT_0037_CANDIDATE_MANIFEST_BYTES,
  OPT_0037_CANDIDATE_MANIFEST_PATH,
  OPT_0037_CANDIDATE_MANIFEST_SHA256,
  OPT_0037_CANDIDATE_RUNTIME_PROFILE,
  OPT_0037_CONTROL_MANIFEST_BYTES,
  OPT_0037_CONTROL_MANIFEST_PATH,
  OPT_0037_CONTROL_MANIFEST_SHA256,
  OPT_0037_CONTROL_RUNTIME_PROFILE,
  OPT_0037_FINAL_LATENT_ELEMENTS,
  compareOpt0037FinalLatents,
} from "./browser/opt-0037-dit-rev7-vs-rev8-contract.js";
import { testDiagnostics, testInitializeMessage } from "./runtime-fixtures.js";

const CONTROL_MANIFEST_URL = new URL(
  "../model/files-fp16-dit-rev7-oracle/manifest.json",
  import.meta.url,
);
const CANDIDATE_MANIFEST_URL = new URL(
  "../model/files-fp16-dit-layer-mixed-experimental/manifest.json",
  import.meta.url,
);
const HAS_LOCAL_PACKAGES = existsSync(CONTROL_MANIFEST_URL) &&
  existsSync(CANDIDATE_MANIFEST_URL);
const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0037-dit-rev7-vs-rev8-worker.ts",
  import.meta.url,
), "utf8");
const CONTRACT_SOURCE = readFileSync(new URL(
  "./browser/opt-0037-dit-rev7-vs-rev8-contract.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0037-dit-rev7-vs-rev8.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0037-dit-rev7-vs-rev8.html",
  import.meta.url,
), "utf8");
const DEMO_SOURCE = readFileSync(new URL("../demo/main.ts", import.meta.url),
  "utf8");
const PIPELINE_SOURCE = readFileSync(new URL(
  "../src/runtime/webgpu-pipeline.ts",
  import.meta.url,
), "utf8");

describe("OPT-0037 authenticated rev7/rev8 runtime dispatch", () => {
  it("resolves only the two exact manifest/profile pairs", () => {
    expect(resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/rev7/manifest.json",
      manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
      runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
    })).toEqual({
      role: "opt-0009-rev7-oracle",
      manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
      manifestByteLength: OPT_0037_CONTROL_MANIFEST_BYTES,
      runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
    });
    expect(resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/rev8/manifest.json",
      manifestSha256: OPT_0037_CANDIDATE_MANIFEST_SHA256,
      runtimeProfile: OPT_0037_CANDIDATE_RUNTIME_PROFILE,
    }).role).toBe("opt-0037-rev8-production");
    expect(() => resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/crossed/manifest.json",
      manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
      runtimeProfile: OPT_0037_CANDIDATE_RUNTIME_PROFILE,
    } as never)).toThrow(/exact authenticated manifest\/profile pair/);
    expect(() => resolveAceDitDensePackageRuntimeIdentity({
      manifestUrl: "https://example.test/crossed/manifest.json",
      manifestSha256: OPT_0037_CANDIDATE_MANIFEST_SHA256,
      runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
    } as never)).toThrow(/exact authenticated manifest\/profile pair/);
  });

  it("accepts rev7 through the public protocol only with its exact profile", () => {
    const base = testInitializeMessage();
    const rev7 = {
      ...base,
      configuration: {
        ...base.configuration,
        ditDensePackage: {
          manifestUrl: "https://example.test/rev7/manifest.json",
          manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
          runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
        },
      },
    };
    expect(isAceClientMessage(rev7)).toBe(true);
    expect(isAceClientMessage({
      ...rev7,
      configuration: {
        ...rev7.configuration,
        ditDensePackage: {
          ...rev7.configuration.ditDensePackage,
          runtimeProfile: OPT_0037_CANDIDATE_RUNTIME_PROFILE,
        },
      },
    })).toBe(false);
  });

  it("validates truthful diagnostics for either exact identity but no mixture", () => {
    const rev7 = testDiagnostics({
      ditDenseManifestId: "ace-test-opt-0009-dit-manifest",
      ditDenseManifestUrl: "https://example.test/rev7/manifest.json",
      ditDenseManifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
      ditDenseManifestByteLength: ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
      ditDenseRuntimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ditDenseKernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
      ditDenseLayerBytes: ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
      ditResidentWeightBytes: ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
    });
    expect(isAceRuntimeDiagnosticsValue(rev7)).toBe(true);
    expect(isAceRuntimeDiagnosticsValue(testDiagnostics())).toBe(true);
    expect(isAceRuntimeDiagnosticsValue({
      ...rev7,
      ditDenseRuntimeProfile: ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
    })).toBe(false);
    expect(isAceRuntimeDiagnosticsValue({
      ...rev7,
      ditDenseManifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
    })).toBe(false);
  });
});

describe.skipIf(!HAS_LOCAL_PACKAGES)(
  "OPT-0037 local oracle and candidate manifests",
  () => {
    it("authenticates both exact manifest byte streams and inventories", () => {
      const controlBytes = readFileSync(CONTROL_MANIFEST_URL);
      const candidateBytes = readFileSync(CANDIDATE_MANIFEST_URL);
      expect(controlBytes.byteLength).toBe(OPT_0037_CONTROL_MANIFEST_BYTES);
      expect(candidateBytes.byteLength).toBe(OPT_0037_CANDIDATE_MANIFEST_BYTES);
      expect(sha256(controlBytes)).toBe(OPT_0037_CONTROL_MANIFEST_SHA256);
      expect(sha256(candidateBytes)).toBe(OPT_0037_CANDIDATE_MANIFEST_SHA256);

      const rawControlManifest = JSON.parse(controlBytes.toString("utf8"));
      expect(() => parseAcePackageManifest(
        rawControlManifest,
        "fp16-dit-dense-experimental",
      )).toThrow(/converter revision/);
      const controlManifest = parseAcePackageManifest(
        rawControlManifest,
        "fp16-dit-dense-experimental",
        { authenticatedDitDenseConverterRevision: 7 },
      );
      const candidateManifest = parseAcePackageManifest(
        JSON.parse(candidateBytes.toString("utf8")),
        "fp16-dit-dense-experimental",
      );
      const controlLoaded = {
        manifest: controlManifest,
        manifestUrl: CONTROL_MANIFEST_URL.href,
        manifestSha256: sha256(controlBytes),
        manifestByteLength: controlBytes.byteLength,
        manifestId: `ace:${sha256(controlBytes)}`,
      };
      const candidateLoaded = {
        manifest: candidateManifest,
        manifestUrl: CANDIDATE_MANIFEST_URL.href,
        manifestSha256: sha256(candidateBytes),
        manifestByteLength: candidateBytes.byteLength,
        manifestId: `ace:${sha256(candidateBytes)}`,
      };
      expect(controlManifest.provenance.converterRevision).toBe(7);
      expect(candidateManifest.provenance.converterRevision).toBe(8);
      expect(() => requireAceOpt0009DitDensePackageIdentity(controlLoaded))
        .not.toThrow();
      expect(() => requireAceOpt0037DitK4PackageIdentity(candidateLoaded))
        .not.toThrow();

      const controlAcquisition =
        createAceOpt0009DitDenseAcquisitionManifest(controlManifest);
      const candidateAcquisition =
        createAceOpt0037DitK4AcquisitionManifest(candidateManifest);
      expect(controlAcquisition.files.map((file) => file.name)).toEqual(
        ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES,
      );
      expect(candidateAcquisition.files.map((file) => file.name)).toEqual(
        ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES,
      );
      expect(controlAcquisition.files.reduce(
        (sum, file) => sum + file.byteLength,
        0,
      )).toBe(ACE_OPT_0009_DIT_MIXED_LAYER_BYTES);
      expect(candidateAcquisition.files.reduce(
        (sum, file) => sum + file.byteLength,
        0,
      )).toBe(ACE_OPT_0009_DIT_MIXED_LAYER_BYTES);
    });

    it("opens rev7 parsing only behind its exact pre-parse SHA trust root", async () => {
      const controlBytes = readFileSync(CONTROL_MANIFEST_URL);
      const fetchControl = async (): Promise<Response> => new Response(
        new Uint8Array(
          controlBytes.buffer,
          controlBytes.byteOffset,
          controlBytes.byteLength,
        ),
        {
          status: 200,
          headers: { "Content-Length": String(controlBytes.byteLength) },
        },
      );
      await expect(loadAcePackageManifest({
        manifestUrl: "https://example.test/rev7/manifest.json",
        expectedManifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
        expectedProfile: "fp16-dit-dense-experimental",
        authenticatedDitDenseConverterRevision: 7,
        fetch: fetchControl,
      })).resolves.toMatchObject({
        manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
        manifestByteLength: OPT_0037_CONTROL_MANIFEST_BYTES,
        manifest: { provenance: { converterRevision: 7 } },
      });
      await expect(loadAcePackageManifest({
        manifestUrl: "https://example.test/rev7/manifest.json",
        expectedManifestSha256: OPT_0037_CANDIDATE_MANIFEST_SHA256,
        expectedProfile: "fp16-dit-dense-experimental",
        authenticatedDitDenseConverterRevision: 7,
        fetch: fetchControl,
      })).rejects.toMatchObject({ code: "MANIFEST_IDENTITY_ERROR" });
      await expect(loadAcePackageManifest({
        manifestUrl: "https://example.test/rev7/manifest.json",
        expectedManifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
        expectedProfile: "fp16-dit-dense-experimental",
        fetch: fetchControl,
      })).rejects.toThrow(/converter revision/);
    });
  },
);

describe("OPT-0037 full final-latent comparison contract", () => {
  it("reports exact bit identity, hashes aside, as a numerical pass", () => {
    const control = deterministicLatent();
    const comparison = compareOpt0037FinalLatents(control, control.slice());
    expect(comparison).toMatchObject({
      count: OPT_0037_FINAL_LATENT_ELEMENTS,
      finitePairCount: OPT_0037_FINAL_LATENT_ELEMENTS,
      differingU32Count: 0,
      signedZeroDifferenceCount: 0,
      classChangeCount: 0,
      controlNonFiniteCount: 0,
      candidateNonFiniteCount: 0,
      signedMeanError: 0,
      meanAbsoluteError: 0,
      rmsError: 0,
      relativeRmsError: 0,
      nrmse: 0,
      snrDecibels: "positive-infinity",
      pearsonCorrelation: 1,
      maximumAbsoluteError: 0,
      maximumRelativeError: 0,
      firstDifference: null,
      passed: true,
    });
  });

  it("reports relative error and rejects maximum-error or finite failures", () => {
    const control = deterministicLatent();
    const candidate = control.slice();
    candidate[17] = candidate[17]! + 0.5;
    const comparison = compareOpt0037FinalLatents(control, candidate);
    expect(comparison.differingU32Count).toBe(1);
    expect(comparison.maximumAbsoluteError).toBeCloseTo(0.5);
    expect(comparison.relativeMaximumAbsoluteError).toBeGreaterThan(0);
    expect(comparison.relativeRmsError).toBeGreaterThan(0);
    expect(comparison.maximumRelativeError).toBeGreaterThan(0);
    expect(comparison.firstDifference).toMatchObject({ index: 17 });
    expect(comparison.worstDifference).toMatchObject({ index: 17 });
    expect(comparison.passed).toBe(false);

    candidate[18] = Number.NaN;
    const nonfinite = compareOpt0037FinalLatents(control, candidate);
    expect(nonfinite.candidateNonFiniteCount).toBe(1);
    expect(nonfinite.classChangeCount).toBeGreaterThan(0);
    expect(nonfinite.passed).toBe(false);
  });

  it("rejects anything other than two complete M2250 F32 latents", () => {
    expect(() => compareOpt0037FinalLatents(
      new Float32Array(1),
      new Float32Array(1),
    )).toThrow(/geometry changed/);
  });
});

describe("OPT-0037 button-gated sequential browser harness", () => {
  it("does not construct its dedicated worker before the explicit click", () => {
    expect(PAGE_SOURCE).toContain('runButton.addEventListener("click"');
    expect(PAGE_SOURCE.indexOf("new Worker(")).toBeGreaterThan(
      PAGE_SOURCE.indexOf('runButton.addEventListener("click"'),
    );
    expect(HTML_SOURCE).toContain('id="run"');
    expect(HTML_SOURCE).toContain("idle — no GPU work");
    expect(HTML_SOURCE).toContain("No worker, WebGPU device, or model load");
  });

  it("runs control then disposal then candidate and never enters VAE", () => {
    const controlRun = WORKER_SOURCE.indexOf("await runArm(CONTROL, request)");
    const controlDisposed = WORKER_SOURCE.indexOf(
      "control.lifecycle.disposeCompletedOrdinal",
      controlRun,
    );
    const candidateRun = WORKER_SOURCE.indexOf(
      "await runArm(CANDIDATE, request)",
    );
    expect(controlRun).toBeGreaterThan(0);
    expect(controlDisposed).toBeGreaterThan(controlRun);
    expect(candidateRun).toBeGreaterThan(controlDisposed);
    expect(WORKER_SOURCE).toContain("await backend.dispose()");
    expect(WORKER_SOURCE).toContain("controller.abort(privateStop)");
    expect(WORKER_SOURCE).toContain("vaeWeightAcquireStarted: false");
    expect(WORKER_SOURCE).toContain("vaeBackendCreated: false");
    expect(WORKER_SOURCE).not.toContain("decodeWindow(");
  });

  it("pins exact package URLs and labels all timing as order-confounded", () => {
    expect(WORKER_SOURCE).toContain("OPT_0037_CONTROL_MANIFEST_PATH");
    expect(WORKER_SOURCE).toContain("OPT_0037_CANDIDATE_MANIFEST_PATH");
    expect(CONTRACT_SOURCE).toContain(OPT_0037_CONTROL_MANIFEST_PATH);
    expect(CONTRACT_SOURCE).toContain(OPT_0037_CANDIDATE_MANIFEST_PATH);
    expect(WORKER_SOURCE).toContain("compareOpt0037FinalLatents(");
    expect(WORKER_SOURCE).toContain("graphSubmitThroughDrainMs");
    expect(WORKER_SOURCE).toContain("stageWalls");
    expect(WORKER_SOURCE).toContain("orderConfounded: true");
    expect(WORKER_SOURCE).toContain("speedupCalculated: false");
    expect(WORKER_SOURCE).toContain("performanceDecisionAuthorized: false");
    expect(WORKER_SOURCE).not.toMatch(/speedup\s*[:=]\s*[^f]/u);
  });

  it("restores the authenticated exact profile after the rev8 gate failed", () => {
    expect(DEMO_SOURCE).toContain(ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256);
    expect(DEMO_SOURCE).toContain(ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE);
    expect(DEMO_SOURCE).toContain(OPT_0037_CONTROL_MANIFEST_PATH);
    expect(DEMO_SOURCE).not.toContain(ACE_OPT_0037_DIT_K4_MANIFEST_SHA256);
    expect(PIPELINE_SOURCE).toContain(
      "createAceOpt0009DitDenseAcquisitionManifest(loaded.manifest)",
    );
    expect(PIPELINE_SOURCE).toContain(
      "createAceOpt0037DitK4AcquisitionManifest(loaded.manifest)",
    );
  });
});

function deterministicLatent(): Float32Array {
  const latent = new Float32Array(OPT_0037_FINAL_LATENT_ELEMENTS);
  for (let index = 0; index < latent.length; index += 1) {
    latent[index] = Math.fround(((index % 257) - 128) / 31);
  }
  return latent;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
