import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES } from
  "../../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  loadAcePackageManifest,
} from "../../src/model/package.js";
import { createAceOpt0011VaeAcquisitionManifest } from
  "../../src/runtime/webgpu-pipeline.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../../src/webgpu/vae-fp16-package.js";
import { serializeOpt0018Failure } from
  "./opt-0018-dit-m2250-production-family-profile.js";

declare global {
  interface Window {
    __ACE_OPT0080_CACHE_STAGE_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

const button = element<HTMLButtonElement>("#stage");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const params = new URL(location.href).searchParams;
const harnessCommit = params.get("harnessCommit");

if (harnessCommit === null || !/^[a-f0-9]{40}$/.test(harnessCommit)) {
  button.disabled = true;
  publishFailure(new Error("OPT-0080 cache staging requires harnessCommit"));
}

button.addEventListener("click", () => {
  if (harnessCommit === null) return;
  button.disabled = true;
  document.body.dataset.status = "running";
  void stage(harnessCommit).catch(publishFailure);
}, { once: true });

async function stage(commit: string): Promise<void> {
  const manifestUrl = new URL(
    "/model/files-fp16-vae-revision7-experimental/manifest.json",
    location.href,
  ).href;
  progress.textContent = "authenticating the exact revision-7 VAE manifest";
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
    authenticatedVaeConverterRevision: 7,
  });
  const manifest = createAceOpt0011VaeAcquisitionManifest(loaded.manifest);
  const cache = await AceOpfsModelCache.open();
  const acquired = await acquireAceModelFiles({
    manifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    storage: navigator.storage,
    onFileProgress(value) {
      progress.textContent =
        `${value.source}: VAE ${value.fileIndex + 1}/${value.fileCount} ` +
        `${value.completedBytes}/${value.totalBytes} bytes`;
    },
  });
  if (
    loaded.manifestByteLength !== ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES ||
    acquired.plan.runtimeBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    acquired.files.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) =>
      acquired.files.get(name)?.size !==
        manifest.files.find((file) => file.name === name)?.byteLength
    )
  ) throw new Error("OPT-0080 staged VAE package identity changed");
  const estimate = await navigator.storage.estimate();
  publish(Object.freeze({
    schema: "ace-opt-0080-product-cache-stage-v1",
    status: "passed",
    harnessCommit: commit,
    manifest: Object.freeze({
      url: loaded.manifestUrl,
      sha256: loaded.manifestSha256,
      byteLength: loaded.manifestByteLength,
      converterRevision: loaded.manifest.provenance.converterRevision,
    }),
    acquisition: Object.freeze({
      fileCount: acquired.files.size,
      runtimeBytes: acquired.plan.runtimeBytes,
      cachedBytesBeforeStage: acquired.plan.cachedBytes,
      downloadedBytes: acquired.plan.downloadBytes,
      files: Object.freeze(manifest.files.map(({ name, byteLength, sha256 }) =>
        Object.freeze({ name, byteLength, sha256 })
      )),
    }),
    storageEstimate: Object.freeze({
      usageBytes: estimate.usage ?? null,
      quotaBytes: estimate.quota ?? null,
    }),
    gpuDeviceRequested: false,
    productGenerationStarted: false,
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  window.__ACE_OPT0080_CACHE_STAGE_RESULT__ = receipt;
  document.body.dataset.status = receipt.status === "passed" ? "complete" : "failed";
  progress.textContent = receipt.status === "passed"
    ? "PASSED — authenticated revision-7 VAE payload is cache-resident"
    : "FAILED — cache staging stopped";
  result.textContent = JSON.stringify(receipt, null, 2);
}

function publishFailure(error: unknown): void {
  publish(Object.freeze({
    schema: "ace-opt-0080-product-cache-stage-failure-v1",
    status: "failed",
    harnessCommit: harnessCommit ?? null,
    error: serializeOpt0018Failure(error),
    gpuDeviceRequested: false,
    productGenerationStarted: false,
  }));
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0080 staging element ${selector}`);
  return value;
}
