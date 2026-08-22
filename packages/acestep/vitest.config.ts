import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

// The OPT-0009 benchmark harnesses authenticate a pinned sibling
// parakeet.wgsl checkout; their contract tests can only run when that
// checkout is present at ../parakeet.wgsl (relative to this package).
const parakeetSiblingPresent = existsSync(
  fileURLToPath(new URL("../parakeet.wgsl", import.meta.url)),
);

// Generated model packages (multi-GB model/convert.py output) are
// gitignored (model/files-*/); the tests below authenticate them and can
// only run on a machine that has produced model/files-reference. They
// re-enable automatically once model/convert.py output exists.
const modelPackagesPresent = existsSync(
  fileURLToPath(new URL("./model/files-reference", import.meta.url)),
);

// Raw local benchmark evidence under optimization/artifacts/ is gitignored;
// the result-contract tests below recompute those receipts and can only run
// on a machine that kept the raw artifacts. They re-enable automatically
// once optimization/artifacts exists.
const optimizationArtifactsPresent = existsSync(
  fileURLToPath(new URL("./optimization/artifacts", import.meta.url)),
);

export default defineConfig({
  test: {
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      ...(parakeetSiblingPresent
        ? []
        : [
          "test/opt-0009-parakeet-gemm-calibration-contract.test.ts",
          "test/opt-0009-ace-gemm-accumulation-ab-contract.test.ts",
        ]),
      ...(modelPackagesPresent
        ? []
        : [
          "test/opt-0069-cache-authentication-hash.test.ts",
          "test/opt-0071-warm-cache-webcrypto-production.test.ts",
          "test/planner-coordinator.test.ts",
          "test/planner-metadata-fsm.test.ts",
          "test/planner-runtime.test.ts",
        ]),
      ...(optimizationArtifactsPresent
        ? []
        : [
          "test/opt-0011-vae-fp16-c512-long-browser-contract.test.ts",
          "test/opt-0014-result-contract.test.ts",
          "test/opt-0016-result-contract.test.ts",
          "test/opt-0017-result-contract.test.ts",
          "test/opt-0019-result-contract.test.ts",
          "test/opt-0020-result-contract.test.ts",
          "test/opt-0021-result-contract.test.ts",
          "test/opt-0022-result-contract.test.ts",
          "test/opt-0023-result-contract.test.ts",
        ]),
    ],
  },
});
