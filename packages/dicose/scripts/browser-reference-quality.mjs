#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";

try {
  const result = await runBrowserHarness({
    label: "DiCoSe deterministic waveform versus upstream PyTorch",
    mode: "probe",
    warmupRuns: 0,
    measuredRuns: 1,
    timeoutMs: 5 * 60_000,
    pagePath: "/test/browser-reference-quality.html",
  });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify({
    ok: false,
    harness: "browser-reference-quality",
    error: error instanceof Error ? error.message : String(error),
  }, null, 2)}\n`);
  process.exitCode = 1;
}
