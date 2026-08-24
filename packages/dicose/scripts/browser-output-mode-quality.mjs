#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";

try {
  const result = await runBrowserHarness({
    label: "DiCoSe deterministic versus refined waveform panel",
    mode: "probe",
    warmupRuns: 0,
    measuredRuns: 1,
    timeoutMs: 5 * 60_000,
    pagePath: "/test/browser-output-mode-quality.html",
  });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify({
    ok: false,
    harness: "browser-output-mode-quality",
    error: error instanceof Error ? error.message : String(error),
  }, null, 2)}\n`);
  process.exitCode = 1;
}
