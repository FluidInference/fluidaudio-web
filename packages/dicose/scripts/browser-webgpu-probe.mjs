#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";

try {
  const result = await runBrowserHarness({
    label: "DiCoSe raw WGSL linear/tanh/GLU probe",
    mode: "probe",
    warmupRuns: 0,
    measuredRuns: 1,
    timeoutMs: readPositiveIntegerEnv("DICOSE_BROWSER_TIMEOUT_MS", 60_000),
    pagePath: "/test/browser-webgpu-probe.html",
  });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        harness: "browser-webgpu-probe",
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    )}\n`,
  );
  process.exitCode = 1;
}

function readPositiveIntegerEnv(name, fallback) {
  const source = process.env[name];
  if (source === undefined) return fallback;
  const parsed = Number(source);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new RangeError(`${name} must be a positive integer in milliseconds`);
  }
  return parsed;
}
