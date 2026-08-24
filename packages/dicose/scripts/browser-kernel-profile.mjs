#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";

try {
  const focus = readProfileFocus(process.env.DICOSE_PROFILE_FOCUS);
  const result = await runBrowserHarness({
    label: "DiCoSe production-shape GPU kernel profile",
    mode: "probe",
    warmupRuns: 0,
    measuredRuns: 1,
    timeoutMs: readPositiveIntegerEnv("DICOSE_BROWSER_TIMEOUT_MS", 3 * 60_000),
    pagePath: `/test/browser-kernel-profile.html${focus === undefined ? "" : `?focus=${focus}`}`,
  });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        harness: "browser-kernel-profile",
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    )}\n`,
  );
  process.exitCode = 1;
}

function readProfileFocus(source) {
  if (source === undefined) return undefined;
  if (!["all", "dense", "conv", "norm", "attention"].includes(source)) {
    throw new RangeError("DICOSE_PROFILE_FOCUS must be all, dense, conv, norm, or attention");
  }
  return source;
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
