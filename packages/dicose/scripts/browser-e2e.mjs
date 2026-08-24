#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";
import { assertFixtureAcceptance } from "./browser-acceptance.mjs";

try {
  const result = await runBrowserHarness({
    label: "DiCoSe full supplied-wave acceptance",
    mode: "e2e",
    warmupRuns: 0,
    measuredRuns: 1,
    timeoutMs: readPositiveIntegerEnv("DICOSE_BROWSER_TIMEOUT_MS", 20 * 60 * 1000),
  });
  assertFixtureAcceptance(result);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        harness: "browser-e2e",
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
