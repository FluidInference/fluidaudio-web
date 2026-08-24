#!/usr/bin/env node

import { runBrowserHarness } from "./browser-harness.mjs";

try {
  const warmupRuns = readNonNegativeIntegerEnv("DICOSE_BENCHMARK_WARMUP_RUNS", 1);
  const measuredRuns = readPositiveIntegerEnv("DICOSE_BENCHMARK_RUNS", 3);
  const outputMode = readChoiceEnv(
    "DICOSE_BENCHMARK_OUTPUT_MODE",
    ["refined", "deterministic"],
    "refined",
  );
  const result = await runBrowserHarness({
    label: `DiCoSe ${outputMode} supplied-wave benchmark`,
    mode: "benchmark",
    warmupRuns,
    measuredRuns,
    pagePath: `/?outputMode=${encodeURIComponent(outputMode)}`,
    timeoutMs: readPositiveIntegerEnv("DICOSE_BROWSER_TIMEOUT_MS", 60 * 60 * 1000),
  });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        harness: "browser-benchmark",
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    )}\n`,
  );
  process.exitCode = 1;
}

function readNonNegativeIntegerEnv(name, fallback) {
  const source = process.env[name];
  if (source === undefined) return fallback;
  const parsed = Number(source);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
  return parsed;
}

function readPositiveIntegerEnv(name, fallback) {
  const source = process.env[name];
  if (source === undefined) return fallback;
  const parsed = Number(source);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return parsed;
}

function readChoiceEnv(name, choices, fallback) {
  const value = process.env[name] ?? fallback;
  if (!choices.includes(value)) {
    throw new RangeError(`${name} must be one of: ${choices.join(", ")}`);
  }
  return value;
}
