/**
 * Shared, fully unattended Chrome runner for the local DiCoSe WebGPU page.
 *
 * Every invocation starts a temporary Vite server and a Chrome process with a
 * brand-new user-data directory.  It talks to that process over its private
 * CDP port, so there is no dependency on a user profile, extension, click, or
 * download prompt.
 */
import { spawn } from "node:child_process";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { constants } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { createServer } from "vite";

export const repositoryRoot = resolve(import.meta.dirname, "..");

const DEFAULT_CHROME_PATH =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const MAX_CAPTURED_MESSAGES = 50;

/**
 * Load the local page and wait for its automatic run to publish a result.
 *
 * The page contract is deliberately small:
 * `globalThis.__DICOSE_BROWSER__.report` and `#result` must contain the same
 * JSON-serializable report, with `ok: true` on success.  The page reads the
 * query parameters written here and starts itself when `autorun=1`.
 */
export async function runBrowserHarness({
  label,
  mode,
  warmupRuns = 0,
  measuredRuns = 1,
  timeoutMs = 20 * 60 * 1000,
  sourcePath = "/Mixture_audio_1.wav",
  pagePath = "/",
} = {}) {
  assertNonEmptyString(label, "label");
  assertOneOf(mode, ["e2e", "benchmark", "probe"], "mode");
  assertNonNegativeInteger(warmupRuns, "warmupRuns");
  assertPositiveInteger(measuredRuns, "measuredRuns");
  assertPositiveInteger(timeoutMs, "timeoutMs");
  if (!sourcePath.startsWith("/")) {
    throw new RangeError("sourcePath must be an absolute URL path");
  }
  if (!pagePath.startsWith("/")) {
    throw new RangeError("pagePath must be an absolute URL path");
  }

  const profileDirectory = await mkdtemp(
    join(tmpdir(), "dicose-wgsl-chrome-profile-"),
  );
  let server;
  let browser;
  let connection;
  let targetId;
  let sessionId;
  const consoleMessages = [];
  const pageExceptions = [];

  try {
    server = await createServer({
      root: repositoryRoot,
      logLevel: "error",
      server: {
        host: "127.0.0.1",
        port: 0,
        strictPort: false,
      },
    });
    await server.listen();

    const chromePath = await resolveChromePath();
    const stderr = [];
    let launchError;
    browser = spawn(
      chromePath,
      [
        `--user-data-dir=${profileDirectory}`,
        "--headless=new",
        "--remote-debugging-address=127.0.0.1",
        "--remote-debugging-port=0",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-features=MediaRouter,OptimizationHints,Translate",
        "about:blank",
      ],
      { detached: true, stdio: ["ignore", "ignore", "pipe"] },
    );
    browser.once("error", (error) => {
      launchError = error;
    });
    browser.stderr?.setEncoding("utf8");
    browser.stderr?.on("data", (chunk) => {
      capture(stderr, String(chunk));
    });

    const webSocketUrl = await waitForDevTools({
      profileDirectory,
      browser,
      stderr,
      launchError: () => launchError,
    });
    connection = await CdpConnection.connect(webSocketUrl);
    const chromeVersion = await connection.send("Browser.getVersion");

    ({ targetId } = await connection.send("Target.createTarget", {
      url: "about:blank",
    }));
    ({ sessionId } = await connection.send(
      "Target.attachToTarget",
      { targetId, flatten: true },
    ));
    connection.on("Runtime.consoleAPICalled", (params, eventSessionId) => {
      if (eventSessionId !== sessionId) return;
      const text = (params.args ?? [])
        .map((argument) => {
          if (typeof argument.value === "string") return argument.value;
          if (argument.value !== undefined) return JSON.stringify(argument.value);
          return argument.description ?? argument.type ?? "unknown";
        })
        .join(" ");
      capture(consoleMessages, `${params.type ?? "log"}: ${text}`);
    });
    connection.on("Runtime.exceptionThrown", (params, eventSessionId) => {
      if (eventSessionId !== sessionId) return;
      const details = params.exceptionDetails ?? {};
      capture(
        pageExceptions,
        details.exception?.description ?? details.text ?? "Page exception",
      );
    });
    await Promise.all([
      connection.send("Page.enable", {}, sessionId),
      connection.send("Runtime.enable", {}, sessionId),
      connection.send("Log.enable", {}, sessionId),
    ]);

    const targetUrl = createRunUrl(serverOrigin(server), {
      mode,
      warmupRuns,
      measuredRuns,
      sourcePath,
      pagePath,
    });
    const navigation = await connection.send(
      "Page.navigate",
      { url: targetUrl },
      sessionId,
    );
    if (typeof navigation.errorText === "string") {
      throw new Error(`Chrome could not navigate to the local test page: ${navigation.errorText}`);
    }

    const report = await waitForBrowserReport({
      connection,
      sessionId,
      timeoutMs,
      consoleMessages,
      pageExceptions,
    });
    if (report.ok !== true) {
      throw new Error(formatFailedReport(report));
    }
    validateSuccessfulReport(report, mode);

    const page = await evaluate(connection, sessionId, `(() => ({
      url: location.href,
      readyState: document.readyState,
      webgpu: "gpu" in navigator,
      crossOriginIsolated: globalThis.crossOriginIsolated === true,
    }))()`);
    return {
      ok: true,
      label,
      run: { mode, warmupRuns, measuredRuns, sourcePath, pagePath },
      chrome: {
        product: chromeVersion.product,
        userAgent: chromeVersion.userAgent,
        jsVersion: chromeVersion.jsVersion,
      },
      page,
      report,
      consoleMessages,
      pageExceptions,
    };
  } finally {
    if (connection !== undefined && targetId !== undefined) {
      await connection.send("Target.closeTarget", { targetId }).catch(() => {});
    }
    connection?.close();
    await terminateProcessGroup(browser);
    await closeViteServer(server);
    await rm(profileDirectory, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 100,
    }).catch(() => {});
  }
}

function createRunUrl(origin, { mode, warmupRuns, measuredRuns, sourcePath, pagePath }) {
  const url = new URL(pagePath, origin);
  url.searchParams.set("autorun", "1");
  url.searchParams.set("mode", mode);
  url.searchParams.set("warmupRuns", String(warmupRuns));
  url.searchParams.set("measuredRuns", String(measuredRuns));
  url.searchParams.set("source", sourcePath);
  return url.href;
}

async function resolveChromePath() {
  const candidates = [
    process.env.DICOSE_CHROME_PATH,
    process.env.CHROME_PATH,
    process.env.CHROME_BIN,
    DEFAULT_CHROME_PATH,
  ].filter((candidate) => typeof candidate === "string" && candidate.length > 0);
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Try the next explicit or conventional Chrome location.
    }
  }
  throw new Error(
    `Could not find an executable Google Chrome. Checked: ${candidates.join(", ")}`,
  );
}

function serverOrigin(server) {
  const address = server?.httpServer?.address();
  if (address === null || address === undefined || typeof address === "string") {
    throw new Error("Vite did not expose a local TCP address");
  }
  return `http://127.0.0.1:${address.port}`;
}

async function closeViteServer(server) {
  if (server === undefined) return;
  if (typeof server.close === "function") {
    await server.close();
    return;
  }
  const httpServer = server.httpServer;
  if (httpServer === undefined || !httpServer.listening) return;
  await new Promise((resolvePromise, rejectPromise) => {
    httpServer.close((error) => {
      if (error === undefined) resolvePromise();
      else rejectPromise(error);
    });
  });
}

async function waitForDevTools({
  profileDirectory,
  browser,
  stderr,
  launchError,
}) {
  const activePort = join(profileDirectory, "DevToolsActivePort");
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    const startError = launchError();
    if (startError !== undefined) throw startError;
    if (browser.exitCode !== null || browser.signalCode !== null) {
      throw new Error(`Chrome exited before CDP was ready:\n${stderr.join("")}`);
    }
    try {
      const [portSource, browserPath] = (await readFile(activePort, "utf8"))
        .trim()
        .split(/\r?\n/);
      const port = Number(portSource);
      if (Number.isSafeInteger(port) && port > 0 && browserPath) {
        const response = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (response.ok) {
          const version = await response.json();
          return version.webSocketDebuggerUrl ?? `ws://127.0.0.1:${port}${browserPath}`;
        }
      }
    } catch {
      // Chrome has not finished writing the private CDP endpoint yet.
    }
    await delay(100);
  }
  throw new Error(`Timed out waiting for Chrome CDP:\n${stderr.join("")}`);
}

async function waitForBrowserReport({
  connection,
  sessionId,
  timeoutMs,
  consoleMessages,
  pageExceptions,
}) {
  const deadline = Date.now() + timeoutMs;
  let lastEvaluationError;
  while (Date.now() < deadline) {
    let report;
    try {
      report = await evaluate(connection, sessionId, browserReportExpression());
      lastEvaluationError = undefined;
    } catch (error) {
      // Navigation can replace the execution context while the first module is loading.
      lastEvaluationError = error;
    }
    if (report !== null && report !== undefined) {
      if (typeof report !== "object" || Array.isArray(report)) {
        throw new Error("DiCoSe browser page published a non-object report");
      }
      if (report.ok === false) throw new Error(formatFailedReport(report));
      if (report.ok === true) return report;
    }
    await delay(150);
  }
  const details = [
    "Timed out waiting for the automatic DiCoSe browser run.",
    lastEvaluationError === undefined
      ? undefined
      : `Last page evaluation error: ${errorMessage(lastEvaluationError)}`,
    pageExceptions.length === 0
      ? undefined
      : `Page exceptions:\n${pageExceptions.join("\n")}`,
    consoleMessages.length === 0
      ? undefined
      : `Console output:\n${consoleMessages.join("\n")}`,
  ]
    .filter(Boolean)
    .join("\n\n");
  throw new Error(details);
}

function browserReportExpression() {
  return `(() => {
    const api = globalThis.__DICOSE_BROWSER__;
    const candidates = [
      api?.report,
      api?.lastReport,
      globalThis.__DICOSE_BROWSER_REPORT__,
    ];
    for (const candidate of candidates) {
      if (candidate !== undefined && candidate !== null) return candidate;
    }
    const text = document.querySelector("#result")?.textContent ?? "";
    try {
      const parsed = JSON.parse(text);
      return parsed !== null && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  })()`;
}

async function evaluate(connection, sessionId, expression) {
  const response = await connection.send(
    "Runtime.evaluate",
    {
      expression,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
  );
  if (response.exceptionDetails !== undefined) {
    throw new Error(
      response.exceptionDetails.exception?.description ?? response.exceptionDetails.text,
    );
  }
  return response.result?.value;
}

function formatFailedReport(report) {
  const error =
    typeof report.error === "string"
      ? report.error
      : report.error !== null && typeof report.error === "object" &&
          typeof report.error.message === "string"
        ? report.error.message
        : "browser run failed";
  const probeValues = Object.fromEntries(
    ["input", "weight", "copy", "linear", "maskStyle", "validationError", "uncapturedErrors"]
      .filter((key) => report[key] !== undefined)
      .map((key) => [key, report[key]]),
  );
  const suffix = Object.keys(probeValues).length === 0
    ? ""
    : `; details=${JSON.stringify(probeValues)}`;
  return `DiCoSe browser report failed: ${error}${suffix}`;
}

/**
 * A page-level `ok: true` only proves that its control flow completed.  For
 * the bundled non-silent fixture, require the actual E2E result to contain
 * finite stem PCM and, when the diagnostic seam is present, nonzero model
 * contributions.  This prevents a GPU validation failure from masquerading
 * as a successful sampler that merely returns its seeded noise.
 */
function validateSuccessfulReport(report, mode) {
  if (mode !== "e2e") return;
  const output = report.output;
  if (output === null || typeof output !== "object") {
    throw new Error("DiCoSe E2E report omitted output summaries");
  }
  const stems = output.stems;
  if (stems === null || typeof stems !== "object") {
    throw new Error("DiCoSe E2E report omitted stem summaries");
  }
  for (const name of ["drums", "bass", "other", "vocals"]) {
    const stem = stems[name];
    if (stem === null || typeof stem !== "object") {
      throw new Error(`DiCoSe E2E report omitted the ${name} stem`);
    }
    if (
      !Number.isSafeInteger(stem.samples) || stem.samples <= 0 ||
      stem.finiteSamples !== stem.samples * 2 ||
      !Number.isFinite(stem.durationSeconds) || stem.durationSeconds <= 0 ||
      !Number.isFinite(stem.peak) || stem.peak < 0 ||
      !Number.isFinite(stem.rms) || stem.rms < 0
    ) {
      throw new Error(`DiCoSe E2E report has invalid ${name} PCM statistics`);
    }
  }

  const diagnostics = output.diagnostics;
  if (diagnostics === undefined) return;
  if (diagnostics === null || typeof diagnostics !== "object") {
    throw new Error("DiCoSe E2E diagnostics are malformed");
  }
  const deterministicOnly = report.metrics?.outputMode === "deterministic";
  const modelStages = deterministicOnly
    ? [diagnostics.deterministic]
    : [diagnostics.deterministic, diagnostics.cdModelOutput];
  let hasModelContribution = false;
  for (const stage of modelStages) {
    if (stage === null || typeof stage !== "object") {
      throw new Error("DiCoSe E2E diagnostics omit a model stage");
    }
    for (const name of ["drums", "bass", "other", "vocals"]) {
      const statistic = stage[name];
      if (statistic === null || typeof statistic !== "object" ||
        !Number.isFinite(statistic.rms) || !Number.isFinite(statistic.peak)) {
        throw new Error(`DiCoSe E2E diagnostics are malformed for ${name}`);
      }
      hasModelContribution ||= statistic.rms > 1e-7 || statistic.peak > 1e-7;
    }
  }
  if (!hasModelContribution) {
    throw new Error("DiCoSe E2E model diagnostics are all zero; refusing a noise-only false pass");
  }
}

function capture(values, value) {
  values.push(value);
  while (values.length > MAX_CAPTURED_MESSAGES) values.shift();
}

async function terminateProcessGroup(child) {
  if (child === undefined || child.pid === undefined) return;
  const signal = (name) => {
    try {
      process.kill(-child.pid, name);
    } catch (error) {
      if (error?.code !== "ESRCH") throw error;
    }
  };
  signal("SIGTERM");
  if (child.exitCode === null && child.signalCode === null) {
    await waitForChildExit(child, 5_000);
  }
  try {
    process.kill(-child.pid, 0);
    signal("SIGKILL");
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
  }
  if (child.exitCode === null && child.signalCode === null) {
    await waitForChildExit(child, 5_000);
  }
  child.stderr?.destroy();
}

function waitForChildExit(child, timeoutMs) {
  return new Promise((resolvePromise) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.off("exit", onExit);
      resolvePromise();
    };
    const onExit = () => finish();
    const timer = setTimeout(finish, timeoutMs);
    child.once("exit", onExit);
    if (child.exitCode !== null || child.signalCode !== null) finish();
  });
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

function assertOneOf(value, values, label) {
  if (!values.includes(value)) {
    throw new RangeError(`${label} must be one of: ${values.join(", ")}`);
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string`);
  }
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative integer`);
  }
}

function assertPositiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

class CdpConnection {
  #socket;
  #nextId = 1;
  #pending = new Map();
  #listeners = new Map();

  static async connect(webSocketUrl) {
    if (typeof WebSocket !== "function") {
      throw new Error("The browser harness requires Node.js with a global WebSocket");
    }
    const socket = new WebSocket(webSocketUrl);
    await new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(
        () => rejectPromise(new Error("CDP connection timed out")),
        10_000,
      );
      socket.addEventListener(
        "open",
        () => {
          clearTimeout(timer);
          resolvePromise();
        },
        { once: true },
      );
      socket.addEventListener(
        "error",
        (event) => {
          clearTimeout(timer);
          rejectPromise(new Error(event.message ?? "CDP connection failed"));
        },
        { once: true },
      );
    });
    return new CdpConnection(socket);
  }

  constructor(socket) {
    this.#socket = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id !== undefined) {
        const pending = this.#pending.get(message.id);
        if (pending === undefined) return;
        this.#pending.delete(message.id);
        clearTimeout(pending.timer);
        if (message.error === undefined) pending.resolve(message.result ?? {});
        else pending.reject(new Error(`${pending.method}: ${message.error.message}`));
        return;
      }
      const listeners = this.#listeners.get(message.method);
      if (listeners === undefined) return;
      for (const listener of listeners) listener(message.params ?? {}, message.sessionId);
    });
    socket.addEventListener("close", () => {
      for (const pending of this.#pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(new Error("CDP connection closed"));
      }
      this.#pending.clear();
    });
  }

  on(method, listener) {
    const listeners = this.#listeners.get(method) ?? new Set();
    listeners.add(listener);
    this.#listeners.set(method, listeners);
    return () => listeners.delete(listener);
  }

  send(method, params = {}, sessionId, timeoutMs = 30_000) {
    const id = this.#nextId++;
    return new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        rejectPromise(new Error(`${method} timed out`));
      }, timeoutMs);
      this.#pending.set(id, {
        method,
        resolve: resolvePromise,
        reject: rejectPromise,
        timer,
      });
      this.#socket.send(
        JSON.stringify({
          id,
          method,
          params,
          ...(sessionId === undefined ? {} : { sessionId }),
        }),
      );
    });
  }

  close() {
    this.#socket.close();
  }
}
