type WorkerMessage =
  | Readonly<{ readonly type: "progress"; readonly message: string }>
  | Readonly<{
      readonly type: "passed";
      readonly result: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{ readonly type: "failed"; readonly error: unknown }>;

if (typeof document !== "undefined") initializeOpt0082Page();

function initializeOpt0082Page(): void {
  const run = requireElement<HTMLButtonElement>("#run");
  const cancel = requireElement<HTMLButtonElement>("#cancel");
  const progress = requireElement<HTMLElement>("#progress");
  const result = requireElement<HTMLElement>("#result");
  let worker: Worker | undefined;
  let settled = false;

  run.addEventListener("click", () => {
    if (worker !== undefined) return;
    run.disabled = true;
    cancel.disabled = false;
    document.body.dataset.status = "running";
    progress.textContent = "starting the OPT-0082 worker";
    result.textContent = "running";
    worker = new Worker(
      new URL("./opt-0082-planner-compact-head-ab-worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
      if (settled) return;
      const message = event.data;
      if (message.type === "progress") {
        progress.textContent = message.message;
        return;
      }
      settled = true;
      cancel.disabled = true;
      if (message.type === "passed") {
        document.body.dataset.status = "passed";
        progress.textContent = "passed";
        const json = JSON.stringify(message.result, null, 2);
        result.textContent = json;
        downloadReceipt(json);
      } else {
        document.body.dataset.status = "failed";
        progress.textContent = "failed";
        result.textContent = JSON.stringify(message.error, null, 2);
      }
      worker?.terminate();
    });
    worker.addEventListener("error", (event) => {
      if (settled) return;
      settled = true;
      cancel.disabled = true;
      document.body.dataset.status = "failed";
      progress.textContent = "worker error";
      result.textContent = event.error instanceof Error
        ? `${event.error.name}: ${event.error.message}\n${event.error.stack ?? ""}`
        : event.message;
      worker?.terminate();
    });
    worker.postMessage({ type: "run" });
  }, { once: true });

  cancel.addEventListener("click", () => {
    if (worker === undefined || settled) return;
    cancel.disabled = true;
    progress.textContent = "cancelling after the current bounded operation";
    worker.postMessage({ type: "cancel" });
  });
}

function downloadReceipt(json: string): void {
  const url = URL.createObjectURL(new Blob([json, "\n"], {
    type: "application/json",
  }));
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "opt-0082-planner-compact-head-ab.json";
    anchor.style.display = "none";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0082 element ${selector}`);
  return element;
}
