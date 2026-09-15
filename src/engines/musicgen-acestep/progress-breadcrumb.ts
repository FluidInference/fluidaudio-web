export const CRASH_BREADCRUMB_KEY = "ace-step-progress-breadcrumb";

/** Terminal progress must not leave a crash marker for the next visit. */
export function writeProgressBreadcrumb(title: string, detail: string, active: boolean, storage?: Pick<Storage, "setItem">): void {
  try {
    (storage ?? globalThis.localStorage).setItem(CRASH_BREADCRUMB_KEY, JSON.stringify({ title, detail, at: Date.now(), open: active }));
  } catch {
    // Diagnostics must not prevent generation when storage is unavailable.
  }
}
