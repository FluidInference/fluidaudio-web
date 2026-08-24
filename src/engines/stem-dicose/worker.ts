/// <reference lib="webworker" />

// Dedicated DiCoSe inference worker: fetch-referrer patch + the vendored
// runtime's worker entry. Workers don't inherit the page's
// <meta name="referrer" content="no-referrer">, and Hugging Face
// hotlink-blocks requests carrying a *.workers.dev Referer (served without
// CORS headers → surfaces as "Failed to fetch"), same as the acestep worker.
// The static import below is hoisted ahead of the patch, but it only
// registers the message listener — every model fetch happens later.
const workerFetch = self.fetch.bind(self);
self.fetch = ((input: RequestInfo | URL, init?: RequestInit) => workerFetch(input, { ...init, referrerPolicy: "no-referrer" })) as typeof fetch;

import "dicose-wgsl/worker";
