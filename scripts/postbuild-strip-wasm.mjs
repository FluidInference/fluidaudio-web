// Remove onnxruntime-web's wasm binaries from dist/ — they're loaded from the
// jsdelivr CDN at runtime (see core/ort.ts wasmPaths). The threaded+jsep wasm is
// ~26 MB, over Cloudflare's 25 MB per-file asset limit, so it must not ship in the
// build output. Safe: with wasmPaths set, ORT never loads the local copies.
import { readdirSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";

const dir = "dist/assets";
let removed = 0, freed = 0;
for (const f of readdirSync(dir)) {
  if (f.startsWith("ort-") && f.endsWith(".wasm")) {
    const p = join(dir, f);
    freed += statSync(p).size;
    unlinkSync(p);
    removed++;
  }
}
console.log(`[postbuild] stripped ${removed} ORT wasm file(s), ${(freed / 1e6).toFixed(0)} MB (loaded from jsdelivr CDN instead)`);
