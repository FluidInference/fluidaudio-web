import { describe, expect, it } from "vitest";
import { writeProgressBreadcrumb } from "../src/engines/musicgen-acestep/progress-breadcrumb.js";

describe("progress crash markers", () => {
  it.each(["Cancelled", "Generation failed", "Stem split failed"])("records %s as a handled end", (title) => {
    let saved = "";
    const storage = {
      setItem: (_key: string, value: string) => {
        saved = value;
      },
    };
    writeProgressBreadcrumb("Downloading model", "", true, storage);
    expect(JSON.parse(saved).open).toBe(true);
    writeProgressBreadcrumb(title, "", false, storage);
    expect(JSON.parse(saved)).toMatchObject({ title, open: false });
  });

  it("does not interrupt work when storage rejects writes", () => {
    expect(() =>
      writeProgressBreadcrumb("Generating song", "", true, {
        setItem() {
          throw new Error("Storage unavailable");
        },
      }),
    ).not.toThrow();
  });
});
