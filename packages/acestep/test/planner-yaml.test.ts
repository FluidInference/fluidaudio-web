import { describe, expect, it } from "vitest";

import {
  formatAcePlannerCotMetadata,
  type AcePlannerMetadata,
} from "../src/runtime/planner.js";
import vectors from "./planner-yaml-vectors.json";

describe("ACE planner pinned PyYAML subset", () => {
  it("matches every independently generated PyYAML 6.0.3 vector", () => {
    expect(vectors.pyyamlVersion).toBe("6.0.3");
    for (const vector of vectors.cases) {
      expect(
        formatAcePlannerCotMetadata(vector.metadata as AcePlannerMetadata),
        vector.id,
      ).toBe(vector.expected);
    }
  });

  it.each([
    "tab\ttext",
    "line\nbreak",
    "carriage\rreturn",
    "next\u0085line",
    "separator\u2028line",
    "bom\ufefftext",
    "noncharacter\uffff",
  ])("fails closed instead of needing a double-quoted emitter: %j", (caption) => {
    expect(() => formatAcePlannerCotMetadata({ caption })).toThrow(
      /unsupported PyYAML escaping/,
    );
  });

  it("fails closed on Python-only Unicode decimal conversion", () => {
    expect(() => formatAcePlannerCotMetadata({ duration: "٠١٢" })).toThrow(
      /unsupported non-ASCII decimal digits/,
    );
  });
});
