import { describe, expect, it } from "vitest";
import { estimateLocalizeCredits } from "./credit-pricing";

describe("estimateLocalizeCredits", () => {
  it("charges per source image, generated language image, and non-PDF output format", () => {
    expect(estimateLocalizeCredits({ fileCount: 2, languageCount: 3, outputFormat: "PNG" })).toBe(38);
  });

  it("charges PDF output per generated image", () => {
    expect(estimateLocalizeCredits({ fileCount: 2, languageCount: 3, outputFormat: "pdf" })).toBe(54);
  });

  it("clamps zero and negative counts to one billable unit", () => {
    expect(estimateLocalizeCredits({ fileCount: 0, languageCount: -4, outputFormat: "PNG" })).toBe(12);
  });
});
