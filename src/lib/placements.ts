export type { Placement } from "@/lib/preview-schema";
export { placementsFromSharedSchema as placements, sharedTemplateSchemaVersion } from "@/lib/preview-schema";

export const languages = [
  { code: "EN", label: "English" },
  { code: "DE", label: "German" },
  { code: "FR", label: "French" },
  { code: "IT", label: "Italian" },
  { code: "ES", label: "Spanish" },
  { code: "PT", label: "Portuguese" },
  { code: "TR", label: "Turkish" },
  { code: "AR", label: "Arabic", rtl: true },
  { code: "ZH", label: "Chinese" },
  { code: "JA", label: "Japanese" },
];

export const outputFormats = ["Original", "PNG", "JPG", "WebP", "PDF"];
