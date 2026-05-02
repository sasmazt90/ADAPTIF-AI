import previewTemplateSchema from "@/shared/preview-templates.json";

export type PlacementDevice = "iphone" | "android" | "desktop";
export type PlacementOverlay = "facebook" | "instagram" | "tiktok" | "youtube" | "snapchat" | "linkedin" | "web";

export type SharedPreviewPlacement = {
  placementId: string;
  platform: string;
  label: string;
  ratio: string;
  dimensions: { width: number; height: number };
  device: PlacementDevice;
  overlay: PlacementOverlay;
  templateId: string;
  templateStatus: "production" | "partial" | "stub";
  assetPlaceholderBox: [number, number, number, number];
  safeArea: {
    warnings: string[];
    zones: Array<{
      id: string;
      label: string;
      x: number;
      y: number;
      width: number;
      height: number;
    }>;
  };
  supportedMetadataFields: string[];
  carouselSupported: boolean;
  uiElements: string[];
  deviceFrame: boolean;
  resizeRules: {
    mode: "cover" | "contain" | "fill";
    protectText: boolean;
    protectProduct: boolean;
  };
  layoutBoxes: Record<string, [number, number, number, number]>;
};

export type SharedPreviewSchema = {
  schemaVersion: string;
  placements: SharedPreviewPlacement[];
};

type RawSharedPreviewPlacement = Omit<SharedPreviewPlacement, "assetPlaceholderBox" | "layoutBoxes"> & {
  assetPlaceholderBox: number[];
  layoutBoxes: Record<string, number[]>;
};

export type Placement = {
  id: string;
  platform: string;
  label: string;
  ratio: string;
  width: number;
  height: number;
  device: PlacementDevice;
  overlay: PlacementOverlay;
  supportsCarousel?: boolean;
  safeZones: Array<{
    id: string;
    label: string;
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
};

function toQuad(value: number[]): [number, number, number, number] {
  return [value[0] ?? 0, value[1] ?? 0, value[2] ?? 0, value[3] ?? 0];
}

function normalizePlacement(input: RawSharedPreviewPlacement): SharedPreviewPlacement {
  return {
    ...input,
    device: input.device as PlacementDevice,
    overlay: input.overlay as PlacementOverlay,
    templateStatus: input.templateStatus as "production" | "partial" | "stub",
    assetPlaceholderBox: toQuad(input.assetPlaceholderBox as number[]),
    safeArea: {
      warnings: [...input.safeArea.warnings],
      zones: input.safeArea.zones.map((zone) => ({ ...zone })),
    },
    layoutBoxes: Object.fromEntries(
      Object.entries(input.layoutBoxes).map(([key, value]) => [key, toQuad(value as number[])]),
    ) as Record<string, [number, number, number, number]>,
  };
}

const sharedPreviewSchema: SharedPreviewSchema = {
  schemaVersion: String((previewTemplateSchema as unknown as { schemaVersion: string }).schemaVersion),
  placements: (previewTemplateSchema as unknown as { placements: RawSharedPreviewPlacement[] }).placements.map(normalizePlacement),
};

export const sharedTemplateSchemaVersion = sharedPreviewSchema.schemaVersion;
export const sharedPreviewPlacements = sharedPreviewSchema.placements;
export const sharedPreviewPlacementMap = Object.fromEntries(
  sharedPreviewPlacements.map((placement) => [placement.placementId, placement]),
) as Record<string, SharedPreviewPlacement>;

export const placementsFromSharedSchema: Placement[] = sharedPreviewPlacements.map((placement) => ({
  id: placement.placementId,
  platform: placement.platform,
  label: placement.label,
  ratio: placement.ratio,
  width: placement.dimensions.width,
  height: placement.dimensions.height,
  device: placement.device,
  overlay: placement.overlay,
  supportsCarousel: placement.carouselSupported,
  safeZones: placement.safeArea.zones,
}));

export function buildPreviewTemplateParitySnapshot() {
  return {
    sharedTemplateSchemaVersion,
    placementIds: sharedPreviewPlacements.map((placement) => placement.placementId),
    dimensions: Object.fromEntries(
      sharedPreviewPlacements.map((placement) => [
        placement.placementId,
        { width: placement.dimensions.width, height: placement.dimensions.height },
      ]),
    ),
    carouselSupport: Object.fromEntries(
      sharedPreviewPlacements.map((placement) => [placement.placementId, placement.carouselSupported]),
    ),
    templateStatus: Object.fromEntries(
      sharedPreviewPlacements.map((placement) => [placement.placementId, placement.templateStatus]),
    ),
  };
}
