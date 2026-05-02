import { sharedPreviewPlacementMap, sharedPreviewPlacements, type Placement } from "@/lib/preview-schema";

export type PreviewTemplateKind =
  | "facebook_feed"
  | "facebook_marketplace"
  | "facebook_right_column"
  | "instagram_feed"
  | "instagram_story"
  | "instagram_reels"
  | "tiktok_infeed"
  | "tiktok_topview"
  | "tiktok_branded_content"
  | "snap_top_snap"
  | "snap_story_ad"
  | "linkedin_single_image_1200x628"
  | "linkedin_single_image_1080x1080"
  | "linkedin_sponsored_content"
  | "gdn_300x250"
  | "gdn_728x90"
  | "gdn_160x600"
  | "gdn_320x50"
  | "gdn_300x600"
  | "youtube_instream"
  | "youtube_shorts"
  | "custom_display";

export type PreviewMetadata = {
  brandName: string;
  username: string;
  accountName?: string;
  sponsorLabel: string;
  headline: string;
  description: string;
  ctaText: string;
  price: string;
  likesLabel: string;
  commentsLabel: string;
  caption?: string;
  carouselAssetsProvided?: boolean;
  carouselAssets?: string[];
  activeSlideIndex?: number;
  carouselMode?: boolean;
  creativeMode?: "single" | "carousel";
  carouselActivationSource?: "user_selected" | "forced_single" | "invalid_missing_assets";
  unusedAssets?: string[];
};

export type PreviewTemplate = {
  id: PreviewTemplateKind;
  placementType: string;
  templateStatus: "production" | "partial" | "stub";
  supportsCarousel: boolean;
  reusedShell: boolean;
  reusedShellReason: string;
  layoutBoxes: Record<string, [number, number, number, number]>;
  assetPlaceholderBox: [number, number, number, number];
  supportedMetadataFields: string[];
  uiElements: string[];
  deviceFrame: boolean;
};

export const placementPreviewTemplates: Record<string, PreviewTemplate> = Object.fromEntries(
  sharedPreviewPlacements.map((placement) => [
    placement.placementId,
    {
      id: placement.templateId as PreviewTemplateKind,
      placementType: placement.templateId,
      templateStatus: placement.templateStatus,
      supportsCarousel: placement.carouselSupported,
      reusedShell: false,
      reusedShellReason: "",
      layoutBoxes: placement.layoutBoxes,
      assetPlaceholderBox: placement.assetPlaceholderBox,
      supportedMetadataFields: placement.supportedMetadataFields,
      uiElements: placement.uiElements,
      deviceFrame: placement.deviceFrame,
    },
  ]),
) as Record<string, PreviewTemplate>;

function toTitleCase(value: string) {
  return value
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function derivePreviewMetadata(
  placement: Placement,
  translatedText: string | undefined,
  sourceName: string | undefined,
  userIdentity: string,
): PreviewMetadata {
  const lines = (translatedText ?? "")
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  const schemaPlacement = sharedPreviewPlacementMap[placement.id];
  const brandSeed = sourceName?.replace(/\.[^.]+$/, "") ?? "AdaptifAI";
  const brandName = toTitleCase(brandSeed.split(/[-_]/).slice(0, 2).join(" ") || "AdaptifAI");
  const username = userIdentity.includes("@") ? userIdentity.split("@")[0] : userIdentity;
  const headline = lines.slice(0, Math.min(2, lines.length)).join(" ").trim() || "Localized campaign headline";
  const description = lines.slice(2).join(" ").trim() || "Adapted creative placed inside a native platform preview.";
  const ctaByPlacement: Record<string, string> = {
    "facebook-feed": "Shop Now",
    "facebook-marketplace": "Shop Now",
    "facebook-right-column": "Learn More",
    "instagram-feed": "Shop Now",
    "instagram-story": "Learn More",
    "instagram-reels": "Learn More",
    "tiktok-in-feed": "Shop Now",
    "tiktok-topview": "Learn More",
    "tiktok-branded-content": "Learn More",
    "snap-top-snap": "Swipe Up",
    "snap-story-ad": "Swipe Up",
    "linkedin-single-wide": "Visit Website",
    "linkedin-single-square": "Visit Website",
    "linkedin-sponsored": "Visit Website",
    "gdn-300x250": "Open",
    "gdn-728x90": "Open",
    "gdn-160x600": "Open",
    "gdn-320x50": "Open",
    "gdn-300x600": "Open",
    "youtube-instream": "Learn More",
    "youtube-shorts": "Shop Now",
    "custom-display": "Read More",
  };
  return {
    brandName,
    username: username || "brand.co",
    accountName: brandName,
    sponsorLabel: placement.platform === "LINKEDIN" ? "Promoted" : "Sponsored",
    headline,
    description,
    ctaText: ctaByPlacement[placement.id] ?? "Learn More",
    price: schemaPlacement?.platform === "META" && placement.id === "facebook-marketplace" ? "€39.90" : "",
    likesLabel: "1,234 likes",
    commentsLabel: "View all 12 comments",
    caption: description,
    carouselAssetsProvided: false,
    carouselAssets: [],
    activeSlideIndex: 0,
    carouselMode: schemaPlacement?.carouselSupported ?? false,
    creativeMode: "single",
    carouselActivationSource: (schemaPlacement?.carouselSupported ?? false) ? "user_selected" : "forced_single",
    unusedAssets: [],
  };
}
