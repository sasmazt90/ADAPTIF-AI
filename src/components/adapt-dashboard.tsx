"use client";

import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CloudUpload,
  CreditCard,
  Download,
  Bell,
  Bookmark,
  BriefcaseBusiness,
  FileArchive,
  Frame,
  Globe2,
  Heart,
  Home,
  Languages,
  Loader2,
  LogIn,
  LogOut,
  Menu,
  MessageCircle,
  MoreHorizontal,
  Scissors,
  Search,
  Send,
  Settings2,
  Shield,
  Sparkles,
  Type,
  User,
  Users,
  X,
  XCircle,
} from "lucide-react";
import type { User as SupabaseUser } from "@supabase/supabase-js";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { creditPricing, estimateEditCredits, estimateLocalizeCredits, estimateResizeCredits } from "@/lib/credit-pricing";
import { languages, outputFormats, Placement, placements } from "@/lib/placements";
import { derivePreviewMetadata, placementPreviewTemplates, type PreviewMetadata as PlatformPreviewMetadata, type PreviewTemplateKind } from "@/lib/preview-templates";
import { getSupabaseBrowser, hasSupabaseBrowserConfig } from "@/lib/supabase-client";

type Mode = "adapt" | "resize";
type AuthMode = "sign-in" | "sign-up" | "forgot-password" | "reset-password";
type ConsentChoice = "necessary" | "all" | null;
type Device = "mobile" | "desktop";
type FitMode = "contain" | "cover" | "fill";
type CreativeMode = "single" | "carousel";
type PreviewMetadata = PlatformPreviewMetadata;
type PipelineOutput = {
  placement_id?: string | null;
  filename: string;
  download_url: string;
  width: number;
  height: number;
  source_name: string;
  language?: string | null;
  source_language?: string | null;
  translated_text?: string;
  extracted_blocks?: Array<{ text: string; translated_text?: string | null; translate?: boolean }>;
};
type PipelineResult = {
  job_id: string;
  outputs: PipelineOutput[];
  credits_remaining?: number;
};
type ReceiptLine = { label: string; formula: string; credits: number };
type AdminUser = { user_id: string; credits: number; updated_at: string };

function formatCreditText(value: number) {
  return `${value} credit${value === 1 ? "" : "s"}`;
}

const platformOrder = ["SOCIAL", "GOOGLE", "CUSTOM"];
type PreviewVariant = {
  id: string;
  label: string;
  device: "Mobile" | "Desktop";
  templateId: PreviewTemplateKind;
};

const previewVariantsByPlacement: Record<string, PreviewVariant[]> = {
  "social-feed-square": [
    { id: "instagram-feed-mobile", label: "Instagram Feed", device: "Mobile", templateId: "instagram_feed" },
    { id: "facebook-feed-mobile", label: "Facebook Feed", device: "Mobile", templateId: "facebook_feed" },
    { id: "linkedin-feed-mobile-square", label: "LinkedIn Feed", device: "Mobile", templateId: "linkedin_mobile_feed" },
    { id: "linkedin-feed-desktop-square", label: "LinkedIn Feed", device: "Desktop", templateId: "linkedin_single_image_1080x1080" },
  ],
  "social-feed-portrait": [
    { id: "instagram-feed-mobile-portrait", label: "Instagram Feed", device: "Mobile", templateId: "social_feed_portrait" },
    { id: "facebook-feed-mobile-portrait", label: "Facebook Feed", device: "Mobile", templateId: "facebook_feed" },
  ],
  "story-image": [
    { id: "instagram-story-mobile", label: "Instagram Story", device: "Mobile", templateId: "story_image" },
    { id: "facebook-story-mobile", label: "Facebook Story", device: "Mobile", templateId: "story_image" },
    { id: "snapchat-story-mobile", label: "Snapchat Story", device: "Mobile", templateId: "snap_story_ad" },
  ],
  "wide-landscape": [
    { id: "linkedin-wide-mobile", label: "LinkedIn Feed", device: "Mobile", templateId: "linkedin_mobile_feed" },
    { id: "linkedin-wide-desktop", label: "LinkedIn Feed", device: "Desktop", templateId: "wide_landscape" },
    { id: "facebook-right-column-desktop", label: "Facebook Right Column", device: "Desktop", templateId: "facebook_right_column" },
  ],
  "google-responsive-landscape": [
    { id: "google-rda-landscape-desktop", label: "Google Responsive Display", device: "Desktop", templateId: "google_responsive_landscape" },
  ],
  "google-responsive-square": [
    { id: "google-rda-square-desktop", label: "Google Responsive Display", device: "Desktop", templateId: "google_responsive_square" },
  ],
  "google-responsive-vertical": [
    { id: "google-rda-vertical-mobile", label: "Google Responsive Display", device: "Mobile", templateId: "google_responsive_vertical" },
  ],
  "custom-display": [
    { id: "custom-web-desktop", label: "Custom Web Preview", device: "Desktop", templateId: "custom_display" },
  ],
};

function previewVariantsForPlacement(placementId: string) {
  return previewVariantsByPlacement[placementId] ?? previewVariantsByPlacement["custom-display"];
}
const sampleCopy = {
  adapt: "Launch faster with localized ads",
  resize: "Creative resized for every paid channel",
};
const pricingPacks = [
  { id: "starter", name: "Starter", credits: "50 credits", price: "€9.90", body: "Small campaign tests and quick localization checks." },
  { id: "studio", name: "Studio", credits: "150 credits", price: "€24.90", body: "Recurring paid social and display production." },
  { id: "scale", name: "Scale", credits: "250 credits", price: "€39.90", body: "High-volume global creative operations." },
] as const;

function cleanCopy(value: string) {
  return value.replaceAll("[BOLD]", "").replaceAll("[/BOLD]", "");
}

function mergeFiles(current: File[], incoming: File[]) {
  const seen = new Set(current.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
  const merged = [...current];
  for (const file of incoming) {
    const key = `${file.name}-${file.size}-${file.lastModified}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(file);
    }
  }
  return merged;
}

function overlaps(zone: Placement["safeZones"][number], box: { x: number; y: number; width: number; height: number }) {
  return zone.x < box.x + box.width && zone.x + zone.width > box.x && zone.y < box.y + box.height && zone.y + zone.height > box.y;
}

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="relative grid h-11 w-11 place-items-center rounded-md bg-[#101414] text-white">
        <span className="absolute left-2 top-2 h-3 w-3 rounded-sm bg-[#7ee1c6]" />
        <span className="absolute bottom-2 right-2 h-3 w-3 rounded-sm bg-[#ee4d6a]" />
        <span className="text-lg font-black">A</span>
      </div>
      <div>
        <p className="text-xl font-black leading-5">AdaptifAI</p>
        <p className="text-xs font-semibold uppercase text-[#0f766e]">Creative localization and resizing</p>
      </div>
    </div>
  );
}

function Collapsible({ title, icon, children, defaultOpen = true }: { title: string; icon?: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-md border border-[#151515]/10 bg-white">
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center justify-between px-4 py-3 text-left font-semibold">
        <span className="flex items-center gap-2">{icon}{title}</span>
        <ChevronDown className={["h-4 w-4 text-[#0f766e] transition", open ? "rotate-180" : ""].join(" ")} />
      </button>
      {open && <div className="border-t border-[#151515]/10 p-4">{children}</div>}
    </section>
  );
}

function ConsentBanner() {
  const [choice, setChoice] = useState<ConsentChoice>(null);
  const [ready, setReady] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const storageKey = "adaptifai:consent:v1";

  useEffect(() => {
    queueMicrotask(() => {
      const saved = window.localStorage.getItem(storageKey);
      setChoice(saved === "all" || saved === "necessary" ? saved : null);
      setReady(true);
    });
  }, []);

  const saveChoice = (nextChoice: Exclude<ConsentChoice, null>) => {
    window.localStorage.setItem(storageKey, nextChoice);
    setChoice(nextChoice);
    setShowDetails(false);
  };

  if (!ready) return null;
  if (choice) {
    return (
      <button
        type="button"
        onClick={() => {
          setChoice(null);
          setShowDetails(true);
        }}
        className="fixed bottom-4 left-4 z-50 rounded-md border border-[#151515]/10 bg-white px-3 py-2 text-xs font-semibold text-[#151515] shadow-lg"
      >
        Privacy settings
      </button>
    );
  }

  return (
    <div className="fixed inset-x-4 bottom-4 z-50 mx-auto max-w-4xl rounded-md border border-[#151515]/10 bg-white p-4 text-[#151515] shadow-2xl">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-sm font-black">Privacy and consent</p>
          <p className="mt-1 max-w-2xl text-sm text-[#555]">
            We use required cookies/storage for sign-in, credits, security and the editor. Optional analytics help improve the product and can be refused.
          </p>
          {showDetails && (
            <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
              <div className="rounded-md bg-[#faf9f5] p-3">
                <p className="font-bold">Necessary</p>
                <p className="mt-1 text-[#666]">Authentication session, credit balance, consent choice and security logs. Always active.</p>
              </div>
              <div className="rounded-md bg-[#faf9f5] p-3">
                <p className="font-bold">Analytics</p>
                <p className="mt-1 text-[#666]">Anonymous usage signals for product improvement. No ad tracking is loaded by default.</p>
              </div>
            </div>
          )}
          <div className="mt-2 flex gap-3 text-xs font-semibold">
            <a href="/privacy" className="text-[#0f766e] hover:text-[#151515]">Privacy GDPR/KVKK</a>
            <a href="/terms" className="text-[#0f766e] hover:text-[#151515]">Terms</a>
          </div>
        </div>
        <div className="grid shrink-0 gap-2 sm:grid-cols-3 md:grid-cols-1">
          <button type="button" onClick={() => saveChoice("necessary")} className="h-10 rounded-md border border-[#151515]/15 px-4 text-sm font-semibold">Necessary only</button>
          <button type="button" onClick={() => saveChoice("all")} className="h-10 rounded-md bg-[#151515] px-4 text-sm font-semibold text-white">Accept all</button>
          <button type="button" onClick={() => setShowDetails((value) => !value)} className="h-10 rounded-md bg-[#e8f7f1] px-4 text-sm font-semibold text-[#064e46]">Manage</button>
        </div>
      </div>
    </div>
  );
}

function AuthPanel({
  authMode,
  setAuthMode,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authError,
  authNotice,
  authPending,
  submitAuth,
}: {
  authMode: AuthMode;
  setAuthMode: (mode: AuthMode) => void;
  authEmail: string;
  setAuthEmail: (value: string) => void;
  authPassword: string;
  setAuthPassword: (value: string) => void;
  authError: string | null;
  authNotice: string | null;
  authPending: boolean;
  submitAuth: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const isSignIn = authMode === "sign-in";
  const isSignUp = authMode === "sign-up";
  const isForgot = authMode === "forgot-password";
  const isReset = authMode === "reset-password";
  const eyebrow = isSignIn ? "Welcome back" : isSignUp ? "Start free" : isForgot ? "Recover access" : "Secure reset";
  const title = isSignIn ? "Sign in" : isSignUp ? "Create account" : isForgot ? "Reset password" : "Set new password";
  const buttonLabel = isSignIn ? "Sign in" : isSignUp ? "Sign up and enter app" : isForgot ? "Send reset email" : "Update password";

  return (
    <form id="auth" onSubmit={submitAuth} className="rounded-md border border-[#151515]/10 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-[#0f766e]">{eyebrow}</p>
          <h2 className="text-2xl font-semibold">{title}</h2>
        </div>
        <LogIn className="h-5 w-5 text-[#0f766e]" />
      </div>
      <div className="mt-5 space-y-3">
        {!isReset && <label className="block text-sm font-semibold">Email<input className="mt-1 h-11 w-full rounded-md border border-[#151515]/15 px-3 outline-none focus:border-[#0f766e]" type="email" autoComplete="email" value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} required /></label>}
        {!isForgot && <label className="block text-sm font-semibold">Password<input className="mt-1 h-11 w-full rounded-md border border-[#151515]/15 px-3 outline-none focus:border-[#0f766e]" type="password" autoComplete={isSignIn ? "current-password" : "new-password"} value={authPassword} onChange={(event) => setAuthPassword(event.target.value)} required minLength={6} /></label>}
      </div>
      {authNotice && <p className="mt-3 rounded-md bg-[#e8f7f1] p-3 text-sm text-[#064e46]">{authNotice}</p>}
      {authError && <p className="mt-3 rounded-md bg-[#fff0d8] p-3 text-sm text-[#6b3b00]">{authError}</p>}
      <button type="submit" disabled={authPending} className="mt-5 flex h-11 w-full items-center justify-center gap-2 rounded-md bg-[#151515] font-semibold text-white disabled:bg-[#d6d0c4]">
        {authPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
        {buttonLabel}
      </button>
      <div className="mt-3 grid gap-2 text-center text-sm font-semibold text-[#0f766e]">
        {isSignIn && <button type="button" onClick={() => setAuthMode("forgot-password")}>Forgot password?</button>}
        {!isReset && <button type="button" onClick={() => setAuthMode(isSignUp ? "sign-in" : "sign-up")}>{isSignUp ? "Already have an account? Sign in" : "Need an account? Sign up"}</button>}
        {(isForgot || isReset) && <button type="button" onClick={() => setAuthMode("sign-in")}>Back to sign in</button>}
      </div>
    </form>
  );
}

function LandingPage({
  authMode,
  setAuthMode,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authError,
  authNotice,
  authPending,
  submitAuth,
}: {
  authMode: AuthMode;
  setAuthMode: (mode: AuthMode) => void;
  authEmail: string;
  setAuthEmail: (value: string) => void;
  authPassword: string;
  setAuthPassword: (value: string) => void;
  authError: string | null;
  authNotice: string | null;
  authPending: boolean;
  submitAuth: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const chooseAuth = (mode: AuthMode) => {
    setAuthMode(mode);
    document.getElementById("auth")?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#f6f3eb] text-[#151515]">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(21,21,21,0.045)_1px,transparent_1px),linear-gradient(0deg,rgba(21,21,21,0.035)_1px,transparent_1px)] bg-[size:44px_44px]" />
        <div className="absolute left-[-8%] top-28 h-40 w-[120%] -skew-y-3 bg-[#7ee1c6]/18" />
        <div className="absolute right-[-10%] top-[520px] h-52 w-[78%] skew-y-6 bg-[#ee4d6a]/12" />
        <div className="absolute left-[8%] top-[650px] h-28 w-[36%] border-y border-[#151515]/10 bg-white/35" />
      </div>
      <header className="sticky top-0 z-30 border-b border-[#151515]/10 bg-[#faf9f5]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1180px] items-center justify-between gap-4 px-5 py-4">
          <Brand />
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => chooseAuth("sign-in")} className="h-10 rounded-md border border-[#151515]/15 bg-white px-4 text-sm font-semibold">Sign In</button>
            <button type="button" onClick={() => chooseAuth("sign-up")} className="h-10 rounded-md bg-[#151515] px-4 text-sm font-semibold text-white">Sign Up</button>
          </div>
        </div>
      </header>

      <section className="relative border-b border-[#151515]/10">
        <div className="mx-auto grid max-w-[1180px] gap-10 px-5 py-14 lg:grid-cols-[1fr_420px] lg:items-center">
          <div>
            <p className="text-sm font-black uppercase text-[#0f766e]">AI creative localization for paid media</p>
            <h1 className="mt-4 max-w-3xl text-5xl font-black leading-[1.02] tracking-normal md:text-7xl">AdaptifAI turns one ad into launch-ready global variants.</h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-[#4f4f4f]">
              Extract campaign copy, translate it contextually, remove source text from the background, and preview exact Meta, TikTok, Google, LinkedIn and native placements before export.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <button type="button" onClick={() => chooseAuth("sign-up")} className="flex h-12 items-center gap-2 rounded-md bg-[#ee4d6a] px-5 font-semibold text-white">Start localizing <ArrowRight className="h-4 w-4" /></button>
              <button type="button" onClick={() => chooseAuth("sign-in")} className="h-12 rounded-md border border-[#151515]/15 bg-white px-5 font-semibold">Open workspace</button>
            </div>
          </div>
          <AuthPanel authMode={authMode} setAuthMode={setAuthMode} authEmail={authEmail} setAuthEmail={setAuthEmail} authPassword={authPassword} setAuthPassword={setAuthPassword} authError={authError} authNotice={authNotice} authPending={authPending} submitAuth={submitAuth} />
        </div>
      </section>

      <section className="relative mx-auto grid max-w-[1180px] gap-5 px-5 py-12 md:grid-cols-3">
        {[
          ["Marketing text only", "Detect headlines and CTAs while leaving ingredients, labels and product text untouched."],
          ["Layout-safe translation", "Preserve emphasis tags, fit translated copy into original bounds and flag platform safe-zone conflicts."],
          ["Export for every channel", "Generate original, PNG, JPG, WebP or PDF outputs across paid social, display and native placements."],
        ].map(([title, body]) => (
          <div key={title} className="border-t border-[#151515]/15 pt-5">
            <p className="text-lg font-black">{title}</p>
            <p className="mt-2 text-sm leading-6 text-[#555]">{body}</p>
          </div>
        ))}
      </section>

      <section className="relative border-y border-[#151515]/10 bg-white/88 backdrop-blur">
        <div className="mx-auto grid max-w-[1180px] gap-10 px-5 py-12 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div>
            <p className="text-sm font-black uppercase text-[#0f766e]">Workflow</p>
            <h2 className="mt-3 text-4xl font-black">Upload once, approve every placement with context.</h2>
            <p className="mt-4 text-[#555]">The app separates Adapt and Resize so translation QA and production resizing stay focused.</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {["OCR and copy filtering", "GPT-4o translation", "Background restoration", "Platform previews", "Manual edit pass", "Credit-based checkout"].map((item, index) => (
              <div key={item} className="flex items-center gap-3 rounded-md bg-[#faf9f5] p-3">
                <span className="grid h-8 w-8 place-items-center rounded-md bg-[#151515] text-xs font-black text-white">{index + 1}</span>
                <span className="text-sm font-semibold">{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-[1180px] px-5 py-12">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-sm font-black uppercase text-[#0f766e]">Pricing</p>
            <h2 className="mt-2 text-4xl font-black">Credits that scale with production volume.</h2>
          </div>
          <button type="button" onClick={() => chooseAuth("sign-up")} className="h-11 rounded-md bg-[#151515] px-5 font-semibold text-white">Create account</button>
        </div>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {pricingPacks.map(({ name, credits, price, body }) => (
            <div key={name} className="rounded-md border border-[#151515]/10 bg-white p-5">
              <p className="text-lg font-black">{name}</p>
              <p className="mt-4 text-4xl font-black">{price}</p>
              <p className="mt-2 text-sm font-semibold text-[#0f766e]">{credits}</p>
              <p className="mt-4 text-sm leading-6 text-[#555]">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="relative mx-auto flex max-w-[1180px] flex-wrap items-center justify-between gap-3 border-t border-[#151515]/10 px-5 py-5 text-xs text-[#666]">
        <div className="space-y-1">
          <p className="font-semibold text-[#151515]">SASMAZ DIGITAL SOLUTIONS / AdaptifAI - CREATIVE LOCALIZATION AND RESIZING TOOL</p>
          <p>İbrahim Tolgar ŞAŞMAZ / 81543, Munich Germany / <a href="mailto:tolgar@sasmaz.digital" className="hover:text-[#151515]">tolgar@sasmaz.digital</a></p>
          <p>Strictly stateless creative processing / temporary files auto-delete after 24h</p>
        </div>
        <nav className="flex gap-4"><a href="/terms" className="hover:text-[#151515]">Terms</a><a href="/privacy" className="hover:text-[#151515]">Privacy GDPR/KVKK</a><a href="/refund" className="hover:text-[#151515]">Refund</a></nav>
      </footer>
      <ConsentBanner />
    </main>
  );
}

function Creative({ placement, copy, mode, x, y, opacity, scale, fit, imageUrl }: { placement: Placement; copy: string; mode: Mode; x: number; y: number; opacity: number; scale: number; fit: FitMode; imageUrl?: string }) {
  const box = placement.ratio === "9:16" ? { x: 10 + x, y: 34 + y, width: 62, height: 14 } : { x: 9 + x, y: 34 + y, width: 58, height: 18 };
  return (
    <div className="relative overflow-hidden bg-[#f0d553]" style={{ aspectRatio: `${placement.width} / ${placement.height}` }}>
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="Generated output"
          className={["absolute inset-0 h-full w-full", fit === "contain" ? "object-contain bg-[#f7f4ed]" : fit === "fill" ? "object-fill" : "object-cover"].join(" ")}
          style={{ transform: `scale(${scale / 100})` }}
        />
      ) : (
        <>
          <div
            className={["absolute inset-0 bg-[linear-gradient(135deg,#f9f4e8_0%,#f0d553_34%,#38b6a6_72%,#172320_100%)]", fit === "fill" ? "blur-[1px]" : ""].join(" ")}
            style={{ transform: `scale(${scale / 100})` }}
          />
          <div className="absolute left-[8%] top-[13%] h-[16%] w-[28%] rounded-full bg-white/70" />
          <div className="absolute bottom-[12%] right-[8%] h-[24%] w-[38%] rounded-sm bg-[#ee4d6a]/70" />
          <div className="absolute left-[12%] top-[25%] h-[18%] w-[54%] rounded bg-white/10" style={{ opacity: Math.max(0.18, opacity / 140) }} />
          <div className="absolute flex flex-col justify-center" style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.width}%`, minHeight: `${box.height}%` }}>
            <p className="max-w-full text-[clamp(9px,1.25vw,14px)] font-black uppercase leading-tight text-[#111] [text-shadow:0_1px_0_rgba(255,255,255,0.55)]">{cleanCopy(copy || sampleCopy[mode]).slice(0, 42)}</p>
            <p className="mt-1 max-w-[82%] text-[clamp(7px,0.8vw,10px)] font-semibold leading-tight text-[#26302d] [text-shadow:0_1px_0_rgba(255,255,255,0.45)]">Preview placeholder</p>
          </div>
        </>
      )}
    </div>
  );
}

function AdFrame({ placement, children }: { placement: Placement; children: ReactNode }) {
  return (
    <div className="overflow-hidden rounded-md border border-[#151515]/15 bg-white">
      <div className="flex items-center justify-between border-b border-[#151515]/10 bg-[#faf9f5] px-2 py-1 text-[10px] font-semibold uppercase tracking-normal text-[#666]">
        <span>Ad creative</span>
        <span>{placement.width} x {placement.height}</span>
      </div>
      {children}
    </div>
  );
}

function CarouselAssetSurface({
  placement,
  mode,
  copy,
  x,
  y,
  opacity,
  scale,
  fit,
  imageUrl,
  carouselAssets,
  activeSlideIndex = 0,
}: {
  placement: Placement;
  mode: Mode;
  copy: string;
  x: number;
  y: number;
  opacity: number;
  scale: number;
  fit: FitMode;
  imageUrl?: string;
  carouselAssets?: string[];
  activeSlideIndex?: number;
}) {
  const slides = carouselAssets?.filter(Boolean) ?? [];
  const supportsCarousel = Boolean(placement.supportsCarousel);
  if (!supportsCarousel || slides.length <= 1) {
    return <Creative placement={placement} mode={mode} copy={copy} x={x} y={y} opacity={opacity} scale={scale} fit={fit} imageUrl={imageUrl} />;
  }

  const currentIndex = Math.max(0, Math.min(activeSlideIndex, slides.length - 1));
  const current = slides[currentIndex] ?? imageUrl;
  const next = slides[(currentIndex + 1) % slides.length];
  const showPeek = next && slides.length > 1;

  return (
    <div className="relative overflow-hidden">
      <div className="overflow-hidden rounded-[inherit]">
        <Creative placement={placement} mode={mode} copy={copy} x={x} y={y} opacity={opacity} scale={scale} fit={fit} imageUrl={current} />
      </div>
      {showPeek ? (
        <>
          <div className="pointer-events-none absolute inset-y-[8%] right-[4%] w-[26%] overflow-hidden rounded-2xl border border-white/70 bg-white/20 shadow-xl">
            <div className="h-full w-[180%] -translate-x-[42%]">
              <Creative placement={placement} mode={mode} copy={copy} x={x} y={y} opacity={opacity} scale={scale} fit={fit} imageUrl={next} />
            </div>
          </div>
          <div className="pointer-events-none absolute left-3 top-1/2 grid h-9 w-9 -translate-y-1/2 place-items-center rounded-full bg-black/45 text-lg font-bold text-white">‹</div>
          <div className="pointer-events-none absolute right-3 top-1/2 grid h-9 w-9 -translate-y-1/2 place-items-center rounded-full bg-black/45 text-lg font-bold text-white">›</div>
          <div className="pointer-events-none absolute inset-x-0 bottom-3 flex items-center justify-center gap-1.5">
            {slides.map((slide, index) => (
              <span
                key={`${slide}-${index}`}
                className={["h-2.5 rounded-full transition-all", index === currentIndex ? "w-5 bg-white" : "w-2.5 bg-white/50"].join(" ")}
              />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}

function AssetSurface({
  placement,
  mode,
  copy,
  x,
  y,
  opacity,
  scale,
  fit,
  imageUrl,
  carouselAssets,
  activeSlideIndex,
}: {
  placement: Placement;
  mode: Mode;
  copy: string;
  x: number;
  y: number;
  opacity: number;
  scale: number;
  fit: FitMode;
  imageUrl?: string;
  carouselAssets?: string[];
  activeSlideIndex?: number;
}) {
  return (
    <CarouselAssetSurface
      placement={placement}
      mode={mode}
      copy={copy}
      x={x}
      y={y}
      opacity={opacity}
      scale={scale}
      fit={fit}
      imageUrl={imageUrl}
      carouselAssets={carouselAssets}
      activeSlideIndex={activeSlideIndex}
    />
  );
}

function PreviewMetadataForm({ metadata, onChange }: { metadata: PreviewMetadata; onChange: (next: PreviewMetadata) => void }) {
  const update = (key: keyof PreviewMetadata, value: string) => onChange({ ...metadata, [key]: value });
  return (
    <div className="grid gap-2 rounded-md border border-[#151515]/10 bg-white p-3">
      <p className="text-xs font-semibold uppercase text-[#0f766e]">Preview metadata</p>
      <div className="grid gap-2">
        <input className="h-9 rounded-md border border-[#151515]/10 px-3 text-xs outline-none focus:border-[#0f766e]" value={metadata.brandName} onChange={(event) => update("brandName", event.target.value)} placeholder="Brand name" />
        <input className="h-9 rounded-md border border-[#151515]/10 px-3 text-xs outline-none focus:border-[#0f766e]" value={metadata.headline} onChange={(event) => update("headline", event.target.value)} placeholder="Headline" />
        <textarea className="min-h-20 rounded-md border border-[#151515]/10 px-3 py-2 text-xs outline-none focus:border-[#0f766e]" value={metadata.description} onChange={(event) => update("description", event.target.value)} placeholder="Description" />
        <div className="grid grid-cols-2 gap-2">
          <input className="h-9 rounded-md border border-[#151515]/10 px-3 text-xs outline-none focus:border-[#0f766e]" value={metadata.ctaText} onChange={(event) => update("ctaText", event.target.value)} placeholder="CTA" />
          <input className="h-9 rounded-md border border-[#151515]/10 px-3 text-xs outline-none focus:border-[#0f766e]" value={metadata.price} onChange={(event) => update("price", event.target.value)} placeholder="Price" />
        </div>
      </div>
    </div>
  );
}

function CreativeModeControl({
  placement,
  value,
  onChange,
}: {
  placement: Placement;
  value: CreativeMode;
  onChange: (next: CreativeMode) => void;
}) {
  const supportsCarousel = Boolean(placement.supportsCarousel);
  return (
    <div className="grid gap-2 rounded-md border border-[#151515]/10 bg-white p-3">
      <p className="text-xs font-semibold uppercase text-[#0f766e]">Creative mode</p>
      <select
        className="h-9 rounded-md border border-[#151515]/10 px-3 text-xs outline-none focus:border-[#0f766e] disabled:bg-[#f4f5f7]"
        value={supportsCarousel ? value : "single"}
        onChange={(event) => onChange(event.target.value as CreativeMode)}
      >
        <option value="single">Single Image</option>
        <option value="carousel" disabled={!supportsCarousel}>Carousel</option>
      </select>
      <p className="text-[11px] text-[#666]">
        {supportsCarousel
          ? "Choose whether this placement should render as a single creative or a carousel preview."
          : "This placement only supports single-image preview. Carousel stays disabled here."}
      </p>
    </div>
  );
}

function PhoneChrome({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return (
    <div className="mx-auto w-full max-w-[328px] rounded-[42px] border-[10px] border-[#111] bg-[#111] shadow-2xl">
      <div className={["relative overflow-hidden rounded-[31px]", dark ? "bg-[#050506]" : "bg-white"].join(" ")}>
        <div className="pointer-events-none absolute left-1/2 top-2 z-30 h-6 w-24 -translate-x-1/2 rounded-full bg-black/90" />
        {children}
      </div>
    </div>
  );
}

function BrandAvatar({ label, className = "" }: { label: string; className?: string }) {
  return (
    <div className={["grid place-items-center rounded-full bg-[#111] text-xs font-black text-white", className].join(" ")}>
      {label.slice(0, 1).toUpperCase()}
    </div>
  );
}

function InstagramActionBar() {
  return (
    <div className="flex items-center gap-4 px-4 py-3 text-[#111]">
      <Heart className="h-6 w-6" strokeWidth={2.2} />
      <MessageCircle className="h-6 w-6" strokeWidth={2.2} />
      <Send className="h-6 w-6" strokeWidth={2.2} />
      <Bookmark className="ml-auto h-6 w-6" strokeWidth={2.2} />
    </div>
  );
}

function Preview({ placement, mode, device, copy, x, y, opacity, scale, fit, imageUrl, metadata, previewTemplateId }: { placement: Placement; mode: Mode; device: Device; copy: string; x: number; y: number; opacity: number; scale: number; fit: FitMode; imageUrl?: string; metadata: PreviewMetadata; previewTemplateId?: PreviewTemplateKind }) {
  const template = placementPreviewTemplates[placement.id] ?? placementPreviewTemplates["custom-display"];
  const shellTemplateId = previewTemplateId ?? template.id;
  const box = placement.ratio === "9:16" ? { x: 9 + x, y: 28 + y, width: 65, height: 24 } : { x: 8 + x, y: 30 + y, width: 62, height: 26 };
  const warnings = placement.safeZones.filter((zone) => overlaps(zone, box));
  const carouselAssets = metadata.carouselAssets?.filter(Boolean) ?? [];
  const creativeMode = metadata.creativeMode ?? "single";
  const carouselWarning = template.supportsCarousel && creativeMode === "carousel" && carouselAssets.length < 2;
  const activeImageUrl = creativeMode === "single" ? (carouselAssets[0] ?? imageUrl) : imageUrl;
  const asset = (
    <AssetSurface
      placement={placement}
      mode={mode}
      copy={copy}
      x={x}
      y={y}
      opacity={opacity}
      scale={scale}
      fit={fit}
      imageUrl={activeImageUrl}
      carouselAssets={creativeMode === "carousel" ? carouselAssets : []}
      activeSlideIndex={creativeMode === "carousel" ? metadata.activeSlideIndex : 0}
    />
  );
  const framedCreative = <AdFrame placement={placement}>{asset}</AdFrame>;
  let shell: ReactNode = null;

  if (shellTemplateId === "social_feed_square" || shellTemplateId === "social_feed_portrait") {
    shell = (
      <PhoneChrome>
        <div className="flex h-10 items-end justify-between px-5 pb-2 text-[12px] font-semibold text-[#111]">
          <span>10:11</span>
          <span className="flex items-center gap-1"><span className="h-2.5 w-4 rounded-sm border border-[#111]" /><span>5G</span></span>
        </div>
        <div className="flex items-center justify-between border-b border-[#eceff3] px-4 py-2">
          <p className="font-serif text-[28px] font-black tracking-normal">Instagram</p>
          <div className="flex items-center gap-4"><Heart className="h-6 w-6" /><Send className="h-6 w-6" /></div>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-gradient-to-tr from-[#f7b733] via-[#d62976] to-[#4f5bd5] p-[2px]">
              <BrandAvatar label={metadata.brandName} className="h-9 w-9 border-2 border-white" />
            </div>
            <div><p className="text-[13px] font-bold leading-4">{metadata.username}</p><p className="text-[10px] text-[#666]">{metadata.sponsorLabel}</p></div>
          </div>
          <MoreHorizontal className="h-5 w-5 text-[#333]" />
        </div>
        <div className="bg-black">{asset}</div>
        <InstagramActionBar />
        <div className="space-y-1 px-4 pb-4 text-[12px] leading-4">
          <p className="font-semibold">{metadata.likesLabel}</p>
          <p><span className="font-bold">{metadata.username}</span> {metadata.headline}</p>
          <p className="text-[#555]">{metadata.description}</p>
          <p className="text-[#777]">{metadata.commentsLabel}</p>
          <button type="button" className="mt-2 rounded-full border border-[#0b66c3] px-4 py-1.5 text-[11px] font-bold text-[#0b66c3]">{metadata.ctaText}</button>
        </div>
        <div className="grid grid-cols-5 border-t border-[#eceff3] px-5 py-3 text-[#111]">
          <Home className="h-5 w-5" /><Search className="h-5 w-5" /><div className="h-5 w-5 rounded-md border-2 border-[#111]" /><BriefcaseBusiness className="h-5 w-5" /><User className="h-5 w-5" />
        </div>
      </PhoneChrome>
    );
  } else if (shellTemplateId === "story_image") {
    shell = (
      <PhoneChrome dark>
        <div className="relative">
          <div className="absolute inset-x-4 top-4 z-20 flex gap-1">{Array.from({ length: 4 }).map((_, index) => <span key={index} className={["h-1 flex-1 rounded-full", index === 0 ? "bg-white" : "bg-white/35"].join(" ")} />)}</div>
          <div className="absolute left-4 right-4 top-9 z-20 flex items-center justify-between text-white">
            <div className="flex items-center gap-2">
              <div className="rounded-full bg-gradient-to-tr from-[#f7b733] via-[#d62976] to-[#4f5bd5] p-[2px]">
                <BrandAvatar label={metadata.brandName} className="h-9 w-9 border-2 border-black/20" />
              </div>
              <div><p className="text-[13px] font-bold leading-4">{metadata.username}</p><p className="text-[10px] text-white/75">{metadata.sponsorLabel}</p></div>
            </div>
            <X className="h-6 w-6" />
          </div>
          <div>{asset}</div>
          <div className="absolute inset-x-7 bottom-20 z-20 grid place-items-center">
            <button type="button" className="rounded-full bg-white px-7 py-3 text-[14px] font-black text-[#111] shadow-lg">{metadata.ctaText}</button>
          </div>
          <div className="absolute inset-x-5 bottom-5 z-20 rounded-full border border-white/55 px-5 py-3 text-[13px] font-semibold text-white">Send message</div>
        </div>
      </PhoneChrome>
    );
  } else if (shellTemplateId === "linkedin_mobile_feed") {
    shell = (
      <PhoneChrome>
        <div className="flex h-10 items-end justify-between px-5 pb-2 text-[12px] font-semibold text-[#111]">
          <span>10:11</span>
          <span className="flex items-center gap-1"><span className="h-2.5 w-4 rounded-sm border border-[#111]" /><span>5G</span></span>
        </div>
        <div className="flex items-center gap-3 border-b border-[#d6d9de] bg-white px-4 py-3">
          <div className="grid h-8 w-8 place-items-center rounded bg-[#0a66c2] text-lg font-black text-white">in</div>
          <div className="flex h-8 flex-1 items-center gap-2 rounded-md bg-[#eef3f8] px-3 text-[#586069]"><Search className="h-4 w-4" /><span className="text-xs">Search</span></div>
          <MessageCircle className="h-5 w-5 text-[#555]" />
        </div>
        <article className="bg-white">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3"><BrandAvatar label={metadata.brandName} className="h-10 w-10" /><div><p className="text-[13px] font-bold">{metadata.brandName}</p><p className="text-[10px] text-[#666]">{metadata.sponsorLabel}</p></div></div>
            <MoreHorizontal className="h-5 w-5 text-[#555]" />
          </div>
          <p className="px-4 pb-3 text-[12px] leading-4 text-[#222]">{metadata.description}</p>
          <div>{asset}</div>
          <div className="px-4 py-3"><p className="text-[14px] font-bold">{metadata.headline}</p><button type="button" className="mt-2 rounded-full border border-[#0a66c2] px-4 py-1.5 text-[11px] font-bold text-[#0a66c2]">{metadata.ctaText}</button></div>
          <div className="grid grid-cols-4 border-t border-[#edf0f3] px-4 py-3 text-center text-[11px] font-semibold text-[#666]"><span>Like</span><span>Comment</span><span>Share</span><span>Send</span></div>
        </article>
        <div className="grid grid-cols-5 border-t border-[#d6d9de] px-5 py-3 text-[#555]">
          <Home className="h-5 w-5" /><Users className="h-5 w-5" /><BriefcaseBusiness className="h-5 w-5" /><Bell className="h-5 w-5" /><User className="h-5 w-5" />
        </div>
      </PhoneChrome>
    );
  } else if (shellTemplateId === "wide_landscape") {
    shell = (
      <div className="mx-auto w-full max-w-[900px] overflow-hidden rounded-xl border border-[#d6d9de] bg-[#f3f2ef] shadow-xl">
        <div className="flex items-center gap-6 border-b border-[#d6d9de] bg-white px-6 py-3">
          <div className="grid h-10 w-10 place-items-center rounded bg-[#0a66c2] text-2xl font-black text-white">in</div>
          <div className="flex h-10 flex-1 items-center gap-2 rounded-md bg-[#eef3f8] px-3 text-[#586069]"><Search className="h-4 w-4" /><span className="text-sm">Search</span></div>
          <Home className="h-5 w-5 text-[#555]" /><Users className="h-5 w-5 text-[#555]" /><BriefcaseBusiness className="h-5 w-5 text-[#555]" /><MessageCircle className="h-5 w-5 text-[#555]" /><Bell className="h-5 w-5 text-[#555]" />
        </div>
        <div className="grid gap-5 p-5 md:grid-cols-[240px_minmax(0,1fr)_220px]">
          <aside className="hidden rounded-xl border border-[#d6d9de] bg-white p-4 md:block"><BrandAvatar label={metadata.brandName} className="mx-auto h-16 w-16" /><p className="mt-3 text-center text-lg font-bold">{metadata.brandName}</p><p className="text-center text-xs text-[#666]">Sponsored profile</p></aside>
          <article className="overflow-hidden rounded-xl border border-[#d6d9de] bg-white">
            <div className="flex items-center justify-between px-4 py-3"><div className="flex items-center gap-3"><BrandAvatar label={metadata.brandName} className="h-11 w-11" /><div><p className="text-sm font-bold">{metadata.brandName}</p><p className="text-xs text-[#666]">{metadata.sponsorLabel}</p></div></div><MoreHorizontal className="h-5 w-5 text-[#555]" /></div>
            <div className="px-4 pb-3 text-sm text-[#222]">{metadata.description}</div>
            <div>{asset}</div>
            <div className="px-4 py-3"><p className="text-lg font-bold">{metadata.headline}</p><button type="button" className="mt-2 rounded-full border border-[#0a66c2] px-4 py-1.5 text-sm font-bold text-[#0a66c2]">{metadata.ctaText}</button></div>
            <div className="grid grid-cols-4 border-t border-[#edf0f3] px-4 py-2 text-xs font-semibold text-[#666]"><span>Like</span><span>Comment</span><span>Share</span><span>Send</span></div>
          </article>
          <aside className="hidden rounded-xl border border-[#d6d9de] bg-white p-4 md:block"><p className="text-lg font-bold">LinkedIn News</p><p className="mt-3 text-sm font-semibold">{metadata.headline}</p><p className="mt-2 text-xs text-[#666]">Promoted insight</p></aside>
        </div>
      </div>
    );
  } else if (shellTemplateId === "google_responsive_landscape" || shellTemplateId === "google_responsive_square" || shellTemplateId === "google_responsive_vertical") {
    shell = (
      <div className="mx-auto w-full max-w-[880px] overflow-hidden rounded-xl border border-[#dfe1e5] bg-white shadow-xl">
        <div className="flex items-center gap-2 border-b border-[#dfe1e5] bg-[#f8fafd] px-4 py-2 text-xs text-[#5f6368]">
          <span className="h-3 w-3 rounded-full bg-[#ea4335]" /><span className="h-3 w-3 rounded-full bg-[#fbbc04]" /><span className="h-3 w-3 rounded-full bg-[#34a853]" /><span className="ml-2 flex flex-1 items-center gap-2 rounded-full bg-white px-3 py-1"><Globe2 className="h-3.5 w-3.5" /> publisher.example/article</span><Menu className="h-4 w-4" />
        </div>
        <div className="grid gap-5 p-5 md:grid-cols-[minmax(0,1fr)_280px]">
          <main className="space-y-3"><h3 className="text-2xl font-black">Article headline with responsive ad slot</h3><p className="text-sm leading-6 text-[#555]">Publisher content surrounds the responsive image asset so the advertiser can preview density, crop pressure and text legibility.</p><div className="h-28 rounded-lg bg-[#f1f3f4]" /></main>
          <aside className="overflow-hidden rounded-xl border border-[#dfe1e5] bg-white">
            <div className="flex items-center justify-between border-b border-[#edf0f3] px-3 py-2 text-[10px] font-bold uppercase text-[#5f6368]"><span>Advertisement</span><span>Ad</span></div>
            <div className="p-3">{framedCreative}</div>
            <div className="space-y-1 border-t border-[#edf0f3] px-3 py-3"><p className="text-sm font-bold">{metadata.headline}</p><p className="text-xs text-[#5f6368]">{metadata.description}</p><button type="button" className="mt-2 rounded bg-[#1a73e8] px-3 py-1.5 text-xs font-bold text-white">{metadata.ctaText}</button></div>
          </aside>
        </div>
      </div>
    );
  }

  if (!shell && shellTemplateId === "facebook_feed") {
    shell = (
      <div className="mx-auto w-full max-w-[440px] overflow-hidden rounded-2xl border border-[#d8dce6] bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-[#eff2f7] px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="grid h-9 w-9 place-items-center rounded-full bg-[#1877f2] text-xs font-black text-white">{metadata.brandName.slice(0, 1)}</div>
            <div><p className="text-[12px] font-bold">{metadata.username}</p><p className="text-[10px] text-[#666]">{metadata.sponsorLabel}</p></div>
          </div>
          <span className="text-[#666]">•••</span>
        </div>
        <div>{asset}</div>
        <div className="space-y-2 px-4 py-3">
          <p className="text-[14px] font-bold">{metadata.headline}</p>
          <p className="text-[12px] text-[#666]">{metadata.description}</p>
          <button type="button" className="rounded-full bg-[#1877f2] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
        </div>
        <div className="flex items-center justify-between border-t border-[#eff2f7] px-4 py-3 text-[11px] text-[#666]">
          <span>Like</span><span>Comment</span><span>Share</span>
        </div>
      </div>
    );
  } else if (!shell && (shellTemplateId === "social_feed_square" || shellTemplateId === "social_feed_portrait" || shellTemplateId === "instagram_feed")) {
    shell = (
      <div className="mx-auto w-full max-w-[420px] overflow-hidden rounded-[28px] border border-[#151515]/10 bg-white shadow-xl">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="grid h-8 w-8 place-items-center rounded-full bg-[#151515] text-xs font-black text-white">{metadata.brandName.slice(0, 1)}</div>
            <div><p className="text-[12px] font-bold">{metadata.username}</p><p className="text-[10px] text-[#666]">{metadata.sponsorLabel}</p></div>
          </div>
          <span className="text-[#666]">•••</span>
        </div>
        <div>{asset}</div>
        <div className="flex items-center gap-4 px-4 py-3 text-[#151515]">
          <span>♡</span><span>💬</span><span>➤</span><span className="ml-auto">🔖</span>
        </div>
        <div className="space-y-1 px-4 pb-4 text-[12px]">
          <p className="font-semibold">{metadata.likesLabel}</p>
          <p><span className="font-bold">{metadata.username}</span> {metadata.headline}</p>
          <p className="text-[#555]">{metadata.description}</p>
          <p className="text-[#777]">{metadata.commentsLabel}</p>
          <button type="button" className="mt-2 rounded-full bg-[#2550a8] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
        </div>
      </div>
    );
  } else if (!shell && (shellTemplateId === "story_image" || shellTemplateId === "instagram_story")) {
    shell = (
      <div className="relative mx-auto w-full max-w-[300px] overflow-hidden rounded-[36px] border-[10px] border-[#111] bg-black shadow-2xl">
        <div className="absolute inset-x-4 top-3 z-20 flex gap-1">{Array.from({ length: 5 }).map((_, index) => <span key={index} className={["h-1 flex-1 rounded-full", index === 0 ? "bg-white" : "bg-white/35"].join(" ")} />)}</div>
        <div className="absolute left-4 right-4 top-6 z-20 flex items-center justify-between text-white">
          <div className="flex items-center gap-2"><div className="grid h-8 w-8 place-items-center rounded-full bg-white/20 text-xs font-black">{metadata.brandName.slice(0, 1)}</div><div><p className="text-[12px] font-bold">{metadata.username}</p><p className="text-[10px] text-white/70">{metadata.sponsorLabel}</p></div></div>
          <span className="text-lg">×</span>
        </div>
        <div className="relative">{asset}</div>
        <div className="absolute inset-x-4 bottom-4 z-20 rounded-full bg-white px-4 py-3 text-center text-[12px] font-black text-[#111] shadow-lg">{metadata.ctaText}</div>
      </div>
    );
  } else if (shellTemplateId === "instagram_reels") {
    shell = (
      <div className="relative mx-auto w-full max-w-[300px] overflow-hidden rounded-[36px] border-[10px] border-[#111] bg-black shadow-2xl text-white">
        <div className="relative">{asset}</div>
        <div className="absolute right-3 top-[34%] z-20 grid gap-3 text-center text-[9px] font-semibold">
          {["♥", "💬", "↗", "⋯"].map((item, index) => <span key={`${item}-${index}`} className="grid h-10 w-10 place-items-center rounded-full bg-black/38 backdrop-blur">{item}</span>)}
        </div>
        <div className="absolute bottom-4 left-4 right-16 z-20 space-y-1 text-[11px]">
          <p className="font-bold">@{metadata.username}</p>
          <p className="font-semibold">{metadata.headline}</p>
          <p className="text-white/80">{metadata.description}</p>
        </div>
      </div>
    );
  } else if (shellTemplateId === "facebook_marketplace") {
    shell = (
      <div className="mx-auto w-full max-w-[440px] overflow-hidden rounded-2xl border border-[#d8dce6] bg-[#f3f4f8] shadow-xl">
        <div className="border-b border-[#dde2ec] bg-white px-4 py-3">
          <p className="text-sm font-black">Marketplace</p>
          <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-[#666]">
            <span className="rounded-full bg-[#eef1f7] px-2 py-1 text-center">For you</span>
            <span className="rounded-full bg-[#eef1f7] px-2 py-1 text-center">Local</span>
            <span className="rounded-full bg-[#eef1f7] px-2 py-1 text-center">Categories</span>
          </div>
        </div>
        <div className="p-4">
          <div className="overflow-hidden rounded-2xl border border-[#d8dce6] bg-white">
            <div className="p-2">{asset}</div>
            <div className="space-y-1 px-3 pb-3 pt-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-black">{metadata.price}</p>
                <span className="rounded-full bg-[#eef7ff] px-2 py-1 text-[10px] font-bold text-[#2550a8]">{metadata.sponsorLabel}</span>
              </div>
              <p className="text-sm font-semibold">{metadata.headline}</p>
              <p className="text-xs text-[#666]">{metadata.description}</p>
              <button type="button" className="mt-2 rounded-full bg-[#2550a8] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
            </div>
          </div>
        </div>
      </div>
    );
  } else if (shellTemplateId === "facebook_right_column") {
    shell = (
      <div className="mx-auto w-full max-w-[560px] overflow-hidden rounded-2xl border border-[#dde2ec] bg-white shadow-xl">
        <div className="border-b border-[#eff2f7] px-4 py-3 text-[12px] font-semibold text-[#666]">facebook.com / Sponsored</div>
        <div className="p-4">{framedCreative}</div>
        <div className="border-t border-[#eff2f7] px-4 py-3">
          <p className="text-[13px] font-black">{metadata.headline}</p>
          <p className="mt-1 text-[12px] text-[#666]">{metadata.description}</p>
          <button type="button" className="mt-3 rounded-md bg-[#2550a8] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
        </div>
      </div>
    );
  } else if (shellTemplateId === "tiktok_infeed") {
    shell = (
      <div className="relative mx-auto w-[286px] overflow-hidden rounded-[42px] border-[10px] border-[#111] bg-[#0c0c0f] text-white shadow-2xl">
        <div className="pointer-events-none absolute left-1/2 top-2 z-20 h-6 w-28 -translate-x-1/2 rounded-full bg-black/90" />
        <div className="relative overflow-hidden rounded-[32px]">{asset}</div>
        <div className="absolute left-4 top-6 z-20 rounded-full bg-black/40 px-2.5 py-1 text-[10px] font-bold backdrop-blur">{metadata.sponsorLabel}</div>
        <div className="absolute right-3 top-[28%] z-20 grid gap-3 text-center text-[9px] font-semibold">
          {["♥", "💬", "↗", "⋯"].map((item, index) => <span key={`${item}-${index}`} className="grid h-9 w-9 place-items-center rounded-full bg-black/38 backdrop-blur">{item}</span>)}
        </div>
        <div className="absolute bottom-4 left-4 right-16 z-20 space-y-1 text-[11px]">
          <p className="font-bold">@{metadata.username}</p>
          <p className="font-semibold">{metadata.headline}</p>
          <p className="text-white/80">{metadata.description}</p>
          <button type="button" className="mt-2 rounded-full bg-white px-4 py-2 text-[11px] font-black text-[#111]">{metadata.ctaText}</button>
        </div>
      </div>
    );
  } else if (shellTemplateId === "tiktok_topview") {
    shell = (
      <div className="relative mx-auto w-[286px] overflow-hidden rounded-[42px] border-[10px] border-[#111] bg-[#0c0c0f] text-white shadow-2xl">
        <div className="pointer-events-none absolute left-1/2 top-2 z-20 h-6 w-28 -translate-x-1/2 rounded-full bg-black/90" />
        <div className="relative overflow-hidden rounded-[32px]">{asset}</div>
        <div className="absolute left-4 top-6 z-20 rounded-full bg-white px-3 py-1 text-[10px] font-black text-[#111]">TopView</div>
        <div className="absolute inset-x-4 bottom-24 z-20 rounded-2xl bg-black/45 px-3 py-3 backdrop-blur">
          <p className="text-[10px] font-semibold uppercase tracking-normal text-white/70">{metadata.sponsorLabel}</p>
          <p className="mt-1 text-[13px] font-black">{metadata.headline}</p>
          <p className="mt-1 text-[11px] text-white/80">{metadata.description}</p>
        </div>
        <div className="absolute right-3 top-[28%] z-20 grid gap-3 text-center text-[9px] font-semibold">
          {["♥", "💬", "↗"].map((item, index) => <span key={`${item}-${index}`} className="grid h-9 w-9 place-items-center rounded-full bg-black/38 backdrop-blur">{item}</span>)}
        </div>
        <button type="button" className="absolute bottom-4 left-4 z-20 rounded-full bg-[#f8d948] px-4 py-2 text-[11px] font-black text-[#111]">{metadata.ctaText}</button>
      </div>
    );
  } else if (shellTemplateId === "tiktok_branded_content") {
    shell = (
      <div className="relative mx-auto w-[286px] overflow-hidden rounded-[42px] border-[10px] border-[#111] bg-[#0c0c0f] text-white shadow-2xl">
        <div className="relative overflow-hidden rounded-[32px]">{asset}</div>
        <div className="absolute left-4 top-6 z-20 rounded-full bg-white/20 px-3 py-1 text-[10px] font-bold backdrop-blur">Branded content</div>
        <div className="absolute right-3 top-[32%] z-20 grid gap-3 text-center text-[9px] font-semibold">
          {["♥", "💬", "↗"].map((item) => <span key={item} className="grid h-9 w-9 place-items-center rounded-full bg-black/38 backdrop-blur">{item}</span>)}
        </div>
        <div className="absolute bottom-4 left-4 right-16 z-20 space-y-1 text-[11px]">
          <p className="font-bold">@{metadata.username}</p>
          <p className="text-white/80">{metadata.description}</p>
          <button type="button" className="mt-2 rounded-full bg-white px-4 py-2 text-[11px] font-black text-[#111]">{metadata.ctaText}</button>
        </div>
      </div>
    );
  } else if (shellTemplateId === "youtube_instream") {
    shell = (
      <div className="mx-auto w-full max-w-[620px] overflow-hidden rounded-2xl bg-[#0f0f0f] text-white shadow-xl">
        <div className="relative">{asset}<div className="absolute left-3 top-3 rounded bg-black/70 px-2 py-1 text-[10px] font-bold">Ad 0:06</div><div className="absolute bottom-0 left-0 right-0 h-1 bg-white/15"><div className="h-1 w-1/3 bg-[#ff0033]" /></div></div>
        <div className="grid gap-2 px-4 py-3">
          <div className="flex items-center justify-between"><div><p className="font-bold">{metadata.brandName}</p><p className="text-[11px] text-white/70">{metadata.sponsorLabel}</p></div><button type="button" className="rounded-full bg-white px-4 py-2 text-[11px] font-black text-[#111]">{metadata.ctaText}</button></div>
          <p className="text-[13px] font-semibold">{metadata.headline}</p>
          <p className="text-[12px] text-white/75">{metadata.description}</p>
        </div>
      </div>
    );
  } else if (shellTemplateId === "youtube_shorts") {
    shell = (
      <div className="relative mx-auto w-[286px] overflow-hidden rounded-[42px] border-[10px] border-[#111] bg-[#0c0c0f] text-white shadow-2xl">
        <div className="relative overflow-hidden rounded-[32px]">{asset}</div>
        <div className="absolute right-3 top-[34%] z-20 grid gap-3 text-center text-[9px] font-semibold">
          {["♥", "💬", "➤"].map((item) => <span key={item} className="grid h-10 w-10 place-items-center rounded-full bg-black/38 backdrop-blur">{item}</span>)}
        </div>
        <div className="absolute bottom-4 left-4 right-16 z-20 space-y-1 text-[11px]">
          <p className="font-bold">{metadata.brandName}</p>
          <p className="font-semibold">{metadata.headline}</p>
          <p className="text-white/80">{metadata.description}</p>
        </div>
      </div>
    );
  } else if (!shell && (
    shellTemplateId === "wide_landscape" ||
    shellTemplateId === "linkedin_single_image_1200x628" ||
    shellTemplateId === "linkedin_single_image_1080x1080" ||
    shellTemplateId === "linkedin_sponsored_content"
  )) {
    shell = (
      <div className="mx-auto w-full max-w-[560px] overflow-hidden rounded-2xl border border-[#d8dce6] bg-white shadow-xl">
        <div className="flex items-center gap-3 border-b border-[#eff2f7] px-4 py-3">
          <div className="grid h-9 w-9 place-items-center rounded-full bg-[#0a66c2] text-xs font-black text-white">in</div>
          <div><p className="text-[12px] font-bold">{metadata.brandName}</p><p className="text-[10px] text-[#666]">{metadata.sponsorLabel}</p></div>
        </div>
        <div className="p-3">{framedCreative}</div>
        <div className="space-y-2 px-4 pb-4">
          <p className="text-[15px] font-black">{metadata.headline}</p>
          <p className="text-[12px] text-[#555]">{metadata.description}</p>
          <div className="flex items-center justify-between rounded-xl border border-[#e4e7ec] bg-[#f8fafc] px-3 py-3">
            <div><p className="text-[11px] font-semibold text-[#667085]">{metadata.brandName}</p><p className="text-[13px] font-semibold">{metadata.ctaText}</p></div>
            <button type="button" className="rounded-md bg-[#0a66c2] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
          </div>
        </div>
      </div>
    );
  } else if (shellTemplateId === "snap_top_snap") {
    shell = (
      <div className="relative mx-auto w-[236px] overflow-hidden rounded-[36px] border-[9px] border-[#111] bg-[#fffc00] shadow-2xl">
        <div className="pointer-events-none absolute left-1/2 top-2 z-20 h-5 w-24 -translate-x-1/2 rounded-full bg-black/85" />
        <div className="overflow-hidden rounded-[28px]">{asset}</div>
        <div className="absolute left-4 right-4 top-6 z-20 flex justify-between text-[10px] font-bold text-white"><span>{metadata.brandName}</span><span>{metadata.sponsorLabel}</span></div>
        <div className="absolute bottom-4 left-6 right-6 z-20 rounded-full bg-white px-3 py-2 text-center text-[10px] font-black">{metadata.ctaText}</div>
      </div>
    );
  } else if (shellTemplateId === "snap_story_ad") {
    shell = (
      <div className="relative mx-auto w-[236px] overflow-hidden rounded-[36px] border-[9px] border-[#111] bg-[#fffc00] shadow-2xl">
        <div className="absolute inset-x-4 top-3 z-20 flex gap-1">{Array.from({ length: 4 }).map((_, index) => <span key={index} className={["h-1 flex-1 rounded-full", index === 0 ? "bg-white" : "bg-white/35"].join(" ")} />)}</div>
        <div className="overflow-hidden rounded-[28px]">{asset}</div>
        <div className="absolute left-4 right-4 top-6 z-20 flex justify-between text-[10px] font-bold text-white"><span>{metadata.brandName}</span><span>{metadata.sponsorLabel}</span></div>
        <div className="absolute bottom-4 left-6 right-6 z-20 rounded-full bg-white px-3 py-2 text-center text-[10px] font-black">{metadata.ctaText}</div>
      </div>
    );
  } else if (!shell && (shellTemplateId === "google_responsive_landscape" || shellTemplateId === "google_responsive_square" || shellTemplateId === "google_responsive_vertical")) {
    shell = (
      <div className="mx-auto w-full max-w-[660px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-[#eef2f7] bg-[#f7f8fb] px-3 py-2 text-[10px] font-semibold text-[#667085]">
          <span>Google Responsive Display</span>
          <span>Ad</span>
        </div>
        <div className="grid gap-3 p-3 md:grid-cols-[minmax(0,1fr)_220px]">
          <div>{framedCreative}</div>
          <div className="flex flex-col justify-center rounded-lg bg-[#f7f8fb] p-3">
            <p className="text-[10px] font-black uppercase text-[#667085]">Responsive ad preview</p>
            <p className="mt-2 text-sm font-black">{metadata.headline}</p>
            <p className="mt-1 text-xs text-[#555]">{metadata.description}</p>
            <button type="button" className="mt-3 w-fit rounded-md bg-[#1a73e8] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
          </div>
        </div>
      </div>
    );
  } else if (shellTemplateId === "gdn_300x250") {
    shell = (
      <div className="mx-auto w-[300px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-[#eef2f7] bg-[#f7f8fb] px-3 py-2 text-[10px] font-semibold text-[#667085]"><span>Display Ad</span><span>Ad</span></div>
        <div className="p-2">{framedCreative}</div>
      </div>
    );
  } else if (shellTemplateId === "gdn_728x90") {
    shell = (
      <div className="mx-auto w-full max-w-[760px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="p-2">{framedCreative}</div>
      </div>
    );
  } else if (shellTemplateId === "gdn_160x600") {
    shell = (
      <div className="mx-auto w-[184px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="p-2">{framedCreative}</div>
      </div>
    );
  } else if (shellTemplateId === "gdn_320x50") {
    shell = (
      <div className="mx-auto w-[344px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="p-2">{framedCreative}</div>
      </div>
    );
  } else if (shellTemplateId === "gdn_300x600") {
    shell = (
      <div className="mx-auto w-[324px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="p-2">{framedCreative}</div>
      </div>
    );
  } else if (shellTemplateId.startsWith("gdn_")) {
    shell = (
      <div className="mx-auto w-full max-w-[640px] overflow-hidden rounded-xl border border-[#d3d7df] bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-[#eef2f7] bg-[#f7f8fb] px-3 py-2 text-[10px] font-semibold text-[#667085]">
          <span>Google Display Network</span>
          <span>Ad</span>
        </div>
        <div className="p-3">{framedCreative}</div>
      </div>
    );
  } else if (!shell) {
    shell = (
      <div className="mx-auto w-full max-w-[640px] overflow-hidden rounded-2xl border border-[#d8dce6] bg-white shadow-xl">
        <div className="flex items-center gap-2 border-b bg-[#f7f7f7] px-3 py-2"><span className="h-2.5 w-2.5 rounded-full bg-[#ee4d6a]" /><span className="h-2.5 w-2.5 rounded-full bg-[#f0d553]" /><span className="h-2.5 w-2.5 rounded-full bg-[#38b6a6]" /><span className="ml-2 rounded bg-white px-2 py-1 text-[9px] text-[#666]">publisher.example/feature</span></div>
        <div className="grid gap-4 p-4 md:grid-cols-[1fr_260px]">
          <div className="space-y-3">
            <p className="text-[10px] font-black uppercase text-[#0f766e]">Native publisher</p>
            <h3 className="text-xl font-black">{metadata.headline}</h3>
            <p className="text-sm leading-6 text-[#555]">{metadata.description}</p>
            <button type="button" className="rounded-full bg-[#151515] px-4 py-2 text-[11px] font-bold text-white">{metadata.ctaText}</button>
          </div>
          <div>
            <p className="mb-1 text-right text-[9px] font-semibold uppercase text-[#777]">Advertisement</p>
            {framedCreative}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[300px] max-h-[calc(100svh-230px)] flex-col justify-center gap-2 overflow-hidden bg-[#f3f0e8] p-2">
      {shell}
      <div className="mx-auto flex w-full max-w-[540px] justify-between rounded-md bg-[#111] px-3 py-2 text-[10px] text-white">
        <span>{template.placementType} / {placement.width}x{placement.height}</span>
        <span className={warnings.length || carouselWarning ? "text-[#ffcf4a]" : "text-[#7ee1c6]"}>{carouselWarning ? "Carousel assets missing" : warnings.length ? `${warnings.length} safe-zone warning` : "Safe zone clear"}</span>
      </div>
    </div>
  );
}

function LocalizePreview({
  originalUrl,
  localizedUrl,
  originalLabel,
  localizedLabel,
  index,
  total,
  onPrevious,
  onNext,
}: {
  originalUrl?: string;
  localizedUrl?: string;
  originalLabel: string;
  localizedLabel: string;
  index: number;
  total: number;
  onPrevious: () => void;
  onNext: () => void;
}) {
  return (
    <div className="bg-[#f3f0e8] p-4">
      <div className="mx-auto w-full max-w-[860px] overflow-hidden rounded-md border border-[#151515]/15 bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-[#151515]/10 bg-[#faf9f5] px-4 py-3">
          <div>
            <p className="text-lg font-semibold">Localization Preview</p>
            <p className="text-xs text-[#666]">Compare original and localized creative side by side</p>
          </div>
          <div className="flex items-center gap-2 text-xs font-semibold text-[#0f766e]">
            <Shield className="h-4 w-4" />
            Show Safe Zones
          </div>
        </div>
        <div className="space-y-4 p-4">
          <div>
            <p className="mb-2 text-xs font-bold uppercase text-[#444]">{localizedLabel}</p>
            <div className="overflow-hidden rounded-md border border-[#151515]/10 bg-[#eef5fb]">
              {localizedUrl ? (
                <img src={localizedUrl} alt="Localized creative" className="block h-auto w-full object-contain" />
              ) : (
                <div className="grid min-h-[220px] place-items-center bg-[linear-gradient(135deg,#edf6ff_0%,#a9d7f7_52%,#f4d28c_100%)]" />
              )}
            </div>
          </div>
          <div className="flex justify-center">
            <span className="grid h-12 w-12 place-items-center rounded-full border border-[#d9e5f3] bg-white text-[#2550a8] shadow-sm">
              <ArrowRight className="h-5 w-5 rotate-90" />
            </span>
          </div>
          <div>
            <p className="mb-2 text-xs font-bold uppercase text-[#444]">{originalLabel}</p>
            <div className="overflow-hidden rounded-md border border-[#151515]/10 bg-[#eef5fb]">
              {originalUrl ? (
                <img src={originalUrl} alt="Original creative" className="block h-auto w-full object-contain" />
              ) : (
                <div className="grid min-h-[220px] place-items-center bg-[linear-gradient(135deg,#edf6ff_0%,#a9d7f7_52%,#f4d28c_100%)]" />
              )}
            </div>
          </div>
        </div>
        {total > 1 && (
          <div className="flex items-center justify-between border-t border-[#151515]/10 bg-white px-3 py-2 text-xs font-semibold">
            <button type="button" onClick={onPrevious} className="flex items-center gap-1 rounded-md border border-[#151515]/10 px-3 py-1.5 hover:border-[#0f766e]"><ChevronLeft className="h-4 w-4" />Previous</button>
            <span>{index + 1} / {total}</span>
            <button type="button" onClick={onNext} className="flex items-center gap-1 rounded-md border border-[#151515]/10 px-3 py-1.5 hover:border-[#0f766e]">Next<ChevronRight className="h-4 w-4" /></button>
          </div>
        )}
      </div>
    </div>
  );
}

export function AdaptDashboard() {
  const supabase = useMemo(() => getSupabaseBrowser(), []);
  const supabaseConfigured = hasSupabaseBrowserConfig();
  const grouped = useMemo(() => platformOrder.map((p) => [p, placements.filter((x) => x.platform === p)] as const), []);
  const [mode, setMode] = useState<Mode>("adapt");
  const [selectedPlacementIds, setSelectedPlacementIds] = useState<string[]>([]);
  const [activePlacementId, setActivePlacementId] = useState("story-image");
  const [activePreviewVariantId, setActivePreviewVariantId] = useState("instagram-story-mobile");
  const [creativeModesByPlacement, setCreativeModesByPlacement] = useState<Record<string, CreativeMode>>({});
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>([]);
  const [selectedFormat, setSelectedFormat] = useState("PNG");
  const [files, setFiles] = useState<File[]>([]);
  const [credits, setCredits] = useState(240);
  const [userId, setUserId] = useState(() => typeof window === "undefined" ? "guest@adaptif.ai" : window.localStorage.getItem("adaptifai:user") || "guest@adaptif.ai");
  const [authUser, setAuthUser] = useState<SupabaseUser | null>(null);
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authMode, setAuthMode] = useState<AuthMode>("sign-in");
  const [sessionToken, setSessionToken] = useState<string | null>(null);
  const [authReady, setAuthReady] = useState(!supabaseConfigured);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const [authPending, setAuthPending] = useState(false);
  const [showCreditStore, setShowCreditStore] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [adminUsers, setAdminUsers] = useState<AdminUser[]>([]);
  const [adminTargetEmail, setAdminTargetEmail] = useState("");
  const [adminAmount, setAdminAmount] = useState(100);
  const [adminAction, setAdminAction] = useState<"add" | "deduct">("add");
  const [adminStatus, setAdminStatus] = useState<string | null>(null);
  const [isAdminUpdating, setIsAdminUpdating] = useState(false);
  const [preserveBold, setPreserveBold] = useState(true);
  const [maskCleanup, setMaskCleanup] = useState(true);
  const [fitBounds, setFitBounds] = useState(true);
  const [copy, setCopy] = useState(sampleCopy.adapt);
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const [opacity, setOpacity] = useState(18);
  const [scale, setScale] = useState(100);
  const [textColor, setTextColor] = useState("#111111");
  const [fontSizeScale, setFontSizeScale] = useState(100);
  const [textItalic, setTextItalic] = useState(false);
  const [textUnderline, setTextUnderline] = useState(false);
  const [textStrike, setTextStrike] = useState(false);
  const [isDraggingPreview, setIsDraggingPreview] = useState(false);
  const previewDragRef = useRef<{ mouseX: number; mouseY: number; startX: number; startY: number } | null>(null);
  const [fit, setFit] = useState<FitMode>("cover");
  const [customWidth, setCustomWidth] = useState(1200);
  const [customHeight, setCustomHeight] = useState(800);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [activeOutputIndex, setActiveOutputIndex] = useState(0);
  const [activeResizeSource, setActiveResizeSource] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [isApplyingEdit, setIsApplyingEdit] = useState(false);
  const [editStatus, setEditStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activePlacement = useMemo(() => {
    const placement = placements.find((p) => p.id === activePlacementId) ?? placements[0];
    if (placement.id !== "custom-display") return placement;
    return {
      ...placement,
      width: customWidth,
      height: customHeight,
      ratio: `${customWidth}x${customHeight}`,
    };
  }, [activePlacementId, customHeight, customWidth]);
  const activePreviewVariants = previewVariantsForPlacement(activePlacement.id);
  const activePreviewVariant = activePreviewVariants.find((variant) => variant.id === activePreviewVariantId) ?? activePreviewVariants[0];
  const currentUserEmail = authUser?.email ?? userId;
  const activeCreativeMode = activePlacement.supportsCarousel ? (creativeModesByPlacement[activePlacement.id] ?? "single") : "single";
  const isAdmin = currentUserEmail.trim().toLowerCase() === (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "tolgar@sasmaz.digital").trim().toLowerCase();
  const filePreviewUrls = useMemo(() => Object.fromEntries(files.map((file) => [file.name, URL.createObjectURL(file)])), [files]);
  const activeOutput = result?.outputs[activeOutputIndex] ?? null;
  const activeOriginalUrl = activeOutput ? filePreviewUrls[activeOutput.source_name] : undefined;
  const resizeSourceNames = useMemo(() => Array.from(new Set((result?.outputs ?? []).map((output) => output.source_name))).filter(Boolean), [result]);
  const effectiveResizeSource = activeResizeSource || resizeSourceNames[0] || "";
  const activeResizeOutput = result?.outputs.find((output) => output.placement_id === activePlacementId && output.source_name === effectiveResizeSource)
    ?? result?.outputs.find((output) => output.source_name === effectiveResizeSource)
    ?? activeOutput;
  const carouselPreviewAssets = useMemo(() => {
    if (!result) return [];
    const placementOutputs = (result.outputs ?? [])
      .filter((output) => output.placement_id === activePlacementId)
      .map((output) => output.download_url)
      .filter(Boolean);
    if (placementOutputs.length > 1) return placementOutputs.slice(0, 6);
    return files.map((file) => filePreviewUrls[file.name]).filter(Boolean).slice(0, 6);
  }, [activePlacementId, filePreviewUrls, files, result]);
  const derivedPreviewMetadata = useMemo(
    () => {
      const base = derivePreviewMetadata(activePlacement, activeResizeOutput?.translated_text ?? copy, activeResizeOutput?.source_name, currentUserEmail);
      const creativeMode = activePlacement.supportsCarousel ? (creativeModesByPlacement[activePlacement.id] ?? "single") : "single";
      const orderedOutputs = (result?.outputs ?? []).filter((output) => output.placement_id === activePlacementId);
      const unusedAssets = creativeMode === "single"
        ? carouselPreviewAssets.slice(1)
        : [];
      const carouselActivationSource: PreviewMetadata["carouselActivationSource"] = !activePlacement.supportsCarousel
        ? "forced_single"
        : creativeMode === "carousel" && carouselPreviewAssets.length < 2
          ? "invalid_missing_assets"
          : "user_selected";
      return {
        ...base,
        carouselAssets: carouselPreviewAssets,
        carouselAssetsProvided: carouselPreviewAssets.length > 1,
        activeSlideIndex: creativeMode === "carousel" && activeResizeOutput
          ? Math.max(0, orderedOutputs.findIndex((output) => output.filename === activeResizeOutput.filename))
          : 0,
        creativeMode,
        carouselActivationSource,
        unusedAssets,
      };
    },
    [activePlacement, activePlacementId, activeResizeOutput, carouselPreviewAssets, copy, creativeModesByPlacement, currentUserEmail, result?.outputs],
  );
  const previewMetadata = derivedPreviewMetadata;
  const estimatedRunCredits = files.length === 0 ? 0 : mode === "adapt"
    ? estimateLocalizeCredits({ fileCount: files.length, languageCount: selectedLanguages.length, outputFormat: selectedFormat })
    : estimateResizeCredits({ fileCount: files.length, dimensionCount: selectedPlacementIds.length, outputFormat: selectedFormat });
  const editCredits = estimateEditCredits(mode);
  const generatedLocalizeCount = files.length * selectedLanguages.length;
  const outputFormatCost = selectedFormat.toLowerCase() === "pdf";
  const outputUnitCount = files.length;
  const receiptLines: ReceiptLine[] = result
    ? [{ label: "Modify edit", formula: `1 x ${formatCreditText(editCredits)}`, credits: editCredits }]
    : mode === "adapt"
      ? [
        { label: "Files", formula: `${files.length} x ${formatCreditText(creditPricing.localizeImage)}`, credits: files.length * creditPricing.localizeImage },
        { label: "Languages", formula: `${generatedLocalizeCount} x ${formatCreditText(creditPricing.localizeLanguagePerGeneratedImage)}`, credits: generatedLocalizeCount * creditPricing.localizeLanguagePerGeneratedImage },
        { label: "Output", formula: outputFormatCost ? `${outputUnitCount} x ${formatCreditText(creditPricing.localizePdfOutput)}` : `${outputUnitCount} x ${formatCreditText(creditPricing.localizeOutputFormat)}`, credits: files.length === 0 ? 0 : outputFormatCost ? outputUnitCount * creditPricing.localizePdfOutput : outputUnitCount * creditPricing.localizeOutputFormat },
      ]
      : [
        { label: "Files", formula: `${files.length} x ${formatCreditText(creditPricing.resizeImage)}`, credits: files.length * creditPricing.resizeImage },
        { label: "Placements", formula: `${selectedPlacementIds.length} x ${formatCreditText(creditPricing.resizeDimension)}`, credits: files.length === 0 ? 0 : selectedPlacementIds.length * creditPricing.resizeDimension },
        { label: "Output", formula: outputFormatCost ? `${outputUnitCount} x ${formatCreditText(creditPricing.resizePdfOutput)}` : `${outputUnitCount} x ${formatCreditText(creditPricing.resizeOutputFormat)}`, credits: files.length === 0 ? 0 : outputFormatCost ? outputUnitCount * creditPricing.resizePdfOutput : outputUnitCount * creditPricing.resizeOutputFormat },
      ];
  const actionCredits = result ? editCredits : receiptLines.reduce((sum, line) => sum + line.credits, 0);
  const remainingAfterAction = credits - actionCredits;
  const canRun = files.length > 0 && (mode === "adapt" || selectedPlacementIds.length > 0) && (mode !== "adapt" || selectedLanguages.length > 0) && credits >= estimatedRunCredits;
  const canApplyCurrentEdit = Boolean(result && activeOutput) && credits >= editCredits;
  const previewDevice = activePlacement.device === "desktop" ? "desktop" : "mobile";

  useEffect(() => () => {
    Object.values(filePreviewUrls).forEach((url) => URL.revokeObjectURL(url));
  }, [filePreviewUrls]);

  useEffect(() => {
    if (!supabase) return;
    let mounted = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setAuthUser(data.session?.user ?? null);
      setSessionToken(data.session?.access_token ?? null);
      if (data.session?.user.email) setUserId(data.session.user.email);
      setAuthReady(true);
    });
    const { data } = supabase.auth.onAuthStateChange((event, session) => {
      setAuthUser(session?.user ?? null);
      setSessionToken(session?.access_token ?? null);
      if (session?.user.email) setUserId(session.user.email);
      if (event === "PASSWORD_RECOVERY") {
        setAuthMode("reset-password");
        setAuthPassword("");
        setAuthNotice("Enter a new password to finish account recovery.");
        setAuthError(null);
      }
    });
    return () => { mounted = false; data.subscription.unsubscribe(); };
  }, [supabase]);

  useEffect(() => {
    if (supabaseConfigured && !sessionToken) return;
    window.localStorage.setItem("adaptifai:user", currentUserEmail);
    fetch(`/api/credits?user_id=${encodeURIComponent(currentUserEmail)}`, { headers: sessionToken ? { authorization: `Bearer ${sessionToken}` } : undefined })
      .then((r) => r.json()).then((p) => setCredits(Number(p.credits ?? 0))).catch(() => undefined);
  }, [currentUserEmail, sessionToken, supabaseConfigured]);

  const togglePlacement = (id: string) => {
    setSelectedPlacementIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
    setActivePlacementId(id);
  };

  const updateCreativeMode = (placementId: string, mode: CreativeMode) => {
    setCreativeModesByPlacement((current) => ({ ...current, [placementId]: mode }));
  };

  const removeFile = (name: string, size: number) => {
    setFiles((current) => current.filter((file) => !(file.name === name && file.size === size)));
    setResult(null);
    setActiveOutputIndex(0);
    setActiveResizeSource("");
    setError(null);
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    setShowAdminPanel(false);
    setCopy(sampleCopy[next]);
    setResult(null);
    setActiveOutputIndex(0);
    setEditStatus(null);
    setError(null);
  };

  const selectOutput = (index: number) => {
    if (!result?.outputs.length) return;
    const nextIndex = (index + result.outputs.length) % result.outputs.length;
    const output = result.outputs[nextIndex];
    setActiveOutputIndex(nextIndex);
    if (output.placement_id) setActivePlacementId(output.placement_id);
    if (output.source_name) setActiveResizeSource(output.source_name);
    if (mode === "adapt" && output.translated_text) setCopy(output.translated_text);
  };

  const runProcess = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canRun) {
      setError(files.length === 0 ? "Upload at least one creative before running." : `This run needs ${estimatedRunCredits} credits. You have ${credits}.`);
      return;
    }
    setIsRunning(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("user_id", currentUserEmail);
    formData.append("mode", mode === "adapt" ? "localize" : "resize");
    formData.append("target_languages", mode === "adapt" ? selectedLanguages.join(",") : "EN");
    formData.append("output_format", selectedFormat);
    formData.append("placements", mode === "adapt" ? "custom-display" : selectedPlacementIds.join(","));
    if (mode === "resize") {
      const creativeModes = Object.fromEntries(
        selectedPlacementIds.map((placementId) => [
          placementId,
          (placements.find((item) => item.id === placementId)?.supportsCarousel
            ? creativeModesByPlacement[placementId] ?? "single"
            : "single"),
        ]),
      );
      formData.append("creative_modes", JSON.stringify(creativeModes));
    }
    if (mode === "resize" && selectedPlacementIds.includes("custom-display")) {
      formData.append("custom_width", String(customWidth));
      formData.append("custom_height", String(customHeight));
    }
    try {
      const response = await fetch("/api/adapt", { method: "POST", body: formData, headers: sessionToken ? { authorization: `Bearer ${sessionToken}` } : undefined });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Pipeline failed.");
      setResult(payload);
      if (payload.outputs?.[0]?.placement_id) setActivePlacementId(payload.outputs[0].placement_id);
      if (payload.outputs?.[0]?.source_name) setActiveResizeSource(payload.outputs[0].source_name);
      setActiveOutputIndex(0);
      setCopy(payload.outputs?.[0]?.translated_text ?? "");
      setCredits(Number(payload.credits_remaining ?? credits));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Pipeline failed.");
    } finally {
      setIsRunning(false);
    }
  };

  const applyManualEdit = async () => {
    setIsApplyingEdit(true);
    setEditStatus(null);
    setError(null);
    try {
      const response = await fetch("/api/edit", {
        method: "POST",
        headers: { "content-type": "application/json", ...(sessionToken ? { authorization: `Bearer ${sessionToken}` } : {}) },
        body: JSON.stringify({
          job_id: result?.job_id,
          filename: activeOutput?.filename,
          mode,
          copy,
          x,
          y,
          opacity,
          scale,
          fit,
          preserve_bold: preserveBold,
          mask_cleanup: maskCleanup,
          fit_bounds: fitBounds,
          text_color: textColor,
          font_size_scale: fontSizeScale,
          text_italic: textItalic,
          text_underline: textUnderline,
          text_strike: textStrike,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Unable to apply edit.");
      if (payload.output && result) {
        setResult({
          ...result,
          outputs: result.outputs.map((output) => output.filename === payload.output.filename ? payload.output : output),
        });
      }
      setCredits(Number(payload.credits_remaining ?? credits));
      setEditStatus(`${mode === "adapt" ? "Translation edit" : "Resize edit"} applied. ${editCredits} credits used.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to apply edit.");
    } finally {
      setIsApplyingEdit(false);
    }
  };

  const buyCredits = async (pack: (typeof pricingPacks)[number]["id"] = "starter") => {
    const response = await fetch("/api/stripe/checkout", {
      method: "POST",
      headers: { "content-type": "application/json", ...(sessionToken ? { authorization: `Bearer ${sessionToken}` } : {}) },
      body: JSON.stringify({ pack, user_id: currentUserEmail }),
    });
    const payload = await response.json();
    if (payload.url) globalThis.location.assign(payload.url);
    else setError(payload.error ?? "Stripe Checkout is not configured yet.");
  };

  const submitAuth = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!supabase) return;
    setAuthError(null);
    setAuthNotice(null);
    setAuthPending(true);
    try {
      const email = authEmail.trim().toLowerCase();
      if (authMode === "forgot-password") {
        if (!email || !email.includes("@")) throw new Error("A valid email is required.");
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${globalThis.location.origin}/`,
        });
        if (error) throw error;
        setAuthMode("sign-in");
        setAuthPassword("");
        setAuthNotice("Password reset email sent. Open the link in your email, then set a new password.");
        return;
      }

      if (authMode === "reset-password") {
        if (authPassword.length < 6) throw new Error("Password must be at least 6 characters.");
        const { error } = await supabase.auth.updateUser({ password: authPassword });
        if (error) throw error;
        await supabase.auth.signOut();
        setAuthMode("sign-in");
        setAuthPassword("");
        setAuthNotice("Password updated. Please sign in with your new password.");
        return;
      }

      if (authMode === "sign-up") {
        const response = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ email, password: authPassword }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "Unable to create account.");
      }

      const { data, error } = await supabase.auth.signInWithPassword({ email, password: authPassword });
      if (error) throw error;
      if (data.user?.email) setUserId(data.user.email);
    } catch (caught) {
      setAuthError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setAuthPending(false);
    }
  };

  const adjustCredits = async () => {
    setIsAdminUpdating(true);
    setAdminStatus(null);
    try {
      const response = await fetch("/api/admin/credits", {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${sessionToken}` },
        body: JSON.stringify({ user_id: adminTargetEmail, amount: adminAmount, action: adminAction }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Unable to adjust credits.");
      setAdminStatus(`${payload.user_id} balance: ${payload.credits} credits`);
      if (payload.user_id === currentUserEmail.toLowerCase()) setCredits(Number(payload.credits));
      setAdminUsers((current) => {
        const next = current.filter((user) => user.user_id !== payload.user_id);
        return [{ user_id: payload.user_id, credits: Number(payload.credits), updated_at: new Date().toISOString() }, ...next];
      });
    } catch (caught) {
      setAdminStatus(caught instanceof Error ? caught.message : "Unable to adjust credits.");
    } finally {
      setIsAdminUpdating(false);
    }
  };

  const openAdminPanel = async () => {
    if (!isAdmin) return;
    setShowCreditStore(false);
    setShowAdminPanel(true);
    setAdminStatus(null);
    try {
      const response = await fetch("/api/admin/users", { headers: sessionToken ? { authorization: `Bearer ${sessionToken}` } : undefined });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Unable to load users.");
      setAdminUsers(payload.users ?? []);
    } catch (caught) {
      setAdminStatus(caught instanceof Error ? caught.message : "Unable to load users.");
    }
  };

  if (!authReady) return <main className="grid min-h-screen place-items-center bg-[#faf9f5]"><Loader2 className="h-7 w-7 animate-spin text-[#0f766e]" /></main>;

  if (supabaseConfigured && (!authUser || authMode === "reset-password")) {
    return (
      <LandingPage authMode={authMode} setAuthMode={setAuthMode} authEmail={authEmail} setAuthEmail={setAuthEmail} authPassword={authPassword} setAuthPassword={setAuthPassword} authError={authError} authNotice={authNotice} authPending={authPending} submitAuth={submitAuth} />
    );
  }

  if (showAdminPanel) {
    return (
      <main className="min-h-screen bg-[#faf9f5] text-[#151515]">
        <header className="sticky top-0 z-20 border-b border-[#151515]/10 bg-[#faf9f5]/95 backdrop-blur">
          <div className="mx-auto flex max-w-[1180px] flex-wrap items-center justify-between gap-3 px-4 py-4">
            <Brand />
            <div className="flex items-center gap-3">
              <div className="flex h-10 items-center gap-2 rounded-md border border-[#151515]/15 bg-white px-3 text-sm font-semibold"><Sparkles className="h-4 w-4 text-[#0f766e]" />{credits} credits</div>
              <button type="button" onClick={() => setShowAdminPanel(false)} className="h-10 rounded-md border border-[#151515]/15 bg-white px-4 text-sm font-semibold">Back to workspace</button>
            </div>
          </div>
        </header>
        <section className="mx-auto max-w-[480px] px-4 py-12">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="h-6 w-6 text-[#0f766e]" />
            <div>
              <p className="text-xs font-black uppercase text-[#0f766e]">Admin</p>
              <h1 className="text-2xl font-black">Credit Management</h1>
            </div>
          </div>
          <div className="rounded-md border border-[#151515]/10 bg-white p-6 shadow-sm space-y-4">
            {adminStatus && <p className="rounded-md bg-[#e8f7f1] p-3 text-sm text-[#064e46]">{adminStatus}</p>}
            <label className="block text-sm font-semibold">
              User email
              <input
                className="mt-1 h-11 w-full rounded-md border border-[#151515]/10 bg-[#faf9f5] px-3 outline-none focus:border-[#0f766e]"
                type="email"
                placeholder="user@example.com"
                value={adminTargetEmail}
                onChange={(e) => setAdminTargetEmail(e.target.value)}
              />
            </label>
            <label className="block text-sm font-semibold">
              Credits to add
              <input
                className="mt-1 h-11 w-full rounded-md border border-[#151515]/10 bg-[#faf9f5] px-3 outline-none focus:border-[#0f766e]"
                type="number"
                min="1"
                value={adminAmount}
                onChange={(e) => setAdminAmount(Number(e.target.value))}
              />
            </label>
            <button
              type="button"
              onClick={adjustCredits}
              disabled={isAdminUpdating || !adminTargetEmail}
              className="flex h-11 w-full items-center justify-center gap-2 rounded-md bg-[#0f766e] text-sm font-semibold text-white disabled:bg-[#d6d0c4]"
            >
              {isAdminUpdating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Shield className="h-4 w-4" />}
              Add credits
            </button>
          </div>
        </section>
      </main>
    );
  }

  if (showCreditStore) {
    return (
      <main className="min-h-screen bg-[#faf9f5] text-[#151515]">
        <header className="sticky top-0 z-20 border-b border-[#151515]/10 bg-[#faf9f5]/95 backdrop-blur">
          <div className="mx-auto flex max-w-[1180px] flex-wrap items-center justify-between gap-3 px-4 py-4">
            <Brand />
            <div className="flex items-center gap-3">
              <div className="flex h-10 items-center gap-2 rounded-md border border-[#151515]/15 bg-white px-3 text-sm font-semibold"><Sparkles className="h-4 w-4 text-[#0f766e]" />{credits} credits</div>
              <button type="button" onClick={() => setShowCreditStore(false)} className="h-10 rounded-md border border-[#151515]/15 bg-white px-4 text-sm font-semibold">Back to workspace</button>
            </div>
          </div>
        </header>
        <section className="mx-auto max-w-[1180px] px-4 py-10">
          <p className="text-xs font-black uppercase text-[#0f766e]">Buy credits</p>
          <h1 className="mt-2 text-4xl font-black">Choose a credit pack</h1>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {pricingPacks.map((pack) => (
              <div key={pack.id} className="rounded-md border border-[#151515]/10 bg-white p-5 shadow-sm">
                <p className="text-lg font-black">{pack.name}</p>
                <p className="mt-4 text-4xl font-black">{pack.price}</p>
                <p className="mt-2 text-sm font-semibold text-[#0f766e]">{pack.credits}</p>
                <p className="mt-4 min-h-12 text-sm leading-6 text-[#555]">{pack.body}</p>
                <button type="button" onClick={() => buyCredits(pack.id)} className="mt-5 flex h-11 w-full items-center justify-center gap-2 rounded-md bg-[#151515] text-sm font-semibold text-white"><CreditCard className="h-4 w-4" />Buy {pack.name}</button>
              </div>
            ))}
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[#faf9f5] text-[#151515]">
      <header className="sticky top-0 z-20 border-b border-[#151515]/10 bg-[#faf9f5]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-3">
          <Brand />
          <div className="flex items-center gap-1 rounded-md bg-[#f1eee6] p-1">
            {(["adapt", "resize"] as const).map((item) => (
              <button key={item} type="button" onClick={() => switchMode(item)} className={["h-9 rounded px-4 text-sm font-semibold transition", mode === item ? "bg-[#151515] text-white" : "text-[#555] hover:bg-white"].join(" ")}>{item === "adapt" ? "Localize" : "Resize"}</button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <div className="flex h-10 items-center gap-2 rounded-md border border-[#151515]/15 bg-white px-3 text-sm font-semibold"><Sparkles className="h-4 w-4 text-[#0f766e]" />{credits} credits</div>
            <button type="button" onClick={() => setShowCreditStore(true)} className="flex h-10 items-center gap-2 rounded-md bg-[#151515] px-4 text-sm font-semibold text-white"><CreditCard className="h-4 w-4" />Buy credits</button>
            {isAdmin && (
              <button type="button" onClick={openAdminPanel} title="Admin panel" className="grid h-10 w-10 place-items-center rounded-md border border-[#0f766e] bg-[#e8f7f1] hover:bg-[#c7efe4]" aria-label="Admin panel">
                <Settings2 className="h-4 w-4 text-[#0f766e]" />
              </button>
            )}
            <div className="flex h-10 items-center gap-2 rounded-md border border-[#151515]/15 bg-white px-3 text-sm font-semibold"><User className="h-4 w-4 text-[#0f766e]" /><span className="hidden max-w-[190px] truncate md:block">{currentUserEmail}</span></div>
            {supabaseConfigured && <button type="button" onClick={() => supabase?.auth.signOut()} className="grid h-10 w-10 place-items-center rounded-md border border-[#151515]/15 bg-white" aria-label="Sign out"><LogOut className="h-4 w-4 text-[#0f766e]" /></button>}
          </div>
        </div>
      </header>

      <form onSubmit={runProcess} className="mx-auto grid max-w-[1440px] gap-3 px-3 py-4 lg:grid-cols-[290px_minmax(0,1fr)_280px] 2xl:grid-cols-[320px_minmax(0,1fr)_300px]">
        <aside className="space-y-4">
          <section className="rounded-md border border-[#151515]/10 bg-white p-4">
            <p className="text-xs font-semibold uppercase text-[#0f766e]">{mode === "adapt" ? "Localize workspace" : "Resize workspace"}</p>
            <h1 className="text-xl font-semibold">{mode === "adapt" ? "Translate and restore" : "Resize placements"}</h1>
            <p className="mt-2 text-xs text-[#666]">{result ? "Modify the generated result, then apply a paid edit pass from the action panel." : mode === "adapt" ? "Upload one or more creatives, choose target languages and export format." : "Upload a creative, then choose platform dimensions to resize."}</p>
          </section>

          {result ? (
            <section className="rounded-md border border-[#151515]/10 bg-white p-4">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="font-semibold">{mode === "adapt" ? "Text Editor" : "Resize Controls"}</h2>
                {mode === "adapt" ? <Type className="h-4 w-4 text-[#0f766e]" /> : <Frame className="h-4 w-4 text-[#0f766e]" />}
              </div>

              {mode === "adapt" ? (
                <>
                  {/* Original text (read-only) */}
                  {activeOutput?.extracted_blocks && activeOutput.extracted_blocks.some((b) => b.translate) && (
                    <div className="mb-3">
                      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Original ({activeOutput.source_language ?? "Source"})</p>
                      <div className="rounded-md border border-[#151515]/10 bg-[#f6f1e7] p-3 text-sm text-[#666]">
                        {activeOutput.extracted_blocks.filter((b) => b.translate).map((b, i) => (
                          <p key={i} className="leading-snug">{b.text}</p>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Translated text (editable) */}
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Translated ({activeOutput?.language ?? "Target"})</p>
                  <textarea
                    className="min-h-[88px] w-full resize-none rounded-md border border-[#151515]/10 bg-[#faf9f5] p-3 text-sm outline-none focus:border-[#0f766e]"
                    value={copy}
                    onChange={(e) => setCopy(e.target.value)}
                    placeholder="Translated text will appear here after localization runs."
                    aria-label="Manual translation override"
                  />

                  {/* Typography */}
                  <div className="mt-4">
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Typography</p>
                    <div className="flex items-center gap-1">
                      <button type="button" title="Bold" onClick={() => setPreserveBold((v) => !v)} className={["h-8 w-8 rounded-md border text-sm font-black", preserveBold ? "border-[#0f766e] bg-[#e8f7f1] text-[#064e46]" : "border-[#151515]/10 bg-[#faf9f5] text-[#555]"].join(" ")}>B</button>
                      <button type="button" title="Italic" onClick={() => setTextItalic((v) => !v)} className={["h-8 w-8 rounded-md border text-sm italic font-semibold", textItalic ? "border-[#0f766e] bg-[#e8f7f1] text-[#064e46]" : "border-[#151515]/10 bg-[#faf9f5] text-[#555]"].join(" ")}>I</button>
                      <button type="button" title="Underline" onClick={() => setTextUnderline((v) => !v)} className={["h-8 w-8 rounded-md border text-sm font-semibold underline", textUnderline ? "border-[#0f766e] bg-[#e8f7f1] text-[#064e46]" : "border-[#151515]/10 bg-[#faf9f5] text-[#555]"].join(" ")}>U</button>
                      <button type="button" title="Strikethrough" onClick={() => setTextStrike((v) => !v)} className={["h-8 w-8 rounded-md border text-sm font-semibold line-through", textStrike ? "border-[#0f766e] bg-[#e8f7f1] text-[#064e46]" : "border-[#151515]/10 bg-[#faf9f5] text-[#555]"].join(" ")}>S</button>
                      <label title="Text color" className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border border-[#151515]/10 bg-[#faf9f5]">
                        <input type="color" className="sr-only" value={textColor} onChange={(e) => setTextColor(e.target.value)} />
                        <span className="h-4 w-4 rounded-sm border border-[#151515]/15 shadow-sm" style={{ background: textColor }} />
                      </label>
                      <div className="ml-auto flex items-center gap-1">
                        <button type="button" onClick={() => setFontSizeScale((v) => Math.max(60, v - 10))} className="h-8 w-8 rounded-md border border-[#151515]/10 bg-[#faf9f5] text-base font-bold text-[#555]">−</button>
                        <span className="w-10 text-center text-[11px] font-semibold text-[#555]">{fontSizeScale}%</span>
                        <button type="button" onClick={() => setFontSizeScale((v) => Math.min(180, v + 10))} className="h-8 w-8 rounded-md border border-[#151515]/10 bg-[#faf9f5] text-base font-bold text-[#555]">+</button>
                      </div>
                    </div>
                  </div>

                  {/* Text position */}
                  <div className="mt-4">
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Text position</p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="w-3 shrink-0 text-[11px] font-bold text-[#555]">X</span>
                        <input type="range" min="-24" max="24" value={x} onChange={(e) => setX(Number(e.target.value))} className="flex-1 accent-[#0f766e]" />
                        <span className="w-9 text-right text-[11px] font-semibold tabular-nums text-[#555]">{x > 0 ? `+${x}` : x}px</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-3 shrink-0 text-[11px] font-bold text-[#555]">Y</span>
                        <input type="range" min="-24" max="24" value={y} onChange={(e) => setY(Number(e.target.value))} className="flex-1 accent-[#0f766e]" />
                        <span className="w-9 text-right text-[11px] font-semibold tabular-nums text-[#555]">{y > 0 ? `+${y}` : y}px</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  {/* Frame mode */}
                  <div>
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Frame mode</p>
                    <div className="grid grid-cols-3 gap-1 rounded-md bg-[#f1eee6] p-1">
                      {(["contain", "cover", "fill"] as const).map((item) => (
                        <button key={item} type="button" onClick={() => setFit(item)} className={["h-9 rounded text-xs font-semibold capitalize transition", fit === item ? "bg-[#151515] text-white" : "text-[#555] hover:bg-white"].join(" ")}>{item}</button>
                      ))}
                    </div>
                  </div>

                  {/* Scale */}
                  <div className="mt-4">
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Content scale</p>
                    <div className="flex items-center gap-2">
                      <input type="range" min="70" max="140" value={scale} onChange={(e) => setScale(Number(e.target.value))} className="flex-1 accent-[#0f766e]" />
                      <span className="w-9 text-right text-[11px] font-semibold tabular-nums text-[#555]">{scale}%</span>
                    </div>
                  </div>

                  {/* Content position */}
                  <div className="mt-4">
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[#999]">Content position</p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="w-3 shrink-0 text-[11px] font-bold text-[#555]">X</span>
                        <input type="range" min="-24" max="24" value={x} onChange={(e) => setX(Number(e.target.value))} className="flex-1 accent-[#0f766e]" />
                        <span className="w-9 text-right text-[11px] font-semibold tabular-nums text-[#555]">{x > 0 ? `+${x}` : x}px</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-3 shrink-0 text-[11px] font-bold text-[#555]">Y</span>
                        <input type="range" min="-24" max="24" value={y} onChange={(e) => setY(Number(e.target.value))} className="flex-1 accent-[#0f766e]" />
                        <span className="w-9 text-right text-[11px] font-semibold tabular-nums text-[#555]">{y > 0 ? `+${y}` : y}px</span>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {editStatus && <p className="mt-3 rounded-md bg-[#e8f7f1] p-3 text-sm text-[#064e46]">{editStatus}</p>}
              <p className="mt-3 text-[11px] text-[#999]">Adjust settings above, then click <span className="font-semibold text-[#555]">Apply edit</span> in the Action panel to regenerate.</p>
              <button type="button" onClick={() => { setResult(null); setEditStatus(null); setError(null); }} className="mt-3 h-10 w-full rounded-md border border-[#151515]/15 bg-white text-sm font-semibold">{mode === "adapt" ? "Localize another creative" : "Resize another creative"}</button>
            </section>
          ) : (
            <>
              <section className="rounded-md border border-[#151515]/10 bg-white p-4">
                <div className="mb-4 flex items-center justify-between"><h2 className="font-semibold">Upload</h2><FileArchive className="h-4 w-4 text-[#0f766e]" /></div>
                <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-[#151515]/30 bg-[#f6f1e7] p-4 text-center hover:border-[#0f766e]"><CloudUpload className="mb-3 h-8 w-8 text-[#0f766e]" /><span className="text-sm font-semibold">Upload PNG, WebP, JPG, JPEG, PDF or ZIP</span><span className="mt-1 text-xs text-[#595959]">Multiple files supported</span><input className="sr-only" multiple accept=".png,.webp,.jpg,.jpeg,.pdf,.zip" type="file" onChange={(e) => setFiles((current) => mergeFiles(current, Array.from(e.target.files ?? [])))} /></label>
                <div className="mt-3 space-y-2">
                  {(files.length ? files : [{ name: "No files selected", size: 0 } as File]).map((file) => (
                    <div key={`${file.name}-${file.size}`} className="flex items-center justify-between rounded-md bg-[#faf9f5] px-3 py-2 text-xs">
                      <span className="max-w-[190px] truncate">{file.name}</span>
                      <div className="ml-3 flex items-center gap-2">
                        <span>{file.size ? `${Math.ceil(file.size / 1024)} KB` : ""}</span>
                        {file.size ? (
                          <button type="button" onClick={() => removeFile(file.name, file.size)} className="rounded-full text-[#777] hover:text-[#ee4d6a]" aria-label={`${file.name} remove`}>
                            <XCircle className="h-4 w-4" />
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {mode === "adapt" ? (
                <>
                  <Collapsible title="Languages" icon={<Languages className="h-4 w-4 text-[#0f766e]" />}>
                    <div className="grid grid-cols-5 gap-1">{languages.map((language) => { const selected = selectedLanguages.includes(language.code); return <button key={language.code} type="button" onClick={() => setSelectedLanguages((current) => selected ? current.filter((code) => code !== language.code) : [...current, language.code])} className={["flex h-9 items-center justify-center gap-1 rounded-md border px-2 text-xs font-semibold", selected ? "border-[#0f766e] bg-[#dff8ef] text-[#064e46]" : "border-[#151515]/10 bg-[#faf9f5]"].join(" ")}>{language.code}{selected && <Check className="h-3 w-3" />}</button>; })}</div>
                  </Collapsible>
                  <Collapsible title="Output Format" icon={<Download className="h-4 w-4 text-[#0f766e]" />}>
                    <div className="grid grid-cols-5 gap-1 rounded-md bg-[#f1eee6] p-1">{outputFormats.map((format) => <button key={format} type="button" onClick={() => setSelectedFormat(format)} className={["h-9 rounded text-xs font-semibold", selectedFormat === format ? "bg-[#151515] text-white" : "text-[#555] hover:bg-white"].join(" ")}>{format}</button>)}</div>
                  </Collapsible>
                </>
              ) : (
                <>
                  <Collapsible title="Dimensions" icon={<Frame className="h-4 w-4 text-[#0f766e]" />}>
                    <div className="max-h-[500px] space-y-4 overflow-auto pr-1">
                      {grouped.map(([platform, items]) => <div key={platform}><p className="mb-2 text-xs font-semibold uppercase text-[#777]">{platform}</p><div className="space-y-2">{items.map((placement) => { const selected = selectedPlacementIds.includes(placement.id); return <label key={placement.id} className={["flex cursor-pointer items-center gap-3 rounded-md border px-3 py-2 text-sm", selected ? "border-[#0f766e] bg-[#e8f7f1]" : "border-[#151515]/10 bg-[#faf9f5]"].join(" ")}><input type="checkbox" className="h-4 w-4 accent-[#0f766e]" checked={selected} onChange={() => togglePlacement(placement.id)} /><span className="min-w-0 flex-1"><span className="block font-semibold">{placement.label}</span><span className="text-xs text-[#666]">{placement.id === "custom-display" ? `${placement.ratio} / ${customWidth}x${customHeight}` : `${placement.ratio} / ${placement.width}x${placement.height}`}</span></span></label>; })}</div></div>)}
                      {selectedPlacementIds.includes("custom-display") && (
                        <div className="rounded-md border border-[#151515]/10 bg-[#faf9f5] p-3">
                          <p className="text-xs font-semibold uppercase text-[#0f766e]">Custom aspect ratio</p>
                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <label className="text-xs font-semibold text-[#555]">Width<input className="mt-1 h-10 w-full rounded-md border border-[#151515]/10 bg-white px-3 outline-none focus:border-[#0f766e]" type="number" min="64" value={customWidth} onChange={(e) => setCustomWidth(Number(e.target.value))} /></label>
                            <label className="text-xs font-semibold text-[#555]">Height<input className="mt-1 h-10 w-full rounded-md border border-[#151515]/10 bg-white px-3 outline-none focus:border-[#0f766e]" type="number" min="64" value={customHeight} onChange={(e) => setCustomHeight(Number(e.target.value))} /></label>
                          </div>
                        </div>
                      )}
                    </div>
                  </Collapsible>
                  <Collapsible title="Output Format" icon={<Download className="h-4 w-4 text-[#0f766e]" />}>
                    <div className="grid grid-cols-5 gap-1 rounded-md bg-[#f1eee6] p-1">{outputFormats.map((format) => <button key={format} type="button" onClick={() => setSelectedFormat(format)} className={["h-9 rounded text-xs font-semibold", selectedFormat === format ? "bg-[#151515] text-white" : "text-[#555] hover:bg-white"].join(" ")}>{format}</button>)}</div>
                  </Collapsible>
                </>
              )}
            </>
          )}
        </aside>

        <section className="overflow-hidden rounded-md border border-[#151515]/10 bg-white">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#151515]/10 px-4 py-3">
            <div><h2 className="font-semibold">{mode === "adapt" ? "Localize Result" : "Resize Result"}</h2><p className="text-xs text-[#666]">{mode === "adapt" ? "Localized creative preview without platform template chrome" : "Selected placement rendered inside platform UI with safe-zone masks"}</p></div>
            {mode === "resize" && (
              <div className="flex flex-wrap items-center gap-2">
                {resizeSourceNames.length > 1 && (
                  <select className="h-9 rounded-md border border-[#151515]/15 bg-white px-3 text-xs font-semibold outline-none focus:border-[#0f766e]" value={effectiveResizeSource} onChange={(e) => setActiveResizeSource(e.target.value)}>
                    {resizeSourceNames.map((sourceName) => <option key={sourceName} value={sourceName}>{sourceName}</option>)}
                  </select>
                )}
                <select className="h-9 rounded-md border border-[#151515]/15 bg-white px-3 text-xs font-semibold outline-none focus:border-[#0f766e]" value={activePlacementId} onChange={(e) => setActivePlacementId(e.target.value)}>
                  {(selectedPlacementIds.length ? placements.filter((placement) => selectedPlacementIds.includes(placement.id)) : placements).map((placement) => <option key={placement.id} value={placement.id}>{placement.platform} / {placement.label} / {placement.id === "custom-display" ? `${customWidth}x${customHeight}` : `${placement.width}x${placement.height}`}</option>)}
                </select>
                <select className="h-9 rounded-md border border-[#151515]/15 bg-white px-3 text-xs font-semibold outline-none focus:border-[#0f766e]" value={activePreviewVariant.id} onChange={(e) => setActivePreviewVariantId(e.target.value)}>
                  {activePreviewVariants.map((variant) => <option key={variant.id} value={variant.id}>{variant.label} / {variant.device}</option>)}
                </select>
              </div>
            )}
          </div>
          {mode === "adapt" ? (
            <LocalizePreview
              originalUrl={activeOriginalUrl}
              localizedUrl={activeOutput?.download_url}
              originalLabel={activeOutput ? `ORIGINAL (${activeOutput.source_language ?? "Source"})` : "ORIGINAL"}
              localizedLabel={activeOutput ? `LOCALIZED (${activeOutput.language ?? "Target"})` : "LOCALIZED"}
              index={activeOutputIndex}
              total={result?.outputs.length ?? 0}
              onPrevious={() => selectOutput(activeOutputIndex - 1)}
              onNext={() => selectOutput(activeOutputIndex + 1)}
            />
          ) : (
            <div className="flex items-center justify-center overflow-x-auto py-6">
              {mode === "resize" ? (
                <Preview placement={activePlacement} mode={mode} device={previewDevice} copy={copy} x={x} y={y} opacity={opacity} scale={scale} fit={fit} imageUrl={activeResizeOutput?.download_url} metadata={previewMetadata} previewTemplateId={activePreviewVariant.templateId} />
              ) : (
                <div className="flex flex-col items-center gap-3 py-16 text-[#aaa]">
                  <Frame className="h-12 w-12 opacity-30" />
                  <p className="text-sm">Select placements and click <span className="font-semibold text-[#555]">Run Resize</span> to see the preview.</p>
                </div>
              )}
            </div>
          )}
          <div className="border-t border-[#151515]/10 p-4">
            {error && <div className="mb-3 flex gap-2 rounded-md bg-[#fff0d8] p-3 text-sm text-[#6b3b00]"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{error}</div>}
            {result && result.outputs.length > 0 ? (
              <div className="space-y-3 text-sm">
                <div className="rounded-md bg-[#e8f7f1] p-3 font-semibold text-[#064e46]">{result.outputs.length} file{result.outputs.length === 1 ? "" : "s"} ready for download</div>
                <div className="grid gap-2 sm:grid-cols-2">{result.outputs.map((output, index) => {
                  const friendlyName = mode === "resize"
                    ? `${output.source_name.replace(/\.[^.]+$/, "")} — ${output.placement_id ? placements.find((p) => p.id === output.placement_id)?.label ?? output.placement_id : "resized"} (${output.width}×${output.height})`
                    : `${output.source_name.replace(/\.[^.]+$/, "")}${output.language ? ` — ${output.language}` : ""}`;
                  return (
                    <div key={output.filename} className={["flex items-center justify-between rounded-md border px-3 py-2", activeOutput?.filename === output.filename ? "border-[#0f766e] bg-[#e8f7f1]" : "border-[#151515]/10"].join(" ")}>
                      <button type="button" onClick={() => selectOutput(index)} className="min-w-0 flex-1 text-left">
                        <span className="block truncate font-semibold text-[13px]">{friendlyName}</span>
                        <span className="text-[11px] text-[#666]">{output.width}×{output.height} px</span>
                      </button>
                      <a href={output.download_url} className="ml-3 shrink-0 rounded-md border border-[#151515]/10 p-2 hover:border-[#0f766e]"><Download className="h-4 w-4 text-[#0f766e]" /></a>
                    </div>
                  );
                })}</div>
              </div>
            ) : result ? (
              <p className="text-sm text-[#b42318]">Processing completed but no output files were generated. Please try again.</p>
            ) : (
              mode === "adapt" ? <p className="text-sm text-[#666]">Localized images will appear here after processing.</p> : null
            )}
          </div>
        </section>

        <aside className="space-y-4">
          <section className="sticky top-24 rounded-md border border-[#151515]/10 bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center justify-between"><h2 className="font-semibold">Action</h2><Sparkles className="h-4 w-4 text-[#0f766e]" /></div>
            <div className="rounded-md bg-[#faf9f5] px-3 py-2 text-sm">
              <div className="flex justify-between gap-3"><span>Operation</span><span className="font-semibold">{result ? "Modify result" : mode === "adapt" ? "Localize" : "Resize"}</span></div>
            </div>
            <div className="mt-3 space-y-2 text-sm">
              {receiptLines.filter((line) => !(mode === "adapt" && line.label === "Placements")).map((line) => (
                <div key={line.label} className="grid grid-cols-[1fr_auto] gap-3 rounded-md bg-[#faf9f5] px-3 py-2">
                  <span><span className="block font-semibold">{line.label}</span><span className="text-[11px] text-[#666]">{line.formula}</span></span>
                  <span className="font-black">{formatCreditText(line.credits)}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 border-t border-[#151515]/10 pt-3 text-sm">
              <div className="flex justify-between gap-3 font-black"><span>Total Credits</span><span>{formatCreditText(actionCredits)}</span></div>
              <div className={["mt-2 flex justify-between gap-3 font-semibold", remainingAfterAction < 0 ? "text-[#b42318]" : "text-[#0f766e]"].join(" ")}><span>Remaining Credits</span><span>{formatCreditText(remainingAfterAction)}</span></div>
            </div>
            <button type={result ? "button" : "submit"} onClick={result ? applyManualEdit : undefined} disabled={result ? isApplyingEdit || !canApplyCurrentEdit : isRunning || !canRun} className="mt-4 flex h-11 w-full items-center justify-center gap-2 rounded-md bg-[#ee4d6a] text-sm font-semibold text-white disabled:bg-[#d6d0c4]">
              {result ? (isApplyingEdit ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />) : isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {result ? `Apply edit / use ${editCredits} credits` : mode === "adapt" ? "Run Localize" : "Run Resize"}
            </button>
            {remainingAfterAction < 0 && <p className="mt-3 text-xs font-semibold text-[#b42318]">Add credits before starting this action.</p>}
          </section>
        </aside>
      </form>

      <footer className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-3 border-t border-[#151515]/10 px-5 py-5 text-xs text-[#666]">
        <div className="space-y-1">
          <p className="font-semibold text-[#151515]">SASMAZ DIGITAL SOLUTIONS / AdaptifAI - CREATIVE LOCALIZATION AND RESIZING TOOL</p>
          <p>İbrahim Tolgar ŞAŞMAZ / 81543, Munich Germany / <a href="mailto:tolgar@sasmaz.digital" className="hover:text-[#151515]">tolgar@sasmaz.digital</a></p>
          <p>Strictly stateless creative processing / temporary files auto-delete after 24h</p>
        </div>
        <nav className="flex gap-4"><a href="/terms" className="hover:text-[#151515]">Terms of Service</a><a href="/privacy" className="hover:text-[#151515]">Privacy GDPR/KVKK</a><a href="/refund" className="hover:text-[#151515]">Refund Policy</a><a href="mailto:tolgar@sasmaz.digital" className="hover:text-[#151515]">Support</a></nav>
      </footer>
      <ConsentBanner />
    </main>
  );
}
