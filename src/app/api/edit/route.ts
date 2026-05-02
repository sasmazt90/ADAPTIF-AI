import { NextResponse } from "next/server";
import { getAuthenticatedEmail } from "@/lib/auth";
import { estimateEditCredits } from "@/lib/credit-pricing";
import { getCredits, spendCredits } from "@/lib/credits";

export async function POST(request: Request) {
  const backendUrl = process.env.ADAPTIFAI_BACKEND_URL ?? "http://127.0.0.1:8000";

  try {
    const userId = await getAuthenticatedEmail(request);
    const body = (await request.json().catch(() => ({}))) as {
      job_id?: string;
      filename?: string;
      mode?: "adapt" | "resize";
      copy?: string;
      x?: number;
      y?: number;
      opacity?: number;
      scale?: number;
      fit?: string;
      preserve_bold?: boolean;
      mask_cleanup?: boolean;
      fit_bounds?: boolean;
      text_color?: string;
      font_size_scale?: number;
      text_italic?: boolean;
      text_underline?: boolean;
      text_strike?: boolean;
    };

    const mode = body.mode === "resize" ? "resize" : "adapt";
    const cost = estimateEditCredits(mode);
    const currentCredits = await getCredits(userId ?? "guest");
    if (currentCredits < cost) {
      return NextResponse.json({ error: "Insufficient credits.", credits: currentCredits }, { status: 402 });
    }

    const response = await fetch(`${backendUrl}/edit`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        job_id: body.job_id,
        filename: body.filename,
        mode: mode === "resize" ? "resize" : "localize",
        copy: body.copy ?? "",
        x: Math.trunc(Number(body.x ?? 0)),
        y: Math.trunc(Number(body.y ?? 0)),
        opacity: Math.trunc(Number(body.opacity ?? 18)),
        scale: Math.trunc(Number(body.scale ?? 100)),
        fit: body.fit ?? "cover",
        preserve_bold: body.preserve_bold ?? true,
        mask_cleanup: body.mask_cleanup ?? true,
        fit_bounds: body.fit_bounds ?? true,
        text_color: body.text_color ?? "",
        font_size_scale: Math.trunc(Number(body.font_size_scale ?? 100)),
        text_italic: body.text_italic ?? false,
        text_underline: body.text_underline ?? false,
        text_strike: body.text_strike ?? false,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      return NextResponse.json(payload, { status: response.status });
    }

    const spend = await spendCredits(userId ?? "guest", cost);
    if (!spend.ok) {
      return NextResponse.json({ error: "Insufficient credits.", credits: spend.credits }, { status: 402 });
    }

    return NextResponse.json({
      ok: true,
      mode,
      output: payload,
      credits_remaining: spend.credits,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to apply edit.";
    return NextResponse.json({ error: message }, { status: message === "Authentication required." ? 401 : 500 });
  }
}
