import { NextRequest, NextResponse } from "next/server";
import { getAuthenticatedOrDevelopmentUser } from "@/lib/auth";
import { estimateLocalizeCredits, estimateResizeCredits } from "@/lib/credit-pricing";
import { getCredits, spendCredits } from "@/lib/credits";

export const runtime = "nodejs";

function estimateCredits(formData: FormData) {
  const fileCount = formData.getAll("files").length;
  const languages = String(formData.get("target_languages") ?? "EN").split(",").filter(Boolean);
  const placements = String(formData.get("placements") ?? "meta-stories").split(",").filter(Boolean);
  const outputFormat = String(formData.get("output_format") ?? "PNG");
  const mode = String(formData.get("mode") ?? "").toLowerCase();
  const isLocalize = mode === "localize" || (placements.length === 1 && placements[0] === "native-custom");

  return isLocalize
    ? estimateLocalizeCredits({ fileCount, languageCount: languages.length, outputFormat })
    : estimateResizeCredits({ fileCount, dimensionCount: placements.length, outputFormat });
}

export async function POST(request: NextRequest) {
  const backendUrl = process.env.ADAPTIFAI_BACKEND_URL ?? "http://127.0.0.1:8000";

  try {
    const formData = await request.formData();
    const userId = await getAuthenticatedOrDevelopmentUser(request, String(formData.get("user_id") ?? "guest"));
    const estimatedCredits = estimateCredits(formData);
    const currentCredits = await getCredits(userId);
    if (currentCredits < estimatedCredits) {
      return NextResponse.json({ error: "Insufficient credits.", credits: currentCredits, credits_required: estimatedCredits }, { status: 402 });
    }

    const response = await fetch(`${backendUrl}/adapt`, {
      method: "POST",
      body: formData,
    });

    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: `Backend returned ${response.status}`, detail: await response.text() };
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : typeof payload.error === "string" ? payload.error : "";
      const isGatewayError = response.status === 502 || response.status === 503 || response.status === 504;
      const message = isGatewayError
        ? "Backend service is temporarily unavailable while processing this creative. Please retry in a minute; if it repeats, the backend deploy/runtime needs attention."
        : detail || `Backend returned ${response.status}`;
      return NextResponse.json(
        {
          ...payload,
          error: message,
          backend_status: response.status,
        },
        { status: response.status },
      );
    }
    if (response.ok) {
      const spend = await spendCredits(userId, estimatedCredits);
      if (!spend.ok) {
        return NextResponse.json({ error: "Insufficient credits.", credits: spend.credits }, { status: 402 });
      }
      payload.credits_estimated = estimatedCredits;
      payload.credits_remaining = spend.credits;
    }
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Adapt pipeline is unavailable.";
    if (message === "Authentication required.") {
      return NextResponse.json({ error: message }, { status: 401 });
    }
    return NextResponse.json(
      {
        error: message,
        hint: "Start the FastAPI backend with: python -m uvicorn app.main:app --reload --port 8000",
      },
      { status: 503 },
    );
  }
}
