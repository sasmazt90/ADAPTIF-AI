import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  if (process.env.VERCEL_ENV !== "preview") {
    return NextResponse.json({ error: "Not available." }, { status: 404 });
  }
  try {
    const response = await fetch("https://adaptifai-backend-production.up.railway.app/health", {
      cache: "no-store",
      signal: AbortSignal.timeout(25000),
    });
    return NextResponse.json({ reachable: response.ok, status: response.status }, { status: response.ok ? 200 : 503 });
  } catch (error) {
    return NextResponse.json({ reachable: false, error: error instanceof Error ? error.message : "Connection error" }, { status: 503 });
  }
}
