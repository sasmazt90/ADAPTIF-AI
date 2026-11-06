import { NextRequest, NextResponse } from "next/server";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ jobId: string; filename: string }> },
) {
  const { jobId, filename } = await context.params;
  const backendUrl = process.env.VERCEL ? "https://adaptifai-backend-production.up.railway.app" : (process.env.ADAPTIFAI_BACKEND_URL ?? "http://127.0.0.1:8000");
  const response = await fetch(`${backendUrl}/outputs/${encodeURIComponent(jobId)}/${encodeURIComponent(filename)}`);

  if (!response.ok || !response.body) {
    return NextResponse.json({ error: "Output file is unavailable or expired." }, { status: response.status || 404 });
  }

  return new NextResponse(response.body, {
    status: 200,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/octet-stream",
      "content-disposition": response.headers.get("content-disposition") ?? `attachment; filename="${filename}"`,
      "cache-control": "no-store",
    },
  });
}
