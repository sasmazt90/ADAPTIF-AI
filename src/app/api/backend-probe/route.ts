import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 120;
const backend = "https://adaptifai-backend-production.up.railway.app";

export async function GET(request: NextRequest) {
  if (process.env.VERCEL_ENV !== "preview") {
    return NextResponse.json({ error: "Not available." }, { status: 404 });
  }
  try {
    if (request.nextUrl.searchParams.get("process") === "1") {
      const formData = new FormData();
      formData.append("files", new Blob([Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAACVUlEQVR42u3cy0tUUQDA4TN6xZZRhNV/EEyG5iLTdIpBTLAWhj3oQQbNInVRGgalBAZSEWGbHpsgKKh9SPayoFUhlIW0igiiIM2lKUwLISIpxnAmnb7f6g5z71w435zDXVxOLJ1OB/27CgwBAAACAEAAAAgAAAEAoNwVZXLSvivPjdRfdCNVYQZYggQAgAAAEAAAAgBAAAAIAAABACAAAAAIAAAB+E+LsvGjF5rj+TpeR2+PmAGWIAEAIAAABACAAAAQAAACAEAAAAgAAAEAoMyKcn/Le4+eXL91J4TwYvjl+rLSEML+5qaO7t7S+JqZE5I11S17d74efXuu//LU9HQUFfb1nHj1ZnT2VfXJxGIHiGWycetcN+vI8K2Istr64aGB2cczNe4+eO3i2ZUlKwYeDN0dfNjfd/p3Z+ayOb0VkclmHdGC/Wt8GRuf/DYZQkjWVi1fttQSlOuOtaZ2HTqSqK7c3lC3oaIcQHabmprec7ht5rizNVVWGm9q3JqsrR58/LT3fH/d5pr2VAuALFZUFN28eunHx7Hxr+/efyhfF9+xrWHLpo0NzQfyFWCBPobGYrG2ru6Pnz6HEMYnJlavKrEE5W4JKl8b72hLnTnZ2Xr81JLi4sLCgr6eLgDz389PkyPP7v/ybaKqMlFV+eerLEECAEAAAAgAAAEAIAAABACAAAAQAAACAEAAAAgAAAEAIAALrKy8mjjvm2uaAQIAQAAACAAAAQAgAAAEAIAAABAAAAIAQAAACAAAAVhEZbR1scwAAAIAQAAACAAAAQAgAHnRd8dLhl9e4AUeAAAAAElFTkSuQmCC", "base64")], { type: "image/png" }), "probe.png");
      formData.append("mode", "resize");
      formData.append("placements", "custom-display");
      formData.append("custom_width", "128");
      formData.append("custom_height", "128");
      formData.append("output_format", "PNG");
      const processed = await fetch(`${backend}/adapt`, {
        method: "POST", body: formData, signal: AbortSignal.timeout(105000),
      });
      const payload = await processed.json().catch(() => ({}));
      const first = payload.outputs?.[0];
      let downloaded: number | null = null;
      if (processed.ok && first?.filename && payload.job_id) {
        const file = await fetch(`${backend}/outputs/${encodeURIComponent(payload.job_id)}/${encodeURIComponent(first.filename)}`, {
          signal: AbortSignal.timeout(15000),
        });
        downloaded = file.status;
      }
      return NextResponse.json({
        processed: processed.ok, processing_status: processed.status,
        output_count: payload.outputs?.length ?? 0,
        width: first?.width, height: first?.height,
        download_status: downloaded,
        error: processed.ok ? undefined : JSON.stringify(payload).slice(0,1000),
      }, { status: processed.ok ? 200 : 503 });
    }
    const response = await fetch(`${backend}/health`, { cache: "no-store", signal: AbortSignal.timeout(25000) });
    return NextResponse.json({ reachable: response.ok, status: response.status }, { status: response.ok ? 200 : 503 });
  } catch (error) {
    return NextResponse.json({ reachable: false, error: error instanceof Error ? error.message : "Connection error" }, { status: 503 });
  }
}
