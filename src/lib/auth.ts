import { getSupabaseAdmin, hasSupabaseServerConfig } from "@/lib/supabase";

export function allowsDevelopmentUserFallback() {
  return !hasSupabaseServerConfig() && process.env.NODE_ENV !== "production" && !process.env.VERCEL;
}

export async function getAuthenticatedEmail(request: Request) {
  if (!hasSupabaseServerConfig()) return null;

  const authorization = request.headers.get("authorization") ?? "";
  const token = authorization.startsWith("Bearer ") ? authorization.slice("Bearer ".length) : "";
  const supabase = getSupabaseAdmin();

  if (!supabase || !token) {
    throw new Error("Authentication required.");
  }

  const { data, error } = await supabase.auth.getUser(token);
  const email = data.user?.email?.trim().toLowerCase();
  if (error || !email) {
    throw new Error("Authentication required.");
  }

  return email;
}

export async function requireAuthenticatedEmail(request: Request) {
  const email = await getAuthenticatedEmail(request);
  if (email) return email;
  throw new Error("Authentication required.");
}

export async function getAuthenticatedOrDevelopmentUser(request: Request, fallbackUserId: string | null | undefined = "guest") {
  const email = await getAuthenticatedEmail(request);
  if (email) return email;
  if (allowsDevelopmentUserFallback()) return fallbackUserId?.trim() || "guest";
  throw new Error("Authentication required.");
}
