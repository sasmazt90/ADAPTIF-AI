import { NextResponse } from "next/server";
import { listCreditUsers } from "@/lib/credits";
import { adminEmail, getSupabaseAdmin } from "@/lib/supabase";

async function requireAdmin(request: Request) {
  const authorization = request.headers.get("authorization") ?? "";
  const token = authorization.startsWith("Bearer ") ? authorization.slice("Bearer ".length) : "";
  const supabase = getSupabaseAdmin();

  if (!supabase || !token) {
    return { ok: false as const, error: "Supabase auth is not configured." };
  }

  const { data, error } = await supabase.auth.getUser(token);
  const email = data.user?.email?.toLowerCase();
  if (error || email !== adminEmail) {
    return { ok: false as const, error: "Admin access required." };
  }

  return { ok: true as const, email };
}

export async function GET(request: Request) {
  const admin = await requireAdmin(request);
  if (!admin.ok) return NextResponse.json({ error: admin.error }, { status: 403 });

  try {
    const users = await listCreditUsers();
    return NextResponse.json({ users });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to list users.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
