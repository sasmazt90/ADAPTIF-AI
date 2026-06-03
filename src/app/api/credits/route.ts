import { NextResponse } from "next/server";
import { getAuthenticatedOrDevelopmentUser } from "@/lib/auth";
import { addCredits, getCredits, normalizeUserId } from "@/lib/credits";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const userId = normalizeUserId(await getAuthenticatedOrDevelopmentUser(request, url.searchParams.get("user_id")));
    return NextResponse.json({
      user_id: userId,
      credits: await getCredits(userId),
      currency: "credits",
      stateless: true,
      source: process.env.NEXT_PUBLIC_SUPABASE_URL ? "supabase_credit_ledger" : "local_credit_ledger",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Authentication required.";
    return NextResponse.json({ error: message }, { status: 401 });
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { user_id?: string; credits?: number };
    const userId = normalizeUserId(await getAuthenticatedOrDevelopmentUser(request, body.user_id));
    const credits = await addCredits(userId, Math.max(0, Number(body.credits ?? 0)), undefined, "system", "legacy_credit_add");
    return NextResponse.json({ user_id: userId, credits });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Authentication required.";
    return NextResponse.json({ error: message }, { status: 401 });
  }
}
