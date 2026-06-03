import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { getSupabaseAdmin, hasSupabaseServerConfig } from "@/lib/supabase";

type CreditLedger = {
  users: Record<string, { credits: number; updatedAt: string }>;
  processedSessions: Record<string, string>;
};

const ledgerRoot = process.env.ADAPTIFAI_LEDGER_DIR ?? (process.env.VERCEL ? "/tmp/adaptifai" : join(process.cwd(), ".adaptifai"));
const ledgerPath = join(ledgerRoot, "credits.json");

function readLedger(): CreditLedger {
  try {
    return JSON.parse(readFileSync(ledgerPath, "utf8")) as CreditLedger;
  } catch {
    return { users: {}, processedSessions: {} };
  }
}

function writeLedger(ledger: CreditLedger) {
  mkdirSync(dirname(ledgerPath), { recursive: true });
  writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2));
}

export function normalizeUserId(userId: string | null | undefined) {
  const normalized = (userId ?? "guest").trim().toLowerCase();
  return normalized || "guest";
}

export async function getCredits(userId: string) {
  if (hasSupabaseServerConfig()) {
    const supabase = getSupabaseAdmin();
    const normalized = normalizeUserId(userId);
    const { data, error } = await supabase!
      .from("credit_accounts")
      .select("credits")
      .eq("user_id", normalized)
      .maybeSingle();

    if (error) throw new Error(error.message);
    if (data) return Number(data.credits ?? 0);
    return Number(process.env.ADAPTIFAI_DEMO_CREDITS ?? 240);
  }

  const ledger = readLedger();
  return ledger.users[normalizeUserId(userId)]?.credits ?? Number(process.env.ADAPTIFAI_DEMO_CREDITS ?? 240);
}

export async function addCredits(userId: string, credits: number, sessionId?: string, actorEmail = "system", reason = "manual") {
  const normalized = normalizeUserId(userId);

  if (hasSupabaseServerConfig()) {
    const supabase = getSupabaseAdmin();
    const { data, error } = await supabase!.rpc("adjust_credits", {
      p_user_id: normalized,
      p_delta: Math.trunc(credits),
      p_actor_email: actorEmail,
      p_reason: reason,
      p_stripe_session_id: sessionId ?? null,
    });

    if (error) throw new Error(error.message);
    return Number(data ?? 0);
  }

  const ledger = readLedger();

  if (sessionId && ledger.processedSessions[sessionId]) {
    return ledger.users[normalized]?.credits ?? 0;
  }

  const existing = ledger.users[normalized]?.credits ?? Number(process.env.ADAPTIFAI_DEMO_CREDITS ?? 240);
  ledger.users[normalized] = {
    credits: existing + credits,
    updatedAt: new Date().toISOString(),
  };

  if (sessionId) ledger.processedSessions[sessionId] = normalized;
  writeLedger(ledger);
  return ledger.users[normalized].credits;
}

export async function spendCredits(userId: string, credits: number) {
  const normalized = normalizeUserId(userId);

  if (hasSupabaseServerConfig()) {
    try {
      const balance = await addCredits(normalized, -Math.trunc(credits), undefined, "system", "adapt_spend");
      return { ok: true, credits: balance };
    } catch {
      try {
        return { ok: false, credits: await getCredits(normalized) };
      } catch {
        return { ok: false, credits: 0 };
      }
    }
  }

  const ledger = readLedger();
  const existing = ledger.users[normalized]?.credits ?? Number(process.env.ADAPTIFAI_DEMO_CREDITS ?? 240);
  if (existing < credits) return { ok: false, credits: existing };

  ledger.users[normalized] = {
    credits: existing - credits,
    updatedAt: new Date().toISOString(),
  };
  writeLedger(ledger);
  return { ok: true, credits: ledger.users[normalized].credits };
}

export async function listCreditUsers() {
  if (hasSupabaseServerConfig()) {
    const supabase = getSupabaseAdmin();
    const { data, error } = await supabase!
      .from("credit_accounts")
      .select("user_id, credits, updated_at")
      .order("updated_at", { ascending: false })
      .limit(100);

    if (error) throw new Error(error.message);
    return (data ?? []).map((item) => ({
      user_id: normalizeUserId(item.user_id),
      credits: Number(item.credits ?? 0),
      updated_at: String(item.updated_at ?? ""),
    }));
  }

  const ledger = readLedger();
  return Object.entries(ledger.users)
    .map(([user_id, value]) => ({
      user_id,
      credits: value.credits,
      updated_at: value.updatedAt,
    }))
    .sort((left, right) => right.updated_at.localeCompare(left.updated_at));
}
