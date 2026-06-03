import { NextRequest, NextResponse } from "next/server";
import { getAuthenticatedOrDevelopmentUser } from "@/lib/auth";
import { creditPacks, CreditPackId, getStripe, getStripeEnv } from "@/lib/stripe";

function cleanOrigin(value: string | null | undefined) {
  const candidate = (value ?? process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000").trim().replace(/^"|"$/g, "");
  return new URL(candidate).origin;
}

async function createCheckoutSessionWithRest({
  origin,
  priceId,
  pack,
  packId,
  userId,
}: {
  origin: string;
  priceId: string;
  pack: (typeof creditPacks)[CreditPackId];
  packId: CreditPackId;
  userId: string;
}) {
  const secretKey = getStripeEnv("STRIPE_SECRET_KEY");
  if (!secretKey) throw new Error("Stripe secret key is not configured.");

  const body = new URLSearchParams({
    mode: "payment",
    success_url: `${origin}?checkout=success&credits=${pack.credits}`,
    cancel_url: `${origin}?checkout=cancelled`,
    "line_items[0][quantity]": "1",
    "line_items[0][price]": priceId,
    "metadata[credits]": String(pack.credits),
    "metadata[pack]": packId,
    "metadata[user_id]": userId,
  });
  if (userId.includes("@")) body.set("customer_email", userId);

  const response = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${secretKey}`,
      "content-type": "application/x-www-form-urlencoded",
    },
    body,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message ?? "Unable to create Stripe Checkout Session.");
  return payload as { url?: string };
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { pack?: CreditPackId; user_id?: string };
    const userId = await getAuthenticatedOrDevelopmentUser(request, body.user_id ?? "guest");
    const packId = body.pack ?? "starter";
    const pack = creditPacks[packId];

    if (!pack) {
      return NextResponse.json({ error: "Unknown credit pack." }, { status: 400 });
    }

    const origin = cleanOrigin(request.headers.get("origin"));
    const configuredPriceId = getStripeEnv(pack.envPriceKey);
    if (!configuredPriceId) {
      return NextResponse.json({ error: `${pack.envPriceKey} is not configured.` }, { status: 500 });
    }

    let session: { url?: string | null };
    try {
      const stripe = getStripe();
      session = await stripe.checkout.sessions.create({
        mode: "payment",
        success_url: `${origin}?checkout=success&credits=${pack.credits}`,
        cancel_url: `${origin}?checkout=cancelled`,
        line_items: [{ quantity: 1, price: configuredPriceId }],
        metadata: {
          credits: String(pack.credits),
          pack: packId,
          user_id: userId,
        },
        customer_email: userId.includes("@") ? userId : undefined,
      });
    } catch {
      session = await createCheckoutSessionWithRest({ origin, priceId: configuredPriceId, pack, packId, userId });
    }

    if (!session.url) {
      return NextResponse.json({ error: "Stripe Checkout Session did not return a URL." }, { status: 502 });
    }

    return NextResponse.json({ url: session.url, mode: "checkout_session" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to create Stripe Checkout Session.";
    return NextResponse.json({ error: message }, { status: message === "Authentication required." ? 401 : 500 });
  }
}
