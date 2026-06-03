import { beforeEach, describe, expect, it, vi } from "vitest";

let signature: string | null = null;
const constructEvent = vi.fn();
const addCredits = vi.fn();

vi.mock("next/headers", () => ({
  headers: async () => new Headers(signature ? { "stripe-signature": signature } : {}),
}));

vi.mock("@/lib/stripe", () => ({
  getStripeEnv: (key: string) => process.env[key]?.trim(),
  getStripe: () => ({
    webhooks: {
      constructEvent,
    },
  }),
}));

vi.mock("@/lib/credits", () => ({
  addCredits,
}));

describe("Stripe webhook route", () => {
  beforeEach(() => {
    signature = null;
    delete process.env.STRIPE_WEBHOOK_SECRET;
    vi.clearAllMocks();
  });

  it("rejects requests without a Stripe signature or webhook secret", async () => {
    const { POST } = await import("./route");
    const response = await POST(new Request("http://localhost/api/stripe/webhook", { method: "POST", body: "{}" }) as never);

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Stripe webhook signing secret is not configured." });
    expect(constructEvent).not.toHaveBeenCalled();
  });

  it("returns a 400 when Stripe event construction fails", async () => {
    signature = "sig_test";
    process.env.STRIPE_WEBHOOK_SECRET = "whsec_test";
    constructEvent.mockImplementationOnce(() => {
      throw new Error("Invalid signature");
    });

    const { POST } = await import("./route");
    const response = await POST(new Request("http://localhost/api/stripe/webhook", { method: "POST", body: "{}" }) as never);

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Invalid signature" });
    expect(addCredits).not.toHaveBeenCalled();
  });
});
