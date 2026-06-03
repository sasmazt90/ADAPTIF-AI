import { beforeEach, describe, expect, it, vi } from "vitest";

const supabaseClient = {
  rpc: vi.fn(),
  from: vi.fn(),
};

vi.mock("@/lib/supabase", () => ({
  hasSupabaseServerConfig: () => true,
  getSupabaseAdmin: () => supabaseClient,
}));

describe("spendCredits", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns the current balance when Supabase addCredits RPC fails", async () => {
    supabaseClient.rpc.mockResolvedValueOnce({ data: null, error: { message: "rpc failed" } });
    supabaseClient.from.mockReturnValueOnce({
      select: () => ({
        eq: () => ({
          maybeSingle: () => Promise.resolve({ data: { credits: 42 }, error: null }),
        }),
      }),
    });

    const { spendCredits } = await import("./credits");
    await expect(spendCredits("USER@Example.com", 7)).resolves.toEqual({ ok: false, credits: 42 });
  });

  it("does not throw if both the spend RPC and fallback balance lookup fail", async () => {
    supabaseClient.rpc.mockResolvedValueOnce({ data: null, error: { message: "rpc failed" } });
    supabaseClient.from.mockReturnValueOnce({
      select: () => ({
        eq: () => ({
          maybeSingle: () => Promise.resolve({ data: null, error: { message: "lookup failed" } }),
        }),
      }),
    });

    const { spendCredits } = await import("./credits");
    await expect(spendCredits("user@example.com", 7)).resolves.toEqual({ ok: false, credits: 0 });
  });
});
