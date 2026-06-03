import { beforeEach, describe, expect, it, vi } from "vitest";

let hasServerConfig = true;
const supabaseAdmin = {
  auth: {
    getUser: vi.fn(),
  },
};

vi.mock("@/lib/supabase", () => ({
  hasSupabaseServerConfig: () => hasServerConfig,
  getSupabaseAdmin: () => supabaseAdmin,
}));

describe("getAuthenticatedOrDevelopmentUser", () => {
  beforeEach(() => {
    hasServerConfig = true;
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    delete process.env.VERCEL;
    vi.stubEnv("NODE_ENV", "test");
  });

  it("requires a bearer token when Supabase server auth is configured", async () => {
    const { getAuthenticatedOrDevelopmentUser } = await import("./auth");

    await expect(getAuthenticatedOrDevelopmentUser(new Request("http://localhost/api/adapt"), "guest"))
      .rejects.toThrow("Authentication required.");
    expect(supabaseAdmin.auth.getUser).not.toHaveBeenCalled();
  });

  it("normalizes the authenticated Supabase email from a valid bearer token", async () => {
    supabaseAdmin.auth.getUser.mockResolvedValueOnce({
      data: { user: { email: "USER@Example.COM " } },
      error: null,
    });
    const { getAuthenticatedOrDevelopmentUser } = await import("./auth");

    await expect(getAuthenticatedOrDevelopmentUser(
      new Request("http://localhost/api/adapt", { headers: { authorization: "Bearer token" } }),
      "guest",
    )).resolves.toBe("user@example.com");
  });

  it("allows fallback users only for local development without Supabase config", async () => {
    hasServerConfig = false;
    const { getAuthenticatedOrDevelopmentUser } = await import("./auth");

    await expect(getAuthenticatedOrDevelopmentUser(new Request("http://localhost/api/adapt"), "guest"))
      .resolves.toBe("guest");

    vi.stubEnv("NODE_ENV", "production");
    await expect(getAuthenticatedOrDevelopmentUser(new Request("http://localhost/api/adapt"), "guest"))
      .rejects.toThrow("Authentication required.");
  });
});
