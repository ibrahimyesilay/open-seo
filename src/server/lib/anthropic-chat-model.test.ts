import type { LanguageModelV3 } from "@openrouter/ai-sdk-provider";
import { APICallError } from "ai";
import { describe, expect, it } from "vitest";
import {
  anthropicCostUsd,
  buildAnthropicChatModel,
  parseAnthropicKeys,
  shouldFailOverToNextKey,
} from "@/server/lib/anthropic-chat-model";
import { requireOpenRouterCostUsd } from "@/server/lib/chatAgent";

const USAGE = {
  inputTokens: { total: 1200, noCache: 1000, cacheRead: 100, cacheWrite: 100 },
  outputTokens: { total: 500, text: 500, reasoning: 0 },
};

function apiError(statusCode: number, responseBody = "") {
  return new APICallError({
    message: `HTTP ${statusCode}`,
    url: "https://api.anthropic.com/v1/messages",
    requestBodyValues: {},
    statusCode,
    responseBody,
  });
}

function fakeModel(
  behavior: () => Promise<Awaited<ReturnType<LanguageModelV3["doGenerate"]>>>,
): LanguageModelV3 {
  return {
    specificationVersion: "v3",
    provider: "fake",
    modelId: "claude-opus-5-5",
    supportedUrls: {},
    doGenerate: behavior,
    doStream: async () => {
      const generated = await behavior();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "text-start", id: "t" });
          controller.enqueue({ type: "text-delta", id: "t", delta: "hi" });
          controller.enqueue({ type: "text-end", id: "t" });
          controller.enqueue({
            type: "finish",
            usage: generated.usage,
            finishReason: generated.finishReason,
          });
          controller.close();
        },
      });
      return { stream };
    },
  } as unknown as LanguageModelV3;
}

const okResult = () =>
  Promise.resolve({
    content: [{ type: "text" as const, text: "hi" }],
    finishReason: { unified: "stop" as const, raw: undefined },
    usage: USAGE,
    warnings: [],
  });

describe("parseAnthropicKeys", () => {
  it("splits keys and optional workspace suffixes", () => {
    expect(parseAnthropicKeys(" sk-a , sk-b@wrkspc_1,, sk-c@ ,@x ")).toEqual([
      { apiKey: "sk-a" },
      { apiKey: "sk-b", workspaceId: "wrkspc_1" },
      { apiKey: "sk-c" },
    ]);
    expect(parseAnthropicKeys(undefined)).toEqual([]);
  });
});

describe("anthropicCostUsd", () => {
  it("prices Opus 5.5 usage from the public table", () => {
    // 1000×$4 + 100×$5 + 100×$0.20 + 500×$20 per MTok
    expect(anthropicCostUsd("claude-opus-5-5", USAGE)).toBeCloseTo(
      (1000 * 4 + 100 * 5 + 100 * 0.2 + 500 * 20) / 1e6,
      10,
    );
  });

  it("derives uncached input from the total when noCache is missing", () => {
    const usage = {
      inputTokens: {
        total: 1200,
        noCache: undefined,
        cacheRead: 100,
        cacheWrite: 100,
      },
      outputTokens: { total: 0 },
    };
    expect(anthropicCostUsd("claude-sonnet-5-5", usage)).toBeCloseTo(
      (1000 * 2 + 100 * 2.5 + 100 * 0.1) / 1e6,
      10,
    );
  });

  it("returns 0 for unknown models instead of guessing", () => {
    expect(anthropicCostUsd("claude-unknown-9", USAGE)).toBe(0);
  });
});

describe("shouldFailOverToNextKey", () => {
  it("fails over on exhausted credits, rejected keys and rate limits", () => {
    expect(
      shouldFailOverToNextKey(
        apiError(400, '{"error":{"message":"Your credit balance is too low"}}'),
      ),
    ).toBe(true);
    expect(shouldFailOverToNextKey(apiError(401))).toBe(true);
    expect(shouldFailOverToNextKey(apiError(403))).toBe(true);
    expect(shouldFailOverToNextKey(apiError(429))).toBe(true);
  });

  it("does not fail over on other errors", () => {
    expect(shouldFailOverToNextKey(apiError(400, "max_tokens invalid"))).toBe(
      false,
    );
    expect(shouldFailOverToNextKey(apiError(529))).toBe(false);
    expect(shouldFailOverToNextKey(new Error("boom"))).toBe(false);
  });
});

describe("buildAnthropicChatModel", () => {
  const keys = [{ apiKey: "k1" }, { apiKey: "k2" }];
  const params = { prompt: [], providerOptions: {} } as never;

  it("uses the next key when the first pool is exhausted", async () => {
    const calls: string[] = [];
    const model = buildAnthropicChatModel(keys, undefined, "max", (modelKeys) =>
      modelKeys.map((key, index) =>
        fakeModel(() => {
          calls.push(key.apiKey);
          return index === 0
            ? Promise.reject(apiError(400, "credit balance is too low"))
            : okResult();
        }),
      ),
    );
    const result = await model.doGenerate(params);
    expect(calls).toEqual(["k1", "k2"]);
    expect(requireOpenRouterCostUsd(result.providerMetadata)).toBeCloseTo(
      anthropicCostUsd("claude-opus-5-5", USAGE),
      10,
    );
  });

  it("raises non-billing errors without trying other keys", async () => {
    const calls: string[] = [];
    const model = buildAnthropicChatModel(keys, undefined, "low", (modelKeys) =>
      modelKeys.map((key) =>
        fakeModel(() => {
          calls.push(key.apiKey);
          return Promise.reject(apiError(400, "invalid request"));
        }),
      ),
    );
    await expect(model.doGenerate(params)).rejects.toMatchObject({
      statusCode: 400,
    });
    expect(calls).toEqual(["k1"]);
  });

  it("raises the last error when every key is exhausted", async () => {
    const model = buildAnthropicChatModel(keys, undefined, "low", (modelKeys) =>
      modelKeys.map(() =>
        fakeModel(() => Promise.reject(apiError(401, "invalid x-api-key"))),
      ),
    );
    await expect(model.doGenerate(params)).rejects.toMatchObject({
      statusCode: 401,
    });
  });

  it("adds the computed cost to the stream finish part", async () => {
    const model = buildAnthropicChatModel(
      [{ apiKey: "k1" }],
      "claude-sonnet-5-5",
      "low",
      (modelKeys) => modelKeys.map(() => fakeModel(okResult)),
    );
    const { stream } = await model.doStream(params);
    const parts: Array<{ type: string; providerMetadata?: unknown }> = [];
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }
    const finish = parts.find((part) => part.type === "finish");
    expect(finish).toBeDefined();
    expect(requireOpenRouterCostUsd(finish?.providerMetadata)).toBeCloseTo(
      anthropicCostUsd("claude-sonnet-5-5", USAGE),
      10,
    );
  });

  it("sets adaptive thinking and the mapped effort on every call", async () => {
    let seen: unknown;
    const model = buildAnthropicChatModel(
      [{ apiKey: "k1" }],
      undefined,
      "max",
      (modelKeys) =>
        modelKeys.map(() => {
          const base = fakeModel(okResult);
          return {
            ...base,
            doGenerate: async (options: unknown) => {
              seen = options;
              return okResult();
            },
          } as unknown as LanguageModelV3;
        }),
    );
    await model.doGenerate(params);
    expect(seen).toMatchObject({
      providerOptions: {
        anthropic: { thinking: { type: "adaptive" }, effort: "high" },
      },
    });
  });
});
