import type { LanguageModelV3 } from "@openrouter/ai-sdk-provider";
import { APICallError, wrapLanguageModel } from "ai";

// Direct Anthropic provider for the SAM chat agent and the research/prompt
// generation flows, as an alternative to OpenRouter. Motivation: Claude Max /
// Team subscriptions grant monthly Claude API credits per Console
// organization, so a self-hoster can run the in-app AI features on those
// instead of paying OpenRouter. Each credit pool lives in its own
// organization, hence a *list* of keys: calls go to the first key and fail
// over to the next when a pool is exhausted (or a key is rejected), so
// several subscriptions chain into one larger monthly allowance.
//
// The rest of the app meters LLM spend through
// `providerMetadata.openrouter.usage.cost` (see chatAgent.ts). Anthropic does
// not return a USD cost, so this module computes one from token usage and the
// public price table and publishes it under the same metadata key — every
// existing cost reader keeps working unchanged.

// Default when ANTHROPIC_MODEL is unset.
const DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5";

type AnthropicKey = { apiKey: string; workspaceId?: string };

/**
 * Parses ANTHROPIC_API_KEYS: comma-separated entries of `key` or
 * `key@wrkspc_…`. The `@workspace` suffix is for user-scoped keys
 * (`sk-ant-usr-…`) that are not bound to a workspace; the API then requires
 * the `anthropic-workspace-id` header on every request.
 */
export function parseAnthropicKeys(raw: string | undefined): AnthropicKey[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const at = entry.indexOf("@");
      if (at === -1) return { apiKey: entry };
      return {
        apiKey: entry.slice(0, at).trim(),
        workspaceId: entry.slice(at + 1).trim() || undefined,
      };
    })
    .filter((key) => key.apiKey.length > 0);
}

// USD per million tokens: [input, cache write (5m), cache read, output].
// Source: platform.claude.com/docs/en/about-claude/pricing, 2026-10-17.
// Haiku 5.5 is priced at its ≤100k-token tier; prompts above that cost 5x.
// Longest prefix wins, so dated/suffixed ids resolve to their family.
const PRICE_TABLE_USD_PER_MTOK: Array<[prefix: string, prices: number[]]> = [
  ["claude-fable-5-1", [10, 12.5, 0.25, 50]],
  ["claude-fable-5", [10, 12.5, 1, 50]],
  ["claude-opus-5-5", [4, 5, 0.2, 20]],
  ["claude-opus-5", [5, 6.25, 0.5, 25]],
  ["claude-opus-4-8", [5, 6.25, 0.5, 25]],
  ["claude-opus-4-7", [5, 6.25, 0.5, 25]],
  ["claude-opus-4-6", [5, 6.25, 0.5, 25]],
  ["claude-opus-4-5", [5, 6.25, 0.5, 25]],
  ["claude-sonnet-5-5", [2, 2.5, 0.1, 10]],
  ["claude-sonnet-5", [2, 2.5, 0.2, 10]],
  ["claude-sonnet-4-6", [3, 3.75, 0.3, 15]],
  ["claude-sonnet-4-5", [3, 3.75, 0.3, 15]],
  ["claude-haiku-5-5", [0.1, 0.125, 0.01, 0.5]],
  ["claude-haiku-4-5", [1, 1.25, 0.1, 5]],
];

function pricesFor(modelId: string): number[] | null {
  let best: [string, number[]] | null = null;
  for (const row of PRICE_TABLE_USD_PER_MTOK) {
    if (
      modelId.startsWith(row[0]) &&
      (!best || row[0].length > best[0].length)
    ) {
      best = row;
    }
  }
  return best?.[1] ?? null;
}

type Usage = {
  inputTokens: {
    total: number | undefined;
    noCache: number | undefined;
    cacheRead: number | undefined;
    cacheWrite: number | undefined;
  };
  outputTokens: { total: number | undefined };
};

/** USD cost of one response from its token usage; 0 for unknown models. */
export function anthropicCostUsd(modelId: string, usage: Usage): number {
  const prices = pricesFor(modelId);
  if (!prices) return 0;
  const [inputUsd, cacheWriteUsd, cacheReadUsd, outputUsd] = prices;
  const cacheRead = usage.inputTokens.cacheRead ?? 0;
  const cacheWrite = usage.inputTokens.cacheWrite ?? 0;
  const noCache =
    usage.inputTokens.noCache ??
    Math.max(0, (usage.inputTokens.total ?? 0) - cacheRead - cacheWrite);
  const output = usage.outputTokens.total ?? 0;
  const usd =
    (noCache * inputUsd +
      cacheWrite * cacheWriteUsd +
      cacheRead * cacheReadUsd +
      output * outputUsd) /
    1_000_000;
  return Math.round(usd * 1e8) / 1e8;
}

/**
 * True for errors that mean "this key/organization cannot serve the call
 * right now but another one might": an exhausted credit pool (400 with
 * Anthropic's "credit balance is too low" message), a rejected key (401/403),
 * or a per-organization rate limit (429). Everything else (bad request,
 * overloaded, network) is raised as-is — retrying it on another org would
 * just fail again or double-spend.
 */
export function shouldFailOverToNextKey(error: unknown): boolean {
  if (!APICallError.isInstance(error)) return false;
  const status = error.statusCode;
  if (status === 401 || status === 403 || status === 429) return true;
  if (status === 400) {
    const body = `${error.responseBody ?? ""} ${error.message}`;
    return /credit balance/i.test(body);
  }
  return false;
}

type ReasoningEffort = "max" | "low";

// The SAM agent runs at effort "max" on OpenRouter; on Claude, `high` is the
// documented sweet spot for agentic tool loops and keeps a $200/month credit
// pool from draining on reasoning tokens. Research/suggestion flows ask for
// "low" and get it unchanged.
function anthropicEffort(effort: ReasoningEffort): "high" | "low" {
  return effort === "low" ? "low" : "high";
}

function withCost(
  modelId: string,
  usage: Usage,
  providerMetadata: Record<string, Record<string, unknown>> | undefined,
) {
  return {
    ...providerMetadata,
    openrouter: {
      ...providerMetadata?.openrouter,
      usage: { cost: anthropicCostUsd(modelId, usage) },
    },
  };
}

// `@ai-sdk/anthropic` is loaded lazily on the first call: the lean-worker
// bundle plugin denylists it from the worker's eager startup graph (isolate
// memory), and the SAM DO's `getModel()` hook is sync, so the provider cannot
// be awaited at construction time. Loaded once per isolate.
const defaultBuildModels: BuildModels = async (modelKeys, resolvedModel) => {
  const { createAnthropic } = await import("@ai-sdk/anthropic");
  return modelKeys.map((key) =>
    createAnthropic({
      apiKey: key.apiKey,
      headers: key.workspaceId
        ? { "anthropic-workspace-id": key.workspaceId }
        : undefined,
    })(resolvedModel),
  );
};

type BuildModels = (
  modelKeys: AnthropicKey[],
  resolvedModel: string,
) => LanguageModelV3[] | Promise<LanguageModelV3[]>;

/**
 * Builds the AI SDK LanguageModel for Claude. `keys` are tried in order on
 * every call (see shouldFailOverToNextKey). The returned model reports the
 * computed USD cost under `providerMetadata.openrouter.usage.cost` on
 * generate results and on the stream `finish` part.
 *
 * `buildModels` is injectable so tests can substitute fake models.
 */
export function buildAnthropicChatModel(
  keys: AnthropicKey[],
  modelId: string | undefined,
  effort: ReasoningEffort,
  buildModels: BuildModels = defaultBuildModels,
): LanguageModelV3 {
  if (keys.length === 0) {
    throw new Error("buildAnthropicChatModel needs at least one API key");
  }
  const resolvedModelId = modelId ?? DEFAULT_ANTHROPIC_MODEL;
  let modelsPromise: Promise<LanguageModelV3[]> | undefined;
  const getModels = () => {
    modelsPromise ??= Promise.resolve(buildModels(keys, resolvedModelId)).then(
      (models) => {
        if (models.length === 0) {
          throw new Error("buildAnthropicChatModel produced no models");
        }
        return models;
      },
    );
    return modelsPromise;
  };

  async function tryEach<T>(
    run: (model: LanguageModelV3, index: number) => PromiseLike<T>,
  ): Promise<T> {
    const models = await getModels();
    let lastError: unknown;
    for (const [index, model] of models.entries()) {
      try {
        return await run(model, index);
      } catch (error) {
        lastError = error;
        const hasNext = index < models.length - 1;
        if (!hasNext || !shouldFailOverToNextKey(error)) throw error;
        console.warn(
          `[anthropic] key #${index + 1} failed (${APICallError.isInstance(error) ? error.statusCode : "?"}); failing over to key #${index + 2}`,
        );
      }
    }
    throw lastError;
  }

  // The wrapped base never runs: the middleware below replaces both call
  // paths and dispatches to the lazily built provider models.
  const base: LanguageModelV3 = {
    specificationVersion: "v3",
    provider: "anthropic",
    modelId: resolvedModelId,
    supportedUrls: {},
    doGenerate: () => {
      throw new Error("unreachable: replaced by the failover middleware");
    },
    doStream: () => {
      throw new Error("unreachable: replaced by the failover middleware");
    },
  };

  return wrapLanguageModel({
    model: base,
    middleware: {
      specificationVersion: "v3",
      transformParams: async ({ params }) => ({
        ...params,
        providerOptions: {
          ...params.providerOptions,
          anthropic: {
            ...params.providerOptions?.anthropic,
            thinking: { type: "adaptive" },
            effort: anthropicEffort(effort),
          },
        },
      }),
      wrapGenerate: async ({ params }) => {
        const result = await tryEach((model) => model.doGenerate(params));
        return {
          ...result,
          providerMetadata: withCost(
            resolvedModelId,
            result.usage,
            result.providerMetadata,
          ),
        };
      },
      wrapStream: async ({ params }) => {
        const result = await tryEach((model) => model.doStream(params));
        type Part =
          Awaited<
            ReturnType<LanguageModelV3["doStream"]>
          >["stream"] extends ReadableStream<infer P>
            ? P
            : never;
        const stream = result.stream.pipeThrough(
          new TransformStream<Part, Part>({
            transform(part, controller) {
              if (part.type === "finish") {
                controller.enqueue({
                  ...part,
                  providerMetadata: withCost(
                    resolvedModelId,
                    part.usage,
                    part.providerMetadata,
                  ),
                });
                return;
              }
              controller.enqueue(part);
            },
          }),
        );
        return { ...result, stream };
      },
    },
  });
}
