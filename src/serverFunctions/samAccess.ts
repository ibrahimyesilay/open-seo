import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import {
  CHAT_MODEL_KEY_MISSING_MESSAGE,
  hasChatModelConfigAsync,
} from "@/server/lib/openrouter";
import { isHostedServerAuthMode } from "@/server/lib/runtime-env";
import { requireProjectContext } from "@/serverFunctions/middleware";

const projectScopedSchema = z.object({ projectId: z.string().min(1) });

type SamAccessStatus = {
  enabled: boolean;
  errorMessage: string | null;
};

// Gates the in-app AI agent (SAM) on an LLM provider key (Claude or OpenRouter) being configured, the
// same way backlinks/AI-search gate on their DataForSEO subscriptions. Hosted
// deployments always have the key provisioned, so only self-hosted is checked.
export const getSamAccessSetupStatus = createServerFn({ method: "GET" })
  .middleware(requireProjectContext)
  .validator(projectScopedSchema)
  .handler(async (): Promise<SamAccessStatus> => {
    if (await isHostedServerAuthMode()) {
      return { enabled: true, errorMessage: null };
    }

    const enabled = await hasChatModelConfigAsync();
    return {
      enabled,
      errorMessage: enabled ? null : CHAT_MODEL_KEY_MISSING_MESSAGE,
    };
  });
