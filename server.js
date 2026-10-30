
"use strict";

// ============================================================
// NVIDIA NIM -> OPENAI-COMPATIBLE CHAT PROXY
// ONE MODEL AT A TIME. NO AUTOMATIC MODEL FALLBACK.
// ============================================================
//
// Requirements: express, cors, axios (same as your old server.js)
// Render environment variables:
//   NIM_API_KEY   = your NVIDIA API key (NVIDIA_API_KEY also works)
//   ACTIVE_MODEL  = one exact model ID from MODEL_PROFILES below
//                   (optional; defaults to Mistral Large 4)
//   NIM_API_BASE  = optional custom NVIDIA URL
//
// To test another model: change ACTIVE_MODEL in Render and redeploy.
// All other models below are RESERVE OPTIONS ONLY. They will never
// receive a request unless you explicitly make one the ACTIVE_MODEL.
// The client's model field cannot override your active model.
// ============================================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const NIM_API_BASE = (
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");
const NIM_API_KEY = process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY;

// ============================================================
// MODEL LIBRARY (ONLY THE SELECTED ONE IS ACTIVE)
// ============================================================
// These are model-specific starting points, not benchmark winners.
// Client temperature/top_p/max_tokens take priority when supplied.
// Optional model flags are applied ONLY to their own model.
// ============================================================

const MODEL_PROFILES = Object.freeze({
  // START HERE: strong, straightforward non-reasoning RP candidate.
  "mistralai/mistral-large-4-1t-a52b": {
  maxTokens: 4096,
  temperature: 0.7,
  advanced: ["frequency_penalty", "presence_penalty", "tools", "tool_choice"]
},

  // RESERVE OPTIONS: never used unless explicitly selected.
  "z-ai/glm-5.3-flash": {
    maxTokens: 4096,
    temperature: 0.7,
    extra: { reasoning_effort: "low", chat_template_kwargs: { clear_thinking: true } },
    advanced: ["frequency_penalty", "presence_penalty", "tools", "tool_choice"]
  },
  "minimaxai/minimax-m3.1-flash": {
    maxTokens: 6144,
    temperature: 1.0,
    topP: 0.95,
    // Reasoning controls on this endpoint may depend on its serving recipe.
    // Don't inject unverified flags that could cause HTTP 400/422.
    advanced: ["tools", "tool_choice"]
  },
  "moonshotai/kimi-k2.6": {
    maxTokens: 4096,
    temperature: 0.7,
    extra: { chat_template_kwargs: { thinking: false } },
    advanced: ["tools", "tool_choice"]
  },
  "nvidia/nemotron-3.5-lightning-30b-a3b": {
    maxTokens: 3072,
    temperature: 1.0,
    topP: 0.95,
    extra: { chat_template_kwargs: { enable_thinking: false } }
  },
  "nvidia/nemotron-3.5-bolt-e4b": {
    maxTokens: 3072,
    temperature: 1.0,
    topP: 0.95,
    extra: { chat_template_kwargs: { enable_thinking: false } },
    advanced: ["tools", "tool_choice", "response_format"]
  },
  "google/gemma-4-31b-it": {
    maxTokens: 3072,
    temperature: 0.8,
    extra: { chat_template_kwargs: { enable_thinking: false } }
  },
  "writer/palmyra-creative-122b": {
    maxTokens: 3072,
    temperature: 0.8
  },
  "nvidia/nemotron-3-ultra-550b-a55b": {
    maxTokens: 4096,
    temperature: 0.7
  },
  "stepfun-ai/step-5-preview": {
    maxTokens: 6144,
    temperature: 0.7,
    // NVIDIA says Step 5 reasoning cannot be turned off.
    advanced: ["frequency_penalty", "presence_penalty", "tools", "tool_choice"]
  },
  "deepseek-ai/deepseek-v4.1-flash": {
    maxTokens: 4096,
    temperature: 0.7
  }
});

const DEFAULT_ACTIVE_MODEL = "mistralai/mistral-large-4-1t-a52b";
const ACTIVE_MODEL = (process.env.ACTIVE_MODEL || DEFAULT_ACTIVE_MODEL).trim();
const ACTIVE_PROFILE = Object.prototype.hasOwnProperty.call(MODEL_PROFILES, ACTIVE_MODEL)
  ? MODEL_PROFILES[ACTIVE_MODEL]
  : null;

// These are informational only; this array is NEVER used for requests.
const RESERVE_MODELS = Object.keys(MODEL_PROFILES).filter(
  (name) => name !== ACTIVE_MODEL
);

const REQUEST_TIMEOUT_MS = 150000; // 2.5 minutes
const MAX_OUTPUT_TOKENS = 8192; // Hard safety cap across all models

app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true, limit: "25mb" }));

function apiError(res, status, message, type) {
  return res.status(status).json({
    error: { message, type, code: status }
  });
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        return isObject(part) && typeof part.text === "string" ? part.text : "";
      })
      .join("");
  }
  return "";
}

function buildPayload(body) {
  const profile = ACTIVE_PROFILE;
  const payload = {
    model: ACTIVE_MODEL, // never body.model
    messages: body.messages,
    stream: false, // forced non-streaming, even if the client asks for true
    max_tokens: profile.maxTokens
  };

  if (Number.isFinite(profile.temperature)) {
    payload.temperature = profile.temperature;
  }
  if (Number.isFinite(profile.topP)) {
    payload.top_p = profile.topP;
  }

  // A small, compatible set of frontend parameters. In particular, do
  // NOT blindly forward top_k, min_p, or a frontend reasoning setting;
  // unsupported parameters can cause otherwise-good models to fail.
  for (const name of ["temperature", "top_p", "stop", "seed"]) {
    if (body[name] !== undefined && body[name] !== null) {
      payload[name] = body[name];
    }
  }
  for (const name of profile.advanced || []) {
    if (body[name] !== undefined && body[name] !== null) {
      payload[name] = body[name];
    }
  }

  const clientTokens = body.max_tokens ?? body.max_completion_tokens;
  if (Number.isInteger(clientTokens) && clientTokens > 0) {
    payload.max_tokens = Math.min(clientTokens, MAX_OUTPUT_TOKENS);
  } else {
    payload.max_tokens = Math.min(profile.maxTokens, MAX_OUTPUT_TOKENS);
  }

  // Model-specific compatibility flags ALWAYS win. Clients cannot
  // accidentally turn thinking back on in our tested non-thinking modes.
  if (profile.extra) {
    Object.assign(payload, profile.extra);
  }

  return payload;
}

function configurationError() {
  if (!NIM_API_KEY) return "Missing NIM_API_KEY (or NVIDIA_API_KEY) in Render.";
  if (!ACTIVE_PROFILE) {
    return "Unknown ACTIVE_MODEL: " + ACTIVE_MODEL +
      ". Choose an exact model ID from MODEL_PROFILES in server.js.";
  }
  return null;
}

// ============================================================
// HEALTH AND MODEL DISCOVERY
// ============================================================
// /v1/models intentionally advertises ONLY the active model.
// It does not claim that NVIDIA has confirmed it is online.
// ============================================================

app.get("/", (_req, res) => {
  res.json({ service: "NVIDIA NIM chat proxy", active_model: ACTIVE_MODEL });
});

app.get("/health", (_req, res) => {
  const problem = configurationError();
  res.status(problem ? 503 : 200).json({
    status: problem ? "misconfigured" : "ok",
    active_model: ACTIVE_MODEL,
    reserve_models: RESERVE_MODELS.length,
    api_key_configured: Boolean(NIM_API_KEY),
    ...(problem ? { error: problem } : {})
  });
});

// Optional full-catalog endpoint for checking what NVIDIA has added.
// This is informational ONLY; catalog models are never auto-activated.
// Keep /v1/models limited to one model so chat clients stay predictable.
let fullCatalog = null;
let fullCatalogFetchedAt = 0;
const CATALOG_TTL_MS = 10 * 60 * 1000;

app.get("/nvidia/models", async (_req, res) => {
  if (!NIM_API_KEY) {
    return apiError(res, 503, "Missing NVIDIA API key.", "configuration_error");
  }
  if (fullCatalog && Date.now() - fullCatalogFetchedAt < CATALOG_TTL_MS) {
    return res.json(fullCatalog);
  }
  try {
    const response = await axios.get(NIM_API_BASE + "/models", {
      headers: { Authorization: "Bearer " + NIM_API_KEY },
      timeout: 30000,
      validateStatus: () => true
    });
    if (response.status !== 200 || !Array.isArray(response.data?.data)) {
      console.error("NVIDIA model catalog HTTP", response.status);
      return apiError(res, 502, "Could not retrieve NVIDIA's model catalog.", "upstream_error");
    }
    fullCatalog = response.data;
    fullCatalogFetchedAt = Date.now();
    return res.json(fullCatalog);
  } catch (err) {
    console.error("NVIDIA model catalog error:", err.code || err.message);
    return apiError(res, 502, "Could not retrieve NVIDIA's model catalog.", "upstream_error");
  }
});

app.get("/v1/models", (_req, res) => {
  if (!ACTIVE_PROFILE) {
    return apiError(res, 503, configurationError(), "configuration_error");
  }
  res.json({
    object: "list",
    data: [{
      id: ACTIVE_MODEL,
      object: "model",
      created: 0,
      owned_by: ACTIVE_MODEL.split("/")[0]
    }]
  });
});

// ============================================================
// CHAT COMPLETIONS: EXACTLY ONE NVIDIA REQUEST, ONE MODEL
// ============================================================

app.post("/v1/chat/completions", async (req, res) => {
  const problem = configurationError();
  if (problem) return apiError(res, 503, problem, "configuration_error");

  const body = req.body;
  if (!isObject(body) || !Array.isArray(body.messages) || body.messages.length === 0) {
    return apiError(res, 400, "Provide a nonempty messages array.", "invalid_request_error");
  }

  const payload = buildPayload(body);
  if (typeof body.model === "string" && body.model !== ACTIVE_MODEL) {
    console.log("Client model override ignored; using ACTIVE_MODEL:", ACTIVE_MODEL);
  }

  console.log("NVIDIA request ->", ACTIVE_MODEL,
    "tokens:", payload.max_tokens, "stream:", payload.stream);

  let upstream;
  try {
    upstream = await axios.post(
      NIM_API_BASE + "/chat/completions",
      payload,
      {
        headers: {
          Authorization: "Bearer " + NIM_API_KEY,
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        timeout: REQUEST_TIMEOUT_MS,
        validateStatus: () => true
      }
    );
  } catch (err) {
    const timedOut = err.code === "ECONNABORTED" || err.code === "ETIMEDOUT";
    console.error("NVIDIA network failure:", ACTIVE_MODEL, err.code || err.message);
    return apiError(
      res,
      timedOut ? 504 : 502,
      timedOut
        ? "The active model timed out. No reserve model was attempted."
        : "Could not reach NVIDIA. No reserve model was attempted.",
      "upstream_error"
    );
  }

  if (upstream.status !== 200) {
    const remoteMessage = upstream.data?.error?.message || upstream.data?.message;
    console.error("NVIDIA refused", ACTIVE_MODEL, "HTTP", upstream.status,
      typeof remoteMessage === "string" ? remoteMessage.slice(0, 500) : "");

    if (upstream.status === 401 || upstream.status === 403) {
      return apiError(res, upstream.status,
        "NVIDIA rejected the API key or access to this model. Check Render credentials and NVIDIA permissions.",
        "authentication_error");
    }
    if (upstream.status === 402) {
      return apiError(res, 402, "NVIDIA reports insufficient API credits.", "billing_error");
    }
    if (upstream.status === 429) {
      if (upstream.headers?.["retry-after"]) {
        res.set("Retry-After", upstream.headers["retry-after"]);
      }
      return apiError(res, 429, "NVIDIA rate-limited the active model. No fallback attempted.", "rate_limit_error");
    }
    if (upstream.status === 400 || upstream.status === 422) {
      const detail = typeof remoteMessage === "string"
        ? remoteMessage.slice(0, 220)
        : "Check model-specific request settings in the Render logs.";
      return apiError(res, 502,
        "NVIDIA rejected the active model request: " + detail,
        "upstream_request_error");
    }
    return apiError(res, 502,
      "NVIDIA returned HTTP " + upstream.status + " for the active model. No fallback attempted.",
      "upstream_error");
  }

  const result = upstream.data;
  const choice = Array.isArray(result?.choices) ? result.choices[0] : null;
  const message = isObject(choice?.message) ? choice.message : null;
  const content = nonEmptyText(message?.content);
  const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];

  // Crucial fix: a choices array is NOT proof that the model answered.
  // Never send an empty "successful" roleplay reply to Janitor/Chub.
  // Reasoning-only outputs are NOT presented as character dialogue.
  if (!content.trim() && toolCalls.length === 0) {
    const finish = typeof choice?.finish_reason === "string"
      ? choice.finish_reason : "unknown";
    console.error("EMPTY MODEL RESPONSE:", ACTIVE_MODEL,
      "finish_reason:", finish,
      "completion_tokens:", result?.usage?.completion_tokens ?? "unknown");
    return apiError(res, 502,
      "The active model returned no visible reply (finish_reason: " + finish +
      "). Try increasing max_tokens in your chat app. No fallback attempted.",
      "empty_model_response");
  }

  const outgoingMessage = {
    role: "assistant",
    content: content || null
  };
  if (toolCalls.length > 0) outgoingMessage.tool_calls = toolCalls;

  console.log("SUCCESS:", ACTIVE_MODEL,
    "characters:", content.length,
    "finish_reason:", choice.finish_reason || "unknown");

  return res.json({
    id: typeof result.id === "string" ? result.id : "chatcmpl-" + Date.now(),
    object: "chat.completion",
    created: Number.isInteger(result.created)
      ? result.created : Math.floor(Date.now() / 1000),
    model: ACTIVE_MODEL,
    choices: [{
      index: 0,
      message: outgoingMessage,
      finish_reason: choice.finish_reason || (toolCalls.length ? "tool_calls" : "stop")
    }],
    usage: isObject(result.usage) ? result.usage : {
      prompt_tokens: 0, completion_tokens: 0, total_tokens: 0
    }
  });
});

// Avoid uncaught errors escaping Express. Do not claim that a model
// succeeded or switch to a reserve on errors.
app.use((err, _req, res, _next) => {
  console.error("Proxy error:", err.message || err);
  if (res.headersSent) return;
  if (err.type === "entity.too.large") {
    return apiError(res, 413, "Request exceeds the 25 MB limit.", "invalid_request_error");
  }
  if (err instanceof SyntaxError && "body" in err) {
    return apiError(res, 400, "Malformed JSON request body.", "invalid_request_error");
  }
  return apiError(res, 500, "Internal proxy error.", "server_error");
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("NVIDIA NIM Proxy listening on port", PORT);
  console.log("ACTIVE MODEL:", ACTIVE_MODEL);
  console.log("RESERVE MODELS:", RESERVE_MODELS.length, "(disabled; no fallback)");
  const problem = configurationError();
  if (problem) console.error("CONFIGURATION ERROR:", problem);
});
