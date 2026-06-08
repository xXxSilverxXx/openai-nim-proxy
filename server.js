// server.js — Production-grade NVIDIA NIM proxy
// Optimized for:
// - JanitorAI
// - Chub
// - RP quality
// - Render stability
// - NVIDIA NIM
// - Proper parameter passthrough
// - Smart fallbacks
// - Anti-looping support
// - Low-jank behavior

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
process.env.NIM_API_BASE ||
"https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// ─────────────────────────────────────────────
// Middleware
// ─────────────────────────────────────────────

app.use(cors());

app.use(express.json({
limit: "25mb"
}));

app.use(express.urlencoded({
extended: true,
limit: "25mb"
}));

// ─────────────────────────────────────────────
// Global Safety
// ─────────────────────────────────────────────

process.on("unhandledRejection", (err) => {
console.error("UNHANDLED REJECTION:", err);
});

process.on("uncaughtException", (err) => {
console.error("UNCAUGHT EXCEPTION:", err);
});

// Axios should NOT throw on 404/429/etc
axios.defaults.validateStatus = () => true;

// ─────────────────────────────────────────────
// Live Model Cache
// ─────────────────────────────────────────────

let cachedModels = [];
let availableModels = new Set();
let lastModelFetch = 0;

// Refresh every 10 minutes
const MODEL_CACHE_MS = 10 * 60 * 1000;

// ─────────────────────────────────────────────
// RP-Optimized Fallback Order
// ─────────────────────────────────────────────

const FALLBACK_MODELS = [
"deepseek-ai/deepseek-v4-pro",
"deepseek-ai/deepseek-v4-flash",
"writer/palmyra-creative-122b",
"qwen/qwen3-next-80b-a3b-instruct",
"nvidia/llama-3.3-nemotron-super-49b-v1.5",
"meta/llama-3.3-70b-instruct"
];

// ─────────────────────────────────────────────
// Fetch Live Models
// ─────────────────────────────────────────────

async function refreshModels() {
try {

```
const now = Date.now();

if (
  cachedModels.length &&
  now - lastModelFetch < MODEL_CACHE_MS
) {
  return;
}

console.log("Refreshing NVIDIA model cache...");

const response = await axios.get(
  NIM_API_BASE + "/models",
  {
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`
    },
    timeout: 15000
  }
);

const rawModels = response.data?.data || [];

cachedModels = rawModels
  .filter((m) => m && m.id)
  .map((m) => ({
    id: m.id,
    object: "model",
    created: m.created || now,
    owned_by: "nvidia-nim"
  }));

availableModels = new Set(
  cachedModels.map((m) => m.id)
);

lastModelFetch = now;

console.log(
  `Loaded ${cachedModels.length} NVIDIA models`
);
```

} catch (err) {

```
console.error(
  "Model refresh failed:",
  err?.response?.data || err.message
);
```

}
}

// Initial startup refresh
refreshModels();

// ─────────────────────────────────────────────
// Models Endpoint
// ─────────────────────────────────────────────

app.get("/v1/models", async (req, res) => {
await refreshModels();

return res.json({
object: "list",
data: cachedModels
});
});

// ─────────────────────────────────────────────
// Health Endpoint
// ─────────────────────────────────────────────

app.get("/health", (req, res) => {
return res.json({
status: "ok",
models_cached: cachedModels.length
});
});

// ─────────────────────────────────────────────
// Chat Completions
// ─────────────────────────────────────────────

app.post("/v1/chat/completions", async (req, res) => {

try {

```
await refreshModels();

// ─────────────────────────────────────────
// Input Sanitization
// ─────────────────────────────────────────

const body = req.body || {};

const requestedModel =
  typeof body.model === "string"
    ? body.model
    : null;

const safeMessages =
  Array.isArray(body.messages)
    ? body.messages
    : [];

// ─────────────────────────────────────────
// Build Model Attempt List
// ─────────────────────────────────────────

let candidateModels = [];

// User-selected model FIRST
if (
  requestedModel &&
  availableModels.has(requestedModel)
) {
  candidateModels.push(requestedModel);
}

// Add fallbacks
candidateModels.push(...FALLBACK_MODELS);

// Remove duplicates
candidateModels = [...new Set(candidateModels)];

// Keep ONLY currently available models
candidateModels = candidateModels.filter((m) =>
  availableModels.has(m)
);

// Absolute emergency fallback
if (!candidateModels.length) {
  candidateModels = [
    "meta/llama-3.3-70b-instruct"
  ];
}

let finalResponse = null;
let finalModel = null;

// ─────────────────────────────────────────
// Attempt Models
// ─────────────────────────────────────────

for (const modelToTry of candidateModels) {

  console.log(`Trying model: ${modelToTry}`);

  // Dynamic passthrough payload
  // IMPORTANT:
  // We ONLY send params frontend provided.
  // This preserves Janitor/Chub tuning.
  const payload = {
    model: modelToTry,
    messages: safeMessages,
    stream: false
  };

  // Safe passthrough params
  const passthroughParams = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "stop"
  ];

  for (const key of passthroughParams) {
    if (body[key] !== undefined) {
      payload[key] = body[key];
    }
  }

  // Small sanity defaults ONLY if absent
  if (payload.max_tokens === undefined) {
    payload.max_tokens = 500;
  }

  try {

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      payload,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json"
        },

        // MUCH safer for Render
        timeout: 30000
      }
    );

    // Handle non-200 gracefully
    if (response.status !== 200) {

      console.warn(
        `Model ${modelToTry} failed with status ${response.status}`
      );

      console.warn(response.data);

      continue;
    }

    // Validate response
    if (
      !response.data ||
      !Array.isArray(response.data.choices) ||
      !response.data.choices.length
    ) {

      console.warn(
        `Model ${modelToTry} returned invalid response`
      );

      continue;
    }

    finalResponse = response.data;
    finalModel = modelToTry;

    console.log(
      `SUCCESS with ${modelToTry}`
    );

    break;

  } catch (err) {

    if (err.code === "ECONNABORTED") {

      console.warn(
        `Timeout from ${modelToTry}`
      );

    } else {

      console.warn(
        `Error from ${modelToTry}:`,
        err?.response?.data || err.message
      );

    }
  }
}

// ─────────────────────────────────────────
// Total Failure
// ─────────────────────────────────────────

if (!finalResponse) {

  return res.json({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),

    model:
      requestedModel ||
      "unavailable",

    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content:
            "[All NVIDIA models are currently unavailable or overloaded. Please retry shortly.]"
        },
        finish_reason: "stop"
      }
    ],

    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  });
}

// ─────────────────────────────────────────
// OpenAI-Compatible Response
// ─────────────────────────────────────────

return res.json({

  id:
    finalResponse.id ||
    `chatcmpl-${Date.now()}`,

  object: "chat.completion",

  created:
    finalResponse.created ||
    Math.floor(Date.now() / 1000),

  // IMPORTANT:
  // Return ACTUAL responding model
  model: finalModel,

  choices: finalResponse.choices.map(
    (choice, index) => ({
      index,

      message: {
        role:
          choice.message?.role ||
          "assistant",

        content:
          choice.message?.content ||
          ""
      },

      finish_reason:
        choice.finish_reason ||
        "stop"
    })
  ),

  usage:
    finalResponse.usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
});
```

} catch (err) {

```
console.error(
  "Proxy crash:",
  err?.response?.data || err.message
);

return res.status(500).json({
  error: {
    message:
      err?.response?.data?.error?.message ||
      err.message ||
      "Internal server error",

    type: "server_error",
    code: 500
  }
});
```

}
});

// ─────────────────────────────────────────────
// Start Server
// ─────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {

console.log(
`NVIDIA NIM Proxy running on port ${PORT}`
);

});
