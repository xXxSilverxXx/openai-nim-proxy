const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────
// Middleware
// ─────────────────────────────
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true }));

// ─────────────────────────────
// Config
// ─────────────────────────────
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// ─────────────────────────────
// Chat-safe model filter
// (prevents embeddings/vision breaking chat)
// ─────────────────────────────
const CHAT_KEYWORDS = [
  "instruct",
  "chat",
  "llama",
  "mistral",
  "mixtral",
  "qwen",
  "deepseek",
  "nemotron",
  "kimi",
  "gpt-oss",
  "solar",
  "step",
  "granite",
  "writer"
];

// ─────────────────────────────
// Basic routes
// ─────────────────────────────
app.get("/", (req, res) => {
  res.send("NIM Proxy Running");
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// ─────────────────────────────
// SAFE MODEL LIST (FIXED)
// ─────────────────────────────
app.get("/v1/models", async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`
      },
      timeout: 15000
    });

    const rawModels = r.data?.data || [];

    const chatModels = rawModels.filter((m) => {
      const id = (m.id || "").toLowerCase();
      return CHAT_KEYWORDS.some((k) => id.includes(k));
    });

    const models = chatModels.map((m) => ({
      id: m.id,
      object: "model",
      created: m.created || Date.now(),
      owned_by: "nvidia-nim"
    }));

    res.json({
      object: "list",
      data: models
    });
  } catch (err) {
    console.error("Model fetch failed:", err.message);

    res.json({
      object: "list",
      data: []
    });
  }
});

// ─────────────────────────────
// CHAT COMPLETIONS (STABLE ROUTER)
// ─────────────────────────────
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    // Safe fallback list (ONLY real chat models)
    const fallbackModels = [
      "deepseek-ai/deepseek-v3.2",
      "meta/llama-3.3-70b-instruct",
      "mistralai/mistral-large-2-instruct",
      "qwen/qwen3-next-80b-a3b-instruct",
      "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    ];

    // If frontend sends unknown model, ignore it safely
    const requested = (model || "").toLowerCase();

    let modelQueue = [];

    const matchedFallback = fallbackModels.find((m) =>
      m.toLowerCase().includes(requested)
    );

    if (matchedFallback) {
      modelQueue.push(matchedFallback);
    }

    modelQueue = [
      ...modelQueue,
      ...fallbackModels.filter((m) => !modelQueue.includes(m))
    ];

    let lastError = null;

    for (const m of modelQueue) {
      try {
        console.log("Trying model:", m);

        const result = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          {
            model: m,
            messages,
            temperature: temperature ?? 0.7,
            max_tokens: max_tokens ?? 300,
            top_p: 0.9,
            stream: false
          },
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              "Content-Type": "application/json"
            },
            timeout: 60000
          }
        );

        if (result.data?.choices) {
          return res.json({
            id: `chatcmpl-${Date.now()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: m,
            choices: result.data.choices.map((c, i) => ({
              index: i,
              message: {
                role: "assistant",
                content: c.message?.content || ""
              },
              finish_reason: c.finish_reason
            })),
            usage: result.data.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
          });
        }
      } catch (err) {
        lastError = err;
        console.warn("Model failed:", m, err.message);
      }
    }

    // Always-safe fallback response
    return res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "All models are temporarily unavailable. Please try again."
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
  } catch (err) {
    console.error("Fatal error:", err.message);

    return res.status(500).json({
      error: {
        message: err.message,
        type: "server_error",
        code: 500
      }
    });
  }
});

// ─────────────────────────────
// 404 fallback
// ─────────────────────────────
app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Route ${req.path} not found`,
      type: "invalid_request_error",
      code: 404
    }
  });
});

// ─────────────────────────────
// Start server
// ─────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`NIM Proxy running on port ${PORT}`);
});
