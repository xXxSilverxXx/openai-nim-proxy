// server.js — Stable OpenAI-compatible NVIDIA NIM proxy (Chub/JANITOR SAFE)

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// Middleware
// ─────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true, limit: "25mb" }));

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// NO STREAMING (important for Janitor / Chub stability)
const STREAMING_ENABLED = false;

// ─────────────────────────────────────────────
// Model cache (fast / stable / prevents lag in /v1/models)
// ─────────────────────────────────────────────
let cachedModels = [];
let lastModelFetch = 0;

// ─────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─────────────────────────────────────────────
// MODEL LIST (cached + filtered clean)
// ─────────────────────────────────────────────
app.get("/v1/models", async (req, res) => {
  try {
    const now = Date.now();

    // refresh every 10 minutes
    if (!cachedModels.length || now - lastModelFetch > 10 * 60 * 1000) {
      const r = await axios.get(`${NIM_API_BASE}/models`, {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
        },
        timeout: 10000,
      });

      const raw = r.data?.data || [];

      cachedModels = raw
        .filter((m) => m?.id) // only valid models
        .map((m) => ({
          id: m.id,
          object: "model",
          created: m.created || now,
          owned_by: "nvidia-nim",
        }));

      lastModelFetch = now;
    }

    return res.json({
      object: "list",
      data: cachedModels,
    });
  } catch (err) {
    console.error("models error:", err.message);

    return res.json({
      object: "list",
      data: cachedModels,
    });
  }
});

// ─────────────────────────────────────────────
// Chat Completions (CORE ROUTER)
// ─────────────────────────────────────────────
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    // ─────────────────────────────────────────────
    // 1. RESPECT CHUB SELECTION (NEVER OVERRIDE)
    // ─────────────────────────────────────────────
    const requestedModel = model;

    // If model is missing, only then fallback
    const safeModel =
      requestedModel || "meta/llama-3.3-70b-instruct";

    // ─────────────────────────────────────────────
    // 2. SMART FALLBACK ORDER (requested FIRST)
    // ─────────────────────────────────────────────
    const fallbackModels = [
      safeModel, // 👈 CRITICAL: user selection ALWAYS FIRST
      "deepseek-ai/deepseek-v3.2",
      "mistralai/mistral-large-2-instruct",
      "meta/llama-3.3-70b-instruct",
      "qwen/qwen3-next-80b-a3b-instruct",
      "mistralai/mistral-small-4-119b-2603"
    ];

    let lastError = null;
    let response = null;

    // ─────────────────────────────────────────────
    // 3. TRY MODELS IN ORDER
    // ─────────────────────────────────────────────
    for (const modelToTry of fallbackModels) {
      try {
        console.log("Trying model:", modelToTry);

        const payload = {
          model: modelToTry,
          messages,
          temperature: temperature ?? 0.7,
          max_tokens: max_tokens ?? 400,
          top_p: 0.9,
          stream: STREAMING_ENABLED,
        };

        const result = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          payload,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              "Content-Type": "application/json",
            },
            timeout: 90000, // stable for large models
          }
        );

        if (result?.data?.choices) {
          response = result;
          break;
        }
      } catch (err) {
        lastError = err;

        console.warn(
          "Model failed:",
          modelToTry,
          err?.response?.data || err.message
        );

        await sleep(800);
      }
    }

    // ─────────────────────────────────────────────
    // 4. HARD FALLBACK RESPONSE (never crash client)
    // ─────────────────────────────────────────────
    if (!response?.data?.choices) {
      return res.json({
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: requestedModel,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content:
                "[All models are currently overloaded. Please retry shortly.]",
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      });
    }

    // ─────────────────────────────────────────────
    // 5. OPENAI-COMPATIBLE RESPONSE
    // ─────────────────────────────────────────────
    return res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),

      // IMPORTANT: return requested model (not fallback model)
      model: requestedModel,

      choices: response.data.choices.map((c, i) => ({
        index: i,
        message: {
          role: "assistant",
          content: c.message?.content || "",
        },
        finish_reason: c.finish_reason || "stop",
      })),

      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    });
  } catch (error) {
    console.error("Proxy crash:", error?.response?.data || error.message);

    return res.status(500).json({
      error: {
        message:
          error?.response?.data?.error?.message ||
          error.message ||
          "Internal server error",
        type: "server_error",
        code: 500,
      },
    });
  }
});

// ─────────────────────────────────────────────
// Health
// ─────────────────────────────────────────────
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// ─────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`NIM Proxy running on port ${PORT}`);
});
    // Always-safe fallback response

