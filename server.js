const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY =
  process.env.NIM_API_KEY;

// Middleware
app.use(cors());

app.use(express.json({
  limit: "25mb"
}));

app.use(express.urlencoded({
  extended: true,
  limit: "25mb"
}));

// Prevent crashes
process.on("unhandledRejection", function(err) {
  console.error("UNHANDLED REJECTION:", err);
});

process.on("uncaughtException", function(err) {
  console.error("UNCAUGHT EXCEPTION:", err);
});

// Axios should not throw on non-200
axios.defaults.validateStatus = function () {
  return true;
};

// Model cache
var cachedModels = [];
var availableModels = {};
var lastModelFetch = 0;

var MODEL_CACHE_MS = 10 * 60 * 1000;

// RP-optimized fallback list
var FALLBACK_MODELS = [
  "deepseek-ai/deepseek-v4-flash",
  "nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "meta/llama-3.3-70b-instruct",
  "deepseek-ai/deepseek-v4-pro"
];

// Refresh model cache
async function refreshModels() {

  try {

    var now = Date.now();

    if (
      cachedModels.length > 0 &&
      (now - lastModelFetch) < MODEL_CACHE_MS
    ) {
      return;
    }

    console.log("Refreshing NVIDIA models...");

    var response = await axios.get(
      NIM_API_BASE + "/models",
      {
        headers: {
          Authorization: "Bearer " + NIM_API_KEY
        },
        timeout: 30000
      }
    );

    var rawModels = [];

    if (
      response &&
      response.data &&
      Array.isArray(response.data.data)
    ) {
      rawModels = response.data.data;
    }

    cachedModels = [];
    availableModels = {};

    for (var i = 0; i < rawModels.length; i++) {

      var m = rawModels[i];

      if (m && m.id) {

        cachedModels.push({
          id: m.id,
          object: "model",
          created: m.created || now,
          owned_by: "nvidia-nim"
        });

        availableModels[m.id] = true;
      }
    }

    lastModelFetch = now;

    console.log(
      "Loaded " +
      cachedModels.length +
      " NVIDIA models"
    );

  } catch (err) {

    console.error(
      "Model refresh failed:",
      err.message || err
    );
  }
}

// Initial model load
refreshModels();

// Models endpoint
app.get("/v1/models", async function(req, res) {

  await refreshModels();

  return res.json({
    object: "list",
    data: cachedModels
  });

});

// Health endpoint
app.get("/health", function(req, res) {

  return res.json({
    status: "ok",
    models_cached: cachedModels.length
  });

});

// Chat completions endpoint
app.post("/v1/chat/completions", async function(req, res) {

  try {

    await refreshModels();

    var body = req.body || {};

    var requestedModel = null;

    if (typeof body.model === "string") {
      requestedModel = body.model;
    }

    var safeMessages = [];

    if (Array.isArray(body.messages)) {
      safeMessages = body.messages;
    }

    // Build model list
    var candidateModels = [];

    if (
      requestedModel &&
      availableModels[requestedModel]
    ) {
      candidateModels.push(requestedModel);
    }

    for (var i = 0; i < FALLBACK_MODELS.length; i++) {

      var fallback = FALLBACK_MODELS[i];

      if (
        availableModels[fallback] &&
        candidateModels.indexOf(fallback) === -1
      ) {
        candidateModels.push(fallback);
      }
    }

    // Emergency fallback
    if (candidateModels.length === 0) {
      candidateModels.push(
        "meta/llama-3.3-70b-instruct"
      );
    }

    var finalResponse = null;
    var finalModel = null;

    // Try models
    for (var j = 0; j < candidateModels.length; j++) {

      var modelToTry = candidateModels[j];

      console.log(
        "Trying model:",
        modelToTry
      );

      var payload = {
        model: modelToTry,
        messages: safeMessages,
        stream: false
      };

      // Pass through frontend params
      var passthrough = [
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

      for (var k = 0; k < passthrough.length; k++) {

        var key = passthrough[k];

        if (body[key] !== undefined) {
          payload[key] = body[key];
        }
      }

      // Safe default
      if (payload.max_tokens === undefined) {
        payload.max_tokens = 1250;
      }

      try {

        var response = await axios.post(
          NIM_API_BASE + "/chat/completions",
          payload,
          {
            headers: {
              Authorization: "Bearer " + NIM_API_KEY,
              "Content-Type": "application/json"
            },
            timeout: 60000
          }
        );

        if (response.status !== 200) {

          console.warn(
            "Model failed:",
            modelToTry,
            "status:",
            response.status
          );

          continue;
        }

        if (
          !response.data ||
          !Array.isArray(response.data.choices) ||
          response.data.choices.length === 0
        ) {

          console.warn(
            "Invalid response from:",
            modelToTry
          );

          continue;
        }

        finalResponse = response.data;
        finalModel = modelToTry;

        console.log(
          "SUCCESS:",
          modelToTry
        );

        break;

      } catch (err) {

        if (err.code === "ECONNABORTED") {

          console.warn(
            "Timeout from:",
            modelToTry
          );

        } else {

          console.warn(
            "Error from:",
            modelToTry,
            err.message || err
          );
        }
      }
    }

    // Total failure
    if (!finalResponse) {

      return res.json({
        id: "chatcmpl-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),

        model:
          requestedModel || "unavailable",

        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content:
                "[All NVIDIA models are currently unavailable. Please retry shortly.]"
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

    // OpenAI-compatible response
    return res.json({

      id:
        finalResponse.id ||
        ("chatcmpl-" + Date.now()),

      object: "chat.completion",

      created:
        finalResponse.created ||
        Math.floor(Date.now() / 1000),

      model: finalModel,

      choices: [
  {
    index: 0,
    message: {
      role: "assistant",
      content:
        (
          finalResponse &&
          finalResponse.choices &&
          finalResponse.choices[0] &&
          finalResponse.choices[0].message &&
          typeof finalResponse.choices[0].message.content === "string"
        )
          ? finalResponse.choices[0].message.content
          : ""
    },
    finish_reason:
      (
        finalResponse &&
        finalResponse.choices &&
        finalResponse.choices[0] &&
        finalResponse.choices[0].finish_reason
      )
        ? finalResponse.choices[0].finish_reason
        : "stop"
  }
],

      usage:
        finalResponse.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
    });

  } catch (err) {

    console.error(
      "Proxy crash:",
      err.message || err
    );

    return res.status(500).json({
      error: {
        message:
          err.message ||
          "Internal server error",

        type: "server_error",
        code: 500
      }
    });
  }
});

// Start server
app.listen(PORT, "0.0.0.0", function() {

  console.log(
    "NVIDIA NIM Proxy running on port " + PORT
  );

});
