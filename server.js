const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY =
  process.env.NIM_API_KEY ||
  process.env.NVIDIA_API_KEY;

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
process.on("unhandledRejection", function (err) {
  console.error("UNHANDLED REJECTION:", err);
});

process.on("uncaughtException", function (err) {
  console.error("UNCAUGHT EXCEPTION:", err);
});

// Axios should not throw automatically on HTTP errors
axios.defaults.validateStatus = function () {
  return true;
};


// ============================================================
// MODEL CACHE
// ============================================================

var cachedModels = [];
var availableModels = {};
var lastModelFetch = 0;

var MODEL_CACHE_MS = 10 * 60 * 1000;


// ============================================================
// RP-OPTIMIZED FALLBACK ORDER
// ============================================================
//
// The proxy checks NVIDIA's LIVE /models endpoint before using
// these models. This list controls priority, not availability.
//
// The order is intentionally optimized toward:
// - Character/RP quality
// - Instruction following
// - Natural dialogue
// - Context handling
// - Reduced unnecessary reasoning
// - Reliability
//
// ============================================================

var FALLBACK_MODELS = [
  "deepseek-ai/deepseek-v4-flash-0731"
];


// ============================================================
// MODELS WHERE THINKING CAN BE DISABLED
// ============================================================

var NON_THINKING_MODELS = {
  "deepseek-ai/deepseek-v4-flash-0731": false
};


// ============================================================
// REFRESH NVIDIA MODEL CACHE
// ============================================================

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

    if (!NIM_API_KEY) {
      console.error(
        "NVIDIA API key is missing. Set NIM_API_KEY or NVIDIA_API_KEY."
      );
      return;
    }

    var response = await axios.get(
      NIM_API_BASE + "/models",
      {
        headers: {
          Authorization: "Bearer " + NIM_API_KEY
        },
        timeout: 30000
      }
    );

    if (response.status === 401 || response.status === 403) {

      console.error(
        "NVIDIA API KEY REJECTED. HTTP " + response.status
      );

      return;
    }

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
          created: m.created || Math.floor(now / 1000),
          owned_by: m.owned_by || "nvidia-nim"
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


// ============================================================
// INITIAL MODEL LOAD
// ============================================================

refreshModels();


// ============================================================
// MODELS ENDPOINT
// ============================================================

app.get("/v1/models", async function (req, res) {

  await refreshModels();

  return res.json({
    object: "list",
    data: cachedModels
  });

});


// ============================================================
// HEALTH ENDPOINT
// ============================================================

app.get("/health", function (req, res) {

  return res.json({
    status: "ok",
    models_cached: cachedModels.length,
    api_key_configured: !!NIM_API_KEY
  });

});


// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post("/v1/chat/completions", async function (req, res) {

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
// Our RP priority list ALWAYS comes first.
// Chub's requested model is only used as a final fallback.

var candidateModels = [];

// Primary RP priority list
for (var i = 0; i < FALLBACK_MODELS.length; i++) {

  var fallback = FALLBACK_MODELS[i];

  if (
    availableModels[fallback] &&
    candidateModels.indexOf(fallback) === -1
  ) {
    candidateModels.push(fallback);
  }
}

// Only use Chub's requested model if none of our
// preferred models are currently available.
if (
  requestedModel &&
  availableModels[requestedModel] &&
  candidateModels.indexOf(requestedModel) === -1
) {
  candidateModels.push(requestedModel);
}

// Emergency fallback
if (candidateModels.length === 0) {

  if (requestedModel) {
    candidateModels.push(requestedModel);
  } else {
    candidateModels.push(
      "meta/llama-3.3-70b-instruct"
    );
  }
}


    // ========================================================
    // EMERGENCY FALLBACK
    // ========================================================

    if (candidateModels.length === 0) {

      console.error(
        "No configured fallback models are currently available."
      );

      return res.status(503).json({
        error: {
          message:
            "No configured NVIDIA models are currently available.",
          type: "service_unavailable",
          code: 503
        }
      });

    }


    var finalResponse = null;
    var finalModel = null;


    // ========================================================
    // TRY MODELS IN PRIORITY ORDER
    // ========================================================

    for (
      var j = 0;
      j < candidateModels.length;
      j++
    ) {

      var modelToTry = candidateModels[j];

      console.log(
        "Trying model:",
        modelToTry
      );


      // ======================================================
      // BASE PAYLOAD
      // ======================================================

      var payload = {
        model: modelToTry,
        messages: safeMessages,
        stream: false
      };


      // ======================================================
      // FRONTEND PASSTHROUGH PARAMETERS
      // ======================================================

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
        "stop",
        "response_format",
        "tools",
        "tool_choice"
      ];


      for (
        var k = 0;
        k < passthrough.length;
        k++
      ) {

        var key = passthrough[k];

        if (body[key] !== undefined) {
          payload[key] = body[key];
        }

      }


      

      // ======================================================
      // SAFE DEFAULT TOKEN LIMIT
      // ======================================================
      //
      // Frontend value ALWAYS takes priority.
      //
      // 4096 gives modern models considerably more room than
      // the previous 1250-token limit while preventing an
      // uncontrolled maximum.
      //
      // ======================================================

      if (payload.max_tokens === undefined) {

        payload.max_tokens = 4096;

      }


      


      // ======================================================
      // SEND REQUEST TO NVIDIA
      // ======================================================

      try {

        var response = await axios.post(
          NIM_API_BASE + "/chat/completions",
          payload,
          {
            headers: {
              Authorization: "Bearer " + NIM_API_KEY,
              "Content-Type": "application/json"
            },
            timeout: 180000
          }
        );


        // ====================================================
        // AUTHENTICATION FAILURE
        // ====================================================
        //
        // There is no point hammering every model if the API
        // key itself has expired or been rejected.
        //
        // ====================================================

        if (
          response.status === 401 ||
          response.status === 403
        ) {

          console.error(
            "NVIDIA API KEY REJECTED:",
            "HTTP " + response.status
          );

          return res.status(response.status).json({
            error: {
              message:
                "NVIDIA rejected the API key. Check or replace NIM_API_KEY / NVIDIA_API_KEY.",
              type: "authentication_error",
              code: response.status
            }
          });

        }


        // ====================================================
        // MODEL FAILURE
        // ====================================================

        if (response.status !== 200) {

          console.warn(
            "Model failed:",
            modelToTry,
            "status:",
            response.status
          );

          if (
            response.data &&
            response.data.error &&
            response.data.error.message
          ) {

            console.warn(
              "NVIDIA error:",
              response.data.error.message
            );

          }

          continue;

        }


        // ====================================================
        // VALIDATE RESPONSE
        // ====================================================

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


        // ====================================================
        // SUCCESS
        // ====================================================

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


    // ========================================================
    // TOTAL MODEL FAILURE
    // ========================================================

    if (!finalResponse) {

      return res.status(503).json({
        error: {
          message:
            "All configured NVIDIA models are currently unavailable. Please retry shortly.",
          type: "service_unavailable",
          code: 503
        }
      });

    }


    // ========================================================
    // OPENAI-COMPATIBLE RESPONSE
    // ========================================================

    var selectedChoice =
      (
        finalResponse &&
        finalResponse.choices &&
        finalResponse.choices[0]
      )
        ? finalResponse.choices[0]
        : null;


    var selectedMessage =
      (
        selectedChoice &&
        selectedChoice.message
      )
        ? selectedChoice.message
        : null;


    var responseContent = "";

    if (
      selectedMessage &&
      typeof selectedMessage.content === "string"
    ) {

      responseContent =
        selectedMessage.content;

    }


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
            role:
              (
                selectedMessage &&
                selectedMessage.role
              )
                ? selectedMessage.role
                : "assistant",

            content: responseContent
          },

          finish_reason:
            (
              selectedChoice &&
              selectedChoice.finish_reason
            )
              ? selectedChoice.finish_reason
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


// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  "0.0.0.0",
  function () {

    console.log(
      "NVIDIA NIM Proxy running on port " + PORT
    );

  }
);
