//go:build openvino

#include "openvino_c.h"

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/runtime/core.hpp>

#include <cstring>
#include <functional>
#include <string>

/* Thread-local error message */
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}
struct ov_llm_pipeline_impl {
    ov::genai::LLMPipeline pipeline;

    ov_llm_pipeline_impl(const std::string& model_dir, const std::string& device)
        : pipeline(model_dir, device) {}
};

extern "C" {

ov_llm_pipeline_t ov_llm_create(const char* model_dir, const char* device) {
    try {
        auto* impl = new ov_llm_pipeline_impl(
            std::string(model_dir),
            std::string(device));
        return impl;
    } catch (const std::exception& e) {
        set_error(std::string("ov_llm_create: ") + e.what());
        return nullptr;
    }
}

void ov_llm_destroy(ov_llm_pipeline_t pipeline) {
    delete pipeline;
}

int ov_llm_generate(ov_llm_pipeline_t pipeline,
                    const ov_llm_config_t* config,
                    ov_llm_token_fn token_fn,
                    void* userdata) {
    if (!pipeline || !config) {
        set_error("null argument");
        return -1;
    }

    try {
        ov::genai::GenerationConfig gen_config;
        gen_config.max_new_tokens = config->max_new_tokens > 0
            ? static_cast<size_t>(config->max_new_tokens) : 256;
        gen_config.temperature = config->temperature > 0.0f
            ? config->temperature : 1.0f;
        gen_config.top_p = config->top_p > 0.0f
            ? config->top_p : 1.0f;
        gen_config.top_k = config->top_k > 0
            ? static_cast<size_t>(config->top_k) : 0;
        gen_config.repetition_penalty = config->repetition_penalty > 0.0f
            ? config->repetition_penalty : 1.0f;
        gen_config.do_sample = config->do_sample;

        bool cancelled = false;

        std::function<bool(std::string)> streamer_fn;
        if (token_fn) {
            /*
             * OpenVINO GenAI streamer lambda:
             *   Return false to continue, true to stop.
             */
            streamer_fn = [&](std::string token) -> bool {
                bool should_continue = token_fn(token.c_str(), userdata);
                if (!should_continue) {
                    cancelled = true;
                }
                return !should_continue;
            };
        }

        if (streamer_fn) {
            pipeline->pipeline.generate(
                std::string(config->prompt), gen_config, streamer_fn);
        } else {
            pipeline->pipeline.generate(
                std::string(config->prompt), gen_config);
        }

        if (cancelled) {
            set_error("generation cancelled");
            return 1;
        }

        return 0;

    } catch (const std::exception& e) {
        set_error(std::string("ov_llm_generate: ") + e.what());
        return -1;
    }
}

const char* ov_llm_last_error(void) {
    return g_last_error.c_str();
}

bool ov_llm_is_available(void) {
    try {
        ov::Core core(std::string{});
        auto devices = core.get_available_devices();
        return !devices.empty();
    } catch (...) {
        return false;
    }
}

} /* extern "C" */
