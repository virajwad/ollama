//go:build openvino

#include "openvino_c.h"

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/streamer_base.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>

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

    ov_llm_pipeline_impl(const std::string& model_dir, const std::string& device,
                         const ov::AnyMap& properties)
        : pipeline(model_dir, device, properties) {}
};

extern "C" {

ov_llm_pipeline_t ov_llm_create(const char* model_dir, const char* device, const char* cache_dir) {
    try {
        ov::AnyMap properties;
        if (cache_dir && cache_dir[0] != '\0') {
            properties[ov::cache_dir.name()] = std::string(cache_dir);
        }
        auto* impl = new ov_llm_pipeline_impl(
            std::string(model_dir),
            std::string(device),
            properties);
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
                    void* userdata,
                    ov_llm_perf_metrics_t* metrics) {
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

        ov::genai::StreamerVariant streamer_var = std::monostate{};
        std::function<ov::genai::StreamingStatus(std::string)> streamer_fn;
        if (token_fn) {
            streamer_fn = [&](std::string token) -> ov::genai::StreamingStatus {
                bool should_continue = token_fn(token.c_str(), userdata);
                if (!should_continue) {
                    cancelled = true;
                    return ov::genai::StreamingStatus::STOP;
                }
                return ov::genai::StreamingStatus::RUNNING;
            };
            streamer_var = streamer_fn;
        }

        ov::genai::DecodedResults result = pipeline->pipeline.generate(
            std::string(config->prompt), gen_config, streamer_var);

        /* Extract perf metrics if caller wants them */
        if (metrics) {
            auto& pm = result.perf_metrics;
            pm.evaluate_statistics();
            metrics->generate_duration = pm.get_generate_duration().mean;
            metrics->ttft = pm.get_ttft().mean;
            metrics->tpot = pm.get_tpot().mean;
            metrics->throughput = pm.get_throughput().mean;
            metrics->load_time = pm.get_load_time();
            metrics->num_generated_tokens = static_cast<int32_t>(pm.get_num_generated_tokens());
            metrics->num_input_tokens = static_cast<int32_t>(pm.get_num_input_tokens());
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
