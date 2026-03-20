//go:build openvino

#ifndef OPENVINO_GENAI_C_H
#define OPENVINO_GENAI_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef OV_WRAPPER_EXPORTS
#  define OV_WRAPPER_API __declspec(dllexport)
#elif defined(OV_WRAPPER_IMPORTS)
#  define OV_WRAPPER_API __declspec(dllimport)
#else
#  define OV_WRAPPER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to the LLMPipeline */
typedef struct ov_llm_pipeline_impl* ov_llm_pipeline_t;

/* Configuration for text generation */
typedef struct {
    const char* prompt;
    int32_t     max_new_tokens;
    float       temperature;
    float       top_p;
    int32_t     top_k;
    float       repetition_penalty;
    bool        do_sample;
} ov_llm_config_t;

/* Performance metrics returned after generation. */
typedef struct {
    float generate_duration;     /* Total generation time in ms */
    float ttft;                  /* Time to first token in ms */
    float tpot;                  /* Time per output token in ms */
    float throughput;            /* Tokens per second */
    float load_time;             /* Model load time in ms */
    int32_t num_generated_tokens; /* Number of generated tokens */
    int32_t num_input_tokens;     /* Number of input tokens */
} ov_llm_perf_metrics_t;

/*
 * Streaming token callback.
 * Called for each generated token.
 * Return true to continue, false to cancel generation.
 */
typedef bool (*ov_llm_token_fn)(const char* token, void* userdata);

/*
 * Create an LLM pipeline from a model directory (OpenVINO IR or optimum-intel export).
 * device: "CPU", "GPU", or "NPU"
 * Returns NULL on failure; call ov_llm_last_error() for details.
 */
OV_WRAPPER_API ov_llm_pipeline_t ov_llm_create(const char* model_dir, const char* device);

/* Destroy a pipeline and free all resources. */
OV_WRAPPER_API void ov_llm_destroy(ov_llm_pipeline_t pipeline);

/*
 * Generate text from a prompt. Blocks until complete or cancelled.
 * token_fn is called for each generated token (may be NULL for non-streaming).
 * metrics is filled with performance data if non-NULL.
 * Returns 0 on success, 1 on cancellation, -1 on error.
 */
OV_WRAPPER_API int ov_llm_generate(ov_llm_pipeline_t pipeline,
                    const ov_llm_config_t* config,
                    ov_llm_token_fn token_fn,
                    void* userdata,
                    ov_llm_perf_metrics_t* metrics);

/* Get last error message (thread-local). */
OV_WRAPPER_API const char* ov_llm_last_error(void);

/* Check if OpenVINO runtime is available. */
OV_WRAPPER_API bool ov_llm_is_available(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENVINO_GENAI_C_H */
