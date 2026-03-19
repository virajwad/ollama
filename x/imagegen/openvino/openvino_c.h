//go:build openvino

#ifndef OPENVINO_GENAI_C_H
#define OPENVINO_GENAI_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to the Text2ImagePipeline */
typedef struct ov_t2i_pipeline_impl* ov_t2i_pipeline_t;

/* Configuration for image generation */
typedef struct {
    const char* prompt;
    int32_t     width;
    int32_t     height;
    int32_t     num_inference_steps;
    int64_t     seed;           /* -1 for random */
    float       guidance_scale; /* SD3 default: 7.0 */
} ov_t2i_config_t;

/* Result from image generation */
typedef struct {
    uint8_t* pixels;   /* RGB row-major, caller must free with ov_t2i_free_pixels */
    int32_t  width;
    int32_t  height;
    int32_t  channels; /* Always 3 (RGB) */
} ov_t2i_result_t;

/*
 * Progress callback.
 * Return true to continue, false to cancel generation.
 */
typedef bool (*ov_t2i_progress_fn)(int32_t step, int32_t total_steps, void* userdata);

/*
 * Create a pipeline from a model directory (OpenVINO IR or HuggingFace optimum export).
 * device: "CPU", "GPU", or "NPU"
 * Returns NULL on failure; call ov_t2i_last_error() for details.
 */
ov_t2i_pipeline_t ov_t2i_create(const char* model_dir, const char* device);

/* Destroy a pipeline and free all resources. */
void ov_t2i_destroy(ov_t2i_pipeline_t pipeline);

/*
 * Generate an image. Blocks until complete or cancelled.
 * progress_fn may be NULL.
 * Returns 0 on success, 1 on cancellation, -1 on error.
 */
int ov_t2i_generate(ov_t2i_pipeline_t pipeline,
                    const ov_t2i_config_t* config,
                    ov_t2i_progress_fn progress_fn,
                    void* userdata,
                    ov_t2i_result_t* out_result);

/* Free pixel data returned by ov_t2i_generate. */
void ov_t2i_free_pixels(uint8_t* pixels);

/* Get last error message (thread-local). */
const char* ov_t2i_last_error(void);

/* Check if OpenVINO runtime is available. */
bool ov_t2i_is_available(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENVINO_GENAI_C_H */
