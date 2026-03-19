//go:build openvino

#include "openvino_c.h"

#include <openvino/genai/text2image/pipeline.hpp>
#include <openvino/openvino.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

/* Thread-local error message */
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

struct ov_t2i_pipeline_impl {
    ov::genai::Text2ImagePipeline pipeline;

    ov_t2i_pipeline_impl(const std::string& model_dir, const std::string& device)
        : pipeline(model_dir, device) {}
};

extern "C" {

ov_t2i_pipeline_t ov_t2i_create(const char* model_dir, const char* device) {
    try {
        auto* impl = new ov_t2i_pipeline_impl(
            std::string(model_dir),
            std::string(device));
        return impl;
    } catch (const std::exception& e) {
        set_error(std::string("ov_t2i_create: ") + e.what());
        return nullptr;
    }
}

void ov_t2i_destroy(ov_t2i_pipeline_t pipeline) {
    delete pipeline;
}

int ov_t2i_generate(ov_t2i_pipeline_t pipeline,
                    const ov_t2i_config_t* config,
                    ov_t2i_progress_fn progress_fn,
                    void* userdata,
                    ov_t2i_result_t* out_result) {
    if (!pipeline || !config || !out_result) {
        set_error("null argument");
        return -1;
    }

    try {
        ov::genai::Text2ImagePipeline::GenerationConfig gen_config;
        gen_config.width = config->width > 0 ? config->width : 1024;
        gen_config.height = config->height > 0 ? config->height : 1024;
        gen_config.num_inference_steps = config->num_inference_steps > 0 ? config->num_inference_steps : 28;
        gen_config.guidance_scale = config->guidance_scale > 0.0f ? config->guidance_scale : 7.0f;

        if (config->seed >= 0) {
            gen_config.generator = std::make_shared<ov::genai::CppStdGenerator>(
                static_cast<uint64_t>(config->seed));
        }

        int32_t total_steps = gen_config.num_inference_steps;
        bool cancelled = false;

        /*
         * OpenVINO GenAI progress callback signature:
         *   bool callback(size_t step, size_t num_steps, ov::Tensor& latent)
         * Return true to continue, false to abort.
         */
        std::function<bool(size_t, size_t, ov::Tensor&)> ov_callback;
        if (progress_fn) {
            ov_callback = [&](size_t step, size_t num_steps, ov::Tensor& /* latent */) -> bool {
                total_steps = static_cast<int32_t>(num_steps);
                bool should_continue = progress_fn(
                    static_cast<int32_t>(step),
                    static_cast<int32_t>(num_steps),
                    userdata);
                if (!should_continue) {
                    cancelled = true;
                }
                return should_continue;
            };
        }

        /* Run the full SD3 pipeline (text encode -> denoise -> VAE decode) */
        ov::Tensor output;
        if (ov_callback) {
            output = pipeline->pipeline.generate(
                std::string(config->prompt), gen_config, ov_callback);
        } else {
            output = pipeline->pipeline.generate(
                std::string(config->prompt), gen_config);
        }

        if (cancelled) {
            set_error("generation cancelled");
            return 1;
        }

        /*
         * The output tensor is uint8 pixels. The layout depends on the
         * OpenVINO GenAI version:
         *   NCHW: [1, 3, H, W]
         *   NHWC: [1, H, W, 3]
         *   HWC:  [H, W, 3]
         */
        auto shape = output.get_shape();

        int32_t out_h, out_w, out_c;
        bool nchw = false;

        if (shape.size() == 4) {
            if (shape[1] == 3) {
                nchw = true;
                out_c = static_cast<int32_t>(shape[1]);
                out_h = static_cast<int32_t>(shape[2]);
                out_w = static_cast<int32_t>(shape[3]);
            } else {
                out_h = static_cast<int32_t>(shape[1]);
                out_w = static_cast<int32_t>(shape[2]);
                out_c = static_cast<int32_t>(shape[3]);
            }
        } else if (shape.size() == 3) {
            out_h = static_cast<int32_t>(shape[0]);
            out_w = static_cast<int32_t>(shape[1]);
            out_c = static_cast<int32_t>(shape[2]);
        } else {
            set_error("unexpected output tensor rank: " + std::to_string(shape.size()));
            return -1;
        }

        size_t pixel_count = static_cast<size_t>(out_h) * out_w * out_c;
        auto* src = output.data<uint8_t>();
        auto* dst = new uint8_t[pixel_count];

        if (nchw) {
            /* Convert NCHW -> HWC (interleave channels) */
            for (int32_t c = 0; c < out_c; ++c) {
                for (int32_t y = 0; y < out_h; ++y) {
                    for (int32_t x = 0; x < out_w; ++x) {
                        dst[(y * out_w + x) * out_c + c] =
                            src[c * out_h * out_w + y * out_w + x];
                    }
                }
            }
        } else {
            std::memcpy(dst, src, pixel_count);
        }

        out_result->pixels = dst;
        out_result->width = out_w;
        out_result->height = out_h;
        out_result->channels = out_c;
        return 0;

    } catch (const std::exception& e) {
        set_error(std::string("ov_t2i_generate: ") + e.what());
        return -1;
    }
}

void ov_t2i_free_pixels(uint8_t* pixels) {
    delete[] pixels;
}

const char* ov_t2i_last_error(void) {
    return g_last_error.c_str();
}

bool ov_t2i_is_available(void) {
    try {
        ov::Core core;
        auto devices = core.get_available_devices();
        return !devices.empty();
    } catch (...) {
        return false;
    }
}

} /* extern "C" */
