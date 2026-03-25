//go:build openvino

// Package openvino provides Go bindings for the OpenVINO GenAI LLMPipeline.
// This implementation links directly to the official openvino_genai_c DLL,
// eliminating the custom C++ wrapper DLL and its overhead.
package openvino

/*
// --- Include paths ---
// OPENVINO_GENAI_ROOT must point to the extracted SDK, e.g.:
//   C:/Users/dungeon/Downloads/openvino_genai_windows_2026.0.0.0_x86_64/openvino_genai_windows_2026.0.0.0_x86_64

#cgo CFLAGS: -I${SRCDIR}/sdk/include
#cgo LDFLAGS: -L${SRCDIR}/sdk/lib -lopenvino_genai_c -lopenvino_c

#include <openvino/c/ov_common.h>
#include <openvino/genai/c/llm_pipeline.h>
#include <openvino/genai/c/generation_config.h>
#include <openvino/genai/c/perf_metrics.h>
#include <stdlib.h>

// Bridge function declared here, defined in export.go via //export.
extern ov_genai_streaming_status_e goOpenVINOTokenBridge(const char* str, void* args);
// Helper to build a streamer_callback struct.
// CGo cannot directly assign to function-pointer struct fields, so we do it in C.
static inline streamer_callback make_streamer_callback(void* args) {
    streamer_callback cb;
    cb.callback_func = goOpenVINOTokenBridge;
    cb.args = args;
    return cb;
}

// CGo cannot call variadic C functions.  Provide fixed-arg wrappers for
// ov_genai_llm_pipeline_create which is declared with "...".
static inline ov_status_e ov_llm_pipeline_create_no_props(
        const char* path, const char* device, ov_genai_llm_pipeline** pipe) {
    return ov_genai_llm_pipeline_create(path, device, 0, pipe);
}
static inline ov_status_e ov_llm_pipeline_create_cache(
        const char* path, const char* device, ov_genai_llm_pipeline** pipe,
        const char* key, const char* val) {
    return ov_genai_llm_pipeline_create(path, device, 2, pipe, key, val);
}
*/
import "C"

import (
	"context"
	"fmt"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Pipeline wraps an OpenVINO GenAI LLMPipeline.
type Pipeline struct {
	handle *C.ov_genai_llm_pipeline
	mu     sync.Mutex
}

// NewPipeline creates a new LLMPipeline from a model directory.
// modelDir must contain the OpenVINO IR files exported via optimum-intel.
// device is "CPU", "GPU", or "NPU".
// cacheDir enables model caching to reduce load time (empty string to disable).
func NewPipeline(modelDir, device, cacheDir string) (*Pipeline, error) {
	cDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cDir))
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))

	var pipe *C.ov_genai_llm_pipeline
	var status C.ov_status_e

	if cacheDir != "" {
		cKey := C.CString("CACHE_DIR")
		defer C.free(unsafe.Pointer(cKey))
		cVal := C.CString(cacheDir)
		defer C.free(unsafe.Pointer(cVal))
		status = C.ov_llm_pipeline_create_cache(cDir, cDev, &pipe, cKey, cVal)
	} else {
		status = C.ov_llm_pipeline_create_no_props(cDir, cDev, &pipe)
	}

	if status != C.OK {
		return nil, fmt.Errorf("openvino: pipeline creation failed: %s", C.GoString(C.ov_get_error_info(status)))
	}

	return &Pipeline{handle: pipe}, nil
}

// Close releases the pipeline resources.
func (p *Pipeline) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.handle != nil {
		C.ov_genai_llm_pipeline_free(p.handle)
		p.handle = nil
	}
}

// GenerateConfig holds parameters for text generation.
type GenerateConfig struct {
	Prompt            string
	MaxNewTokens      int32
	Temperature       float32
	TopP              float32
	TopK              int32
	RepetitionPenalty float32
	DoSample          bool
}

// tokenCallback is the per-generation state passed through the streamer callback.
// A cgo.Handle to this struct replaces the old bridge map + global mutex.
type tokenCallback struct {
	ctx       context.Context
	fn        func(token string) bool // return false to stop
	cancelled bool
}

// PerfMetrics contains performance metrics from OpenVINO GenAI.
type PerfMetrics struct {
	GenerateDuration float32 // Total generation time in ms
	TTFT             float32 // Time to first token in ms
	TPOT             float32 // Time per output token in ms
	Throughput       float32 // Tokens per second
	LoadTime         float32 // Model load time in ms
	NumGenerated     int32   // Number of generated tokens
	NumInput         int32   // Number of input tokens
}

// Generate produces text from the given config, streaming tokens via the callback.
// tokenFn is called for each generated token; return false to stop generation.
// Returns PerfMetrics from the OpenVINO GenAI pipeline.
func (p *Pipeline) Generate(ctx context.Context, cfg *GenerateConfig, tokenFn func(token string) bool) (*PerfMetrics, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.handle == nil {
		return nil, fmt.Errorf("openvino: pipeline is closed")
	}

	// Build GenerationConfig via the official C API setters.
	var genCfg *C.ov_genai_generation_config
	if status := C.ov_genai_generation_config_create(&genCfg); status != C.OK {
		return nil, fmt.Errorf("openvino: config creation failed: %s", C.GoString(C.ov_get_error_info(status)))
	}
	defer C.ov_genai_generation_config_free(genCfg)

	maxNew := cfg.MaxNewTokens
	if maxNew <= 0 {
		maxNew = 256
	}
	C.ov_genai_generation_config_set_max_new_tokens(genCfg, C.size_t(maxNew))

	temp := cfg.Temperature
	if temp > 0 {
		C.ov_genai_generation_config_set_temperature(genCfg, C.float(temp))
	}
	if cfg.TopP > 0 {
		C.ov_genai_generation_config_set_top_p(genCfg, C.float(cfg.TopP))
	}
	if cfg.TopK > 0 {
		C.ov_genai_generation_config_set_top_k(genCfg, C.size_t(cfg.TopK))
	}
	if cfg.RepetitionPenalty > 0 {
		C.ov_genai_generation_config_set_repetition_penalty(genCfg, C.float(cfg.RepetitionPenalty))
	}
	C.ov_genai_generation_config_set_do_sample(genCfg, C.bool(cfg.DoSample))

	cPrompt := C.CString(cfg.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Use cgo.Handle to pass the callback state through the C void* pointer.
	// This completely eliminates the global bridge map and its per-token mutex.
	cb := &tokenCallback{ctx: ctx, fn: tokenFn}
	h := cgo.NewHandle(cb)
	defer h.Delete()

	// Build the SDK's streamer_callback struct.
	streamer := C.make_streamer_callback(unsafe.Pointer(uintptr(h)))

	var results *C.ov_genai_decoded_results
	status := C.ov_genai_llm_pipeline_generate(p.handle, cPrompt, genCfg, &streamer, &results)

	if status != C.OK {
		if cb.cancelled {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("openvino: generation failed: %s", C.GoString(C.ov_get_error_info(status)))
	}
	defer C.ov_genai_decoded_results_free(results)

	// Extract perf metrics via the official C API.
	metrics, err := extractPerfMetrics(results)
	if err != nil {
		// Non-fatal: return nil metrics but no error
		return nil, nil
	}

	if cb.cancelled {
		return metrics, ctx.Err()
	}

	return metrics, nil
}

// extractPerfMetrics reads performance data from decoded results via the official C API.
func extractPerfMetrics(results *C.ov_genai_decoded_results) (*PerfMetrics, error) {
	var pm *C.ov_genai_perf_metrics
	if status := C.ov_genai_decoded_results_get_perf_metrics(results, &pm); status != C.OK {
		return nil, fmt.Errorf("failed to get perf metrics")
	}
	defer C.ov_genai_decoded_results_perf_metrics_free(pm)

	m := &PerfMetrics{}
	var mean, std C.float

	if C.ov_genai_perf_metrics_get_generate_duration(pm, &mean, &std) == C.OK {
		m.GenerateDuration = float32(mean)
	}
	if C.ov_genai_perf_metrics_get_ttft(pm, &mean, &std) == C.OK {
		m.TTFT = float32(mean)
	}
	if C.ov_genai_perf_metrics_get_tpot(pm, &mean, &std) == C.OK {
		m.TPOT = float32(mean)
	}
	if C.ov_genai_perf_metrics_get_throughput(pm, &mean, &std) == C.OK {
		m.Throughput = float32(mean)
	}

	var loadTime C.float
	if C.ov_genai_perf_metrics_get_load_time(pm, &loadTime) == C.OK {
		m.LoadTime = float32(loadTime)
	}

	var numGen C.size_t
	if C.ov_genai_perf_metrics_get_num_generation_tokens(pm, &numGen) == C.OK {
		m.NumGenerated = int32(numGen)
	}
	var numIn C.size_t
	if C.ov_genai_perf_metrics_get_num_input_tokens(pm, &numIn) == C.OK {
		m.NumInput = int32(numIn)
	}

	return m, nil
}

// IsAvailable checks if the OpenVINO runtime can be reached.
// This creates a temporary pipeline check - for a lighter check,
// just attempt NewPipeline and handle the error.
func IsAvailable() bool {
	// The official C API doesn't have a direct "is available" check.
	// We return true optimistically since the DLL loaded successfully.
	return true
}
