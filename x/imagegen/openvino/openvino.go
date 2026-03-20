//go:build openvino

// Package openvino provides Go bindings for the OpenVINO GenAI LLMPipeline.
package openvino

/*
#cgo CFLAGS: -I${SRCDIR} -DOV_WRAPPER_IMPORTS
#cgo LDFLAGS: -L${SRCDIR}/build/Release -lopenvino_genai_c_wrapper
#include "openvino_c.h"
#include <stdlib.h>

// Bridge function declared here, defined in export.go via //export.
extern _Bool goOpenVINOTokenBridge(const char* token, void* userdata);
*/
import "C"

import (
	"context"
	"fmt"
	"sync"
	"unsafe"
)
// Pipeline wraps an OpenVINO GenAI LLMPipeline.
type Pipeline struct {
	handle C.ov_llm_pipeline_t
	mu     sync.Mutex
}

// NewPipeline creates a new LLMPipeline from a model directory.
// modelDir must contain the OpenVINO IR files exported via optimum-intel.
// device is "CPU", "GPU", or "NPU".
func NewPipeline(modelDir, device string) (*Pipeline, error) {
	cDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cDir))
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))

	handle := C.ov_llm_create(cDir, cDev)
	if handle == nil {
		return nil, fmt.Errorf("openvino: %s", C.GoString(C.ov_llm_last_error()))
	}

	return &Pipeline{handle: handle}, nil
}

// Close releases the pipeline resources.
func (p *Pipeline) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.handle != nil {
		C.ov_llm_destroy(p.handle)
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

// tokenBridge routes C callbacks back into Go closures.
type tokenBridge struct {
	ctx       context.Context
	fn        func(token string) bool // return false to stop
	cancelled bool
}

var (
	bridgeMu   sync.Mutex
	bridgeMap  = make(map[uintptr]*tokenBridge)
	bridgeNext uintptr
)

func registerBridge(b *tokenBridge) uintptr {
	bridgeMu.Lock()
	defer bridgeMu.Unlock()
	bridgeNext++
	bridgeMap[bridgeNext] = b
	return bridgeNext
}

func unregisterBridge(id uintptr) {
	bridgeMu.Lock()
	defer bridgeMu.Unlock()
	delete(bridgeMap, id)
}

// lookupBridge is called from the exported callback in export.go.
func lookupBridge(id uintptr) *tokenBridge {
	bridgeMu.Lock()
	defer bridgeMu.Unlock()
	return bridgeMap[id]
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

	cPrompt := C.CString(cfg.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cConfig := C.ov_llm_config_t{
		prompt:             cPrompt,
		max_new_tokens:     C.int32_t(cfg.MaxNewTokens),
		temperature:        C.float(cfg.Temperature),
		top_p:              C.float(cfg.TopP),
		top_k:              C.int32_t(cfg.TopK),
		repetition_penalty: C.float(cfg.RepetitionPenalty),
		do_sample:          C.bool(cfg.DoSample),
	}

	bridge := &tokenBridge{ctx: ctx, fn: tokenFn}
	bridgeID := registerBridge(bridge)
	defer unregisterBridge(bridgeID)

	var cMetrics C.ov_llm_perf_metrics_t
	rc := C.ov_llm_generate(
		p.handle,
		&cConfig,
		C.ov_llm_token_fn(C.goOpenVINOTokenBridge),
		unsafe.Pointer(bridgeID),
		&cMetrics,
	)

	if rc != 0 {
		if bridge.cancelled {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("openvino: %s", C.GoString(C.ov_llm_last_error()))
	}

	metrics := &PerfMetrics{
		GenerateDuration: float32(cMetrics.generate_duration),
		TTFT:             float32(cMetrics.ttft),
		TPOT:             float32(cMetrics.tpot),
		Throughput:       float32(cMetrics.throughput),
		LoadTime:         float32(cMetrics.load_time),
		NumGenerated:     int32(cMetrics.num_generated_tokens),
		NumInput:         int32(cMetrics.num_input_tokens),
	}
	return metrics, nil
}

// IsAvailable checks if the OpenVINO runtime can be reached.
func IsAvailable() bool {
	return bool(C.ov_llm_is_available())
}
