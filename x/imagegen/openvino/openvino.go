//go:build openvino

// Package openvino provides Go bindings for the OpenVINO GenAI Text2ImagePipeline.
package openvino

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lopenvino_genai_c_wrapper -lopenvino -lopenvino_genai -lstdc++
#include "openvino_c.h"
#include <stdlib.h>

// Bridge function declared here, defined in export.go via //export.
extern _Bool goOpenVINOProgressBridge(int step, int total, void* userdata);

// C-callable trampoline that forwards to the Go bridge.
static _Bool ov_progress_trampoline(int32_t step, int32_t total, void* userdata) {
    return goOpenVINOProgressBridge((int)step, (int)total, userdata);
}
*/
import "C"

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"sync"
	"unsafe"
)

// Pipeline wraps an OpenVINO GenAI Text2ImagePipeline.
type Pipeline struct {
	handle C.ov_t2i_pipeline_t
	mu     sync.Mutex
}

// NewPipeline creates a new Text2ImagePipeline from a model directory.
// modelDir must contain the OpenVINO IR files exported via optimum-intel.
// device is "CPU", "GPU", or "NPU".
func NewPipeline(modelDir, device string) (*Pipeline, error) {
	cDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cDir))
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))

	handle := C.ov_t2i_create(cDir, cDev)
	if handle == nil {
		return nil, fmt.Errorf("openvino: %s", C.GoString(C.ov_t2i_last_error()))
	}

	return &Pipeline{handle: handle}, nil
}

// Close releases the pipeline resources.
func (p *Pipeline) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.handle != nil {
		C.ov_t2i_destroy(p.handle)
		p.handle = nil
	}
}

// GenerateConfig holds parameters for image generation.
type GenerateConfig struct {
	Prompt        string
	Width         int32
	Height        int32
	Steps         int32
	Seed          int64
	GuidanceScale float32
}

// progressBridge routes C callbacks back into Go closures.
type progressBridge struct {
	ctx       context.Context
	fn        func(step, total int)
	cancelled bool
}

var (
	bridgeMu   sync.Mutex
	bridgeMap  = make(map[uintptr]*progressBridge)
	bridgeNext uintptr
)

func registerBridge(b *progressBridge) uintptr {
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
func lookupBridge(id uintptr) *progressBridge {
	bridgeMu.Lock()
	defer bridgeMu.Unlock()
	return bridgeMap[id]
}

// Generate produces an image from the given config.
func (p *Pipeline) Generate(ctx context.Context, cfg *GenerateConfig, progress func(step, total int)) (image.Image, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.handle == nil {
		return nil, fmt.Errorf("openvino: pipeline is closed")
	}

	cPrompt := C.CString(cfg.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cConfig := C.ov_t2i_config_t{
		prompt:              cPrompt,
		width:               C.int32_t(cfg.Width),
		height:              C.int32_t(cfg.Height),
		num_inference_steps: C.int32_t(cfg.Steps),
		seed:                C.int64_t(cfg.Seed),
		guidance_scale:      C.float(cfg.GuidanceScale),
	}

	bridge := &progressBridge{ctx: ctx, fn: progress}
	bridgeID := registerBridge(bridge)
	defer unregisterBridge(bridgeID)

	var result C.ov_t2i_result_t
	rc := C.ov_t2i_generate(
		p.handle,
		&cConfig,
		C.ov_t2i_progress_fn(C.ov_progress_trampoline),
		unsafe.Pointer(bridgeID),
		&result,
	)

	if rc != 0 {
		if bridge.cancelled {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("openvino: %s", C.GoString(C.ov_t2i_last_error()))
	}
	defer C.ov_t2i_free_pixels(result.pixels)

	w := int(result.width)
	h := int(result.height)
	ch := int(result.channels)
	srcSize := w * h * ch
	src := C.GoBytes(unsafe.Pointer(result.pixels), C.int(srcSize))

	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := (y*w + x) * ch
			img.SetNRGBA(x, y, color.NRGBA{
				R: src[off],
				G: src[off+1],
				B: src[off+2],
				A: 255,
			})
		}
	}

	return img, nil
}

// IsAvailable checks if the OpenVINO runtime can be reached.
func IsAvailable() bool {
	return bool(C.ov_t2i_is_available())
}
