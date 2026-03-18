//go:build !openvino

// Package openvino provides a stub when OpenVINO is not compiled in.
package openvino

import (
	"context"
	"fmt"
	"image"
)

// Pipeline is a stub that always returns errors.
type Pipeline struct{}

// GenerateConfig holds parameters for image generation.
type GenerateConfig struct {
	Prompt        string
	Width         int32
	Height        int32
	Steps         int32
	Seed          int64
	GuidanceScale float32
}

// NewPipeline returns an error when OpenVINO is not compiled in.
func NewPipeline(_, _ string) (*Pipeline, error) {
	return nil, fmt.Errorf("openvino: not compiled in (build with -tags openvino)")
}

// Close is a no-op.
func (p *Pipeline) Close() {}

// Generate returns an error when OpenVINO is not compiled in.
func (p *Pipeline) Generate(_ context.Context, _ *GenerateConfig, _ func(int, int)) (image.Image, error) {
	return nil, fmt.Errorf("openvino: not compiled in")
}

// IsAvailable always returns false.
func IsAvailable() bool { return false }
