//go:build !openvino

// Package openvino provides a stub when OpenVINO is not compiled in.
package openvino

import (
	"context"
	"fmt"
)

// Pipeline is a stub that always returns errors.
type Pipeline struct{}

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

// NewPipeline returns an error when OpenVINO is not compiled in.
func NewPipeline(_, _ string) (*Pipeline, error) {
	return nil, fmt.Errorf("openvino: not compiled in (build with -tags openvino)")
}

// Close is a no-op.
func (p *Pipeline) Close() {}

// Generate returns an error when OpenVINO is not compiled in.
// tokenFn receives each generated token; return false to stop.
func (p *Pipeline) Generate(_ context.Context, _ *GenerateConfig, _ func(string) bool) error {
	return fmt.Errorf("openvino: not compiled in")
}

// IsAvailable always returns false.
func IsAvailable() bool { return false }
