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

// PerfMetrics contains performance metrics from OpenVINO GenAI.
type PerfMetrics struct {
	GenerateDuration float32
	TTFT             float32
	TPOT             float32
	Throughput       float32
	LoadTime         float32
	NumGenerated     int32
	NumInput         int32
}

// Generate returns an error when OpenVINO is not compiled in.
// tokenFn receives each generated token; return false to stop.
func (p *Pipeline) Generate(_ context.Context, _ *GenerateConfig, _ func(string) bool) (*PerfMetrics, error) {
	return nil, fmt.Errorf("openvino: not compiled in")
}

// IsAvailable always returns false.
func IsAvailable() bool { return false }
