package imagegen

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/openvino"
)

// ExecuteOpenVINO is the entry point for the OpenVINO LLM subprocess.
// Invoked via: ollama runner --openvino-llm-engine --model <dir> --device <dev> --port <port>
func ExecuteOpenVINO(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("openvino-llm-runner", flag.ExitOnError)
	modelDir := fs.String("model", "", "path to OpenVINO IR model directory")
	device := fs.String("device", "CPU", "OpenVINO device (CPU, GPU, NPU)")
	port := fs.Int("port", 0, "port to listen on")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelDir == "" {
		return fmt.Errorf("--model is required")
	}
	if *port == 0 {
		return fmt.Errorf("--port is required")
	}

	slog.Info("starting openvino llm runner", "model", *modelDir, "device", *device, "port", *port)

	pipeline, err := openvino.NewPipeline(*modelDir, *device)
	if err != nil {
		return fmt.Errorf("failed to create openvino pipeline: %w", err)
	}
	defer pipeline.Close()

	slog.Info("openvino llm pipeline loaded successfully")

	srv := &openvinoLLMSubprocess{pipeline: pipeline}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.healthHandler)
	mux.HandleFunc("/completion", srv.completionHandler)

	httpServer := &http.Server{
		Addr:    fmt.Sprintf("127.0.0.1:%d", *port),
		Handler: mux,
	}

	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("shutting down openvino llm runner")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		httpServer.Shutdown(ctx)
		close(done)
	}()

	slog.Info("openvino llm runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}

	<-done
	return nil
}

type openvinoLLMSubprocess struct {
	pipeline *openvino.Pipeline
	mu       sync.Mutex
}

func (s *openvinoLLMSubprocess) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

func (s *openvinoLLMSubprocess) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Serialize - the OpenVINO pipeline is not thread-safe
	s.mu.Lock()
	defer s.mu.Unlock()

	maxTokens := int32(512)
	temperature := float32(0.7)
	topP := float32(0.9)
	topK := int32(40)
	repPenalty := float32(1.0)
	doSample := true

	if req.Options != nil {
		if req.Options.NumPredict > 0 {
			maxTokens = int32(req.Options.NumPredict)
		}
		if req.Options.Temperature > 0 {
			temperature = float32(req.Options.Temperature)
		}
		if req.Options.TopP > 0 {
			topP = float32(req.Options.TopP)
		}
		if req.Options.TopK > 0 {
			topK = int32(req.Options.TopK)
		}
	}

	// Temperature of 0 means greedy (no sampling)
	if temperature == 0 {
		doSample = false
		temperature = 1.0
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	enc := json.NewEncoder(w)
	tokenCount := 0
	start := time.Now()

	cfg := &openvino.GenerateConfig{
		Prompt:            req.Prompt,
		MaxNewTokens:      maxTokens,
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		RepetitionPenalty: repPenalty,
		DoSample:          doSample,
	}

	tokenFn := func(token string) bool {
		tokenCount++
		enc.Encode(Response{Content: token})
		flusher.Flush()
		return true
	}

	err := s.pipeline.Generate(ctx, cfg, tokenFn)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		slog.Error("openvino generation failed", "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	elapsed := time.Since(start)
	enc.Encode(Response{
		Done:         true,
		EvalCount:    tokenCount,
		EvalDuration: int(elapsed.Milliseconds()),
	})
	flusher.Flush()
}
