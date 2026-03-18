package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image/png"
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

// ExecuteOpenVINO is the entry point for the OpenVINO imagegen subprocess.
// Invoked via: ollama runner --openvino-imagegen-engine --model <dir> --device <dev> --port <port>
func ExecuteOpenVINO(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("openvino-imagegen-runner", flag.ExitOnError)
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

	slog.Info("starting openvino imagegen runner", "model", *modelDir, "device", *device, "port", *port)

	pipeline, err := openvino.NewPipeline(*modelDir, *device)
	if err != nil {
		return fmt.Errorf("failed to create openvino pipeline: %w", err)
	}
	defer pipeline.Close()

	slog.Info("openvino pipeline loaded successfully")

	srv := &openvinoSubprocess{pipeline: pipeline}

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
		slog.Info("shutting down openvino runner")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		httpServer.Shutdown(ctx)
		close(done)
	}()

	slog.Info("openvino runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}

	<-done
	return nil
}

type openvinoSubprocess struct {
	pipeline *openvino.Pipeline
	mu       sync.Mutex
}

func (s *openvinoSubprocess) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

func (s *openvinoSubprocess) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Serialize — the OpenVINO pipeline is not thread-safe
	s.mu.Lock()
	defer s.mu.Unlock()

	width := req.Width
	if width <= 0 {
		width = 1024
	}
	height := req.Height
	if height <= 0 {
		height = 1024
	}
	steps := req.Steps
	if steps <= 0 {
		steps = 28 // SD3 default
	}
	seed := req.Seed
	if seed <= 0 {
		seed = time.Now().UnixNano()
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

	cfg := &openvino.GenerateConfig{
		Prompt:        req.Prompt,
		Width:         width,
		Height:        height,
		Steps:         int32(steps),
		Seed:          seed,
		GuidanceScale: 7.0, // SD3 default
	}

	progress := func(step, total int) {
		enc.Encode(Response{Step: step, Total: total})
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	img, err := s.pipeline.Generate(ctx, cfg, progress)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		slog.Error("openvino generation failed", "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Encode to PNG then base64
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		slog.Error("png encode failed", "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	enc.Encode(Response{
		Image: base64.StdEncoding.EncodeToString(buf.Bytes()),
		Done:  true,
	})
	w.Write([]byte("\n"))
	flusher.Flush()
}
