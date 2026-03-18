package imagegen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// OpenVINOServer wraps an OpenVINO runner subprocess to implement llm.LlamaServer.
// It is a parallel path to the MLX-based Server and shares no MLX dependencies.
type OpenVINOServer struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelDir    string // local path to the OpenVINO IR model directory
	device      string // "CPU", "GPU", or "NPU"
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string
	lastErrLock sync.Mutex
}

// NewOpenVINOServer creates a new OpenVINO image generation server.
// modelDir is the local directory containing the exported OpenVINO IR model.
func NewOpenVINOServer(modelDir string) (*OpenVINOServer, error) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		return nil, fmt.Errorf("openvino imagegen is supported on Windows and Linux, got %s", runtime.GOOS)
	}

	device := os.Getenv("OLLAMA_OPENVINO_DEVICE")
	if device == "" {
		device = "CPU"
	}

	return &OpenVINOServer{
		modelDir: modelDir,
		device:   device,
		vramSize: 8 * 1024 * 1024 * 1024, // 8 GB estimate for SD3
		done:     make(chan error, 1),
		client:   &http.Client{Timeout: 10 * time.Minute},
	}, nil
}

// ModelPath returns the model directory path.
func (s *OpenVINOServer) ModelPath() string {
	return s.modelDir
}

// Load starts the OpenVINO subprocess.
func (s *OpenVINOServer) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	// Find a free port
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}
	s.port = port

	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn: ollama runner --openvino-imagegen-engine --model <dir> --device <dev> --port <port>
	cmd := exec.Command(exe, "runner", "--openvino-imagegen-engine",
		"--model", s.modelDir,
		"--device", s.device,
		"--port", strconv.Itoa(port),
	)
	cmd.Env = os.Environ()

	// On Linux, ensure OpenVINO runtime libraries are on LD_LIBRARY_PATH
	if runtime.GOOS == "linux" {
		ovinoRoot := os.Getenv("INTEL_OPENVINO_DIR")
		if ovinoRoot != "" {
			libraryPaths := []string{
				filepath.Join(ovinoRoot, "runtime", "lib", "intel64"),
			}
			if existing, ok := os.LookupEnv("LD_LIBRARY_PATH"); ok {
				libraryPaths = append(libraryPaths, filepath.SplitList(existing)...)
			}
			pathVal := strings.Join(libraryPaths, string(filepath.ListSeparator))
			found := false
			for i := range cmd.Env {
				if strings.HasPrefix(cmd.Env[i], "LD_LIBRARY_PATH=") {
					cmd.Env[i] = "LD_LIBRARY_PATH=" + pathVal
					found = true
					break
				}
			}
			if !found {
				cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+pathVal)
			}
			slog.Debug("openvino subprocess library path", "LD_LIBRARY_PATH", pathVal)
		}
	}

	s.cmd = cmd

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			slog.Info("openvino-runner", "msg", scanner.Text())
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			slog.Warn("openvino-runner", "msg", line)
			s.lastErrLock.Lock()
			s.lastErr = line
			s.lastErrLock.Unlock()
		}
	}()

	slog.Info("starting openvino runner subprocess", "model", s.modelDir, "device", s.device, "port", s.port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start openvino runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		err := cmd.Wait()
		s.done <- err
	}()

	return nil, nil
}

// Ping checks if the subprocess is healthy.
func (s *OpenVINOServer) Ping(ctx context.Context) error {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.port)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

// WaitUntilRunning waits for the subprocess to be ready.
func (s *OpenVINOServer) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(envconfig.LoadTimeout())
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-s.done:
			s.lastErrLock.Lock()
			errMsg := s.lastErr
			s.lastErrLock.Unlock()
			if errMsg != "" {
				return fmt.Errorf("openvino runner failed: %s (exit: %v)", errMsg, err)
			}
			return fmt.Errorf("openvino runner exited unexpectedly: %w", err)
		case <-timeout:
			s.lastErrLock.Lock()
			errMsg := s.lastErr
			s.lastErrLock.Unlock()
			if errMsg != "" {
				return fmt.Errorf("timeout waiting for openvino runner: %s", errMsg)
			}
			return errors.New("timeout waiting for openvino runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("openvino runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// Completion sends an image generation request to the subprocess.
func (s *OpenVINOServer) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	creq := Request{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  int(req.Steps),
		Seed:   seed,
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s", strings.TrimSpace(string(b)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024) // 16MB max for base64 image
	for scanner.Scan() {
		var raw struct {
			Image      string `json:"image,omitempty"`
			Done       bool   `json:"done"`
			Step       int    `json:"step,omitempty"`
			Total      int    `json:"total,omitempty"`
			StopReason string `json:"stop_reason,omitempty"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			slog.Debug("openvino response parse error", "error", err, "line", string(scanner.Bytes()))
			continue
		}

		cresp := llm.CompletionResponse{
			Done:       raw.Done,
			Step:       raw.Step,
			TotalSteps: raw.Total,
			Image:      raw.Image,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	scanErr := scanner.Err()
	if scanErr != nil {
		slog.Error("openvino scanner error", "error", scanErr)
	} else {
		slog.Warn("openvino scanner EOF without Done response - subprocess may have crashed")
	}

	if s.HasExited() {
		slog.Error("openvino subprocess has exited unexpectedly")
	}

	return scanErr
}

// Close terminates the subprocess.
func (s *OpenVINOServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		slog.Info("stopping openvino runner subprocess", "pid", s.cmd.Process.Pid)
		s.cmd.Process.Signal(os.Interrupt)
		select {
		case <-s.done:
		case <-time.After(5 * time.Second):
			s.cmd.Process.Kill()
		}
		s.cmd = nil
	}
	return nil
}

// MemorySize returns estimated total and VRAM memory.
func (s *OpenVINOServer) MemorySize() (total, vram uint64) {
	return s.vramSize, s.vramSize
}

// VRAMByGPU returns VRAM usage for a specific GPU.
func (s *OpenVINOServer) VRAMByGPU(id ml.DeviceID) uint64 {
	return s.vramSize
}

// ContextLength is not applicable for image generation.
func (s *OpenVINOServer) ContextLength() int { return 0 }

// Embedding is not supported for image generation.
func (s *OpenVINOServer) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("embeddings not supported for openvino image generation")
}

// Tokenize is not supported for image generation.
func (s *OpenVINOServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenization not supported for openvino image generation")
}

// Detokenize is not supported for image generation.
func (s *OpenVINOServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("detokenization not supported for openvino image generation")
}

// Pid returns the subprocess PID.
func (s *OpenVINOServer) Pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

// GetPort returns the subprocess listening port.
func (s *OpenVINOServer) GetPort() int { return s.port }

// GetDeviceInfos returns device information.
func (s *OpenVINOServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }

// HasExited returns whether the subprocess has exited.
func (s *OpenVINOServer) HasExited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

// Ensure OpenVINOServer implements llm.LlamaServer at compile time.
var _ llm.LlamaServer = (*OpenVINOServer)(nil)
