package openvino

// Request is the IPC request format for the OpenVINO runner subprocess.
type Request struct {
	Prompt  string          `json:"prompt"`
	Options *RequestOptions `json:"options,omitempty"`
}

// RequestOptions contains LLM-specific generation options.
type RequestOptions struct {
	NumPredict  int      `json:"num_predict,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// Response is streamed back for each progress update from the runner subprocess.
type Response struct {
	Content string `json:"content,omitempty"`

	Done       bool   `json:"done"`
	DoneReason int    `json:"done_reason,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`

	// Statistics
	PromptEvalCount    int `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int `json:"prompt_eval_duration,omitempty"`
	EvalCount          int `json:"eval_count,omitempty"`
	EvalDuration       int `json:"eval_duration,omitempty"`

	// OpenVINO GenAI perf metrics (ms unless noted)
	GenerateDuration float32 `json:"ov_generate_duration,omitempty"`
	TTFT             float32 `json:"ov_ttft,omitempty"`
	TPOT             float32 `json:"ov_tpot,omitempty"`
	Throughput       float32 `json:"ov_throughput,omitempty"`
	NumInputTokens   int     `json:"ov_num_input_tokens,omitempty"`
}

// HealthResponse is returned by the health endpoint.
type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress,omitempty"`
}
