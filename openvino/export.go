//go:build openvino

package openvino

// This file is separate because CGo requires //export functions
// to live in a file with no other C preamble beyond the bare minimum.

/*
#include <openvino/genai/c/llm_pipeline.h>
*/
import "C"
import (
	"runtime/cgo"
	"unsafe"
)

//export goOpenVINOTokenBridge
func goOpenVINOTokenBridge(str *C.char, args unsafe.Pointer) C.ov_genai_streaming_status_e {
	cb := cgo.Handle(uintptr(args)).Value().(*tokenCallback)

	// Fast context cancellation check — ctx.Err() is a direct field read,
	// no channel operation in the common (non-cancelled) case.
	if cb.ctx.Err() != nil {
		cb.cancelled = true
		return C.OV_GENAI_STREAMING_STATUS_STOP
	}

	// cb.fn is guaranteed non-nil by Generate(); skip the nil check.
	if !cb.fn(C.GoString(str)) {
		return C.OV_GENAI_STREAMING_STATUS_STOP
	}
	return C.OV_GENAI_STREAMING_STATUS_RUNNING
}
