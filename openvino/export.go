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
	// Recover the tokenCallback directly from the cgo.Handle.
	// This is a simple integer→pointer lookup with no locking.
	h := cgo.Handle(uintptr(args))
	cb := h.Value().(*tokenCallback)

	// Fast context cancellation check (no channel operation in the common case).
	if err := cb.ctx.Err(); err != nil {
		cb.cancelled = true
		return C.OV_GENAI_STREAMING_STATUS_STOP
	}

	goToken := C.GoString(str)
	if cb.fn != nil {
		if !cb.fn(goToken) {
			return C.OV_GENAI_STREAMING_STATUS_STOP
		}
	}
	return C.OV_GENAI_STREAMING_STATUS_RUNNING
}
