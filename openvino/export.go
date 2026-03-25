//go:build openvino

package openvino

// This file is separate because CGo requires //export functions
// to live in a file with no other C preamble beyond the bare minimum.

/*
#include <stdbool.h>
*/
import "C"
import "unsafe"

//export goOpenVINOTokenBridge
func goOpenVINOTokenBridge(token *C.char, userdata unsafe.Pointer) C.bool {
	id := uintptr(userdata)
	b := lookupBridge(id)
	if b == nil {
		return C.bool(false)
	}

	// Check context cancellation
	select {
	case <-b.ctx.Done():
		b.cancelled = true
		return C.bool(false)
	default:
	}

	goToken := C.GoString(token)
	if b.fn != nil {
		if !b.fn(goToken) {
			return C.bool(false)
		}
	}
	return C.bool(true)
}
