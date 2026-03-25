cmake_minimum_required(VERSION 3.16)
project(openvino_genai_c_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow users to specify the OpenVINO GenAI SDK root directory, e.g.:
#   cmake -DOPENVINO_GENAI_ROOT=C:/path/to/openvino_genai_windows_2026.0.0.0_x86_64
# The SDK ships OpenVINOGenAIConfig.cmake under <root>/runtime/cmake/.
set(OPENVINO_GENAI_ROOT "" CACHE PATH "Root directory of the OpenVINO GenAI SDK")
if(OPENVINO_GENAI_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${OPENVINO_GENAI_ROOT}/runtime/cmake")
endif()

find_package(OpenVINOGenAI REQUIRED)

add_library(openvino_genai_c_wrapper SHARED csrc/openvino_c.cpp)

target_include_directories(openvino_genai_c_wrapper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(openvino_genai_c_wrapper PRIVATE openvino::genai)

if(WIN32)
    target_compile_definitions(openvino_genai_c_wrapper PRIVATE OV_WRAPPER_EXPORTS)
endif()

# Place the built library where CGo expects it (see #cgo LDFLAGS in openvino.go).
# The Go cgo directive references ${SRCDIR}/build/Release, which resolves to
# <this-directory>/build/Release. Override the per-target output directories so
# the library lands there regardless of the top-level CMAKE output settings.
set(_OV_CGO_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")
set_target_properties(openvino_genai_c_wrapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         "${_OV_CGO_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_OV_CGO_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${_OV_CGO_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY         "${_OV_CGO_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_OV_CGO_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${_OV_CGO_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY         "${_OV_CGO_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${_OV_CGO_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${_OV_CGO_OUTPUT_DIR}"
)

install(TARGETS openvino_genai_c_wrapper
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
