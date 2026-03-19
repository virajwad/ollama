cmake_minimum_required(VERSION 3.16)
project(openvino_genai_c_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINOGenAI REQUIRED)

add_library(openvino_genai_c_wrapper SHARED csrc/openvino_c.cpp)

target_include_directories(openvino_genai_c_wrapper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(openvino_genai_c_wrapper PRIVATE openvino::genai)

if(WIN32)
    target_compile_definitions(openvino_genai_c_wrapper PRIVATE OV_WRAPPER_EXPORTS)
endif()

install(TARGETS openvino_genai_c_wrapper
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
