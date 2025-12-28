# CMakeLists.txt for MardukNativeCore
# Builds the native bridge between mad9ml TypeScript and ARM64 libraries

cmake_minimum_required(VERSION 3.18.1)
project(marduk_native_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android-specific settings
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Library paths
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/../../../../../libs/${ANDROID_ABI})
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../../../include)

# Log configuration
message(STATUS "Building MardukNativeCore for ${ANDROID_ABI}")
message(STATUS "Libraries directory: ${LIBS_DIR}")

#=============================================================================
# Pre-built Native Libraries
#=============================================================================

# C++ Standard Library
add_library(c++_shared SHARED IMPORTED)
set_target_properties(c++_shared PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libc++_shared.so
)

# ggml - Tensor operations
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml.so
)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml-base.so
)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml-cpu.so
)

add_library(ggml-blas SHARED IMPORTED)
set_target_properties(ggml-blas PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml-blas.so
)

# GPU backends
add_library(ggml-opencl SHARED IMPORTED)
set_target_properties(ggml-opencl PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml-opencl.so
)

add_library(ggml-vulkan SHARED IMPORTED)
set_target_properties(ggml-vulkan PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libggml-vulkan.so
)

add_library(OpenCL SHARED IMPORTED)
set_target_properties(OpenCL PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libOpenCL.so
)

# llama.cpp - LLM inference
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libllama.so
)

add_library(llama-jni SHARED IMPORTED)
set_target_properties(llama-jni PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libllama-jni.so
)

# ONNX Runtime - Neural network inference
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libonnxruntime.so
)

add_library(onnxruntime4j_jni SHARED IMPORTED)
set_target_properties(onnxruntime4j_jni PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libonnxruntime4j_jni.so
)

# ncnn - Lightweight inference
add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libncnn.so
)

# TVM - Optimized models
add_library(tvm4j_runtime_packed SHARED IMPORTED)
set_target_properties(tvm4j_runtime_packed PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libtvm4j_runtime_packed.so
)

# Qualcomm QNN - NPU acceleration
add_library(QnnHtpV73Stub SHARED IMPORTED)
set_target_properties(QnnHtpV73Stub PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libQnnHtpV73Stub.so
)

add_library(laylaQNN SHARED IMPORTED)
set_target_properties(laylaQNN PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/liblaylaQNN.so
)

# Speech - Sherpa ONNX (STT) + Piper (TTS)
add_library(sherpa-onnx-jni SHARED IMPORTED)
set_target_properties(sherpa-onnx-jni PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libsherpa-onnx-jni.so
)

add_library(kaldi-decoder-core SHARED IMPORTED)
set_target_properties(kaldi-decoder-core PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libkaldi-decoder-core.so
)

add_library(piper_phonemize SHARED IMPORTED)
set_target_properties(piper_phonemize PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libpiper_phonemize.so
)

add_library(espeak-ng SHARED IMPORTED)
set_target_properties(espeak-ng PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libespeak-ng.so
)

# Tokenization
add_library(sentencepiece SHARED IMPORTED)
set_target_properties(sentencepiece PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libsentencepiece.so
)

add_library(tokenizers-jni SHARED IMPORTED)
set_target_properties(tokenizers-jni PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libtokenizers-jni.so
)

# Storage - MMKV
add_library(mmkv SHARED IMPORTED)
set_target_properties(mmkv PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libmmkv.so
)

# Math - OpenBLAS
add_library(openblas SHARED IMPORTED)
set_target_properties(openblas PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libopenblas.so
)

# Parallelism - OpenMP
add_library(omp SHARED IMPORTED)
set_target_properties(omp PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libomp.so
)

# Logging
add_library(glog SHARED IMPORTED)
set_target_properties(glog PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libglog.so
)

add_library(spdlog SHARED IMPORTED)
set_target_properties(spdlog PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libspdlog.so
)

# React Native / Hermes
add_library(hermes SHARED IMPORTED)
set_target_properties(hermes PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libhermes.so
)

add_library(jsi SHARED IMPORTED)
set_target_properties(jsi PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libjsi.so
)

add_library(fbjni SHARED IMPORTED)
set_target_properties(fbjni PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libfbjni.so
)

add_library(folly_runtime SHARED IMPORTED)
set_target_properties(folly_runtime PROPERTIES
    IMPORTED_LOCATION ${LIBS_DIR}/libfolly_runtime.so
)

#=============================================================================
# MardukNativeCore Library
#=============================================================================

add_library(marduk_native_core SHARED
    marduk_native_core.cpp
)

# Include directories
target_include_directories(marduk_native_core PRIVATE
    ${INCLUDE_DIR}
    ${INCLUDE_DIR}/ggml
    ${INCLUDE_DIR}/llama
    ${INCLUDE_DIR}/onnxruntime
)

# Link libraries
target_link_libraries(marduk_native_core
    # Core tensor/ML
    ggml
    ggml-base
    ggml-cpu
    ggml-blas
    llama
    onnxruntime
    ncnn
    
    # GPU acceleration
    ggml-opencl
    ggml-vulkan
    OpenCL
    
    # NPU acceleration
    QnnHtpV73Stub
    laylaQNN
    
    # Speech
    sherpa-onnx-jni
    kaldi-decoder-core
    piper_phonemize
    espeak-ng
    
    # Tokenization
    sentencepiece
    tokenizers-jni
    
    # Storage
    mmkv
    
    # Math/Parallelism
    openblas
    omp
    
    # Logging
    glog
    spdlog
    
    # React Native
    fbjni
    folly_runtime
    
    # Android
    log
    android
)

# Compiler flags
target_compile_options(marduk_native_core PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
    -fno-rtti
)

# ARM64 optimizations
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(marduk_native_core PRIVATE
        -march=armv8-a+fp+simd
        -mtune=cortex-a76
    )
endif()

#=============================================================================
# JSI TurboModule Bridge (for React Native New Architecture)
#=============================================================================

add_library(marduk_jsi_bridge SHARED
    jsi_bridge.cpp
)

target_include_directories(marduk_jsi_bridge PRIVATE
    ${INCLUDE_DIR}
)

target_link_libraries(marduk_jsi_bridge
    marduk_native_core
    jsi
    hermes
    fbjni
    folly_runtime
    log
)

target_compile_options(marduk_jsi_bridge PRIVATE
    -Wall
    -O3
    -fno-rtti
)
