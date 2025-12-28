# Mad9ml Integration Architecture for Marduk-Commander Mobile

## Executive Summary

This document describes the architecture for integrating the `mad9ml` cognitive engine from `OzCog/mad9ml` into the `marduk-commander` mobile application. The integration leverages the native ARM64 libraries already present in the repository to run the full cognitive architecture on-device, enabling truly autonomous, intelligent file management.

## Architecture Overview

The integration follows a **three-tier architecture** that bridges TypeScript cognitive code with native C++ inference engines:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    React Native UI (marduk-commander)                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  ││
│  │  │ Voice Input │  │ Text Input  │  │ File Browser│  │ Status Display│  ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘  ││
│  └─────────┼────────────────┼────────────────┼─────────────────┼──────────┘│
├────────────┼────────────────┼────────────────┼─────────────────┼────────────┤
│            │                │                │                 │            │
│            └────────────────┴────────────────┴─────────────────┘            │
│                                      │                                       │
│  ┌───────────────────────────────────▼──────────────────────────────────────┐│
│  │                      COGNITIVE LAYER (mad9ml)                            ││
│  │  ┌──────────────────────────────────────────────────────────────────┐   ││
│  │  │                    Mad9mlMobileRuntime (TypeScript)               │   ││
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐ │   ││
│  │  │  │  Tensor    │ │ Hypergraph │ │   ECAN     │ │ Meta-Cognitive │ │   ││
│  │  │  │ Operations │ │  Network   │ │ Attention  │ │   Reflection   │ │   ││
│  │  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └───────┬────────┘ │   ││
│  │  │        │              │              │                │          │   ││
│  │  │  ┌─────▼──────────────▼──────────────▼────────────────▼────────┐ │   ││
│  │  │  │                  NativeBridge (JSI/TurboModules)            │ │   ││
│  │  │  └─────────────────────────────┬───────────────────────────────┘ │   ││
│  │  └────────────────────────────────┼─────────────────────────────────┘   ││
│  └───────────────────────────────────┼─────────────────────────────────────┘│
├──────────────────────────────────────┼──────────────────────────────────────┤
│                                      │                                       │
│  ┌───────────────────────────────────▼──────────────────────────────────────┐│
│  │                      NATIVE LAYER (C++/ARM64)                            ││
│  │  ┌────────────────────────────────────────────────────────────────────┐  ││
│  │  │                    MardukNativeCore (C++)                          │  ││
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  ││
│  │  │  │  ggml    │ │  llama   │ │   ONNX   │ │  ncnn    │ │  Sherpa  │ │  ││
│  │  │  │ Tensor   │ │   LLM    │ │ Runtime  │ │ Infer    │ │   STT    │ │  ││
│  │  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │  ││
│  │  │       │            │            │            │            │       │  ││
│  │  │  ┌────▼────────────▼────────────▼────────────▼────────────▼────┐  │  ││
│  │  │  │              Hardware Abstraction Layer                     │  │  ││
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │  │  ││
│  │  │  │  │   CPU   │  │ OpenCL  │  │ Vulkan  │  │  Qualcomm QNN   │ │  │  ││
│  │  │  │  │  ARM64  │  │   GPU   │  │   GPU   │  │  NPU (HTP)      │ │  │  ││
│  │  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │  │  ││
│  │  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │  └────────────────────────────────────────────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Mad9mlMobileRuntime (TypeScript)

The mobile runtime adapts the `mad9ml` TypeScript code for on-device execution by replacing server-side operations with native library calls.

```typescript
// src/native/mad9ml-mobile-runtime.ts

interface Mad9mlMobileConfig {
  // Model paths
  llamaModelPath: string;
  onnxModelsPath: string;
  
  // Hardware settings
  useGPU: boolean;
  useNPU: boolean;
  maxThreads: number;
  
  // Memory settings
  memoryBudgetMB: number;
  tensorCacheSize: number;
  
  // Cognitive settings
  attentionBudget: number;
  evolutionEnabled: boolean;
  metaCognitionDepth: number;
}

class Mad9mlMobileRuntime {
  private nativeBridge: NativeBridge;
  private tensorEngine: NativeTensorEngine;
  private llamaEngine: NativeLlamaEngine;
  private attentionAllocator: NativeECANAllocator;
  private memorySystem: NativeMemorySystem;
  
  async initialize(config: Mad9mlMobileConfig): Promise<void>;
  async cognitiveCycle(): Promise<CognitiveState>;
  async processCommand(text: string): Promise<ExecutionPlan>;
  async evolvePersona(fitness: number): Promise<void>;
  async performReflection(): Promise<MetaCognitiveInsight>;
}
```

### 2. NativeBridge (JSI/TurboModules)

The bridge layer uses React Native's JavaScript Interface (JSI) for synchronous, high-performance communication with native code.

```typescript
// src/native/bridge/native-bridge.ts

interface NativeBridge {
  // Tensor operations (libggml)
  tensor: {
    create(shape: number[], data?: Float32Array): TensorHandle;
    add(a: TensorHandle, b: TensorHandle): TensorHandle;
    matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
    scale(t: TensorHandle, factor: number): TensorHandle;
    similarity(a: TensorHandle, b: TensorHandle): number;
    free(t: TensorHandle): void;
  };
  
  // LLM inference (libllama)
  llama: {
    loadModel(path: string, params: LlamaParams): ModelHandle;
    generate(model: ModelHandle, prompt: string, maxTokens: number): string;
    embed(model: ModelHandle, text: string): Float32Array;
    unloadModel(model: ModelHandle): void;
  };
  
  // ONNX inference (libonnxruntime)
  onnx: {
    loadModel(path: string): OnnxModelHandle;
    run(model: OnnxModelHandle, inputs: Record<string, Float32Array>): Record<string, Float32Array>;
    unloadModel(model: OnnxModelHandle): void;
  };
  
  // Speech (libsherpa-onnx, libpiper)
  speech: {
    transcribe(audioPath: string): Promise<string>;
    synthesize(text: string): Promise<string>; // Returns audio path
  };
  
  // Storage (libmmkv)
  storage: {
    set(key: string, value: string): void;
    get(key: string): string | null;
    delete(key: string): void;
    getAllKeys(): string[];
  };
}
```

### 3. MardukNativeCore (C++)

The C++ layer wraps the native libraries and exposes them through JNI/JSI.

```cpp
// android/app/src/main/cpp/marduk_native_core.h

#pragma once

#include <jni.h>
#include <jsi/jsi.h>
#include "ggml.h"
#include "llama.h"
#include "onnxruntime_cxx_api.h"

namespace marduk {

class MardukNativeCore : public facebook::jsi::HostObject {
public:
    MardukNativeCore();
    ~MardukNativeCore();
    
    // Initialization
    bool initialize(const std::string& modelsPath);
    
    // Tensor operations
    ggml_tensor* createTensor(const std::vector<int64_t>& shape);
    ggml_tensor* addTensors(ggml_tensor* a, ggml_tensor* b);
    ggml_tensor* matmul(ggml_tensor* a, ggml_tensor* b);
    float cosineSimilarity(ggml_tensor* a, ggml_tensor* b);
    
    // LLM operations
    bool loadLlamaModel(const std::string& path, int contextSize);
    std::string generate(const std::string& prompt, int maxTokens);
    std::vector<float> embed(const std::string& text);
    
    // ONNX operations
    bool loadOnnxModel(const std::string& name, const std::string& path);
    std::vector<float> runOnnx(const std::string& name, 
                                const std::vector<float>& input);
    
    // Hardware detection
    bool hasGPU() const;
    bool hasNPU() const;
    int getOptimalThreadCount() const;

private:
    struct ggml_context* m_ggmlCtx;
    llama_model* m_llamaModel;
    llama_context* m_llamaCtx;
    std::map<std::string, Ort::Session*> m_onnxSessions;
    Ort::Env m_ortEnv;
};

} // namespace marduk
```

## Data Flow

### Command Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           COMMAND PROCESSING FLOW                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. INPUT CAPTURE                                                             │
│     ┌─────────────┐      ┌─────────────┐                                     │
│     │ Voice Input │──────│ Sherpa STT  │──────┐                              │
│     └─────────────┘      └─────────────┘      │                              │
│                                               ▼                              │
│     ┌─────────────┐                    ┌─────────────┐                       │
│     │ Text Input  │────────────────────│ Raw Command │                       │
│     └─────────────┘                    └──────┬──────┘                       │
│                                               │                              │
│  2. COGNITIVE PROCESSING                      ▼                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                    Mad9ml Cognitive Engine                       │     │
│     │                                                                  │     │
│     │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │     │
│     │  │   Tokenize   │───▶│    Embed     │───▶│   Classify   │       │     │
│     │  │ (SentencePc) │    │   (llama)    │    │   (ONNX)     │       │     │
│     │  └──────────────┘    └──────────────┘    └───────┬──────┘       │     │
│     │                                                  │              │     │
│     │  ┌──────────────┐    ┌──────────────┐    ┌───────▼──────┐       │     │
│     │  │  Hypergraph  │◀───│    ECAN      │◀───│    Intent    │       │     │
│     │  │   Update     │    │  Attention   │    │   + Entities │       │     │
│     │  └──────────────┘    └──────────────┘    └──────────────┘       │     │
│     │                                                                  │     │
│     │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │     │
│     │  │  Procedural  │───▶│  Deliberate  │───▶│   Generate   │       │     │
│     │  │   Memory     │    │   (llama)    │    │    Plan      │       │     │
│     │  └──────────────┘    └──────────────┘    └───────┬──────┘       │     │
│     └──────────────────────────────────────────────────┼──────────────┘     │
│                                                        │                     │
│  3. EXECUTION                                          ▼                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                    TC-Connector Module                           │     │
│     │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │     │
│     │  │   Validate   │───▶│   Generate   │───▶│   Execute    │       │     │
│     │  │    Paths     │    │   Intents    │    │   via ADB    │       │     │
│     │  └──────────────┘    └──────────────┘    └───────┬──────┘       │     │
│     └──────────────────────────────────────────────────┼──────────────┘     │
│                                                        │                     │
│  4. FEEDBACK & LEARNING                                ▼                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                    Meta-Cognitive Loop                           │     │
│     │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │     │
│     │  │   Observe    │───▶│   Reflect    │───▶│    Evolve    │       │     │
│     │  │   Outcome    │    │   (llama)    │    │   Persona    │       │     │
│     │  └──────────────┘    └──────────────┘    └──────────────┘       │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Memory Architecture

### Cognitive Memory Mapping to Native Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    TypeScript Memory Layer (mad9ml)                     ││
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌───────────┐││
│  │  │   Declarative  │ │    Episodic    │ │   Procedural   │ │  Semantic │││
│  │  │    (Facts)     │ │   (Events)     │ │   (Skills)     │ │ (Concepts)│││
│  │  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └─────┬─────┘││
│  └──────────┼──────────────────┼──────────────────┼────────────────┼──────┘│
│             │                  │                  │                │       │
│             ▼                  ▼                  ▼                ▼       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Native Storage Layer                                 ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                      MMKV (libmmkv.so)                          │   ││
│  │  │  ┌──────────────────────────────────────────────────────────┐  │   ││
│  │  │  │  Key-Value Store (mmap-based, lock-free)                 │  │   ││
│  │  │  │                                                          │  │   ││
│  │  │  │  memory:declarative:{id} → JSON blob                     │  │   ││
│  │  │  │  memory:episodic:{id} → JSON blob                        │  │   ││
│  │  │  │  memory:procedural:{id} → JSON blob                      │  │   ││
│  │  │  │  memory:semantic:{id} → JSON blob                        │  │   ││
│  │  │  │  tensor:{id} → Binary tensor data                        │  │   ││
│  │  │  │  hypergraph:nodes → Serialized node index                │  │   ││
│  │  │  │  hypergraph:edges → Serialized edge index                │  │   ││
│  │  │  │  persona:current → Current persona tensor                │  │   ││
│  │  │  │  persona:history → Evolution history                     │  │   ││
│  │  │  └──────────────────────────────────────────────────────────┘  │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                    Tensor Cache (ggml context)                  │   ││
│  │  │  ┌──────────────────────────────────────────────────────────┐  │   ││
│  │  │  │  In-memory tensor pool for active cognitive operations   │  │   ││
│  │  │  │  - Working memory tensors                                │  │   ││
│  │  │  │  - Attention state tensors                               │  │   ││
│  │  │  │  - Persona trait tensors                                 │  │   ││
│  │  │  │  - Hypergraph activation tensors                         │  │   ││
│  │  │  └──────────────────────────────────────────────────────────┘  │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Hardware Acceleration Strategy

### Automatic Backend Selection

```typescript
// src/native/hardware/backend-selector.ts

interface HardwareCapabilities {
  cpu: {
    cores: number;
    architecture: 'arm64' | 'arm32' | 'x86_64';
    features: string[]; // NEON, SVE, etc.
  };
  gpu: {
    available: boolean;
    type: 'adreno' | 'mali' | 'powervr' | 'unknown';
    openclVersion?: string;
    vulkanVersion?: string;
  };
  npu: {
    available: boolean;
    type: 'qnn' | 'nnapi' | 'none';
    version?: string; // HTP version for Qualcomm
  };
}

class BackendSelector {
  selectOptimalBackend(operation: OperationType): Backend {
    // Priority: NPU > GPU > CPU
    
    if (this.capabilities.npu.available && this.supportsNPU(operation)) {
      return this.createQNNBackend();
    }
    
    if (this.capabilities.gpu.available && this.supportsGPU(operation)) {
      if (this.capabilities.gpu.vulkanVersion) {
        return this.createVulkanBackend();
      }
      return this.createOpenCLBackend();
    }
    
    return this.createCPUBackend();
  }
  
  private supportsNPU(op: OperationType): boolean {
    // NPU optimal for: matrix multiply, convolution, attention
    return ['matmul', 'conv2d', 'attention', 'embedding'].includes(op);
  }
  
  private supportsGPU(op: OperationType): boolean {
    // GPU optimal for: large tensor ops, parallel operations
    return ['matmul', 'elementwise', 'reduction', 'softmax'].includes(op);
  }
}
```

## Integration with TC-Connector

The existing `marduk-commander` TC-Connector is enhanced to receive execution plans from the mad9ml cognitive engine:

```typescript
// src/connectors/tc_connector_enhanced.ts

class EnhancedTCConnector extends TCConnector {
  private cognitiveEngine: Mad9mlMobileRuntime;
  
  async processNaturalLanguage(input: string): Promise<ExecutionResult> {
    // 1. Cognitive processing via mad9ml
    const cognitiveResult = await this.cognitiveEngine.processCommand(input);
    
    // 2. Extract execution plan
    const plan = cognitiveResult.executionPlan;
    
    // 3. Validate with procedural memory
    const procedure = await this.cognitiveEngine.getProcedure(plan.intent);
    
    // 4. Execute via TC-Connector
    const operations = this.buildOperations(plan, procedure);
    const results = await this.executeBatch(operations);
    
    // 5. Feedback to cognitive engine
    await this.cognitiveEngine.recordOutcome({
      plan,
      results,
      success: results.every(r => r.success)
    });
    
    // 6. Trigger meta-cognitive reflection if needed
    if (this.shouldReflect(results)) {
      await this.cognitiveEngine.performReflection();
    }
    
    return results;
  }
}
```

## Model Requirements

### Required Model Files

| Model | Purpose | Size | Location |
|-------|---------|------|----------|
| `marduk-persona.gguf` | LLM for deliberation/reflection | ~2-4GB | `assets/models/` |
| `intent-classifier.onnx` | Intent classification | ~50MB | `assets/models/` |
| `entity-extractor.onnx` | Entity extraction | ~100MB | `assets/models/` |
| `attention-allocator.onnx` | ECAN attention model | ~20MB | `assets/models/` |
| `sherpa-stt.onnx` | Speech-to-text | ~50MB | `assets/models/` |
| `piper-tts.onnx` | Text-to-speech | ~30MB | `assets/models/` |

### Model Quantization

For mobile deployment, models should be quantized:

```bash
# Quantize LLM to 4-bit
./llama-quantize marduk-persona-f16.gguf marduk-persona-q4_k_m.gguf Q4_K_M

# Quantize ONNX models to INT8
python -m onnxruntime.quantization.quantize \
  --input intent-classifier.onnx \
  --output intent-classifier-int8.onnx \
  --quant_format QDQ
```

## Build Configuration

### Android CMakeLists.txt

```cmake
# android/app/src/main/cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.18.1)
project(marduk_native_core)

set(CMAKE_CXX_STANDARD 17)

# Pre-built libraries
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/../../../../../libs/${ANDROID_ABI})

# ggml
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${LIBS_DIR}/libggml.so)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${LIBS_DIR}/libggml-base.so)

# llama
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LIBS_DIR}/libllama.so)

# ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${LIBS_DIR}/libonnxruntime.so)

# MMKV
add_library(mmkv SHARED IMPORTED)
set_target_properties(mmkv PROPERTIES IMPORTED_LOCATION ${LIBS_DIR}/libmmkv.so)

# Main library
add_library(marduk_native_core SHARED
    marduk_native_core.cpp
    tensor_engine.cpp
    llama_engine.cpp
    onnx_engine.cpp
    storage_engine.cpp
    jsi_bindings.cpp
)

target_link_libraries(marduk_native_core
    ggml
    ggml-base
    llama
    onnxruntime
    mmkv
    log
    android
)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Command latency | <500ms | From voice input to execution start |
| Cognitive cycle | <100ms | Single mad9ml cognitive cycle |
| Memory footprint | <500MB | Total app memory usage |
| Battery impact | <5%/hour | During active use |
| Model load time | <3s | Cold start model loading |

## Security Considerations

1. **Model Integrity**: Verify model checksums before loading
2. **Memory Protection**: Use secure memory for persona data
3. **Sandboxing**: Isolate cognitive engine from sensitive data
4. **Audit Logging**: Log all file operations for accountability

## Next Steps

1. **Phase 1**: Implement NativeBridge JSI bindings
2. **Phase 2**: Create Mad9mlMobileRuntime adapter
3. **Phase 3**: Integrate with existing TC-Connector
4. **Phase 4**: Add voice interface
5. **Phase 5**: Optimize for hardware acceleration
