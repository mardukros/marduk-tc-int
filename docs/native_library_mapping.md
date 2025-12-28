# Native Library to Mad9ml Component Mapping

This document maps the ARM64 native libraries in `libs/arm64-v8a/` to their corresponding mad9ml cognitive engine components.

## Library Categories and Mad9ml Integration Points

### 1. Core Tensor Engine (ggml)

These libraries form the foundation for mad9ml's tensor-based cognitive operations.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libggml.so` | Core | `tensor/operations.ts` | Base tensor operations (add, scale, matmul) |
| `libggml-base.so` | Core | `tensor/operations.ts` | Tensor memory management and allocation |
| `libggml-cpu.so` | Backend | `tensor/operations.ts` | CPU-optimized tensor computations |
| `libggml-blas.so` | Backend | `tensor/operations.ts` | BLAS-accelerated linear algebra |
| `libggml-opencl.so` | Backend | `tensor/operations.ts` | OpenCL GPU acceleration |
| `libggml-vulkan.so` | Backend | `tensor/operations.ts` | Vulkan GPU acceleration |
| `libOpenCL.so` | Runtime | GPU Backend | OpenCL runtime for GPU operations |

**Integration**: These libraries directly implement the tensor operations defined in `mad9ml/tensor/operations.ts`. The TypeScript code generates computation graphs that are executed by these native libraries.

### 2. LLM Inference Engine (llama.cpp)

Core inference engine for the Marduk persona and language understanding.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libllama.so` | 35MB | `persona/evolution.ts` | LLM inference engine |
| `libllama-jni.so` | 1.4MB | JNI Bridge | Java/Kotlin native interface |
| `libexecutorch_llama_jni.so` | 4.4MB | JNI Bridge | ExecuTorch Llama integration |

**Integration**: Powers the Marduk persona's language generation, intent understanding, and meta-cognitive reflection. The persona evolution system uses this for generating self-modifications.

### 3. Neural Network Inference (ONNX/ncnn/TVM)

Multi-framework inference for specialized models.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libonnxruntime.so` | 22MB | `hypergraph/` | ONNX model inference |
| `libonnxruntime4j_jni.so` | 1.2MB | JNI Bridge | Java ONNX interface |
| `libonnxruntimejsihelper.so` | 0.3MB | JSI Bridge | JavaScript interface |
| `libncnn.so` | 4.4MB | `attention/` | Lightweight neural inference |
| `libtvm4j_runtime_packed.so` | 16MB | `tensor/` | TVM compiled models |

**Integration**: 
- ONNX Runtime: Runs pre-trained models for pattern recognition in the hypergraph
- ncnn: Lightweight inference for ECAN attention allocation
- TVM: Optimized tensor operations for mobile

### 4. Qualcomm NPU Acceleration (QNN)

Hardware acceleration for Qualcomm Snapdragon devices.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libQnnHtpV68Stub.so` | - | NPU Backend | Snapdragon 865 HTP |
| `libQnnHtpV69Stub.so` | - | NPU Backend | Snapdragon 888 HTP |
| `libQnnHtpV73Stub.so` | - | NPU Backend | Snapdragon 8 Gen 1 HTP |
| `libQnnHtpV75Stub.so` | - | NPU Backend | Snapdragon 8 Gen 2 HTP |
| `libQnnHtpV79Stub.so` | - | NPU Backend | Snapdragon 8 Gen 3 HTP |
| `liblaylaQNN.so` | 0.8MB | QNN Interface | Layla QNN wrapper |

**Integration**: Enables hardware-accelerated inference on Qualcomm devices, critical for real-time cognitive processing.

### 5. Speech Processing (STT/TTS)

Voice interface for natural language commands.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libsherpa-onnx-jni.so` | 4.4MB | Voice Input | Speech-to-text (Sherpa ONNX) |
| `libkaldi-decoder-core.so` | 1.7MB | Voice Input | Kaldi speech decoder |
| `libkaldi-native-fbank-core.so` | 0.5MB | Voice Input | Audio feature extraction |
| `libpiper_phonemize.so` | 0.7MB | Voice Output | Text-to-speech phonemization |
| `libespeak-ng.so` | 0.4MB | Voice Output | eSpeak NG TTS engine |

**Integration**: Enables voice-controlled file management. Speech is transcribed, processed by the cognitive core, and responses are spoken back.

### 6. Text Processing (Tokenization/Translation)

Natural language preprocessing for the cognitive engine.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libsentencepiece.so` | 1.1MB | `vocabulary/` | Tokenization (SentencePiece) |
| `libsentencepiece_train.so` | 1.7MB | `vocabulary/` | Tokenizer training |
| `libssentencepiece_core.so` | 0.9MB | `vocabulary/` | Core tokenization |
| `libtokenizers-jni.so` | 0.3MB | JNI Bridge | HuggingFace tokenizers |
| `libctranslate2.so` | 7.3MB | Translation | CTranslate2 inference |
| `libctranslate2-jni.so` | 0.1MB | JNI Bridge | CTranslate2 Java interface |

**Integration**: Tokenizes user commands and file names for semantic understanding. Supports multilingual file management.

### 7. Image/Vision Processing

Visual understanding for content-aware file organization.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libmediapipe_tasks_vision_image_generator_jni.so` | 1.9MB | Vision | MediaPipe vision tasks |
| `libimagegenerator_gpu.so` | 0.5MB | Vision | GPU image generation |
| `libimagepipeline.so` | 0.1MB | Vision | Image processing pipeline |
| `libnative-filters.so` | 0.1MB | Vision | Image filters |
| `libnative-imagetranscoder.so` | 0.1MB | Vision | Image format conversion |
| `libsd-jni.so` | 11MB | Vision | Stable Diffusion inference |

**Integration**: Enables visual file categorization (photos, documents, screenshots) and content-aware organization.

### 8. React Native Framework

Application framework for the mobile app.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libhermes.so` | 4.9MB | JS Runtime | Hermes JavaScript engine |
| `libhermes_executor.so` | 0.3MB | JS Runtime | Hermes execution |
| `libjsi.so` | 0.3MB | JS Bridge | JavaScript Interface |
| `libreactnativejni.so` | 0.3MB | RN Core | React Native JNI |
| `libfabricjni.so` | 0.5MB | RN Core | Fabric renderer |
| `libreanimated.so` | 1.6MB | RN Animation | Reanimated library |
| `libexpo-modules-core.so` | 0.2MB | Expo | Expo modules |

**Integration**: The React Native framework hosts the JavaScript/TypeScript mad9ml code, bridging to native libraries via JSI and JNI.

### 9. Storage and Utilities

Supporting infrastructure.

| Library | Size | Mad9ml Component | Purpose |
|---------|------|------------------|---------|
| `libmmkv.so` | 0.3MB | `memory/` | High-performance key-value storage |
| `liblvdb-jni.so` | 0.5MB | `memory/` | LevelDB storage |
| `libfolly_runtime.so` | 1.1MB | Runtime | Facebook Folly utilities |
| `libglog.so` | 0.2MB | Logging | Google logging |
| `libspdlog.so` | 0.5MB | Logging | Fast C++ logging |
| `libomp.so` | 0.6MB | Runtime | OpenMP parallelism |
| `libopenblas.so` | 4.9MB | Math | Optimized BLAS |

**Integration**: MMKV provides fast persistence for episodic and procedural memory. OpenBLAS accelerates tensor operations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        React Native App (TypeScript)                     │
│                              marduk-commander                            │
├─────────────────────────────────────────────────────────────────────────┤
│                          mad9ml Cognitive Engine                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Tensor    │  │ Hypergraph  │  │   ECAN      │  │ Meta-Cognitive  │ │
│  │ Operations  │  │  Networks   │  │ Attention   │  │   Reflection    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
├─────────┼────────────────┼────────────────┼──────────────────┼──────────┤
│         │                │                │                  │          │
│    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐        ┌────▼────┐    │
│    │   JSI   │      │   JSI   │      │   JSI   │        │   JSI   │    │
│    │ Bridge  │      │ Bridge  │      │ Bridge  │        │ Bridge  │    │
│    └────┬────┘      └────┬────┘      └────┬────┘        └────┬────┘    │
├─────────┼────────────────┼────────────────┼──────────────────┼──────────┤
│         │                │                │                  │          │
│    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐        ┌────▼────┐    │
│    │  ggml   │      │  ONNX   │      │  ncnn   │        │  llama  │    │
│    │ Tensor  │      │ Runtime │      │ Infer   │        │  .cpp   │    │
│    └────┬────┘      └────┬────┘      └────┬────┘        └────┬────┘    │
├─────────┼────────────────┼────────────────┼──────────────────┼──────────┤
│         │                │                │                  │          │
│         └────────────────┴────────────────┴──────────────────┘          │
│                                    │                                     │
│                          ┌─────────▼─────────┐                          │
│                          │   Hardware Layer   │                          │
│                          │  CPU / GPU / NPU   │                          │
│                          │  (QNN/OpenCL/Vulkan)│                          │
│                          └───────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Integration Priority

### Phase 1: Core Tensor Operations
1. `libggml*.so` - Tensor math foundation
2. `libllama.so` - LLM inference for persona

### Phase 2: Cognitive Subsystems
3. `libonnxruntime.so` - Pattern recognition models
4. `libncnn.so` - Lightweight attention models
5. `libmmkv.so` - Memory persistence

### Phase 3: Voice Interface
6. `libsherpa-onnx-jni.so` - Speech-to-text
7. `libpiper_phonemize.so` + `libespeak-ng.so` - Text-to-speech

### Phase 4: Visual Intelligence
8. `libmediapipe_*.so` - Image understanding
9. `libsd-jni.so` - Visual generation (optional)

### Phase 5: Hardware Optimization
10. `libQnn*.so` - NPU acceleration
11. `libtvm4j_runtime_packed.so` - Optimized models
