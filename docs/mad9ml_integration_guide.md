# Marduk-Commander with mad9ml Cognitive Engine

**Version**: 2.0
**Date**: 2025-12-29

## 1. Overview

This document details the successful integration of the **mad9ml cognitive engine** into the **Marduk-Commander** mobile application. This upgrade transforms the application from a rule-based command interpreter into a sophisticated, on-device cognitive agent capable of learning, reasoning, and autonomous operation.

The integration leverages the extensive set of pre-compiled ARM64 native libraries found in the repository, which previously seemed like overkill but are now revealed to be the essential foundation for running the advanced `mad9ml` architecture on mobile hardware.

## 2. Integration Architecture

The new architecture is a multi-layered system designed to bridge the high-level TypeScript-based `mad9ml` framework with low-level, high-performance native code.

```mermaid
graph TD
    subgraph React Native (TypeScript)
        A[MardukCommanderUI.tsx] --> B[MardukCommanderApp.ts];
        B --> C[Mad9mlMobileRuntime.ts];
        C --> D[NativeBridge.ts];
    end

    subgraph JNI Bridge
        D -- JSI/JNI --> E[marduk_native_core.cpp];
    end

    subgraph Native Libraries (C++/ARM64)
        E --> F[ggml / llama.cpp];
        E --> G[ONNX Runtime / ncnn];
        E --> H[Sherpa / Piper];
        E --> I[MMKV Storage];
        E --> J[Hardware Abstraction];
    end

    B --> K[TC Intent Builder];
    K -- ADB/Tasker --> L[Total Commander App];

    style A fill:#e94560,stroke:#fff,stroke-width:2px
    style L fill:#4ecca3,stroke:#fff,stroke-width:2px
```

### Architectural Layers

| Layer | Component(s) | Responsibility |
| :--- | :--- | :--- |
| **1. UI Layer** | `MardukCommanderUI.tsx` | Provides the user interface, including voice input, command history, and cognitive state visualization. |
| **2. Application Logic** | `MardukCommanderApp.ts` | Orchestrates the entire application flow, manages user commands, history, and feedback, and interfaces with the cognitive engine. |
| **3. Cognitive Adapter** | `Mad9mlMobileRuntime.ts` | Adapts the abstract `mad9ml` architecture for mobile execution. Manages the cognitive cycle, attention, memory, and persona evolution. |
| **4. TypeScript Bridge** | `NativeBridge.ts` | A high-level TypeScript wrapper that provides a clean, promise-based API over the native C++ functions. It abstracts away the complexities of JNI/JSI. |
| **5. C++ Native Bridge** | `marduk_native_core.h/.cpp` | The core JNI/JSI implementation. It exposes native functions to the JavaScript runtime and acts as a central hub, dispatching calls to the appropriate specialized native libraries. |
| **6. Native Engines** | `*.so` libraries | High-performance, pre-compiled libraries for tensor math (`ggml`), LLM inference (`llama.cpp`), ML model execution (`ONNX`), speech processing, and storage. |
| **7. TC Integration** | `TCIntentBuilder.ts` | Generates Android Intents to control Total Commander, executed via ADB or Tasker. |

## 3. Core Components Implemented

### 3.1. Native Interface Layer

- **`marduk_native_core.h` / `.cpp`**: A comprehensive C++ JNI bridge that creates a unified `MardukNativeCore` class. This class manages the lifecycle and provides access to five specialized engines:
  - **TensorEngine**: Wraps `ggml` for all tensor and matrix operations.
  - **LlamaEngine**: Wraps `llama.cpp` for GGUF model loading and LLM inference/embedding.
  - **OnnxEngine**: Wraps `ONNX Runtime` for executing trained ML models (e.g., intent classifiers).
  - **SpeechEngine**: Wraps `Sherpa-ONNX` (STT) and `Piper` (TTS) for voice interaction.
  - **StorageEngine**: Wraps `MMKV` for high-performance, persistent key-value storage.

- **`CMakeLists.txt`**: Configures the native build, linking over 20 pre-compiled `.so` libraries to the `marduk_native_core` bridge.

- **`NativeBridge.ts`**: A TypeScript facade that exposes all the C++ functionality through a clean, singleton `nativeBridge` object. It provides high-level classes like `TensorEngine`, `LlamaEngine`, etc., for use in the application.

### 3.2. mad9ml Mobile Runtime

- **`Mad9mlMobileRuntime.ts`**: This is the heart of the new cognitive capabilities. It adapts the conceptual `mad9ml` architecture to a concrete mobile implementation.
  - **Cognitive Cycle**: Implements a `cognitiveCycle()` method that runs periodically to manage attention decay, meta-cognition, and persona evolution.
  - **ECAN Attention**: An implementation of the ECAN (Elman-Context-Attention-Network) model to manage Short-Term (STI), Long-Term (LTI), and Very-Long-Term (VLTI) importance of concepts.
  - **Hypergraph Memory**: A memory system built on top of the native `StorageEngine` that uses embeddings from the `LlamaEngine` for similarity search.
  - **Persona Evolution**: A module that uses the LLM to generate responses based on a dynamic trait tensor. It updates its "fitness" based on command success and user feedback, and periodically "evolves" its traits.
  - **Intent & Entity Processing**: Combines a native ONNX model with rule-based fallbacks for robust command understanding.

### 3.3. Application & UI

- **`MardukCommanderApp.ts`**: The main application class was refactored to replace the old, simple Python logic with the new `Mad9mlMobileRuntime`. It now orchestrates the full cognitive loop: receiving a command, processing it through the runtime, executing the resulting plan, and recording the outcome for learning.

- **`MardukCommanderUI.tsx`**: A new React Native component that provides a rich user interface to interact with the powerful backend:
  - **Cognitive Visualizer**: Displays the real-time state of the cognitive engine, including the current cycle count, persona fitness, and top active concepts in the attention network.
  - **Insight Panel**: Shows meta-cognitive insights generated by the `MetaCognitiveEngine` as it reflects on its own performance.
  - **Voice Interface**: Integrated voice button for hands-free commands.
  - **Feedback Mechanism**: Allows the user to provide feedback (👍/👎) on any command, which directly influences the persona's fitness and evolution.

## 4. How it Works: A Command's Journey

1.  **Input**: The user issues a command, either via text or voice (which is transcribed to text by the `SpeechEngine`).
2.  **Cognitive Processing**: `MardukCommanderApp` sends the command to the `Mad9mlMobileRuntime`.
3.  **Embedding**: The `LlamaEngine` generates a vector embedding for the command text.
4.  **Understanding**: 
    - The `IntentClassifier` (ONNX model) uses the embedding to determine the user's intent (e.g., `move`).
    - The `EntityExtractor` parses the text to find entities (e.g., source path, destination path).
5.  **Attention**: The `ECANAllocator` increases the attention score for the identified intent and entities.
6.  **Reasoning & Planning**: The `PersonaEvolution` engine uses the LLM to generate a brief reasoning step, and the runtime builds a concrete `ExecutionPlan` of file operations.
7.  **Memory**: The command, its plan, and its embedding are stored in the `HypergraphMemory` as an episodic memory entry.
8.  **Execution**: `MardukCommanderApp` receives the plan and uses the `TCIntentBuilder` to generate and execute the corresponding Android Intents.
9.  **Feedback & Learning**: The success or failure of the execution is reported back to the `Mad9mlMobileRuntime`, which updates the `PersonaEvolution` engine's fitness score.
10. **Reflection**: Periodically, the `MetaCognitiveEngine` analyzes recent performance and history to generate insights, which may lead to suggestions or warnings displayed in the UI.

## 5. Build and Run

### Prerequisites

- Android SDK and NDK
- React Native environment
- All native libraries (`*.so`) placed correctly in the `libs/arm64-v8a` directory.

### Build Steps

1.  **Configure Gradle**: Ensure `android/app/build.gradle` is configured to use the `CMakeLists.txt` for the external native build.

    ```groovy
    // android/app/build.gradle
    android {
        defaultConfig {
            // ...
            externalNativeBuild {
                cmake {
                    cppFlags ""
                }
            }
        }
        externalNativeBuild {
            cmake {
                path "src/main/cpp/CMakeLists.txt"
                version "3.18.1"
            }
        }
    }
    ```

2.  **Install Dependencies**:

    ```bash
    pnpm install
    ```

3.  **Build and Run the App**:

    ```bash
    npx react-native run-android
    ```

This process will compile the C++ JNI bridge, link it against all the native libraries, and package it into the final APK.

## 6. Conclusion

By integrating the `mad9ml` cognitive engine, Marduk-Commander has evolved from a simple proof-of-concept into a powerful demonstration of on-device AGI. It successfully combines a high-level, abstract cognitive framework written in TypeScript with the raw performance of native C++ and specialized ARM64 libraries, creating a learning, reasoning, and self-improving file management assistant.
