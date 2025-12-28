/**
 * MardukNativeCore - Native C++ bridge for mad9ml cognitive engine
 * 
 * This header defines the native interface layer that bridges the TypeScript
 * mad9ml cognitive engine with ARM64 native libraries (ggml, llama, ONNX, etc.)
 */

#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

// Forward declarations for native libraries
struct ggml_context;
struct ggml_tensor;
struct llama_model;
struct llama_context;

namespace marduk {

//=============================================================================
// Type Definitions
//=============================================================================

using TensorHandle = int64_t;
using ModelHandle = int64_t;

struct TensorShape {
    std::vector<int64_t> dims;
    size_t size() const;
};

struct TensorData {
    TensorHandle handle;
    TensorShape shape;
    std::vector<float> data;
};

struct LlamaParams {
    int contextSize = 2048;
    int batchSize = 512;
    int threads = 4;
    bool useGPU = false;
    bool useNPU = false;
    float temperature = 0.7f;
    float topP = 0.9f;
    int topK = 40;
};

struct OnnxParams {
    int threads = 4;
    bool useGPU = false;
    std::string executionProvider = "CPU"; // CPU, CUDA, QNN, NNAPI
};

struct HardwareCapabilities {
    // CPU
    int cpuCores;
    std::string cpuArch;
    bool hasNeon;
    bool hasSve;
    
    // GPU
    bool hasGPU;
    std::string gpuType;
    std::string openclVersion;
    std::string vulkanVersion;
    
    // NPU
    bool hasNPU;
    std::string npuType;
    std::string htpVersion;
};

//=============================================================================
// Tensor Engine - ggml wrapper
//=============================================================================

class TensorEngine {
public:
    TensorEngine();
    ~TensorEngine();
    
    bool initialize(size_t memoryPoolSize);
    void shutdown();
    
    // Tensor creation
    TensorHandle create(const TensorShape& shape);
    TensorHandle create(const TensorShape& shape, const float* data);
    TensorHandle zeros(const TensorShape& shape);
    TensorHandle ones(const TensorShape& shape);
    TensorHandle random(const TensorShape& shape, float min = 0.0f, float max = 1.0f);
    
    // Tensor operations
    TensorHandle add(TensorHandle a, TensorHandle b);
    TensorHandle sub(TensorHandle a, TensorHandle b);
    TensorHandle mul(TensorHandle a, TensorHandle b);
    TensorHandle div(TensorHandle a, TensorHandle b);
    TensorHandle matmul(TensorHandle a, TensorHandle b);
    TensorHandle transpose(TensorHandle t);
    TensorHandle reshape(TensorHandle t, const TensorShape& newShape);
    TensorHandle scale(TensorHandle t, float factor);
    TensorHandle softmax(TensorHandle t, int axis = -1);
    TensorHandle relu(TensorHandle t);
    TensorHandle gelu(TensorHandle t);
    TensorHandle layerNorm(TensorHandle t, TensorHandle gamma, TensorHandle beta);
    
    // Reduction operations
    float sum(TensorHandle t);
    float mean(TensorHandle t);
    float max(TensorHandle t);
    float min(TensorHandle t);
    
    // Similarity operations
    float cosineSimilarity(TensorHandle a, TensorHandle b);
    float dotProduct(TensorHandle a, TensorHandle b);
    float euclideanDistance(TensorHandle a, TensorHandle b);
    
    // Data access
    TensorData getData(TensorHandle t);
    void setData(TensorHandle t, const float* data, size_t size);
    TensorShape getShape(TensorHandle t);
    
    // Memory management
    void free(TensorHandle t);
    void freeAll();
    size_t getMemoryUsage() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

//=============================================================================
// Llama Engine - llama.cpp wrapper
//=============================================================================

class LlamaEngine {
public:
    LlamaEngine();
    ~LlamaEngine();
    
    bool initialize();
    void shutdown();
    
    // Model management
    ModelHandle loadModel(const std::string& path, const LlamaParams& params);
    void unloadModel(ModelHandle model);
    bool isModelLoaded(ModelHandle model) const;
    
    // Text generation
    std::string generate(ModelHandle model, const std::string& prompt, int maxTokens);
    std::string generateWithCallback(ModelHandle model, const std::string& prompt, 
                                      int maxTokens, std::function<bool(const std::string&)> callback);
    
    // Embeddings
    std::vector<float> embed(ModelHandle model, const std::string& text);
    std::vector<std::vector<float>> embedBatch(ModelHandle model, 
                                                const std::vector<std::string>& texts);
    
    // Tokenization
    std::vector<int> tokenize(ModelHandle model, const std::string& text);
    std::string detokenize(ModelHandle model, const std::vector<int>& tokens);
    int getVocabSize(ModelHandle model) const;
    
    // Context management
    void clearContext(ModelHandle model);
    int getContextLength(ModelHandle model) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

//=============================================================================
// ONNX Engine - ONNX Runtime wrapper
//=============================================================================

class OnnxEngine {
public:
    OnnxEngine();
    ~OnnxEngine();
    
    bool initialize(const OnnxParams& params);
    void shutdown();
    
    // Model management
    ModelHandle loadModel(const std::string& name, const std::string& path);
    void unloadModel(ModelHandle model);
    bool isModelLoaded(ModelHandle model) const;
    
    // Inference
    std::vector<float> run(ModelHandle model, const std::vector<float>& input,
                           const std::vector<int64_t>& inputShape);
    std::unordered_map<std::string, std::vector<float>> runMultiOutput(
        ModelHandle model,
        const std::unordered_map<std::string, std::vector<float>>& inputs,
        const std::unordered_map<std::string, std::vector<int64_t>>& inputShapes);
    
    // Model info
    std::vector<std::string> getInputNames(ModelHandle model) const;
    std::vector<std::string> getOutputNames(ModelHandle model) const;
    std::vector<int64_t> getInputShape(ModelHandle model, const std::string& name) const;
    std::vector<int64_t> getOutputShape(ModelHandle model, const std::string& name) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

//=============================================================================
// Speech Engine - Sherpa ONNX + Piper wrapper
//=============================================================================

class SpeechEngine {
public:
    SpeechEngine();
    ~SpeechEngine();
    
    bool initialize(const std::string& modelsPath);
    void shutdown();
    
    // Speech-to-Text
    bool loadSTTModel(const std::string& modelPath);
    std::string transcribe(const std::string& audioPath);
    std::string transcribeBuffer(const float* samples, size_t numSamples, int sampleRate);
    
    // Text-to-Speech
    bool loadTTSModel(const std::string& modelPath);
    std::string synthesize(const std::string& text, const std::string& outputPath);
    std::vector<float> synthesizeBuffer(const std::string& text, int& sampleRate);
    
    // Voice Activity Detection
    bool isVoiceActive(const float* samples, size_t numSamples, int sampleRate);
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

//=============================================================================
// Storage Engine - MMKV wrapper
//=============================================================================

class StorageEngine {
public:
    StorageEngine();
    ~StorageEngine();
    
    bool initialize(const std::string& storagePath);
    void shutdown();
    
    // Key-Value operations
    void set(const std::string& key, const std::string& value);
    std::string get(const std::string& key, const std::string& defaultValue = "");
    bool has(const std::string& key) const;
    void remove(const std::string& key);
    std::vector<std::string> getAllKeys() const;
    
    // Binary data
    void setBytes(const std::string& key, const std::vector<uint8_t>& data);
    std::vector<uint8_t> getBytes(const std::string& key) const;
    
    // Tensor storage
    void setTensor(const std::string& key, const TensorData& tensor);
    TensorData getTensor(const std::string& key) const;
    
    // Batch operations
    void setBatch(const std::unordered_map<std::string, std::string>& items);
    std::unordered_map<std::string, std::string> getBatch(const std::vector<std::string>& keys) const;
    
    // Maintenance
    void sync();
    void clear();
    size_t getSize() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

//=============================================================================
// Hardware Manager - Device capability detection and backend selection
//=============================================================================

class HardwareManager {
public:
    static HardwareManager& getInstance();
    
    HardwareCapabilities getCapabilities() const;
    
    // Backend selection
    std::string selectOptimalBackend(const std::string& operation) const;
    bool supportsGPU() const;
    bool supportsNPU() const;
    int getOptimalThreadCount() const;
    
    // Memory info
    size_t getTotalMemory() const;
    size_t getAvailableMemory() const;
    
private:
    HardwareManager();
    void detectCapabilities();
    
    HardwareCapabilities m_capabilities;
};

//=============================================================================
// MardukNativeCore - Main interface class
//=============================================================================

class MardukNativeCore {
public:
    MardukNativeCore();
    ~MardukNativeCore();
    
    // Initialization
    bool initialize(const std::string& modelsPath, size_t memoryBudgetMB = 512);
    void shutdown();
    bool isInitialized() const;
    
    // Engine access
    TensorEngine& getTensorEngine();
    LlamaEngine& getLlamaEngine();
    OnnxEngine& getOnnxEngine();
    SpeechEngine& getSpeechEngine();
    StorageEngine& getStorageEngine();
    
    // Hardware info
    HardwareCapabilities getHardwareCapabilities() const;
    
    // Convenience methods for common operations
    std::string processCommand(const std::string& command);
    std::vector<float> getEmbedding(const std::string& text);
    float computeSimilarity(const std::string& text1, const std::string& text2);
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    bool m_initialized;
};

} // namespace marduk

//=============================================================================
// JNI Interface
//=============================================================================

extern "C" {

// Initialization
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_create(JNIEnv* env, jobject thiz);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_destroy(JNIEnv* env, jobject thiz, jlong handle);
JNIEXPORT jboolean JNICALL Java_com_marduk_commander_native_MardukNative_initialize(JNIEnv* env, jobject thiz, jlong handle, jstring modelsPath, jint memoryBudgetMB);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_shutdown(JNIEnv* env, jobject thiz, jlong handle);

// Tensor operations
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorCreate(JNIEnv* env, jobject thiz, jlong handle, jlongArray shape, jfloatArray data);
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorAdd(JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b);
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorMatmul(JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b);
JNIEXPORT jfloat JNICALL Java_com_marduk_commander_native_MardukNative_tensorCosineSimilarity(JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_tensorFree(JNIEnv* env, jobject thiz, jlong handle, jlong tensor);

// LLM operations
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_llamaLoadModel(JNIEnv* env, jobject thiz, jlong handle, jstring path, jint contextSize);
JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_llamaGenerate(JNIEnv* env, jobject thiz, jlong handle, jlong model, jstring prompt, jint maxTokens);
JNIEXPORT jfloatArray JNICALL Java_com_marduk_commander_native_MardukNative_llamaEmbed(JNIEnv* env, jobject thiz, jlong handle, jlong model, jstring text);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_llamaUnloadModel(JNIEnv* env, jobject thiz, jlong handle, jlong model);

// ONNX operations
JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_onnxLoadModel(JNIEnv* env, jobject thiz, jlong handle, jstring name, jstring path);
JNIEXPORT jfloatArray JNICALL Java_com_marduk_commander_native_MardukNative_onnxRun(JNIEnv* env, jobject thiz, jlong handle, jlong model, jfloatArray input, jlongArray shape);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_onnxUnloadModel(JNIEnv* env, jobject thiz, jlong handle, jlong model);

// Speech operations
JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_speechTranscribe(JNIEnv* env, jobject thiz, jlong handle, jstring audioPath);
JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_speechSynthesize(JNIEnv* env, jobject thiz, jlong handle, jstring text, jstring outputPath);

// Storage operations
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_storageSet(JNIEnv* env, jobject thiz, jlong handle, jstring key, jstring value);
JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_storageGet(JNIEnv* env, jobject thiz, jlong handle, jstring key);
JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_storageRemove(JNIEnv* env, jobject thiz, jlong handle, jstring key);

// Hardware info
JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_getHardwareInfo(JNIEnv* env, jobject thiz, jlong handle);

} // extern "C"
