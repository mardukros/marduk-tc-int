/**
 * MardukNativeCore - Implementation
 * 
 * Native C++ implementation bridging mad9ml cognitive engine with ARM64 libraries
 */

#include "marduk_native_core.h"
#include <android/log.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>

// Logging macros
#define LOG_TAG "MardukNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Include native library headers
// Note: These would be the actual headers from the native libraries
// For now, we define stub structures for compilation

// ggml stub (actual header: ggml.h)
extern "C" {
    struct ggml_context;
    struct ggml_tensor;
    struct ggml_init_params {
        size_t mem_size;
        void* mem_buffer;
        bool no_alloc;
    };
    
    ggml_context* ggml_init(ggml_init_params params);
    void ggml_free(ggml_context* ctx);
    ggml_tensor* ggml_new_tensor_1d(ggml_context* ctx, int type, int64_t ne0);
    ggml_tensor* ggml_new_tensor_2d(ggml_context* ctx, int type, int64_t ne0, int64_t ne1);
    ggml_tensor* ggml_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
    ggml_tensor* ggml_mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
    float* ggml_get_data_f32(ggml_tensor* tensor);
    int64_t ggml_nelements(ggml_tensor* tensor);
}

// llama stub (actual header: llama.h)
extern "C" {
    struct llama_model;
    struct llama_context;
    struct llama_model_params {
        int n_gpu_layers;
        bool use_mmap;
        bool use_mlock;
    };
    struct llama_context_params {
        uint32_t n_ctx;
        uint32_t n_batch;
        uint32_t n_threads;
        bool embedding;
    };
    
    llama_model* llama_load_model_from_file(const char* path, llama_model_params params);
    void llama_free_model(llama_model* model);
    llama_context* llama_new_context_with_model(llama_model* model, llama_context_params params);
    void llama_free(llama_context* ctx);
}

namespace marduk {

//=============================================================================
// TensorShape Implementation
//=============================================================================

size_t TensorShape::size() const {
    if (dims.empty()) return 0;
    size_t s = 1;
    for (auto d : dims) s *= d;
    return s;
}

//=============================================================================
// TensorEngine Implementation
//=============================================================================

struct TensorEngine::Impl {
    ggml_context* ctx = nullptr;
    std::unordered_map<TensorHandle, ggml_tensor*> tensors;
    TensorHandle nextHandle = 1;
    std::mutex mutex;
    size_t memoryPoolSize = 0;
    
    TensorHandle registerTensor(ggml_tensor* t) {
        std::lock_guard<std::mutex> lock(mutex);
        TensorHandle h = nextHandle++;
        tensors[h] = t;
        return h;
    }
    
    ggml_tensor* getTensor(TensorHandle h) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = tensors.find(h);
        return (it != tensors.end()) ? it->second : nullptr;
    }
};

TensorEngine::TensorEngine() : m_impl(std::make_unique<Impl>()) {}
TensorEngine::~TensorEngine() { shutdown(); }

bool TensorEngine::initialize(size_t memoryPoolSize) {
    LOGI("TensorEngine::initialize with %zu MB", memoryPoolSize / (1024 * 1024));
    
    ggml_init_params params = {
        .mem_size = memoryPoolSize,
        .mem_buffer = nullptr,
        .no_alloc = false
    };
    
    m_impl->ctx = ggml_init(params);
    m_impl->memoryPoolSize = memoryPoolSize;
    
    return m_impl->ctx != nullptr;
}

void TensorEngine::shutdown() {
    if (m_impl->ctx) {
        ggml_free(m_impl->ctx);
        m_impl->ctx = nullptr;
    }
    m_impl->tensors.clear();
}

TensorHandle TensorEngine::create(const TensorShape& shape) {
    if (!m_impl->ctx || shape.dims.empty()) return 0;
    
    ggml_tensor* t = nullptr;
    if (shape.dims.size() == 1) {
        t = ggml_new_tensor_1d(m_impl->ctx, 0 /* GGML_TYPE_F32 */, shape.dims[0]);
    } else if (shape.dims.size() == 2) {
        t = ggml_new_tensor_2d(m_impl->ctx, 0, shape.dims[0], shape.dims[1]);
    }
    
    return t ? m_impl->registerTensor(t) : 0;
}

TensorHandle TensorEngine::create(const TensorShape& shape, const float* data) {
    TensorHandle h = create(shape);
    if (h && data) {
        ggml_tensor* t = m_impl->getTensor(h);
        if (t) {
            float* dst = ggml_get_data_f32(t);
            std::memcpy(dst, data, shape.size() * sizeof(float));
        }
    }
    return h;
}

TensorHandle TensorEngine::add(TensorHandle a, TensorHandle b) {
    ggml_tensor* ta = m_impl->getTensor(a);
    ggml_tensor* tb = m_impl->getTensor(b);
    if (!ta || !tb) return 0;
    
    ggml_tensor* result = ggml_add(m_impl->ctx, ta, tb);
    return result ? m_impl->registerTensor(result) : 0;
}

TensorHandle TensorEngine::matmul(TensorHandle a, TensorHandle b) {
    ggml_tensor* ta = m_impl->getTensor(a);
    ggml_tensor* tb = m_impl->getTensor(b);
    if (!ta || !tb) return 0;
    
    ggml_tensor* result = ggml_mul_mat(m_impl->ctx, ta, tb);
    return result ? m_impl->registerTensor(result) : 0;
}

float TensorEngine::cosineSimilarity(TensorHandle a, TensorHandle b) {
    ggml_tensor* ta = m_impl->getTensor(a);
    ggml_tensor* tb = m_impl->getTensor(b);
    if (!ta || !tb) return 0.0f;
    
    float* da = ggml_get_data_f32(ta);
    float* db = ggml_get_data_f32(tb);
    int64_t n = ggml_nelements(ta);
    
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        dot += da[i] * db[i];
        normA += da[i] * da[i];
        normB += db[i] * db[i];
    }
    
    float denom = std::sqrt(normA) * std::sqrt(normB);
    return (denom > 0) ? (dot / denom) : 0.0f;
}

TensorData TensorEngine::getData(TensorHandle t) {
    TensorData result;
    result.handle = t;
    
    ggml_tensor* tensor = m_impl->getTensor(t);
    if (tensor) {
        int64_t n = ggml_nelements(tensor);
        float* data = ggml_get_data_f32(tensor);
        result.data.assign(data, data + n);
        // Shape would need proper extraction from ggml_tensor
    }
    
    return result;
}

void TensorEngine::free(TensorHandle t) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->tensors.erase(t);
}

void TensorEngine::freeAll() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->tensors.clear();
}

size_t TensorEngine::getMemoryUsage() const {
    return m_impl->memoryPoolSize; // Simplified
}

//=============================================================================
// LlamaEngine Implementation
//=============================================================================

struct LlamaEngine::Impl {
    std::unordered_map<ModelHandle, std::pair<llama_model*, llama_context*>> models;
    ModelHandle nextHandle = 1;
    std::mutex mutex;
};

LlamaEngine::LlamaEngine() : m_impl(std::make_unique<Impl>()) {}
LlamaEngine::~LlamaEngine() { shutdown(); }

bool LlamaEngine::initialize() {
    LOGI("LlamaEngine::initialize");
    return true;
}

void LlamaEngine::shutdown() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    for (auto& [handle, pair] : m_impl->models) {
        if (pair.second) llama_free(pair.second);
        if (pair.first) llama_free_model(pair.first);
    }
    m_impl->models.clear();
}

ModelHandle LlamaEngine::loadModel(const std::string& path, const LlamaParams& params) {
    LOGI("LlamaEngine::loadModel: %s", path.c_str());
    
    llama_model_params modelParams = {
        .n_gpu_layers = params.useGPU ? 99 : 0,
        .use_mmap = true,
        .use_mlock = false
    };
    
    llama_model* model = llama_load_model_from_file(path.c_str(), modelParams);
    if (!model) {
        LOGE("Failed to load model: %s", path.c_str());
        return 0;
    }
    
    llama_context_params ctxParams = {
        .n_ctx = static_cast<uint32_t>(params.contextSize),
        .n_batch = static_cast<uint32_t>(params.batchSize),
        .n_threads = static_cast<uint32_t>(params.threads),
        .embedding = true
    };
    
    llama_context* ctx = llama_new_context_with_model(model, ctxParams);
    if (!ctx) {
        LOGE("Failed to create context for model: %s", path.c_str());
        llama_free_model(model);
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    ModelHandle handle = m_impl->nextHandle++;
    m_impl->models[handle] = {model, ctx};
    
    LOGI("Model loaded successfully, handle: %lld", static_cast<long long>(handle));
    return handle;
}

void LlamaEngine::unloadModel(ModelHandle model) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->models.find(model);
    if (it != m_impl->models.end()) {
        if (it->second.second) llama_free(it->second.second);
        if (it->second.first) llama_free_model(it->second.first);
        m_impl->models.erase(it);
    }
}

std::string LlamaEngine::generate(ModelHandle model, const std::string& prompt, int maxTokens) {
    // Actual implementation would use llama.cpp generation API
    LOGI("LlamaEngine::generate: prompt=%s, maxTokens=%d", prompt.c_str(), maxTokens);
    
    // Placeholder - actual implementation uses llama_decode, llama_sample, etc.
    return "[Generated response for: " + prompt + "]";
}

std::vector<float> LlamaEngine::embed(ModelHandle model, const std::string& text) {
    LOGI("LlamaEngine::embed: %s", text.c_str());
    
    // Placeholder - actual implementation uses llama embedding mode
    std::vector<float> embedding(4096, 0.0f); // Typical embedding size
    
    // Simple hash-based placeholder embedding
    std::hash<std::string> hasher;
    size_t h = hasher(text);
    for (size_t i = 0; i < embedding.size(); i++) {
        embedding[i] = static_cast<float>((h >> (i % 64)) & 1) * 0.1f - 0.05f;
    }
    
    return embedding;
}

//=============================================================================
// OnnxEngine Implementation
//=============================================================================

struct OnnxEngine::Impl {
    // Ort::Env env; // Actual ONNX Runtime environment
    std::unordered_map<ModelHandle, void*> sessions; // Would be Ort::Session*
    ModelHandle nextHandle = 1;
    std::mutex mutex;
    OnnxParams params;
};

OnnxEngine::OnnxEngine() : m_impl(std::make_unique<Impl>()) {}
OnnxEngine::~OnnxEngine() { shutdown(); }

bool OnnxEngine::initialize(const OnnxParams& params) {
    LOGI("OnnxEngine::initialize with provider: %s", params.executionProvider.c_str());
    m_impl->params = params;
    return true;
}

void OnnxEngine::shutdown() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->sessions.clear();
}

ModelHandle OnnxEngine::loadModel(const std::string& name, const std::string& path) {
    LOGI("OnnxEngine::loadModel: %s from %s", name.c_str(), path.c_str());
    
    // Actual implementation would create Ort::Session
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    ModelHandle handle = m_impl->nextHandle++;
    m_impl->sessions[handle] = nullptr; // Placeholder
    
    return handle;
}

void OnnxEngine::unloadModel(ModelHandle model) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->sessions.erase(model);
}

std::vector<float> OnnxEngine::run(ModelHandle model, const std::vector<float>& input,
                                    const std::vector<int64_t>& inputShape) {
    LOGI("OnnxEngine::run with input size: %zu", input.size());
    
    // Placeholder - actual implementation uses Ort::Session::Run
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::tanh(input[i]); // Placeholder transformation
    }
    
    return output;
}

//=============================================================================
// SpeechEngine Implementation
//=============================================================================

struct SpeechEngine::Impl {
    std::string modelsPath;
    bool sttLoaded = false;
    bool ttsLoaded = false;
};

SpeechEngine::SpeechEngine() : m_impl(std::make_unique<Impl>()) {}
SpeechEngine::~SpeechEngine() { shutdown(); }

bool SpeechEngine::initialize(const std::string& modelsPath) {
    LOGI("SpeechEngine::initialize: %s", modelsPath.c_str());
    m_impl->modelsPath = modelsPath;
    return true;
}

void SpeechEngine::shutdown() {
    m_impl->sttLoaded = false;
    m_impl->ttsLoaded = false;
}

bool SpeechEngine::loadSTTModel(const std::string& modelPath) {
    LOGI("SpeechEngine::loadSTTModel: %s", modelPath.c_str());
    m_impl->sttLoaded = true;
    return true;
}

std::string SpeechEngine::transcribe(const std::string& audioPath) {
    LOGI("SpeechEngine::transcribe: %s", audioPath.c_str());
    // Actual implementation would use Sherpa ONNX
    return "[Transcribed audio from: " + audioPath + "]";
}

bool SpeechEngine::loadTTSModel(const std::string& modelPath) {
    LOGI("SpeechEngine::loadTTSModel: %s", modelPath.c_str());
    m_impl->ttsLoaded = true;
    return true;
}

std::string SpeechEngine::synthesize(const std::string& text, const std::string& outputPath) {
    LOGI("SpeechEngine::synthesize: %s -> %s", text.c_str(), outputPath.c_str());
    // Actual implementation would use Piper TTS
    return outputPath;
}

//=============================================================================
// StorageEngine Implementation
//=============================================================================

struct StorageEngine::Impl {
    std::unordered_map<std::string, std::string> store;
    std::unordered_map<std::string, std::vector<uint8_t>> binaryStore;
    std::mutex mutex;
    std::string storagePath;
};

StorageEngine::StorageEngine() : m_impl(std::make_unique<Impl>()) {}
StorageEngine::~StorageEngine() { shutdown(); }

bool StorageEngine::initialize(const std::string& storagePath) {
    LOGI("StorageEngine::initialize: %s", storagePath.c_str());
    m_impl->storagePath = storagePath;
    // Actual implementation would initialize MMKV
    return true;
}

void StorageEngine::shutdown() {
    sync();
}

void StorageEngine::set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->store[key] = value;
}

std::string StorageEngine::get(const std::string& key, const std::string& defaultValue) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->store.find(key);
    return (it != m_impl->store.end()) ? it->second : defaultValue;
}

bool StorageEngine::has(const std::string& key) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->store.find(key) != m_impl->store.end();
}

void StorageEngine::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->store.erase(key);
}

std::vector<std::string> StorageEngine::getAllKeys() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    std::vector<std::string> keys;
    keys.reserve(m_impl->store.size());
    for (const auto& [k, v] : m_impl->store) {
        keys.push_back(k);
    }
    return keys;
}

void StorageEngine::sync() {
    // Actual implementation would sync MMKV to disk
    LOGI("StorageEngine::sync");
}

void StorageEngine::clear() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->store.clear();
    m_impl->binaryStore.clear();
}

size_t StorageEngine::getSize() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->store.size();
}

//=============================================================================
// HardwareManager Implementation
//=============================================================================

HardwareManager& HardwareManager::getInstance() {
    static HardwareManager instance;
    return instance;
}

HardwareManager::HardwareManager() {
    detectCapabilities();
}

void HardwareManager::detectCapabilities() {
    LOGI("HardwareManager::detectCapabilities");
    
    // CPU detection
    m_capabilities.cpuCores = std::thread::hardware_concurrency();
    m_capabilities.cpuArch = "arm64";
    m_capabilities.hasNeon = true; // ARM64 always has NEON
    m_capabilities.hasSve = false; // Would need runtime detection
    
    // GPU detection (simplified)
    m_capabilities.hasGPU = true; // Assume GPU available on modern devices
    m_capabilities.gpuType = "adreno"; // Would need actual detection
    m_capabilities.openclVersion = "2.0";
    m_capabilities.vulkanVersion = "1.1";
    
    // NPU detection (simplified)
    m_capabilities.hasNPU = true; // Assume QNN available
    m_capabilities.npuType = "qnn";
    m_capabilities.htpVersion = "v73"; // Would need actual detection
    
    LOGI("Detected: %d cores, GPU=%s, NPU=%s", 
         m_capabilities.cpuCores,
         m_capabilities.hasGPU ? "yes" : "no",
         m_capabilities.hasNPU ? "yes" : "no");
}

HardwareCapabilities HardwareManager::getCapabilities() const {
    return m_capabilities;
}

std::string HardwareManager::selectOptimalBackend(const std::string& operation) const {
    // NPU optimal for: matmul, attention, embedding
    if (m_capabilities.hasNPU && 
        (operation == "matmul" || operation == "attention" || operation == "embedding")) {
        return "QNN";
    }
    
    // GPU optimal for: large tensor ops
    if (m_capabilities.hasGPU && 
        (operation == "matmul" || operation == "softmax" || operation == "layernorm")) {
        return "OpenCL";
    }
    
    return "CPU";
}

bool HardwareManager::supportsGPU() const {
    return m_capabilities.hasGPU;
}

bool HardwareManager::supportsNPU() const {
    return m_capabilities.hasNPU;
}

int HardwareManager::getOptimalThreadCount() const {
    return std::max(1, static_cast<int>(m_capabilities.cpuCores) - 1);
}

//=============================================================================
// MardukNativeCore Implementation
//=============================================================================

struct MardukNativeCore::Impl {
    TensorEngine tensorEngine;
    LlamaEngine llamaEngine;
    OnnxEngine onnxEngine;
    SpeechEngine speechEngine;
    StorageEngine storageEngine;
    std::string modelsPath;
};

MardukNativeCore::MardukNativeCore() 
    : m_impl(std::make_unique<Impl>()), m_initialized(false) {}

MardukNativeCore::~MardukNativeCore() {
    shutdown();
}

bool MardukNativeCore::initialize(const std::string& modelsPath, size_t memoryBudgetMB) {
    LOGI("MardukNativeCore::initialize: %s, %zu MB", modelsPath.c_str(), memoryBudgetMB);
    
    m_impl->modelsPath = modelsPath;
    
    // Initialize tensor engine
    size_t tensorMemory = memoryBudgetMB * 1024 * 1024 / 2; // Half for tensors
    if (!m_impl->tensorEngine.initialize(tensorMemory)) {
        LOGE("Failed to initialize TensorEngine");
        return false;
    }
    
    // Initialize LLM engine
    if (!m_impl->llamaEngine.initialize()) {
        LOGE("Failed to initialize LlamaEngine");
        return false;
    }
    
    // Initialize ONNX engine
    OnnxParams onnxParams;
    onnxParams.threads = HardwareManager::getInstance().getOptimalThreadCount();
    onnxParams.useGPU = HardwareManager::getInstance().supportsGPU();
    if (!m_impl->onnxEngine.initialize(onnxParams)) {
        LOGE("Failed to initialize OnnxEngine");
        return false;
    }
    
    // Initialize speech engine
    if (!m_impl->speechEngine.initialize(modelsPath)) {
        LOGE("Failed to initialize SpeechEngine");
        return false;
    }
    
    // Initialize storage engine
    std::string storagePath = modelsPath + "/storage";
    if (!m_impl->storageEngine.initialize(storagePath)) {
        LOGE("Failed to initialize StorageEngine");
        return false;
    }
    
    m_initialized = true;
    LOGI("MardukNativeCore initialized successfully");
    return true;
}

void MardukNativeCore::shutdown() {
    if (!m_initialized) return;
    
    LOGI("MardukNativeCore::shutdown");
    
    m_impl->storageEngine.shutdown();
    m_impl->speechEngine.shutdown();
    m_impl->onnxEngine.shutdown();
    m_impl->llamaEngine.shutdown();
    m_impl->tensorEngine.shutdown();
    
    m_initialized = false;
}

bool MardukNativeCore::isInitialized() const {
    return m_initialized;
}

TensorEngine& MardukNativeCore::getTensorEngine() { return m_impl->tensorEngine; }
LlamaEngine& MardukNativeCore::getLlamaEngine() { return m_impl->llamaEngine; }
OnnxEngine& MardukNativeCore::getOnnxEngine() { return m_impl->onnxEngine; }
SpeechEngine& MardukNativeCore::getSpeechEngine() { return m_impl->speechEngine; }
StorageEngine& MardukNativeCore::getStorageEngine() { return m_impl->storageEngine; }

HardwareCapabilities MardukNativeCore::getHardwareCapabilities() const {
    return HardwareManager::getInstance().getCapabilities();
}

} // namespace marduk

//=============================================================================
// JNI Implementation
//=============================================================================

extern "C" {

JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_create(
    JNIEnv* env, jobject thiz) {
    auto* core = new marduk::MardukNativeCore();
    return reinterpret_cast<jlong>(core);
}

JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_destroy(
    JNIEnv* env, jobject thiz, jlong handle) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    delete core;
}

JNIEXPORT jboolean JNICALL Java_com_marduk_commander_native_MardukNative_initialize(
    JNIEnv* env, jobject thiz, jlong handle, jstring modelsPath, jint memoryBudgetMB) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    const char* path = env->GetStringUTFChars(modelsPath, nullptr);
    bool result = core->initialize(path, static_cast<size_t>(memoryBudgetMB));
    env->ReleaseStringUTFChars(modelsPath, path);
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_shutdown(
    JNIEnv* env, jobject thiz, jlong handle) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    core->shutdown();
}

JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorCreate(
    JNIEnv* env, jobject thiz, jlong handle, jlongArray shape, jfloatArray data) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    
    jsize shapeLen = env->GetArrayLength(shape);
    jlong* shapeData = env->GetLongArrayElements(shape, nullptr);
    
    marduk::TensorShape tensorShape;
    tensorShape.dims.assign(shapeData, shapeData + shapeLen);
    env->ReleaseLongArrayElements(shape, shapeData, 0);
    
    marduk::TensorHandle result;
    if (data != nullptr) {
        jsize dataLen = env->GetArrayLength(data);
        jfloat* floatData = env->GetFloatArrayElements(data, nullptr);
        result = core->getTensorEngine().create(tensorShape, floatData);
        env->ReleaseFloatArrayElements(data, floatData, 0);
    } else {
        result = core->getTensorEngine().create(tensorShape);
    }
    
    return static_cast<jlong>(result);
}

JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorAdd(
    JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    return static_cast<jlong>(core->getTensorEngine().add(a, b));
}

JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_tensorMatmul(
    JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    return static_cast<jlong>(core->getTensorEngine().matmul(a, b));
}

JNIEXPORT jfloat JNICALL Java_com_marduk_commander_native_MardukNative_tensorCosineSimilarity(
    JNIEnv* env, jobject thiz, jlong handle, jlong a, jlong b) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    return core->getTensorEngine().cosineSimilarity(a, b);
}

JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_tensorFree(
    JNIEnv* env, jobject thiz, jlong handle, jlong tensor) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    core->getTensorEngine().free(tensor);
}

JNIEXPORT jlong JNICALL Java_com_marduk_commander_native_MardukNative_llamaLoadModel(
    JNIEnv* env, jobject thiz, jlong handle, jstring path, jint contextSize) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    const char* modelPath = env->GetStringUTFChars(path, nullptr);
    
    marduk::LlamaParams params;
    params.contextSize = contextSize;
    params.threads = marduk::HardwareManager::getInstance().getOptimalThreadCount();
    
    marduk::ModelHandle result = core->getLlamaEngine().loadModel(modelPath, params);
    env->ReleaseStringUTFChars(path, modelPath);
    
    return static_cast<jlong>(result);
}

JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_llamaGenerate(
    JNIEnv* env, jobject thiz, jlong handle, jlong model, jstring prompt, jint maxTokens) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    
    std::string result = core->getLlamaEngine().generate(model, promptStr, maxTokens);
    env->ReleaseStringUTFChars(prompt, promptStr);
    
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT jfloatArray JNICALL Java_com_marduk_commander_native_MardukNative_llamaEmbed(
    JNIEnv* env, jobject thiz, jlong handle, jlong model, jstring text) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    const char* textStr = env->GetStringUTFChars(text, nullptr);
    
    std::vector<float> embedding = core->getLlamaEngine().embed(model, textStr);
    env->ReleaseStringUTFChars(text, textStr);
    
    jfloatArray result = env->NewFloatArray(embedding.size());
    env->SetFloatArrayRegion(result, 0, embedding.size(), embedding.data());
    
    return result;
}

JNIEXPORT void JNICALL Java_com_marduk_commander_native_MardukNative_llamaUnloadModel(
    JNIEnv* env, jobject thiz, jlong handle, jlong model) {
    auto* core = reinterpret_cast<marduk::MardukNativeCore*>(handle);
    core->getLlamaEngine().unloadModel(model);
}

JNIEXPORT jstring JNICALL Java_com_marduk_commander_native_MardukNative_getHardwareInfo(
    JNIEnv* env, jobject thiz, jlong handle) {
    auto caps = marduk::HardwareManager::getInstance().getCapabilities();
    
    std::string info = "{"
        "\"cpuCores\":" + std::to_string(caps.cpuCores) + ","
        "\"cpuArch\":\"" + caps.cpuArch + "\","
        "\"hasNeon\":" + (caps.hasNeon ? "true" : "false") + ","
        "\"hasGPU\":" + (caps.hasGPU ? "true" : "false") + ","
        "\"gpuType\":\"" + caps.gpuType + "\","
        "\"hasNPU\":" + (caps.hasNPU ? "true" : "false") + ","
        "\"npuType\":\"" + caps.npuType + "\""
        "}";
    
    return env->NewStringUTF(info.c_str());
}

} // extern "C"
