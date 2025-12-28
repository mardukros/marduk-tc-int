/**
 * NativeBridge - TypeScript interface to MardukNativeCore
 * 
 * This module provides the bridge between the mad9ml TypeScript cognitive engine
 * and the native ARM64 libraries (ggml, llama, ONNX, etc.)
 */

import { NativeModules, Platform } from 'react-native';

// Type definitions
export type TensorHandle = number;
export type ModelHandle = number;

export interface TensorShape {
  dims: number[];
}

export interface TensorData {
  handle: TensorHandle;
  shape: TensorShape;
  data: Float32Array;
}

export interface LlamaParams {
  contextSize?: number;
  batchSize?: number;
  threads?: number;
  useGPU?: boolean;
  useNPU?: boolean;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export interface OnnxParams {
  threads?: number;
  useGPU?: boolean;
  executionProvider?: 'CPU' | 'CUDA' | 'QNN' | 'NNAPI';
}

export interface HardwareCapabilities {
  cpuCores: number;
  cpuArch: string;
  hasNeon: boolean;
  hasSve: boolean;
  hasGPU: boolean;
  gpuType: string;
  openclVersion: string;
  vulkanVersion: string;
  hasNPU: boolean;
  npuType: string;
  htpVersion: string;
}

export interface GenerationCallback {
  (token: string): boolean; // Return false to stop generation
}

// Native module interface
interface MardukNativeModule {
  // Lifecycle
  create(): number;
  destroy(handle: number): void;
  initialize(handle: number, modelsPath: string, memoryBudgetMB: number): boolean;
  shutdown(handle: number): void;
  
  // Tensor operations
  tensorCreate(handle: number, shape: number[], data?: number[]): number;
  tensorAdd(handle: number, a: number, b: number): number;
  tensorSub(handle: number, a: number, b: number): number;
  tensorMul(handle: number, a: number, b: number): number;
  tensorMatmul(handle: number, a: number, b: number): number;
  tensorScale(handle: number, t: number, factor: number): number;
  tensorSoftmax(handle: number, t: number, axis: number): number;
  tensorCosineSimilarity(handle: number, a: number, b: number): number;
  tensorGetData(handle: number, t: number): number[];
  tensorFree(handle: number, t: number): void;
  tensorFreeAll(handle: number): void;
  
  // LLM operations
  llamaLoadModel(handle: number, path: string, contextSize: number): number;
  llamaGenerate(handle: number, model: number, prompt: string, maxTokens: number): Promise<string>;
  llamaGenerateStream(handle: number, model: number, prompt: string, maxTokens: number): Promise<string>;
  llamaEmbed(handle: number, model: number, text: string): Promise<number[]>;
  llamaTokenize(handle: number, model: number, text: string): number[];
  llamaDetokenize(handle: number, model: number, tokens: number[]): string;
  llamaUnloadModel(handle: number, model: number): void;
  
  // ONNX operations
  onnxLoadModel(handle: number, name: string, path: string): number;
  onnxRun(handle: number, model: number, input: number[], shape: number[]): Promise<number[]>;
  onnxUnloadModel(handle: number, model: number): void;
  
  // Speech operations
  speechLoadSTT(handle: number, modelPath: string): boolean;
  speechTranscribe(handle: number, audioPath: string): Promise<string>;
  speechLoadTTS(handle: number, modelPath: string): boolean;
  speechSynthesize(handle: number, text: string, outputPath: string): Promise<string>;
  
  // Storage operations
  storageSet(handle: number, key: string, value: string): void;
  storageGet(handle: number, key: string): string | null;
  storageHas(handle: number, key: string): boolean;
  storageRemove(handle: number, key: string): void;
  storageGetAllKeys(handle: number): string[];
  storageSync(handle: number): void;
  storageClear(handle: number): void;
  
  // Hardware info
  getHardwareInfo(handle: number): string;
}

// Get native module
const MardukNative: MardukNativeModule = Platform.select({
  android: NativeModules.MardukNative,
  ios: NativeModules.MardukNative,
  default: null,
}) as MardukNativeModule;

/**
 * TensorEngine - High-level tensor operations using ggml
 */
export class TensorEngine {
  private handle: number;
  private nativeHandle: number;
  
  constructor(nativeHandle: number) {
    this.nativeHandle = nativeHandle;
    this.handle = 0;
  }
  
  /**
   * Create a new tensor with the given shape
   */
  create(shape: number[], data?: Float32Array): TensorHandle {
    const dataArray = data ? Array.from(data) : undefined;
    return MardukNative.tensorCreate(this.nativeHandle, shape, dataArray);
  }
  
  /**
   * Create a tensor filled with zeros
   */
  zeros(shape: number[]): TensorHandle {
    const size = shape.reduce((a, b) => a * b, 1);
    return this.create(shape, new Float32Array(size));
  }
  
  /**
   * Create a tensor filled with ones
   */
  ones(shape: number[]): TensorHandle {
    const size = shape.reduce((a, b) => a * b, 1);
    return this.create(shape, new Float32Array(size).fill(1));
  }
  
  /**
   * Create a tensor with random values
   */
  random(shape: number[], min = 0, max = 1): TensorHandle {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = min + Math.random() * (max - min);
    }
    return this.create(shape, data);
  }
  
  /**
   * Element-wise addition
   */
  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    return MardukNative.tensorAdd(this.nativeHandle, a, b);
  }
  
  /**
   * Element-wise subtraction
   */
  sub(a: TensorHandle, b: TensorHandle): TensorHandle {
    return MardukNative.tensorSub(this.nativeHandle, a, b);
  }
  
  /**
   * Element-wise multiplication
   */
  mul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return MardukNative.tensorMul(this.nativeHandle, a, b);
  }
  
  /**
   * Matrix multiplication
   */
  matmul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return MardukNative.tensorMatmul(this.nativeHandle, a, b);
  }
  
  /**
   * Scale tensor by a factor
   */
  scale(t: TensorHandle, factor: number): TensorHandle {
    return MardukNative.tensorScale(this.nativeHandle, t, factor);
  }
  
  /**
   * Softmax along axis
   */
  softmax(t: TensorHandle, axis = -1): TensorHandle {
    return MardukNative.tensorSoftmax(this.nativeHandle, t, axis);
  }
  
  /**
   * Compute cosine similarity between two tensors
   */
  cosineSimilarity(a: TensorHandle, b: TensorHandle): number {
    return MardukNative.tensorCosineSimilarity(this.nativeHandle, a, b);
  }
  
  /**
   * Get tensor data as Float32Array
   */
  getData(t: TensorHandle): Float32Array {
    const data = MardukNative.tensorGetData(this.nativeHandle, t);
    return new Float32Array(data);
  }
  
  /**
   * Free a tensor
   */
  free(t: TensorHandle): void {
    MardukNative.tensorFree(this.nativeHandle, t);
  }
  
  /**
   * Free all tensors
   */
  freeAll(): void {
    MardukNative.tensorFreeAll(this.nativeHandle);
  }
}

/**
 * LlamaEngine - LLM inference using llama.cpp
 */
export class LlamaEngine {
  private nativeHandle: number;
  private loadedModels: Map<string, ModelHandle> = new Map();
  
  constructor(nativeHandle: number) {
    this.nativeHandle = nativeHandle;
  }
  
  /**
   * Load a GGUF model
   */
  loadModel(name: string, path: string, params: LlamaParams = {}): ModelHandle {
    const contextSize = params.contextSize ?? 2048;
    const handle = MardukNative.llamaLoadModel(this.nativeHandle, path, contextSize);
    if (handle > 0) {
      this.loadedModels.set(name, handle);
    }
    return handle;
  }
  
  /**
   * Generate text from a prompt
   */
  async generate(modelOrName: ModelHandle | string, prompt: string, maxTokens = 256): Promise<string> {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    if (model === 0) {
      throw new Error(`Model not found: ${modelOrName}`);
    }
    
    return MardukNative.llamaGenerate(this.nativeHandle, model, prompt, maxTokens);
  }
  
  /**
   * Generate text with streaming callback
   */
  async generateStream(
    modelOrName: ModelHandle | string, 
    prompt: string, 
    maxTokens: number,
    onToken: GenerationCallback
  ): Promise<string> {
    // Note: Actual streaming would require native event emitter
    // This is a simplified version
    const result = await this.generate(modelOrName, prompt, maxTokens);
    
    // Simulate streaming by emitting tokens
    const tokens = result.split(' ');
    for (const token of tokens) {
      if (!onToken(token + ' ')) break;
    }
    
    return result;
  }
  
  /**
   * Get embeddings for text
   */
  async embed(modelOrName: ModelHandle | string, text: string): Promise<Float32Array> {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    if (model === 0) {
      throw new Error(`Model not found: ${modelOrName}`);
    }
    
    const embedding = await MardukNative.llamaEmbed(this.nativeHandle, model, text);
    return new Float32Array(embedding);
  }
  
  /**
   * Tokenize text
   */
  tokenize(modelOrName: ModelHandle | string, text: string): number[] {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    return MardukNative.llamaTokenize(this.nativeHandle, model, text);
  }
  
  /**
   * Detokenize tokens to text
   */
  detokenize(modelOrName: ModelHandle | string, tokens: number[]): string {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    return MardukNative.llamaDetokenize(this.nativeHandle, model, tokens);
  }
  
  /**
   * Unload a model
   */
  unloadModel(modelOrName: ModelHandle | string): void {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    if (model > 0) {
      MardukNative.llamaUnloadModel(this.nativeHandle, model);
      if (typeof modelOrName === 'string') {
        this.loadedModels.delete(modelOrName);
      }
    }
  }
  
  /**
   * Unload all models
   */
  unloadAll(): void {
    for (const [name, handle] of this.loadedModels) {
      MardukNative.llamaUnloadModel(this.nativeHandle, handle);
    }
    this.loadedModels.clear();
  }
}

/**
 * OnnxEngine - Neural network inference using ONNX Runtime
 */
export class OnnxEngine {
  private nativeHandle: number;
  private loadedModels: Map<string, ModelHandle> = new Map();
  
  constructor(nativeHandle: number) {
    this.nativeHandle = nativeHandle;
  }
  
  /**
   * Load an ONNX model
   */
  loadModel(name: string, path: string): ModelHandle {
    const handle = MardukNative.onnxLoadModel(this.nativeHandle, name, path);
    if (handle > 0) {
      this.loadedModels.set(name, handle);
    }
    return handle;
  }
  
  /**
   * Run inference on a model
   */
  async run(
    modelOrName: ModelHandle | string, 
    input: Float32Array, 
    shape: number[]
  ): Promise<Float32Array> {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    if (model === 0) {
      throw new Error(`Model not found: ${modelOrName}`);
    }
    
    const output = await MardukNative.onnxRun(
      this.nativeHandle, 
      model, 
      Array.from(input), 
      shape
    );
    
    return new Float32Array(output);
  }
  
  /**
   * Unload a model
   */
  unloadModel(modelOrName: ModelHandle | string): void {
    const model = typeof modelOrName === 'string' 
      ? this.loadedModels.get(modelOrName) ?? 0 
      : modelOrName;
    
    if (model > 0) {
      MardukNative.onnxUnloadModel(this.nativeHandle, model);
      if (typeof modelOrName === 'string') {
        this.loadedModels.delete(modelOrName);
      }
    }
  }
}

/**
 * SpeechEngine - Speech-to-text and text-to-speech
 */
export class SpeechEngine {
  private nativeHandle: number;
  private sttLoaded = false;
  private ttsLoaded = false;
  
  constructor(nativeHandle: number) {
    this.nativeHandle = nativeHandle;
  }
  
  /**
   * Load speech-to-text model
   */
  loadSTT(modelPath: string): boolean {
    this.sttLoaded = MardukNative.speechLoadSTT(this.nativeHandle, modelPath);
    return this.sttLoaded;
  }
  
  /**
   * Transcribe audio file to text
   */
  async transcribe(audioPath: string): Promise<string> {
    if (!this.sttLoaded) {
      throw new Error('STT model not loaded');
    }
    return MardukNative.speechTranscribe(this.nativeHandle, audioPath);
  }
  
  /**
   * Load text-to-speech model
   */
  loadTTS(modelPath: string): boolean {
    this.ttsLoaded = MardukNative.speechLoadTTS(this.nativeHandle, modelPath);
    return this.ttsLoaded;
  }
  
  /**
   * Synthesize text to audio file
   */
  async synthesize(text: string, outputPath: string): Promise<string> {
    if (!this.ttsLoaded) {
      throw new Error('TTS model not loaded');
    }
    return MardukNative.speechSynthesize(this.nativeHandle, text, outputPath);
  }
}

/**
 * StorageEngine - Persistent key-value storage using MMKV
 */
export class StorageEngine {
  private nativeHandle: number;
  
  constructor(nativeHandle: number) {
    this.nativeHandle = nativeHandle;
  }
  
  /**
   * Set a string value
   */
  set(key: string, value: string): void {
    MardukNative.storageSet(this.nativeHandle, key, value);
  }
  
  /**
   * Get a string value
   */
  get(key: string): string | null {
    return MardukNative.storageGet(this.nativeHandle, key);
  }
  
  /**
   * Check if key exists
   */
  has(key: string): boolean {
    return MardukNative.storageHas(this.nativeHandle, key);
  }
  
  /**
   * Remove a key
   */
  remove(key: string): void {
    MardukNative.storageRemove(this.nativeHandle, key);
  }
  
  /**
   * Get all keys
   */
  getAllKeys(): string[] {
    return MardukNative.storageGetAllKeys(this.nativeHandle);
  }
  
  /**
   * Set a JSON object
   */
  setObject<T>(key: string, value: T): void {
    this.set(key, JSON.stringify(value));
  }
  
  /**
   * Get a JSON object
   */
  getObject<T>(key: string): T | null {
    const value = this.get(key);
    if (value === null) return null;
    try {
      return JSON.parse(value) as T;
    } catch {
      return null;
    }
  }
  
  /**
   * Sync to disk
   */
  sync(): void {
    MardukNative.storageSync(this.nativeHandle);
  }
  
  /**
   * Clear all data
   */
  clear(): void {
    MardukNative.storageClear(this.nativeHandle);
  }
}

/**
 * NativeBridge - Main interface to native functionality
 */
export class NativeBridge {
  private nativeHandle: number = 0;
  private initialized = false;
  
  public tensor: TensorEngine;
  public llama: LlamaEngine;
  public onnx: OnnxEngine;
  public speech: SpeechEngine;
  public storage: StorageEngine;
  
  constructor() {
    // Create placeholder engines (will be initialized later)
    this.tensor = null as any;
    this.llama = null as any;
    this.onnx = null as any;
    this.speech = null as any;
    this.storage = null as any;
  }
  
  /**
   * Initialize the native bridge
   */
  async initialize(modelsPath: string, memoryBudgetMB = 512): Promise<boolean> {
    if (this.initialized) {
      return true;
    }
    
    if (!MardukNative) {
      console.warn('MardukNative module not available, using mock implementation');
      return this.initializeMock();
    }
    
    // Create native core
    this.nativeHandle = MardukNative.create();
    if (this.nativeHandle === 0) {
      throw new Error('Failed to create native core');
    }
    
    // Initialize native core
    const success = MardukNative.initialize(this.nativeHandle, modelsPath, memoryBudgetMB);
    if (!success) {
      MardukNative.destroy(this.nativeHandle);
      throw new Error('Failed to initialize native core');
    }
    
    // Create engine instances
    this.tensor = new TensorEngine(this.nativeHandle);
    this.llama = new LlamaEngine(this.nativeHandle);
    this.onnx = new OnnxEngine(this.nativeHandle);
    this.speech = new SpeechEngine(this.nativeHandle);
    this.storage = new StorageEngine(this.nativeHandle);
    
    this.initialized = true;
    return true;
  }
  
  /**
   * Initialize with mock implementation for testing
   */
  private initializeMock(): boolean {
    console.log('Initializing NativeBridge with mock implementation');
    
    // Create mock engines
    this.tensor = new MockTensorEngine() as any;
    this.llama = new MockLlamaEngine() as any;
    this.onnx = new MockOnnxEngine() as any;
    this.speech = new MockSpeechEngine() as any;
    this.storage = new MockStorageEngine() as any;
    
    this.initialized = true;
    return true;
  }
  
  /**
   * Shutdown the native bridge
   */
  shutdown(): void {
    if (!this.initialized) return;
    
    if (MardukNative && this.nativeHandle !== 0) {
      MardukNative.shutdown(this.nativeHandle);
      MardukNative.destroy(this.nativeHandle);
      this.nativeHandle = 0;
    }
    
    this.initialized = false;
  }
  
  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Get hardware capabilities
   */
  getHardwareCapabilities(): HardwareCapabilities {
    if (!MardukNative || this.nativeHandle === 0) {
      return {
        cpuCores: 4,
        cpuArch: 'unknown',
        hasNeon: false,
        hasSve: false,
        hasGPU: false,
        gpuType: 'unknown',
        openclVersion: '',
        vulkanVersion: '',
        hasNPU: false,
        npuType: 'none',
        htpVersion: '',
      };
    }
    
    const infoJson = MardukNative.getHardwareInfo(this.nativeHandle);
    return JSON.parse(infoJson) as HardwareCapabilities;
  }
}

// Mock implementations for testing without native module
class MockTensorEngine {
  private tensors: Map<number, Float32Array> = new Map();
  private nextHandle = 1;
  
  create(shape: number[], data?: Float32Array): number {
    const handle = this.nextHandle++;
    const size = shape.reduce((a, b) => a * b, 1);
    this.tensors.set(handle, data ?? new Float32Array(size));
    return handle;
  }
  
  zeros(shape: number[]): number { return this.create(shape); }
  ones(shape: number[]): number { 
    const size = shape.reduce((a, b) => a * b, 1);
    return this.create(shape, new Float32Array(size).fill(1)); 
  }
  
  add(a: number, b: number): number {
    const ta = this.tensors.get(a);
    const tb = this.tensors.get(b);
    if (!ta || !tb) return 0;
    const result = new Float32Array(ta.length);
    for (let i = 0; i < ta.length; i++) result[i] = ta[i] + tb[i];
    return this.create([ta.length], result);
  }
  
  cosineSimilarity(a: number, b: number): number {
    const ta = this.tensors.get(a);
    const tb = this.tensors.get(b);
    if (!ta || !tb) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < ta.length; i++) {
      dot += ta[i] * tb[i];
      normA += ta[i] * ta[i];
      normB += tb[i] * tb[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
  
  getData(t: number): Float32Array { return this.tensors.get(t) ?? new Float32Array(); }
  free(t: number): void { this.tensors.delete(t); }
  freeAll(): void { this.tensors.clear(); }
}

class MockLlamaEngine {
  async generate(model: any, prompt: string, maxTokens: number): Promise<string> {
    return `[Mock response for: ${prompt.substring(0, 50)}...]`;
  }
  
  async embed(model: any, text: string): Promise<Float32Array> {
    const embedding = new Float32Array(4096);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] = Math.random() * 0.1 - 0.05;
    }
    return embedding;
  }
  
  loadModel(name: string, path: string): number { return 1; }
  unloadModel(model: any): void {}
  unloadAll(): void {}
}

class MockOnnxEngine {
  async run(model: any, input: Float32Array, shape: number[]): Promise<Float32Array> {
    return new Float32Array(input.length).map(x => Math.tanh(x));
  }
  
  loadModel(name: string, path: string): number { return 1; }
  unloadModel(model: any): void {}
}

class MockSpeechEngine {
  loadSTT(path: string): boolean { return true; }
  loadTTS(path: string): boolean { return true; }
  async transcribe(audioPath: string): Promise<string> { return '[Mock transcription]'; }
  async synthesize(text: string, outputPath: string): Promise<string> { return outputPath; }
}

class MockStorageEngine {
  private store: Map<string, string> = new Map();
  
  set(key: string, value: string): void { this.store.set(key, value); }
  get(key: string): string | null { return this.store.get(key) ?? null; }
  has(key: string): boolean { return this.store.has(key); }
  remove(key: string): void { this.store.delete(key); }
  getAllKeys(): string[] { return Array.from(this.store.keys()); }
  setObject<T>(key: string, value: T): void { this.set(key, JSON.stringify(value)); }
  getObject<T>(key: string): T | null {
    const v = this.get(key);
    return v ? JSON.parse(v) : null;
  }
  sync(): void {}
  clear(): void { this.store.clear(); }
}

// Export singleton instance
export const nativeBridge = new NativeBridge();
export default nativeBridge;
