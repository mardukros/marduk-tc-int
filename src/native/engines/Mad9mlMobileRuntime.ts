/**
 * Mad9mlMobileRuntime - Mobile adapter for the mad9ml cognitive engine
 * 
 * This module adapts the mad9ml TypeScript cognitive architecture to run
 * on mobile devices using native ARM64 libraries for tensor operations,
 * LLM inference, and neural network execution.
 */

import { nativeBridge, TensorHandle, ModelHandle, HardwareCapabilities } from '../bridge/NativeBridge';

//=============================================================================
// Type Definitions
//=============================================================================

export interface Mad9mlConfig {
  // Model paths
  modelsPath: string;
  llamaModelPath?: string;
  intentClassifierPath?: string;
  entityExtractorPath?: string;
  attentionModelPath?: string;
  sttModelPath?: string;
  ttsModelPath?: string;
  
  // Hardware settings
  memoryBudgetMB?: number;
  useGPU?: boolean;
  useNPU?: boolean;
  maxThreads?: number;
  
  // Cognitive settings
  attentionBudget?: number;
  evolutionEnabled?: boolean;
  metaCognitionDepth?: number;
  contextSize?: number;
}

export interface CognitiveState {
  timestamp: number;
  cycleCount: number;
  
  // Attention state
  attention: {
    sti: Map<string, number>;  // Short-term importance
    lti: Map<string, number>;  // Long-term importance
    vlti: Set<string>;         // Very long-term importance
  };
  
  // Working memory
  workingMemory: {
    focus: string[];
    context: Map<string, any>;
    goals: string[];
  };
  
  // Persona state
  persona: {
    traits: Float32Array;
    grammar: any;
    fitness: number;
  };
}

export interface ExecutionPlan {
  intent: string;
  confidence: number;
  entities: Map<string, string>;
  operations: FileOperation[];
  reasoning: string;
}

export interface FileOperation {
  type: 'copy' | 'move' | 'delete' | 'create' | 'search' | 'organize' | 'archive';
  source?: string;
  destination?: string;
  pattern?: string;
  recursive?: boolean;
  options?: Record<string, any>;
}

export interface MetaCognitiveInsight {
  type: 'performance' | 'pattern' | 'suggestion' | 'warning';
  content: string;
  confidence: number;
  actionable: boolean;
  suggestedAction?: string;
}

export interface MemoryEntry {
  id: string;
  type: 'declarative' | 'episodic' | 'procedural' | 'semantic';
  content: any;
  embedding?: Float32Array;
  attention: {
    sti: number;
    lti: number;
  };
  timestamp: number;
  accessCount: number;
}

//=============================================================================
// ECAN Attention Allocator
//=============================================================================

class ECANAllocator {
  private stiMap: Map<string, number> = new Map();
  private ltiMap: Map<string, number> = new Map();
  private vltiSet: Set<string> = new Set();
  
  private readonly stiDecayRate = 0.1;
  private readonly ltiDecayRate = 0.01;
  private readonly attentionBudget: number;
  
  constructor(attentionBudget = 100) {
    this.attentionBudget = attentionBudget;
  }
  
  /**
   * Stimulate an atom (increase its STI)
   */
  stimulate(atomId: string, amount: number): void {
    const currentSti = this.stiMap.get(atomId) ?? 0;
    this.stiMap.set(atomId, Math.min(100, currentSti + amount));
    
    // Spread activation to LTI
    const currentLti = this.ltiMap.get(atomId) ?? 0;
    this.ltiMap.set(atomId, Math.min(100, currentLti + amount * 0.1));
  }
  
  /**
   * Decay all attention values
   */
  decay(): void {
    for (const [id, sti] of this.stiMap) {
      const newSti = sti * (1 - this.stiDecayRate);
      if (newSti < 0.01) {
        this.stiMap.delete(id);
      } else {
        this.stiMap.set(id, newSti);
      }
    }
    
    for (const [id, lti] of this.ltiMap) {
      if (!this.vltiSet.has(id)) {
        const newLti = lti * (1 - this.ltiDecayRate);
        if (newLti < 0.01) {
          this.ltiMap.delete(id);
        } else {
          this.ltiMap.set(id, newLti);
        }
      }
    }
  }
  
  /**
   * Get top attention atoms
   */
  getTopAtoms(n: number): string[] {
    const sorted = [...this.stiMap.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n);
    return sorted.map(([id]) => id);
  }
  
  /**
   * Mark an atom as very long-term important
   */
  markVLTI(atomId: string): void {
    this.vltiSet.add(atomId);
    this.ltiMap.set(atomId, 100);
  }
  
  /**
   * Get attention state
   */
  getState(): CognitiveState['attention'] {
    return {
      sti: new Map(this.stiMap),
      lti: new Map(this.ltiMap),
      vlti: new Set(this.vltiSet),
    };
  }
}

//=============================================================================
// Hypergraph Memory
//=============================================================================

class HypergraphMemory {
  private nodes: Map<string, MemoryEntry> = new Map();
  private edges: Map<string, { source: string; target: string; type: string; weight: number }> = new Map();
  private embeddings: Map<string, TensorHandle> = new Map();
  
  constructor(private storage: typeof nativeBridge.storage) {}
  
  /**
   * Add a memory entry
   */
  async add(entry: MemoryEntry): Promise<void> {
    this.nodes.set(entry.id, entry);
    
    // Store in persistent storage
    this.storage.setObject(`memory:${entry.type}:${entry.id}`, entry);
    
    // Store embedding if present
    if (entry.embedding) {
      const handle = nativeBridge.tensor.create([entry.embedding.length], entry.embedding);
      this.embeddings.set(entry.id, handle);
    }
  }
  
  /**
   * Get a memory entry by ID
   */
  get(id: string): MemoryEntry | undefined {
    return this.nodes.get(id);
  }
  
  /**
   * Find similar memories by embedding
   */
  async findSimilar(embedding: Float32Array, topK = 5): Promise<MemoryEntry[]> {
    const queryHandle = nativeBridge.tensor.create([embedding.length], embedding);
    
    const similarities: { entry: MemoryEntry; score: number }[] = [];
    
    for (const [id, handle] of this.embeddings) {
      const score = nativeBridge.tensor.cosineSimilarity(queryHandle, handle);
      const entry = this.nodes.get(id);
      if (entry) {
        similarities.push({ entry, score });
      }
    }
    
    nativeBridge.tensor.free(queryHandle);
    
    return similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(s => s.entry);
  }
  
  /**
   * Add an edge between nodes
   */
  addEdge(id: string, source: string, target: string, type: string, weight = 1.0): void {
    this.edges.set(id, { source, target, type, weight });
  }
  
  /**
   * Get connected nodes
   */
  getConnected(nodeId: string): string[] {
    const connected: string[] = [];
    for (const [_, edge] of this.edges) {
      if (edge.source === nodeId) connected.push(edge.target);
      if (edge.target === nodeId) connected.push(edge.source);
    }
    return connected;
  }
  
  /**
   * Load from persistent storage
   */
  async load(): Promise<void> {
    const keys = this.storage.getAllKeys();
    for (const key of keys) {
      if (key.startsWith('memory:')) {
        const entry = this.storage.getObject<MemoryEntry>(key);
        if (entry) {
          this.nodes.set(entry.id, entry);
        }
      }
    }
  }
  
  /**
   * Save to persistent storage
   */
  async save(): Promise<void> {
    for (const [id, entry] of this.nodes) {
      this.storage.setObject(`memory:${entry.type}:${id}`, entry);
    }
    this.storage.sync();
  }
}

//=============================================================================
// Persona Evolution Engine
//=============================================================================

class PersonaEvolution {
  private currentTraits: TensorHandle = 0;
  private grammarRules: Map<string, any> = new Map();
  private fitness = 0.5;
  private evolutionHistory: { traits: Float32Array; fitness: number; timestamp: number }[] = [];
  
  constructor(private llamaEngine: typeof nativeBridge.llama) {}
  
  /**
   * Initialize persona from model
   */
  async initialize(modelPath: string): Promise<void> {
    // Load the persona model
    await this.llamaEngine.loadModel('persona', modelPath, {
      contextSize: 2048,
      useGPU: true,
    });
    
    // Initialize trait tensor
    this.currentTraits = nativeBridge.tensor.random([128], -1, 1);
  }
  
  /**
   * Generate response using persona
   */
  async generate(prompt: string, maxTokens = 256): Promise<string> {
    // Construct persona-aware prompt
    const personaPrompt = this.buildPersonaPrompt(prompt);
    return this.llamaEngine.generate('persona', personaPrompt, maxTokens);
  }
  
  /**
   * Build persona-aware prompt
   */
  private buildPersonaPrompt(userPrompt: string): string {
    const traitDescription = this.describeTraits();
    return `You are Marduk, an intelligent file management assistant.
${traitDescription}

User request: ${userPrompt}

Respond helpfully and concisely:`;
  }
  
  /**
   * Describe current traits
   */
  private describeTraits(): string {
    const traits = nativeBridge.tensor.getData(this.currentTraits);
    
    // Map trait dimensions to descriptions
    const descriptions: string[] = [];
    
    if (traits[0] > 0.5) descriptions.push('You are highly analytical.');
    if (traits[1] > 0.5) descriptions.push('You prefer concise responses.');
    if (traits[2] > 0.5) descriptions.push('You are proactive in suggestions.');
    if (traits[3] > 0.5) descriptions.push('You explain your reasoning.');
    
    return descriptions.join(' ');
  }
  
  /**
   * Update fitness based on outcome
   */
  updateFitness(success: boolean, userFeedback?: number): void {
    const delta = success ? 0.1 : -0.1;
    const feedbackDelta = userFeedback !== undefined ? (userFeedback - 0.5) * 0.2 : 0;
    
    this.fitness = Math.max(0, Math.min(1, this.fitness + delta + feedbackDelta));
  }
  
  /**
   * Evolve persona traits (MOSES-inspired)
   */
  async evolve(): Promise<void> {
    // Save current state to history
    const currentTraitData = nativeBridge.tensor.getData(this.currentTraits);
    this.evolutionHistory.push({
      traits: currentTraitData,
      fitness: this.fitness,
      timestamp: Date.now(),
    });
    
    // Mutate traits based on fitness
    if (this.fitness < 0.5) {
      // Low fitness: larger mutations
      const mutation = nativeBridge.tensor.random([128], -0.2, 0.2);
      const newTraits = nativeBridge.tensor.add(this.currentTraits, mutation);
      nativeBridge.tensor.free(this.currentTraits);
      nativeBridge.tensor.free(mutation);
      this.currentTraits = newTraits;
    } else {
      // High fitness: smaller mutations
      const mutation = nativeBridge.tensor.random([128], -0.05, 0.05);
      const newTraits = nativeBridge.tensor.add(this.currentTraits, mutation);
      nativeBridge.tensor.free(this.currentTraits);
      nativeBridge.tensor.free(mutation);
      this.currentTraits = newTraits;
    }
    
    // Reset fitness for new evaluation period
    this.fitness = 0.5;
  }
  
  /**
   * Get current persona state
   */
  getState(): CognitiveState['persona'] {
    return {
      traits: nativeBridge.tensor.getData(this.currentTraits),
      grammar: Object.fromEntries(this.grammarRules),
      fitness: this.fitness,
    };
  }
}

//=============================================================================
// Intent Classifier
//=============================================================================

class IntentClassifier {
  private modelHandle: ModelHandle = 0;
  private intentLabels = [
    'copy', 'move', 'delete', 'create', 'search', 
    'organize', 'archive', 'compress', 'extract', 'rename',
    'info', 'help', 'cancel', 'undo', 'unknown'
  ];
  
  constructor(private onnxEngine: typeof nativeBridge.onnx) {}
  
  /**
   * Load the intent classifier model
   */
  async load(modelPath: string): Promise<void> {
    this.modelHandle = this.onnxEngine.loadModel('intent', modelPath);
  }
  
  /**
   * Classify intent from text
   */
  async classify(text: string, embedding: Float32Array): Promise<{ intent: string; confidence: number; alternatives: { intent: string; confidence: number }[] }> {
    if (this.modelHandle === 0) {
      // Fallback to rule-based classification
      return this.classifyRuleBased(text);
    }
    
    // Run ONNX model
    const output = await this.onnxEngine.run(
      this.modelHandle,
      embedding,
      [1, embedding.length]
    );
    
    // Apply softmax and get top predictions
    const probs = this.softmax(output);
    const sorted = probs
      .map((p, i) => ({ intent: this.intentLabels[i], confidence: p }))
      .sort((a, b) => b.confidence - a.confidence);
    
    return {
      intent: sorted[0].intent,
      confidence: sorted[0].confidence,
      alternatives: sorted.slice(1, 4),
    };
  }
  
  /**
   * Rule-based fallback classification
   */
  private classifyRuleBased(text: string): { intent: string; confidence: number; alternatives: { intent: string; confidence: number }[] } {
    const lower = text.toLowerCase();
    
    const patterns: { pattern: RegExp; intent: string }[] = [
      { pattern: /\b(copy|duplicate)\b/, intent: 'copy' },
      { pattern: /\b(move|transfer)\b/, intent: 'move' },
      { pattern: /\b(delete|remove|trash)\b/, intent: 'delete' },
      { pattern: /\b(create|new|make)\b/, intent: 'create' },
      { pattern: /\b(find|search|look for|locate)\b/, intent: 'search' },
      { pattern: /\b(organize|sort|arrange)\b/, intent: 'organize' },
      { pattern: /\b(archive|backup)\b/, intent: 'archive' },
      { pattern: /\b(compress|zip)\b/, intent: 'compress' },
      { pattern: /\b(extract|unzip)\b/, intent: 'extract' },
      { pattern: /\b(rename)\b/, intent: 'rename' },
      { pattern: /\b(info|details|properties)\b/, intent: 'info' },
      { pattern: /\b(help)\b/, intent: 'help' },
      { pattern: /\b(cancel|stop|abort)\b/, intent: 'cancel' },
      { pattern: /\b(undo)\b/, intent: 'undo' },
    ];
    
    for (const { pattern, intent } of patterns) {
      if (pattern.test(lower)) {
        return {
          intent,
          confidence: 0.85,
          alternatives: [],
        };
      }
    }
    
    return {
      intent: 'unknown',
      confidence: 0.5,
      alternatives: [],
    };
  }
  
  /**
   * Softmax function
   */
  private softmax(logits: Float32Array): Float32Array {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return new Float32Array(exps.map(x => x / sum));
  }
}

//=============================================================================
// Entity Extractor
//=============================================================================

class EntityExtractor {
  private pathAliases: Map<string, string> = new Map([
    ['downloads', '/sdcard/Download'],
    ['documents', '/sdcard/Documents'],
    ['pictures', '/sdcard/Pictures'],
    ['photos', '/sdcard/DCIM'],
    ['music', '/sdcard/Music'],
    ['videos', '/sdcard/Movies'],
    ['desktop', '/sdcard'],
    ['home', '/sdcard'],
  ]);
  
  /**
   * Extract entities from text
   */
  extract(text: string): Map<string, string> {
    const entities = new Map<string, string>();
    
    // Extract paths
    const pathMatch = text.match(/(?:from|in|to|at)\s+([\/\w\-\.]+)/gi);
    if (pathMatch) {
      for (const match of pathMatch) {
        const path = match.replace(/^(from|in|to|at)\s+/i, '');
        const resolved = this.resolvePath(path);
        
        if (match.toLowerCase().startsWith('from') || match.toLowerCase().startsWith('in')) {
          entities.set('source', resolved);
        } else if (match.toLowerCase().startsWith('to')) {
          entities.set('destination', resolved);
        }
      }
    }
    
    // Extract file patterns
    const patternMatch = text.match(/\*\.\w+|\b\w+\s+files?\b/gi);
    if (patternMatch) {
      entities.set('pattern', patternMatch[0]);
    }
    
    // Extract file types
    const typePatterns: { pattern: RegExp; type: string }[] = [
      { pattern: /\bpdf(s)?\b/i, type: '*.pdf' },
      { pattern: /\bimage(s)?\b|\bphoto(s)?\b|\bpicture(s)?\b/i, type: '*.jpg,*.png,*.gif' },
      { pattern: /\bdocument(s)?\b/i, type: '*.doc,*.docx,*.pdf,*.txt' },
      { pattern: /\bvideo(s)?\b|\bmovie(s)?\b/i, type: '*.mp4,*.mkv,*.avi' },
      { pattern: /\bmusic\b|\baudio\b|\bsong(s)?\b/i, type: '*.mp3,*.wav,*.flac' },
    ];
    
    for (const { pattern, type } of typePatterns) {
      if (pattern.test(text)) {
        entities.set('fileType', type);
        break;
      }
    }
    
    // Extract size constraints
    const sizeMatch = text.match(/\b(large|big|huge|small|tiny)\b|\b(\d+)\s*(mb|gb|kb)\b/i);
    if (sizeMatch) {
      entities.set('sizeConstraint', sizeMatch[0]);
    }
    
    // Extract time constraints
    const timeMatch = text.match(/\b(today|yesterday|last\s+week|last\s+month|old|recent|new)\b/i);
    if (timeMatch) {
      entities.set('timeConstraint', timeMatch[0]);
    }
    
    return entities;
  }
  
  /**
   * Resolve path aliases
   */
  private resolvePath(path: string): string {
    const lower = path.toLowerCase();
    
    for (const [alias, resolved] of this.pathAliases) {
      if (lower === alias || lower.startsWith(alias + '/')) {
        return path.replace(new RegExp(`^${alias}`, 'i'), resolved);
      }
    }
    
    // If it doesn't start with /, assume it's relative to /sdcard
    if (!path.startsWith('/')) {
      return `/sdcard/${path}`;
    }
    
    return path;
  }
}

//=============================================================================
// Meta-Cognitive Engine
//=============================================================================

class MetaCognitiveEngine {
  private reflectionHistory: MetaCognitiveInsight[] = [];
  private performanceMetrics: {
    successRate: number;
    avgLatency: number;
    commandCount: number;
  } = { successRate: 1.0, avgLatency: 0, commandCount: 0 };
  
  constructor(private llamaEngine: typeof nativeBridge.llama) {}
  
  /**
   * Perform meta-cognitive reflection
   */
  async reflect(state: CognitiveState, recentOutcomes: { success: boolean; latency: number }[]): Promise<MetaCognitiveInsight[]> {
    const insights: MetaCognitiveInsight[] = [];
    
    // Update performance metrics
    this.updateMetrics(recentOutcomes);
    
    // Performance analysis
    if (this.performanceMetrics.successRate < 0.7) {
      insights.push({
        type: 'warning',
        content: 'Success rate has dropped below 70%. Consider reviewing recent failures.',
        confidence: 0.9,
        actionable: true,
        suggestedAction: 'Review failed commands and adjust intent classification.',
      });
    }
    
    if (this.performanceMetrics.avgLatency > 2000) {
      insights.push({
        type: 'performance',
        content: 'Average response time exceeds 2 seconds. Consider optimizing.',
        confidence: 0.85,
        actionable: true,
        suggestedAction: 'Enable GPU acceleration or reduce context size.',
      });
    }
    
    // Pattern analysis using LLM
    if (this.reflectionHistory.length > 0 && this.reflectionHistory.length % 10 === 0) {
      const llmInsight = await this.generateLLMInsight(state);
      if (llmInsight) {
        insights.push(llmInsight);
      }
    }
    
    // Store insights
    this.reflectionHistory.push(...insights);
    
    return insights;
  }
  
  /**
   * Update performance metrics
   */
  private updateMetrics(outcomes: { success: boolean; latency: number }[]): void {
    if (outcomes.length === 0) return;
    
    const successCount = outcomes.filter(o => o.success).length;
    const totalLatency = outcomes.reduce((sum, o) => sum + o.latency, 0);
    
    // Exponential moving average
    const alpha = 0.3;
    this.performanceMetrics.successRate = 
      alpha * (successCount / outcomes.length) + 
      (1 - alpha) * this.performanceMetrics.successRate;
    
    this.performanceMetrics.avgLatency = 
      alpha * (totalLatency / outcomes.length) + 
      (1 - alpha) * this.performanceMetrics.avgLatency;
    
    this.performanceMetrics.commandCount += outcomes.length;
  }
  
  /**
   * Generate insight using LLM
   */
  private async generateLLMInsight(state: CognitiveState): Promise<MetaCognitiveInsight | null> {
    const prompt = `Analyze the following cognitive state and provide one actionable insight:

Cycle count: ${state.cycleCount}
Active focus items: ${state.workingMemory.focus.length}
Current goals: ${state.workingMemory.goals.join(', ')}
Persona fitness: ${state.persona.fitness.toFixed(2)}

Provide a brief, actionable insight:`;

    try {
      const response = await this.llamaEngine.generate('persona', prompt, 100);
      
      return {
        type: 'suggestion',
        content: response.trim(),
        confidence: 0.7,
        actionable: true,
      };
    } catch {
      return null;
    }
  }
  
  /**
   * Get performance summary
   */
  getPerformanceSummary(): typeof this.performanceMetrics {
    return { ...this.performanceMetrics };
  }
}

//=============================================================================
// Mad9ml Mobile Runtime
//=============================================================================

export class Mad9mlMobileRuntime {
  private config: Mad9mlConfig;
  private initialized = false;
  private cycleCount = 0;
  
  // Cognitive components
  private attention: ECANAllocator;
  private memory: HypergraphMemory;
  private persona: PersonaEvolution;
  private intentClassifier: IntentClassifier;
  private entityExtractor: EntityExtractor;
  private metaCognition: MetaCognitiveEngine;
  
  // Working memory
  private workingMemory: CognitiveState['workingMemory'] = {
    focus: [],
    context: new Map(),
    goals: [],
  };
  
  // Recent outcomes for reflection
  private recentOutcomes: { success: boolean; latency: number }[] = [];
  
  constructor(config: Mad9mlConfig) {
    this.config = {
      memoryBudgetMB: 512,
      useGPU: true,
      useNPU: true,
      maxThreads: 4,
      attentionBudget: 100,
      evolutionEnabled: true,
      metaCognitionDepth: 3,
      contextSize: 2048,
      ...config,
    };
    
    // Initialize components (will be fully initialized in initialize())
    this.attention = new ECANAllocator(this.config.attentionBudget);
    this.memory = null as any;
    this.persona = null as any;
    this.intentClassifier = null as any;
    this.entityExtractor = new EntityExtractor();
    this.metaCognition = null as any;
  }
  
  /**
   * Initialize the runtime
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    console.log('Initializing Mad9ml Mobile Runtime...');
    
    // Initialize native bridge
    await nativeBridge.initialize(
      this.config.modelsPath,
      this.config.memoryBudgetMB
    );
    
    // Initialize cognitive components
    this.memory = new HypergraphMemory(nativeBridge.storage);
    await this.memory.load();
    
    this.persona = new PersonaEvolution(nativeBridge.llama);
    if (this.config.llamaModelPath) {
      await this.persona.initialize(this.config.llamaModelPath);
    }
    
    this.intentClassifier = new IntentClassifier(nativeBridge.onnx);
    if (this.config.intentClassifierPath) {
      await this.intentClassifier.load(this.config.intentClassifierPath);
    }
    
    this.metaCognition = new MetaCognitiveEngine(nativeBridge.llama);
    
    // Load speech models if available
    if (this.config.sttModelPath) {
      nativeBridge.speech.loadSTT(this.config.sttModelPath);
    }
    if (this.config.ttsModelPath) {
      nativeBridge.speech.loadTTS(this.config.ttsModelPath);
    }
    
    this.initialized = true;
    console.log('Mad9ml Mobile Runtime initialized');
  }
  
  /**
   * Shutdown the runtime
   */
  async shutdown(): Promise<void> {
    if (!this.initialized) return;
    
    // Save memory
    await this.memory.save();
    
    // Shutdown native bridge
    nativeBridge.shutdown();
    
    this.initialized = false;
  }
  
  /**
   * Run a cognitive cycle
   */
  async cognitiveCycle(): Promise<CognitiveState> {
    this.cycleCount++;
    
    // Decay attention
    this.attention.decay();
    
    // Perform meta-cognitive reflection periodically
    if (this.cycleCount % 10 === 0 && this.config.metaCognitionDepth! > 0) {
      const state = this.getState();
      await this.metaCognition.reflect(state, this.recentOutcomes);
      this.recentOutcomes = [];
    }
    
    // Evolve persona periodically
    if (this.config.evolutionEnabled && this.cycleCount % 100 === 0) {
      await this.persona.evolve();
    }
    
    return this.getState();
  }
  
  /**
   * Process a natural language command
   */
  async processCommand(text: string): Promise<ExecutionPlan> {
    const startTime = Date.now();
    
    try {
      // Get embedding for the command
      const embedding = await nativeBridge.llama.embed('persona', text);
      
      // Classify intent
      const { intent, confidence, alternatives } = await this.intentClassifier.classify(text, embedding);
      
      // Extract entities
      const entities = this.entityExtractor.extract(text);
      
      // Stimulate attention for relevant concepts
      this.attention.stimulate(`intent:${intent}`, confidence * 10);
      for (const [key, value] of entities) {
        this.attention.stimulate(`entity:${key}:${value}`, 5);
      }
      
      // Generate reasoning using persona
      const reasoning = await this.persona.generate(
        `Explain briefly why you would ${intent} ${entities.get('source') ?? 'files'}: ${text}`,
        100
      );
      
      // Build execution plan
      const operations = this.buildOperations(intent, entities);
      
      // Store in episodic memory
      await this.memory.add({
        id: `cmd_${Date.now()}`,
        type: 'episodic',
        content: { text, intent, entities: Object.fromEntries(entities) },
        embedding,
        attention: { sti: confidence * 10, lti: 1 },
        timestamp: Date.now(),
        accessCount: 1,
      });
      
      const latency = Date.now() - startTime;
      this.recentOutcomes.push({ success: true, latency });
      
      return {
        intent,
        confidence,
        entities,
        operations,
        reasoning,
      };
    } catch (error) {
      const latency = Date.now() - startTime;
      this.recentOutcomes.push({ success: false, latency });
      throw error;
    }
  }
  
  /**
   * Build file operations from intent and entities
   */
  private buildOperations(intent: string, entities: Map<string, string>): FileOperation[] {
    const operations: FileOperation[] = [];
    
    const source = entities.get('source');
    const destination = entities.get('destination');
    const pattern = entities.get('pattern') ?? entities.get('fileType');
    
    switch (intent) {
      case 'copy':
        operations.push({
          type: 'copy',
          source,
          destination,
          pattern,
          recursive: true,
        });
        break;
        
      case 'move':
        operations.push({
          type: 'move',
          source,
          destination,
          pattern,
          recursive: true,
        });
        break;
        
      case 'delete':
        operations.push({
          type: 'delete',
          source,
          pattern,
          recursive: false, // Safety: don't recursive delete by default
        });
        break;
        
      case 'search':
        operations.push({
          type: 'search',
          source: source ?? '/sdcard',
          pattern,
          recursive: true,
        });
        break;
        
      case 'organize':
        operations.push({
          type: 'organize',
          source: source ?? '/sdcard/Download',
          destination: destination ?? '/sdcard',
          options: {
            byType: true,
            byDate: false,
          },
        });
        break;
        
      case 'archive':
        operations.push({
          type: 'archive',
          source,
          destination: destination ?? `${source}.zip`,
        });
        break;
        
      default:
        // Unknown intent - return empty operations
        break;
    }
    
    return operations;
  }
  
  /**
   * Record outcome of an execution
   */
  recordOutcome(success: boolean, userFeedback?: number): void {
    this.persona.updateFitness(success, userFeedback);
    this.recentOutcomes.push({ success, latency: 0 });
  }
  
  /**
   * Perform meta-cognitive reflection
   */
  async performReflection(): Promise<MetaCognitiveInsight[]> {
    const state = this.getState();
    return this.metaCognition.reflect(state, this.recentOutcomes);
  }
  
  /**
   * Transcribe speech to text
   */
  async transcribe(audioPath: string): Promise<string> {
    return nativeBridge.speech.transcribe(audioPath);
  }
  
  /**
   * Synthesize text to speech
   */
  async synthesize(text: string, outputPath: string): Promise<string> {
    return nativeBridge.speech.synthesize(text, outputPath);
  }
  
  /**
   * Get current cognitive state
   */
  getState(): CognitiveState {
    return {
      timestamp: Date.now(),
      cycleCount: this.cycleCount,
      attention: this.attention.getState(),
      workingMemory: {
        focus: [...this.workingMemory.focus],
        context: new Map(this.workingMemory.context),
        goals: [...this.workingMemory.goals],
      },
      persona: this.persona.getState(),
    };
  }
  
  /**
   * Get hardware capabilities
   */
  getHardwareCapabilities(): HardwareCapabilities {
    return nativeBridge.getHardwareCapabilities();
  }
}

// Export factory function
export function createMad9mlRuntime(config: Mad9mlConfig): Mad9mlMobileRuntime {
  return new Mad9mlMobileRuntime(config);
}

export default Mad9mlMobileRuntime;
