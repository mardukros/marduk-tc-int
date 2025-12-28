/**
 * MardukCommanderApp - Main application integrating mad9ml cognitive engine
 * with Total Commander file management
 * 
 * This is the unified entry point that combines:
 * - mad9ml cognitive engine (via native ARM64 libraries)
 * - Total Commander integration (via Android Intents)
 * - Voice interface (via Sherpa ONNX + Piper)
 * - Persistent memory and learning
 */

import { Mad9mlMobileRuntime, Mad9mlConfig, ExecutionPlan, CognitiveState, MetaCognitiveInsight } from './native/engines/Mad9mlMobileRuntime';
import { nativeBridge } from './native/bridge/NativeBridge';

//=============================================================================
// Type Definitions
//=============================================================================

export interface TCIntentConfig {
  package: string;
  actions: {
    open: string;
    openPath: string;
    sendTo: string;
    pick: string;
    view: string;
  };
}

export interface ExecutionResult {
  success: boolean;
  message: string;
  details?: any;
  duration: number;
}

export interface VoiceCommand {
  audioPath: string;
  transcription?: string;
}

export interface AppConfig extends Mad9mlConfig {
  // TC Integration
  tcPackage?: string;
  adbPath?: string;
  useTasker?: boolean;
  taskerTaskName?: string;
  
  // Voice settings
  voiceEnabled?: boolean;
  voiceFeedback?: boolean;
  wakeWord?: string;
  
  // UI settings
  showReasoning?: boolean;
  confirmDestructive?: boolean;
  maxHistorySize?: number;
}

//=============================================================================
// TC Intent Builder
//=============================================================================

class TCIntentBuilder {
  private config: TCIntentConfig = {
    package: 'com.ghisler.android.TotalCommander',
    actions: {
      open: 'com.ghisler.android.TotalCommander.OPEN',
      openPath: 'com.ghisler.android.TotalCommander.OPENPATH',
      sendTo: 'com.ghisler.android.TotalCommander.SENDTO',
      pick: 'com.ghisler.android.TotalCommander.PICK',
      view: 'com.ghisler.android.TotalCommander.VIEW',
    },
  };
  
  /**
   * Build an ADB command to send an Intent to Total Commander
   */
  buildAdbCommand(action: string, extras: Record<string, string> = {}): string {
    let cmd = `adb shell am start -a ${action}`;
    
    for (const [key, value] of Object.entries(extras)) {
      cmd += ` --es "${key}" "${value}"`;
    }
    
    cmd += ` -n ${this.config.package}/.TotalCommander`;
    
    return cmd;
  }
  
  /**
   * Build Intent for opening a path
   */
  openPath(path: string): string {
    return this.buildAdbCommand(this.config.actions.openPath, {
      'com.ghisler.android.TotalCommander.Path': path,
    });
  }
  
  /**
   * Build Intent for file operations
   */
  fileOperation(operation: string, source: string, destination?: string): string {
    const extras: Record<string, string> = {
      'com.ghisler.android.TotalCommander.Source': source,
    };
    
    if (destination) {
      extras['com.ghisler.android.TotalCommander.Destination'] = destination;
    }
    
    extras['com.ghisler.android.TotalCommander.Operation'] = operation;
    
    return this.buildAdbCommand(this.config.actions.sendTo, extras);
  }
  
  /**
   * Build Intent for search
   */
  search(path: string, pattern: string): string {
    return this.buildAdbCommand(this.config.actions.open, {
      'com.ghisler.android.TotalCommander.Path': path,
      'com.ghisler.android.TotalCommander.Search': pattern,
    });
  }
  
  /**
   * Build Tasker command
   */
  buildTaskerCommand(taskName: string, params: Record<string, string>): string {
    const paramStr = Object.entries(params)
      .map(([k, v]) => `${k}=${v}`)
      .join(',');
    
    return `am broadcast -a net.dinglisch.android.tasker.ACTION_TASK -e task_name "${taskName}" -e params "${paramStr}"`;
  }
}

//=============================================================================
// Command History
//=============================================================================

interface HistoryEntry {
  id: string;
  timestamp: number;
  command: string;
  plan: ExecutionPlan;
  result: ExecutionResult;
  feedback?: number;
}

class CommandHistory {
  private entries: HistoryEntry[] = [];
  private maxSize: number;
  
  constructor(maxSize = 100) {
    this.maxSize = maxSize;
  }
  
  add(entry: HistoryEntry): void {
    this.entries.unshift(entry);
    if (this.entries.length > this.maxSize) {
      this.entries.pop();
    }
  }
  
  getRecent(n: number): HistoryEntry[] {
    return this.entries.slice(0, n);
  }
  
  getById(id: string): HistoryEntry | undefined {
    return this.entries.find(e => e.id === id);
  }
  
  updateFeedback(id: string, feedback: number): void {
    const entry = this.getById(id);
    if (entry) {
      entry.feedback = feedback;
    }
  }
  
  getSuccessRate(): number {
    if (this.entries.length === 0) return 1;
    const successes = this.entries.filter(e => e.result.success).length;
    return successes / this.entries.length;
  }
  
  toJSON(): HistoryEntry[] {
    return this.entries;
  }
  
  fromJSON(data: HistoryEntry[]): void {
    this.entries = data.slice(0, this.maxSize);
  }
}

//=============================================================================
// Marduk Commander App
//=============================================================================

export class MardukCommanderApp {
  private config: AppConfig;
  private runtime: Mad9mlMobileRuntime;
  private intentBuilder: TCIntentBuilder;
  private history: CommandHistory;
  private initialized = false;
  
  // Event callbacks
  private onStateChange?: (state: CognitiveState) => void;
  private onInsight?: (insight: MetaCognitiveInsight) => void;
  private onExecutionStart?: (plan: ExecutionPlan) => void;
  private onExecutionComplete?: (result: ExecutionResult) => void;
  
  constructor(config: AppConfig) {
    this.config = {
      // Defaults
      tcPackage: 'com.ghisler.android.TotalCommander',
      adbPath: 'adb',
      useTasker: false,
      voiceEnabled: true,
      voiceFeedback: true,
      wakeWord: 'marduk',
      showReasoning: true,
      confirmDestructive: true,
      maxHistorySize: 100,
      ...config,
    };
    
    this.runtime = new Mad9mlMobileRuntime(this.config);
    this.intentBuilder = new TCIntentBuilder();
    this.history = new CommandHistory(this.config.maxHistorySize);
  }
  
  /**
   * Initialize the application
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    console.log('Initializing Marduk Commander...');
    console.log(`Models path: ${this.config.modelsPath}`);
    
    // Initialize cognitive runtime
    await this.runtime.initialize();
    
    // Load history from storage
    const savedHistory = nativeBridge.storage.getObject<HistoryEntry[]>('command_history');
    if (savedHistory) {
      this.history.fromJSON(savedHistory);
    }
    
    // Log hardware capabilities
    const hw = this.runtime.getHardwareCapabilities();
    console.log(`Hardware: ${hw.cpuCores} cores, GPU: ${hw.hasGPU ? hw.gpuType : 'none'}, NPU: ${hw.hasNPU ? hw.npuType : 'none'}`);
    
    this.initialized = true;
    console.log('Marduk Commander initialized');
  }
  
  /**
   * Shutdown the application
   */
  async shutdown(): Promise<void> {
    if (!this.initialized) return;
    
    // Save history
    nativeBridge.storage.setObject('command_history', this.history.toJSON());
    
    // Shutdown runtime
    await this.runtime.shutdown();
    
    this.initialized = false;
    console.log('Marduk Commander shutdown');
  }
  
  /**
   * Process a text command
   */
  async processCommand(text: string): Promise<ExecutionResult> {
    if (!this.initialized) {
      throw new Error('App not initialized');
    }
    
    const startTime = Date.now();
    const commandId = `cmd_${startTime}`;
    
    try {
      // Run cognitive cycle
      const state = await this.runtime.cognitiveCycle();
      this.onStateChange?.(state);
      
      // Process command through cognitive engine
      const plan = await this.runtime.processCommand(text);
      this.onExecutionStart?.(plan);
      
      // Check for destructive operations
      if (this.config.confirmDestructive && this.isDestructive(plan)) {
        // In a real app, this would prompt the user
        console.log(`⚠️ Destructive operation: ${plan.intent}`);
      }
      
      // Execute the plan
      const result = await this.executePlan(plan);
      
      // Record outcome for learning
      this.runtime.recordOutcome(result.success);
      
      // Add to history
      this.history.add({
        id: commandId,
        timestamp: startTime,
        command: text,
        plan,
        result,
      });
      
      // Perform reflection periodically
      if (this.history.getRecent(10).length % 5 === 0) {
        const insights = await this.runtime.performReflection();
        for (const insight of insights) {
          this.onInsight?.(insight);
        }
      }
      
      this.onExecutionComplete?.(result);
      return result;
      
    } catch (error) {
      const result: ExecutionResult = {
        success: false,
        message: `Error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
      };
      
      this.runtime.recordOutcome(false);
      this.onExecutionComplete?.(result);
      return result;
    }
  }
  
  /**
   * Process a voice command
   */
  async processVoiceCommand(audioPath: string): Promise<ExecutionResult> {
    if (!this.initialized) {
      throw new Error('App not initialized');
    }
    
    // Transcribe audio
    const transcription = await this.runtime.transcribe(audioPath);
    console.log(`Transcribed: ${transcription}`);
    
    // Process as text command
    const result = await this.processCommand(transcription);
    
    // Provide voice feedback if enabled
    if (this.config.voiceFeedback) {
      const feedbackText = result.success 
        ? `Done. ${result.message}`
        : `Sorry, ${result.message}`;
      
      const outputPath = `/tmp/feedback_${Date.now()}.wav`;
      await this.runtime.synthesize(feedbackText, outputPath);
      // In a real app, this would play the audio
    }
    
    return result;
  }
  
  /**
   * Execute a plan
   */
  private async executePlan(plan: ExecutionPlan): Promise<ExecutionResult> {
    const startTime = Date.now();
    const results: string[] = [];
    
    for (const op of plan.operations) {
      try {
        const cmd = this.buildCommand(op);
        console.log(`Executing: ${cmd}`);
        
        // In a real app, this would execute the command
        // For now, we simulate success
        results.push(`${op.type}: ${op.source ?? ''} → ${op.destination ?? ''}`);
        
      } catch (error) {
        return {
          success: false,
          message: `Failed to execute ${op.type}: ${error instanceof Error ? error.message : String(error)}`,
          duration: Date.now() - startTime,
        };
      }
    }
    
    return {
      success: true,
      message: results.join('; ') || 'Command executed',
      details: {
        intent: plan.intent,
        confidence: plan.confidence,
        operations: plan.operations.length,
        reasoning: this.config.showReasoning ? plan.reasoning : undefined,
      },
      duration: Date.now() - startTime,
    };
  }
  
  /**
   * Build command for an operation
   */
  private buildCommand(op: ExecutionPlan['operations'][0]): string {
    if (this.config.useTasker) {
      return this.intentBuilder.buildTaskerCommand(
        this.config.taskerTaskName ?? 'MardukTC',
        {
          operation: op.type,
          source: op.source ?? '',
          destination: op.destination ?? '',
          pattern: op.pattern ?? '',
        }
      );
    }
    
    switch (op.type) {
      case 'copy':
      case 'move':
      case 'delete':
        return this.intentBuilder.fileOperation(
          op.type.toUpperCase(),
          op.source ?? '',
          op.destination
        );
        
      case 'search':
        return this.intentBuilder.search(
          op.source ?? '/sdcard',
          op.pattern ?? '*'
        );
        
      default:
        return this.intentBuilder.openPath(op.source ?? '/sdcard');
    }
  }
  
  /**
   * Check if operation is destructive
   */
  private isDestructive(plan: ExecutionPlan): boolean {
    const destructiveIntents = ['delete', 'move', 'organize'];
    return destructiveIntents.includes(plan.intent);
  }
  
  /**
   * Provide feedback on a command
   */
  provideFeedback(commandId: string, rating: number): void {
    this.history.updateFeedback(commandId, rating);
    this.runtime.recordOutcome(rating > 0.5, rating);
  }
  
  /**
   * Undo the last command
   */
  async undo(): Promise<ExecutionResult> {
    const recent = this.history.getRecent(1)[0];
    if (!recent) {
      return {
        success: false,
        message: 'Nothing to undo',
        duration: 0,
      };
    }
    
    // Build reverse operation
    const reverseOp = this.buildReverseOperation(recent.plan);
    if (!reverseOp) {
      return {
        success: false,
        message: `Cannot undo ${recent.plan.intent}`,
        duration: 0,
      };
    }
    
    return this.processCommand(reverseOp);
  }
  
  /**
   * Build reverse operation for undo
   */
  private buildReverseOperation(plan: ExecutionPlan): string | null {
    switch (plan.intent) {
      case 'move':
        // Reverse move: move back
        const dest = plan.entities.get('destination');
        const src = plan.entities.get('source');
        if (dest && src) {
          return `move files from ${dest} to ${src}`;
        }
        return null;
        
      case 'copy':
        // Reverse copy: delete the copy
        const copyDest = plan.entities.get('destination');
        if (copyDest) {
          return `delete files in ${copyDest}`;
        }
        return null;
        
      default:
        return null;
    }
  }
  
  /**
   * Get command suggestions based on context
   */
  async getSuggestions(partialCommand: string): Promise<string[]> {
    const suggestions: string[] = [];
    
    // Get similar commands from history
    const recent = this.history.getRecent(20);
    for (const entry of recent) {
      if (entry.command.toLowerCase().includes(partialCommand.toLowerCase())) {
        suggestions.push(entry.command);
      }
    }
    
    // Add common command templates
    const templates = [
      'move all PDFs from Downloads to Documents',
      'find large files in Pictures',
      'organize Downloads by file type',
      'delete old files in Downloads',
      'search for documents containing',
      'archive Photos folder',
      'copy music to external storage',
    ];
    
    for (const template of templates) {
      if (template.toLowerCase().includes(partialCommand.toLowerCase())) {
        suggestions.push(template);
      }
    }
    
    return [...new Set(suggestions)].slice(0, 5);
  }
  
  /**
   * Get current cognitive state
   */
  getState(): CognitiveState {
    return this.runtime.getState();
  }
  
  /**
   * Get command history
   */
  getHistory(n = 10): HistoryEntry[] {
    return this.history.getRecent(n);
  }
  
  /**
   * Get performance metrics
   */
  getMetrics(): { successRate: number; totalCommands: number } {
    return {
      successRate: this.history.getSuccessRate(),
      totalCommands: this.history.getRecent(1000).length,
    };
  }
  
  /**
   * Set event callbacks
   */
  on(event: 'stateChange', callback: (state: CognitiveState) => void): void;
  on(event: 'insight', callback: (insight: MetaCognitiveInsight) => void): void;
  on(event: 'executionStart', callback: (plan: ExecutionPlan) => void): void;
  on(event: 'executionComplete', callback: (result: ExecutionResult) => void): void;
  on(event: string, callback: (...args: any[]) => void): void {
    switch (event) {
      case 'stateChange':
        this.onStateChange = callback;
        break;
      case 'insight':
        this.onInsight = callback;
        break;
      case 'executionStart':
        this.onExecutionStart = callback;
        break;
      case 'executionComplete':
        this.onExecutionComplete = callback;
        break;
    }
  }
}

//=============================================================================
// CLI Interface
//=============================================================================

export async function runCLI(config: AppConfig): Promise<void> {
  const app = new MardukCommanderApp(config);
  
  // Set up event handlers
  app.on('stateChange', (state) => {
    console.log(`\n[Cycle ${state.cycleCount}] Attention: ${state.attention.sti.size} active atoms`);
  });
  
  app.on('insight', (insight) => {
    console.log(`\n💡 Insight (${insight.type}): ${insight.content}`);
    if (insight.suggestedAction) {
      console.log(`   Suggested: ${insight.suggestedAction}`);
    }
  });
  
  app.on('executionStart', (plan) => {
    console.log(`\n🎯 Intent: ${plan.intent} (${(plan.confidence * 100).toFixed(0)}%)`);
    if (plan.reasoning) {
      console.log(`💭 Reasoning: ${plan.reasoning}`);
    }
  });
  
  app.on('executionComplete', (result) => {
    const icon = result.success ? '✅' : '❌';
    console.log(`${icon} ${result.message} (${result.duration}ms)`);
  });
  
  // Initialize
  await app.initialize();
  
  console.log('\n🤖 Marduk Commander Ready');
  console.log('Type commands in natural language, or "quit" to exit.\n');
  
  // Simple REPL (in a real app, this would use readline or similar)
  const commands = [
    'move all PDFs from Downloads to Documents',
    'find large files in Pictures',
    'organize my Downloads folder',
  ];
  
  for (const cmd of commands) {
    console.log(`\nmarduk> ${cmd}`);
    await app.processCommand(cmd);
  }
  
  // Show metrics
  const metrics = app.getMetrics();
  console.log(`\n📊 Session: ${metrics.totalCommands} commands, ${(metrics.successRate * 100).toFixed(0)}% success rate`);
  
  await app.shutdown();
}

//=============================================================================
// React Native Integration
//=============================================================================

export function createMardukCommanderApp(config: AppConfig): MardukCommanderApp {
  return new MardukCommanderApp(config);
}

// Export for React Native
export default MardukCommanderApp;
