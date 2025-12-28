/**
 * MardukCommanderUI - React Native UI component for Marduk Commander
 * 
 * Provides a voice-enabled, cognitive-aware interface for controlling
 * Total Commander through natural language.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Animated,
  Vibration,
  Platform,
} from 'react-native';

import { MardukCommanderApp, ExecutionResult, AppConfig } from '../MardukCommanderApp';
import { CognitiveState, ExecutionPlan, MetaCognitiveInsight } from '../native/engines/Mad9mlMobileRuntime';

//=============================================================================
// Types
//=============================================================================

interface HistoryItem {
  id: string;
  command: string;
  result: ExecutionResult;
  timestamp: number;
  isVoice?: boolean;
}

interface UIState {
  isListening: boolean;
  isProcessing: boolean;
  currentCommand: string;
  history: HistoryItem[];
  suggestions: string[];
  cognitiveState: CognitiveState | null;
  insights: MetaCognitiveInsight[];
  error: string | null;
}

interface Props {
  config: AppConfig;
  onReady?: () => void;
  onError?: (error: Error) => void;
}

//=============================================================================
// Cognitive State Visualizer
//=============================================================================

const CognitiveStateVisualizer: React.FC<{ state: CognitiveState | null }> = ({ state }) => {
  if (!state) return null;
  
  const topAtoms = Array.from(state.attention.sti.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  
  return (
    <View style={styles.cognitivePanel}>
      <Text style={styles.panelTitle}>🧠 Cognitive State</Text>
      
      <View style={styles.stateRow}>
        <Text style={styles.stateLabel}>Cycle:</Text>
        <Text style={styles.stateValue}>{state.cycleCount}</Text>
      </View>
      
      <View style={styles.stateRow}>
        <Text style={styles.stateLabel}>Persona Fitness:</Text>
        <View style={styles.fitnessBar}>
          <View style={[styles.fitnessProgress, { width: `${state.persona.fitness * 100}%` }]} />
        </View>
        <Text style={styles.stateValue}>{(state.persona.fitness * 100).toFixed(0)}%</Text>
      </View>
      
      <Text style={styles.subTitle}>Active Attention:</Text>
      {topAtoms.map(([id, sti]) => (
        <View key={id} style={styles.atomRow}>
          <Text style={styles.atomId}>{id.substring(0, 20)}...</Text>
          <View style={styles.stiBar}>
            <View style={[styles.stiProgress, { width: `${sti}%` }]} />
          </View>
        </View>
      ))}
      
      {state.workingMemory.goals.length > 0 && (
        <>
          <Text style={styles.subTitle}>Current Goals:</Text>
          {state.workingMemory.goals.map((goal, i) => (
            <Text key={i} style={styles.goalText}>• {goal}</Text>
          ))}
        </>
      )}
    </View>
  );
};

//=============================================================================
// Insight Panel
//=============================================================================

const InsightPanel: React.FC<{ insights: MetaCognitiveInsight[] }> = ({ insights }) => {
  if (insights.length === 0) return null;
  
  const recentInsights = insights.slice(-3);
  
  const getIcon = (type: string) => {
    switch (type) {
      case 'performance': return '📊';
      case 'pattern': return '🔍';
      case 'suggestion': return '💡';
      case 'warning': return '⚠️';
      default: return '💭';
    }
  };
  
  return (
    <View style={styles.insightPanel}>
      <Text style={styles.panelTitle}>💡 Insights</Text>
      {recentInsights.map((insight, i) => (
        <View key={i} style={styles.insightItem}>
          <Text style={styles.insightIcon}>{getIcon(insight.type)}</Text>
          <View style={styles.insightContent}>
            <Text style={styles.insightText}>{insight.content}</Text>
            {insight.suggestedAction && (
              <Text style={styles.insightAction}>→ {insight.suggestedAction}</Text>
            )}
          </View>
        </View>
      ))}
    </View>
  );
};

//=============================================================================
// Command History
//=============================================================================

const CommandHistory: React.FC<{ 
  history: HistoryItem[];
  onFeedback: (id: string, rating: number) => void;
}> = ({ history, onFeedback }) => {
  return (
    <ScrollView style={styles.historyContainer}>
      {history.map((item) => (
        <View key={item.id} style={styles.historyItem}>
          <View style={styles.historyHeader}>
            <Text style={styles.historyCommand}>
              {item.isVoice ? '🎤 ' : '⌨️ '}{item.command}
            </Text>
            <Text style={styles.historyTime}>
              {new Date(item.timestamp).toLocaleTimeString()}
            </Text>
          </View>
          
          <View style={[
            styles.historyResult,
            item.result.success ? styles.resultSuccess : styles.resultError
          ]}>
            <Text style={styles.resultText}>
              {item.result.success ? '✅' : '❌'} {item.result.message}
            </Text>
            {item.result.details?.reasoning && (
              <Text style={styles.reasoningText}>
                💭 {item.result.details.reasoning}
              </Text>
            )}
          </View>
          
          <View style={styles.feedbackRow}>
            <Text style={styles.feedbackLabel}>Was this helpful?</Text>
            <TouchableOpacity 
              style={styles.feedbackButton}
              onPress={() => onFeedback(item.id, 1)}
            >
              <Text>👍</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={styles.feedbackButton}
              onPress={() => onFeedback(item.id, 0)}
            >
              <Text>👎</Text>
            </TouchableOpacity>
          </View>
        </View>
      ))}
    </ScrollView>
  );
};

//=============================================================================
// Voice Button
//=============================================================================

const VoiceButton: React.FC<{
  isListening: boolean;
  isProcessing: boolean;
  onPress: () => void;
}> = ({ isListening, isProcessing, onPress }) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  
  useEffect(() => {
    if (isListening) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 500,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 500,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isListening, pulseAnim]);
  
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isProcessing}
      style={styles.voiceButtonContainer}
    >
      <Animated.View
        style={[
          styles.voiceButton,
          isListening && styles.voiceButtonActive,
          isProcessing && styles.voiceButtonProcessing,
          { transform: [{ scale: pulseAnim }] },
        ]}
      >
        <Text style={styles.voiceButtonText}>
          {isProcessing ? '⏳' : isListening ? '🎤' : '🎙️'}
        </Text>
      </Animated.View>
    </TouchableOpacity>
  );
};

//=============================================================================
// Suggestions
//=============================================================================

const Suggestions: React.FC<{
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}> = ({ suggestions, onSelect }) => {
  if (suggestions.length === 0) return null;
  
  return (
    <View style={styles.suggestionsContainer}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {suggestions.map((suggestion, i) => (
          <TouchableOpacity
            key={i}
            style={styles.suggestionChip}
            onPress={() => onSelect(suggestion)}
          >
            <Text style={styles.suggestionText}>{suggestion}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};

//=============================================================================
// Main Component
//=============================================================================

export const MardukCommanderUI: React.FC<Props> = ({ config, onReady, onError }) => {
  const appRef = useRef<MardukCommanderApp | null>(null);
  
  const [state, setState] = useState<UIState>({
    isListening: false,
    isProcessing: false,
    currentCommand: '',
    history: [],
    suggestions: [],
    cognitiveState: null,
    insights: [],
    error: null,
  });
  
  // Initialize app
  useEffect(() => {
    const initApp = async () => {
      try {
        const app = new MardukCommanderApp(config);
        
        app.on('stateChange', (cogState) => {
          setState(prev => ({ ...prev, cognitiveState: cogState }));
        });
        
        app.on('insight', (insight) => {
          setState(prev => ({ 
            ...prev, 
            insights: [...prev.insights, insight].slice(-10) 
          }));
        });
        
        app.on('executionStart', (plan) => {
          console.log('Execution started:', plan.intent);
        });
        
        app.on('executionComplete', (result) => {
          setState(prev => ({ ...prev, isProcessing: false }));
        });
        
        await app.initialize();
        appRef.current = app;
        
        onReady?.();
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        setState(prev => ({ ...prev, error: err.message }));
        onError?.(err);
      }
    };
    
    initApp();
    
    return () => {
      appRef.current?.shutdown();
    };
  }, [config, onReady, onError]);
  
  // Handle command submission
  const handleSubmit = useCallback(async () => {
    const app = appRef.current;
    if (!app || !state.currentCommand.trim()) return;
    
    setState(prev => ({ ...prev, isProcessing: true, error: null }));
    
    try {
      const result = await app.processCommand(state.currentCommand);
      
      const historyItem: HistoryItem = {
        id: `cmd_${Date.now()}`,
        command: state.currentCommand,
        result,
        timestamp: Date.now(),
        isVoice: false,
      };
      
      setState(prev => ({
        ...prev,
        currentCommand: '',
        history: [historyItem, ...prev.history].slice(0, 50),
        suggestions: [],
      }));
      
      if (result.success) {
        Vibration.vibrate(50);
      } else {
        Vibration.vibrate([0, 100, 50, 100]);
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : String(error),
        isProcessing: false,
      }));
    }
  }, [state.currentCommand]);
  
  // Handle voice input
  const handleVoicePress = useCallback(async () => {
    const app = appRef.current;
    if (!app) return;
    
    if (state.isListening) {
      // Stop listening and process
      setState(prev => ({ ...prev, isListening: false, isProcessing: true }));
      
      // In a real app, this would stop recording and get the audio file
      // For now, we simulate with a placeholder
      try {
        const result = await app.processVoiceCommand('/tmp/voice_recording.wav');
        
        const historyItem: HistoryItem = {
          id: `voice_${Date.now()}`,
          command: '[Voice Command]',
          result,
          timestamp: Date.now(),
          isVoice: true,
        };
        
        setState(prev => ({
          ...prev,
          history: [historyItem, ...prev.history].slice(0, 50),
        }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: error instanceof Error ? error.message : String(error),
        }));
      }
    } else {
      // Start listening
      setState(prev => ({ ...prev, isListening: true }));
      Vibration.vibrate(100);
    }
  }, [state.isListening]);
  
  // Handle text change and get suggestions
  const handleTextChange = useCallback(async (text: string) => {
    setState(prev => ({ ...prev, currentCommand: text }));
    
    const app = appRef.current;
    if (app && text.length > 2) {
      const suggestions = await app.getSuggestions(text);
      setState(prev => ({ ...prev, suggestions }));
    } else {
      setState(prev => ({ ...prev, suggestions: [] }));
    }
  }, []);
  
  // Handle feedback
  const handleFeedback = useCallback((id: string, rating: number) => {
    appRef.current?.provideFeedback(id, rating);
    Vibration.vibrate(30);
  }, []);
  
  // Handle suggestion selection
  const handleSuggestionSelect = useCallback((suggestion: string) => {
    setState(prev => ({ ...prev, currentCommand: suggestion, suggestions: [] }));
  }, []);
  
  // Handle undo
  const handleUndo = useCallback(async () => {
    const app = appRef.current;
    if (!app) return;
    
    setState(prev => ({ ...prev, isProcessing: true }));
    
    try {
      const result = await app.undo();
      
      const historyItem: HistoryItem = {
        id: `undo_${Date.now()}`,
        command: '[Undo]',
        result,
        timestamp: Date.now(),
      };
      
      setState(prev => ({
        ...prev,
        history: [historyItem, ...prev.history].slice(0, 50),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, []);
  
  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>🤖 Marduk Commander</Text>
        <TouchableOpacity onPress={handleUndo} style={styles.undoButton}>
          <Text style={styles.undoText}>↩️ Undo</Text>
        </TouchableOpacity>
      </View>
      
      {/* Error display */}
      {state.error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>⚠️ {state.error}</Text>
          <TouchableOpacity onPress={() => setState(prev => ({ ...prev, error: null }))}>
            <Text style={styles.errorDismiss}>✕</Text>
          </TouchableOpacity>
        </View>
      )}
      
      {/* Cognitive state panel (collapsible) */}
      <CognitiveStateVisualizer state={state.cognitiveState} />
      
      {/* Insights panel */}
      <InsightPanel insights={state.insights} />
      
      {/* Command history */}
      <CommandHistory 
        history={state.history} 
        onFeedback={handleFeedback}
      />
      
      {/* Suggestions */}
      <Suggestions 
        suggestions={state.suggestions}
        onSelect={handleSuggestionSelect}
      />
      
      {/* Input area */}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={state.currentCommand}
          onChangeText={handleTextChange}
          onSubmitEditing={handleSubmit}
          placeholder="Type a command..."
          placeholderTextColor="#888"
          editable={!state.isProcessing}
          returnKeyType="send"
        />
        
        <TouchableOpacity
          style={[styles.sendButton, !state.currentCommand.trim() && styles.sendButtonDisabled]}
          onPress={handleSubmit}
          disabled={!state.currentCommand.trim() || state.isProcessing}
        >
          <Text style={styles.sendButtonText}>
            {state.isProcessing ? '⏳' : '➤'}
          </Text>
        </TouchableOpacity>
      </View>
      
      {/* Voice button */}
      {config.voiceEnabled && (
        <VoiceButton
          isListening={state.isListening}
          isProcessing={state.isProcessing}
          onPress={handleVoicePress}
        />
      )}
    </View>
  );
};

//=============================================================================
// Styles
//=============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#16213e',
    borderBottomWidth: 1,
    borderBottomColor: '#0f3460',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#e94560',
  },
  undoButton: {
    padding: 8,
  },
  undoText: {
    color: '#fff',
    fontSize: 14,
  },
  errorBanner: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#e94560',
    padding: 12,
  },
  errorText: {
    color: '#fff',
    flex: 1,
  },
  errorDismiss: {
    color: '#fff',
    fontSize: 18,
    paddingLeft: 12,
  },
  cognitivePanel: {
    backgroundColor: '#16213e',
    padding: 12,
    margin: 8,
    borderRadius: 8,
  },
  panelTitle: {
    color: '#e94560',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  stateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  stateLabel: {
    color: '#888',
    width: 100,
  },
  stateValue: {
    color: '#fff',
    marginLeft: 8,
  },
  fitnessBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#0f3460',
    borderRadius: 4,
    overflow: 'hidden',
  },
  fitnessProgress: {
    height: '100%',
    backgroundColor: '#4ecca3',
  },
  subTitle: {
    color: '#888',
    fontSize: 12,
    marginTop: 8,
    marginBottom: 4,
  },
  atomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 2,
  },
  atomId: {
    color: '#666',
    fontSize: 10,
    width: 120,
  },
  stiBar: {
    flex: 1,
    height: 4,
    backgroundColor: '#0f3460',
    borderRadius: 2,
    overflow: 'hidden',
  },
  stiProgress: {
    height: '100%',
    backgroundColor: '#e94560',
  },
  goalText: {
    color: '#4ecca3',
    fontSize: 12,
    marginLeft: 8,
  },
  insightPanel: {
    backgroundColor: '#16213e',
    padding: 12,
    marginHorizontal: 8,
    marginBottom: 8,
    borderRadius: 8,
  },
  insightItem: {
    flexDirection: 'row',
    marginVertical: 4,
  },
  insightIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  insightContent: {
    flex: 1,
  },
  insightText: {
    color: '#fff',
    fontSize: 12,
  },
  insightAction: {
    color: '#4ecca3',
    fontSize: 11,
    marginTop: 2,
  },
  historyContainer: {
    flex: 1,
    padding: 8,
  },
  historyItem: {
    backgroundColor: '#16213e',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  historyCommand: {
    color: '#fff',
    fontWeight: 'bold',
    flex: 1,
  },
  historyTime: {
    color: '#666',
    fontSize: 12,
  },
  historyResult: {
    padding: 8,
    borderRadius: 4,
  },
  resultSuccess: {
    backgroundColor: 'rgba(78, 204, 163, 0.2)',
  },
  resultError: {
    backgroundColor: 'rgba(233, 69, 96, 0.2)',
  },
  resultText: {
    color: '#fff',
    fontSize: 13,
  },
  reasoningText: {
    color: '#888',
    fontSize: 11,
    marginTop: 4,
    fontStyle: 'italic',
  },
  feedbackRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#0f3460',
  },
  feedbackLabel: {
    color: '#666',
    fontSize: 12,
    flex: 1,
  },
  feedbackButton: {
    padding: 8,
    marginLeft: 8,
  },
  suggestionsContainer: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  suggestionChip: {
    backgroundColor: '#0f3460',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
  },
  suggestionText: {
    color: '#fff',
    fontSize: 12,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 8,
    backgroundColor: '#16213e',
    borderTopWidth: 1,
    borderTopColor: '#0f3460',
  },
  textInput: {
    flex: 1,
    backgroundColor: '#0f3460',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: '#fff',
    fontSize: 16,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#e94560',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  sendButtonDisabled: {
    backgroundColor: '#666',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 20,
  },
  voiceButtonContainer: {
    position: 'absolute',
    bottom: 80,
    alignSelf: 'center',
  },
  voiceButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#e94560',
    justifyContent: 'center',
    alignItems: 'center',
    ...Platform.select({
      ios: {
        shadowColor: '#e94560',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 8,
      },
      android: {
        elevation: 8,
      },
    }),
  },
  voiceButtonActive: {
    backgroundColor: '#4ecca3',
  },
  voiceButtonProcessing: {
    backgroundColor: '#666',
  },
  voiceButtonText: {
    fontSize: 28,
  },
});

export default MardukCommanderUI;
