import { ref } from 'vue';
import type { AgentResponse, ToolCall } from '@/types/AgentResponse';
import type { ApiMessage } from '@/types/ChatMessage';

// Customer info interface
export interface CustomerInfo {
  customer_id: string;
  display_name: string;
}

// Track the current agent response being shown
export const currentAgentResponse = ref<AgentResponse | null>(null);

// Create event emitter to stream agent results
export const agentResultsEmitter = {
  listeners: new Map<string, Function>(),
  
  addListener(id: string, callback: Function) {
    this.listeners.set(id, callback);
  },
  
  removeListener(id: string) {
    this.listeners.delete(id);
  },
  
  emit(id: string, data: any) {
    const callback = this.listeners.get(id);
    if (callback) {
      callback(data);
    }
  }
};

/**
 * Get list of demo customers from the backend
 */
export const getDemoCustomers = async (): Promise<CustomerInfo[]> => {
  try {
    const response = await fetch('/api/customers');
    if (!response.ok) {
      throw new Error('Failed to fetch customers');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching demo customers:', error);
    // Return default customers if API fails
    return [
      { customer_id: 'CUS-10001', display_name: 'Customer 10001' },
      { customer_id: 'CUS-10002', display_name: 'Customer 10002' },
      { customer_id: 'CUS-10006', display_name: 'Customer 10006' },
    ];
  }
};

/**
 * Convert API messages to conversation history format expected by backend
 */
const convertToConversationHistory = (messages: ApiMessage[]) => {
  // Convert to backend format (ApiMessage only has 'user' | 'assistant')
  return messages.map(msg => ({
    role: msg.role,
    content: msg.content
  }));
};

/**
 * Convert backend response to frontend AgentResponse format
 */
const convertBackendToAgentResponse = (backendResponse: any): AgentResponse => {
  const tools: ToolCall[] = [];
  
  // Use execution_steps if available for more detailed information
  const executionSteps = backendResponse.custom_outputs?.execution_steps || [];
  
  if (executionSteps.length > 0) {
    executionSteps.forEach((step: any, index: number) => {
      if (step.step_type === 'tool_call') {
        tools.push({
          tool_name: step.tool_name || `Tool ${index + 1}`,
          description: step.description || `Called ${step.tool_name}`,
          reasoning: step.reasoning || `Tool executed with arguments: ${JSON.stringify(step.arguments)}`,
          type: 'EXTERNAL_API',
          informations: [
            step.tool_name,
            ...(step.arguments ? [JSON.stringify(step.arguments, null, 2)] : [])
          ]
        });
      } else if (step.step_type === 'tool_result') {
        // Add result information to the last tool
        if (tools.length > 0) {
          const lastTool = tools[tools.length - 1];
          lastTool.informations.push(
            `Result: ${typeof step.result === 'string' ? step.result.substring(0, 200) + '...' : JSON.stringify(step.result).substring(0, 200) + '...'}`
          );
        }
      }
    });
  } else if (backendResponse.tools_used) {
    // Fallback to basic tools_used format
    backendResponse.tools_used.forEach((tool: any, index: number) => {
      tools.push({
        tool_name: tool.name || `Tool ${index + 1}`,
        description: `Called ${tool.name}`,
        reasoning: `Tool executed with arguments: ${JSON.stringify(tool.arguments)}`,
        type: 'EXTERNAL_API',
        informations: [tool.name, JSON.stringify(tool.arguments, null, 2)]
      });
    });
  }

  // Build final informations with more detail
  const finalInformations = [
    backendResponse.agent_type ? `Handled by ${backendResponse.agent_type} agent` : 'Processed by AI assistant'
  ];

  if (tools.length > 0) {
    finalInformations.push(`Used ${tools.length} tool(s) to gather information`);
  }

  // Add routing information if available
  const routingSteps = executionSteps.filter((step: any) => step.step_type === 'routing');
  routingSteps.forEach((step: any) => {
    finalInformations.push(step.description);
  });

  return {
    question: '', // Will be set by caller
    tools: tools,
    final_answer: backendResponse.response,
    final_informations: finalInformations,
    non_intelligent_answer: backendResponse.response
  };
};

/**
 * Send a message to the telco support agent backend
 * @param messages The conversation history
 * @param messageId The ID of the message being responded to
 * @param intelligenceEnabled Whether to show the full intelligence process
 * @param customerID The customer ID for the request
 * @returns A promise that resolves when the agent response is emitted
 */
export const sendMessageToAgent = async (
  messages: ApiMessage[], 
  messageId: string, 
  intelligenceEnabled: boolean = true,
  customerID: string = 'CUS-10001'
): Promise<void> => {
  try {
    // Get the last user message
    const userMessages = messages.filter(msg => msg.role === 'user');
    const lastUserMessage = userMessages[userMessages.length - 1];
    
    if (!lastUserMessage) {
      throw new Error('No user message found');
    }

    // Convert conversation history (exclude the current message)
    const conversationHistory = convertToConversationHistory(
      messages.slice(0, -1) // Exclude the last message as it's the current one
    );

    // Build request payload for your backend
    const requestPayload = {
      message: lastUserMessage.content,
      customer_id: customerID,
      conversation_history: conversationHistory
    };

    // Emit thinking start event
    agentResultsEmitter.emit(messageId, {
      type: 'thinking-start'
    });

    // Make request to your backend
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestPayload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const backendResponse = await response.json();
    
    // Convert backend response to frontend format
    const agentResponse = convertBackendToAgentResponse(backendResponse);
    agentResponse.question = lastUserMessage.content;

    // If intelligence is enabled, emit tool events
    if (intelligenceEnabled && agentResponse.tools.length > 0) {
      // Emit tools one by one with slight delays for animation
      for (let i = 0; i < agentResponse.tools.length; i++) {
        // Add delay between tools for smoother animation
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        agentResultsEmitter.emit(messageId, {
          type: 'tool',
          data: agentResponse.tools[i]
        });
      }
      
      // Small delay before final answer
      await new Promise(resolve => setTimeout(resolve, 800));
    }

    // Emit final answer
    agentResultsEmitter.emit(messageId, {
      type: 'final-answer',
      data: {
        final_answer: agentResponse.final_answer,
        final_informations: agentResponse.final_informations
      }
    });

  } catch (error) {
    console.error('Error in sendMessageToAgent:', error);
    
    // Emit error response
    agentResultsEmitter.emit(messageId, {
      type: 'final-answer',
      data: {
        final_answer: 'I apologize, but I encountered an error processing your request. Please try again.',
        final_informations: ['Error occurred during processing']
      }
    });
    
    throw error;
  }
};

/**
 * Get predefined questions for telco use case
 * These are hardcoded since we're telco-specific now
 */
export const getPredefinedQuestions = async () => {
  // Return telco-specific questions
  return [
    {
      preview: "What plan am I on?",
      text: "What plan am I currently on?"
    },
    {
      preview: "Check my bill",
      text: "Can you show me my billing details for this month?"
    },
    {
      preview: "Data usage",
      text: "How much data have I used this month?"
    },
    {
      preview: "Account info",
      text: "Can you show me my account information?"
    },
    {
      preview: "Device info",
      text: "What devices do I have on my account?"
    },
    {
      preview: "Available plans",
      text: "What plans are available and how do they compare?"
    },
    {
      preview: "WiFi issues",
      text: "My phone won't connect to WiFi, can you help?"
    },
    {
      preview: "Promotions",
      text: "Are there any current promotions I can take advantage of?"
    }
  ];
};