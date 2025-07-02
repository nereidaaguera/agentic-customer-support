<template>
  <v-app>
    <Header 
      :title="appTitle"
      @refresh="handleRefresh"
    />
    
    <v-main>
      <v-container fluid class="fill-height pa-0 pa-sm-4">
        <v-row class="fill-height ma-0">
          <v-col cols="12" lg="8" class="fill-height pa-0 pa-sm-2">
            <v-card class="fill-height d-flex flex-column" elevation="2">
              <v-card-title class="pb-0 pt-4 px-4 d-flex align-center justify-space-between">
                <div class="title-section">
                  <h1 class="text-h5">{{ appTitle }}</h1>
                  <p class="text-body-2 text-medium-emphasis mt-1">
                    Ask me anything about your telecom account and I'll help you.
                  </p>
                  
                  <!-- Customer Selector -->
                  <div class="customer-selector-container mt-3">
                    <CustomerSelector 
                      v-model="selectedCustomerId"
                      @customer-change="handleCustomerChange"
                    />
                  </div>
                </div>
                
                <div class="header-controls">
                  <!-- MLflow Experiment Link -->
                  <v-btn
                    v-if="mlflowUrl"
                    :href="mlflowUrl"
                    target="_blank"
                    variant="text"
                    size="small"
                    class="mlflow-link"
                  >
                    <v-icon icon="mdi-chart-box" size="small" class="mr-1" />
                    MLflow Experiment
                  </v-btn>
                  
                  <!-- Intelligence Toggle -->
                  <div class="intelligence-toggle">
                    <v-tooltip
                      location="bottom"
                      text="Toggle AI intelligence visualization"
                    >
                      <template v-slot:activator="{ props }">
                        <div class="d-flex align-center">
                          <span class="mr-2">Intelligence:</span>
                          <v-switch
                            v-bind="props"
                            v-model="intelligenceEnabled"
                            color="success"
                            hide-details
                            density="compact"
                            inset
                          ></v-switch>
                          <span 
                            class="ml-1 text-caption" 
                            :style="{ 
                              color: intelligenceEnabled ? '#2e7d32' : '#757575', 
                              fontWeight: 500
                            }"
                          >
                            {{ intelligenceEnabled ? 'On' : 'Off' }}
                          </span>
                        </div>
                      </template>
                    </v-tooltip>
                  </div>
                </div>
              </v-card-title>
              
              <v-card-text class="flex-grow-1 d-flex flex-column pa-0 pa-sm-4">
                <ChatBot 
                  ref="chatBot"
                  class="fill-height" 
                  :thinking="isThinking"
                  :agents="agentTools"
                  :finalAnswer="finalAnswer"
                  :finalInformations="finalInformations"
                  :intelligenceEnabled="intelligenceEnabled"
                  :customerId="selectedCustomerId"
                  @update:thinking="handleThinkingUpdate"
                  @update:agents="handleAgentsUpdate"
                  @update:final-answer="handleFinalAnswerUpdate"
                  @update:final-informations="handleFinalInformationsUpdate"
                />
              </v-card-text>
            </v-card>
          </v-col>
          
          <v-col cols="12" lg="4" class="fill-height pa-0 pa-sm-2" :class="{ 'd-none': !intelligenceEnabled, 'd-lg-block': intelligenceEnabled }">
            <IntelligentPanel 
              :show-thinking="isThinking"
              :tools="agentTools"
              :final-answer="finalAnswer"
              :final-informations="finalInformations"
            />
          </v-col>
        </v-row>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import Header from '@/components/Header.vue'
import ChatBot from '@/components/ChatBot.vue'
import IntelligentPanel from '@/components/IntelligentPanel.vue'
import CustomerSelector from '@/components/CustomerSelector.vue'
import type { ToolCall } from '@/types/AgentResponse'

const appTitle = ref('Telco Support Assistant')
const chatBot = ref<InstanceType<typeof ChatBot> | null>(null)
const isThinking = ref(false)
const agentTools = ref<ToolCall[]>([])
const finalAnswer = ref('')
const finalInformations = ref<string[]>([])
const intelligenceEnabled = ref(true) // Default state for the toggle

// Customer management
const selectedCustomerId = ref('CUS-10001') // Default customer

// MLflow experiment URL
const mlflowUrl = ref<string | null>(null)

// Handle refresh
const handleRefresh = () => {
  window.location.reload()
}

// Handle customer changes
const handleCustomerChange = (customerId: string) => {
  console.log('Customer changed to:', customerId)
  selectedCustomerId.value = customerId
  
  // Reset chat when customer changes
  if (chatBot.value && chatBot.value.resetChat) {
    // Reset the intelligence panel as well
    isThinking.value = false
    agentTools.value = []
    finalAnswer.value = ''
    finalInformations.value = []
    
    // Reset the chat interface
    chatBot.value.resetChat()
  }
}

const handleThinkingUpdate = (value: boolean) => {
  isThinking.value = value
  
  // When thinking stops, reset the agent data after a delay
  if (!value) {
    setTimeout(() => {
      if (!isThinking.value) {
        agentTools.value = []
        finalAnswer.value = ''
        finalInformations.value = []
      }
    }, 5000)
  }
}

// Watch for changes to the intelligence toggle
watch(intelligenceEnabled, (newValue) => {
  console.log(`Intelligence panel ${newValue ? 'enabled' : 'disabled'}`)
  
  // Reset the thinking state when toggling intelligence mode
  isThinking.value = false
  
  // Clear the previous agents and responses when toggling
  agentTools.value = []
  finalAnswer.value = ''
  finalInformations.value = []

  // Reset the chat when intelligence is toggled
  if (chatBot.value && chatBot.value.resetChat) {
    chatBot.value.resetChat()
  }
})

const handleAgentsUpdate = (tools: ToolCall[]) => {
  agentTools.value = tools
}

const handleFinalAnswerUpdate = (answer: string) => {
  finalAnswer.value = answer
}

const handleFinalInformationsUpdate = (informations: string[]) => {
  finalInformations.value = informations
}

// Fetch MLflow experiment URL on mount
onMounted(async () => {
  try {
    const response = await fetch('/api/mlflow-experiment')
    if (response.ok) {
      const data = await response.json()
      mlflowUrl.value = data.mlflow_url
    }
  } catch (error) {
    console.error('Failed to fetch MLflow experiment URL:', error)
  }
})
</script>

<style>
:root {
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body {
  height: 100%;
}

.v-application {
  background-color: #f5f7fa;
}

.fill-height {
  height: 100%;
}

.title-section {
  flex-grow: 1;
  min-width: 0; /* Allow content to shrink */
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  margin-left: 16px;
}

.mlflow-link {
  text-transform: none !important;
  font-weight: 500;
  letter-spacing: normal;
  font-style: italic;
  color: #43ADE8 !important;
  font-size: 0.95rem !important;
}

.mlflow-link:hover {
  background-color: rgba(67, 173, 232, 0.08);
  color: #2196F3 !important;
}

.mlflow-link .v-icon {
  color: #43ADE8 !important;
}

.intelligence-toggle {
  flex-shrink: 0;
  border-radius: 8px;
}

.customer-selector-container {
  max-width: 300px;
}

/* Responsive design adjustments */
@media (max-width: 600px) {
  .v-card-title {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .header-controls {
    margin-left: 0;
    margin-top: 12px;
    align-self: flex-end;
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  
  .mlflow-link {
    padding: 4px 8px;
    font-size: 0.875rem;
  }
  
  .intelligence-toggle {
    margin-left: 0;
  }
  
  .customer-selector-container {
    max-width: 100%;
    order: -1; /* Show customer selector first on mobile */
    margin-bottom: 16px;
    margin-top: 0;
  }
  
  .title-section {
    width: 100%;
  }
}

@media (max-width: 960px) {
  .customer-selector-container {
    max-width: 250px;
  }
}

/* Ensure proper spacing in the header area */
.v-card-title {
  gap: 16px;
}

.v-card-title .title-section h1 {
  line-height: 1.2;
  margin-bottom: 4px;
}

.v-card-title .title-section p {
  line-height: 1.3;
  margin-bottom: 0;
}

/* Customer selector specific styling */
.customer-selector-container :deep(.v-select) {
  margin-bottom: 0;
}

.customer-selector-container :deep(.v-field) {
  background-color: rgba(0, 0, 0, 0.02);
}

.customer-selector-container :deep(.v-field:hover) {
  background-color: rgba(0, 0, 0, 0.04);
}

/* Intelligence toggle styling */
.intelligence-toggle .v-switch {
  margin: 0;
}

.intelligence-toggle .v-switch :deep(.v-switch__track) {
  opacity: 1;
}

/* Ensure consistent spacing */
@media (min-width: 601px) {
  .v-card-title {
    align-items: flex-start;
  }
  
  .header-controls {
    margin-top: 0;
  }
}

/* Loading states and transitions */
.customer-selector-container {
  transition: all 0.3s ease;
}

/* Focus and interaction states */
.customer-selector-container :deep(.v-field--focused) {
  background-color: rgba(79, 109, 245, 0.08);
}

/* Ensure proper text contrast */
.customer-selector-container :deep(.v-field__input) {
  color: rgba(0, 0, 0, 0.87);
}

.customer-selector-container :deep(.v-field__prepend-inner) {
  color: rgba(0, 0, 0, 0.6);
}

/* Handle very small screens */
@media (max-width: 400px) {
  .v-card-title {
    padding: 12px 16px 0 16px;
  }
  
  .customer-selector-container {
    margin-top: 8px;
    margin-bottom: 12px;
  }
  
  .header-controls {
    margin-top: 8px;
    gap: 8px;
  }
  
  .mlflow-link {
    padding: 2px 6px;
    font-size: 0.75rem;
  }
  
  .intelligence-toggle .d-flex {
    font-size: 0.875rem;
  }
}

/* Improve visual hierarchy */
.title-section h1 {
  color: rgba(0, 0, 0, 0.87);
  font-weight: 600;
}

.title-section p {
  color: rgba(0, 0, 0, 0.6);
}

/* Animation for customer selector appearance */
.customer-selector-container {
  opacity: 1;
  transform: translateY(0);
  transition: opacity 0.3s ease, transform 0.3s ease;
}

/* Enhance the intelligence toggle appearance */
.intelligence-toggle {
  background-color: rgba(0, 0, 0, 0.02);
  padding: 8px 12px;
  border-radius: 8px;
  transition: background-color 0.2s ease;
}

.intelligence-toggle:hover {
  background-color: rgba(0, 0, 0, 0.04);
}

/* Ensure proper alignment in all screen sizes */
@media (min-width: 1280px) {
  .customer-selector-container {
    max-width: 350px;
  }
}
</style>