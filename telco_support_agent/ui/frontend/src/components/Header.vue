<template>
  <v-app-bar color="white" elevation="1" height="64">
    <div class="d-flex align-center w-100 px-4">
      <!-- Logo and Title section -->
      <div class="d-flex align-center">
        <img 
          src="@/assets/databricks-logo.svg" 
          alt="Databricks" 
          height="32"
          class="mr-3 logo"
        />
        <div class="title-section">
          <h1 class="text-h6 font-weight-bold text-primary header-title">
            {{ title }}
          </h1>
          <p class="text-caption text-medium-emphasis ma-0 header-subtitle">
            Powered by Databricks Agent Framework
          </p>
        </div>
      </div>

      <v-spacer />

      <!-- Right side actions -->
      <div class="d-flex align-center actions-section">
        <v-tooltip location="bottom" text="Refresh Application">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-refresh"
              variant="text"
              color="primary"
              size="default"
              @click="handleRefresh"
              class="refresh-btn"
            />
          </template>
        </v-tooltip>
      </div>
    </div>
  </v-app-bar>
</template>

<script setup lang="ts">
interface Props {
  title: string;
}

interface Emits {
  (e: 'refresh'): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const handleRefresh = () => {
  emit('refresh');
};
</script>

<style scoped>
.v-app-bar {
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  backdrop-filter: blur(10px);
  background-color: rgba(255, 255, 255, 0.95) !important;
}

.logo {
  transition: transform 0.2s ease;
}

.logo:hover {
  transform: scale(1.05);
}

.title-section {
  min-width: 0; /* Allow text to wrap if needed */
}

.header-title {
  line-height: 1.2;
  margin-bottom: 2px;
  color: #1565c0;
  font-size: 1.25rem;
  font-weight: 600;
}

.header-subtitle {
  line-height: 1.2;
  color: rgba(0, 0, 0, 0.6);
  font-size: 0.75rem;
}

.actions-section {
  gap: 8px;
}

.refresh-btn {
  transition: all 0.2s ease;
}

.refresh-btn:hover {
  background-color: rgba(21, 101, 192, 0.08);
  transform: rotate(90deg);
}

/* Responsive design for different screen sizes */
@media (max-width: 960px) {
  .header-title {
    font-size: 1.1rem;
  }
  
  .header-subtitle {
    font-size: 0.7rem;
  }
  
  .logo {
    height: 28px;
    margin-right: 12px;
  }
}

@media (max-width: 600px) {
  .v-app-bar {
    height: 56px;
  }
  
  .px-4 {
    padding-left: 16px !important;
    padding-right: 16px !important;
  }
  
  .header-title {
    font-size: 1rem;
  }
  
  .header-subtitle {
    font-size: 0.65rem;
  }
  
  .logo {
    height: 24px;
    margin-right: 8px;
  }
  
  .refresh-btn {
    min-width: 40px !important;
    width: 40px;
    height: 40px;
  }
}

@media (max-width: 400px) {
  .header-title {
    font-size: 0.9rem;
  }
  
  .header-subtitle {
    display: none; /* Hide subtitle on very small screens */
  }
  
  .px-4 {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }
}

/* Enhanced visual effects */
.v-app-bar:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #1565c0, #42a5f5, #1565c0);
  background-size: 200% 100%;
  animation: shimmer 3s ease-in-out infinite;
  opacity: 0.6;
}

@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

/* Accessibility improvements */
.refresh-btn:focus {
  outline: 2px solid #1565c0;
  outline-offset: 2px;
}

/* Dark mode support (if needed later) */
.v-theme--dark .v-app-bar {
  background-color: rgba(18, 18, 18, 0.95) !important;
  border-bottom-color: rgba(255, 255, 255, 0.1);
}

.v-theme--dark .header-title {
  color: #90caf9;
}

.v-theme--dark .header-subtitle {
  color: rgba(255, 255, 255, 0.7);
}

.v-theme--dark .v-app-bar:before {
  background: linear-gradient(90deg, #90caf9, #64b5f6, #90caf9);
}

/* Loading state styles (if refresh takes time) */
.refresh-btn.loading {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* Ensure proper z-index for header */
.v-app-bar {
  z-index: 1005;
}

/* Smooth transitions for all interactive elements */
* {
  transition: color 0.2s ease, background-color 0.2s ease, transform 0.2s ease;
}

/* Print styles */
@media print {
  .v-app-bar {
    background-color: white !important;
    box-shadow: none !important;
    border-bottom: 1px solid #000;
  }
  
  .v-app-bar:before {
    display: none;
  }
  
  .actions-section {
    display: none;
  }
}
</style>