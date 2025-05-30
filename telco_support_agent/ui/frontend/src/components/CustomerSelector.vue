<template>
  <div class="customer-selector">
    <v-select
      v-model="selectedCustomer"
      :items="customers"
      item-title="display_name"
      item-value="customer_id"
      label="Select Customer"
      variant="outlined"
      density="compact"
      prepend-inner-icon="mdi-account"
      :loading="loading"
      @update:model-value="handleCustomerChange"
    >
      <template v-slot:item="{ props: itemProps, item }">
        <v-list-item v-bind="itemProps">
          <template v-slot:prepend>
            <v-avatar size="32" color="primary">
              <span class="text-white text-caption">
                {{ item.raw.customer_id.replace('CUS-', '') }}
              </span>
            </v-avatar>
          </template>
          <v-list-item-title>{{ item.raw.display_name }}</v-list-item-title>
          <v-list-item-subtitle>{{ item.raw.customer_id }}</v-list-item-subtitle>
        </v-list-item>
      </template>
      
      <template v-slot:selection="{ item }">
        <div class="d-flex align-center">
          <v-avatar size="24" color="primary" class="mr-2">
            <span class="text-white text-caption">
              {{ item.raw.customer_id.replace('CUS-', '') }}
            </span>
          </v-avatar>
          <span>{{ item.raw.display_name }}</span>
        </div>
      </template>
    </v-select>
    
    <div v-if="selectedCustomer" class="customer-info mt-2">
      <v-chip size="small" color="success" variant="outlined">
        <v-icon start>mdi-check-circle</v-icon>
        Customer: {{ selectedCustomer }}
      </v-chip>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { getDemoCustomers, type CustomerInfo } from '@/services/api';

interface Props {
  modelValue?: string;
}

interface Emits {
  (e: 'update:modelValue', value: string): void;
  (e: 'customer-change', customerId: string): void;
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: 'CUS-10001'
});

const emit = defineEmits<Emits>();

const selectedCustomer = ref<string>(props.modelValue);
const customers = ref<CustomerInfo[]>([]);
const loading = ref(false);

const loadCustomers = async () => {
  loading.value = true;
  try {
    customers.value = await getDemoCustomers();
    
    // If no customer is selected, select the first one
    if (!selectedCustomer.value && customers.value.length > 0) {
      selectedCustomer.value = customers.value[0].customer_id;
      handleCustomerChange(selectedCustomer.value);
    }
  } catch (error) {
    console.error('Error loading customers:', error);
  } finally {
    loading.value = false;
  }
};

const handleCustomerChange = (customerId: string) => {
  selectedCustomer.value = customerId;
  emit('update:modelValue', customerId);
  emit('customer-change', customerId);
};

onMounted(() => {
  loadCustomers();
});
</script>

<style scoped>
.customer-selector {
  min-width: 250px;
}

.customer-info {
  display: flex;
  justify-content: center;
}

/* Responsive adjustments */
@media (max-width: 600px) {
  .customer-selector {
    min-width: 100%;
  }
}
</style>