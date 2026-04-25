<template>
  <div class="dropdown" @click="toggle">
    <div class="selected">
      <span class="left">
        <span class="flag">{{ currentFlag }}</span>
        <span class="label">{{ currentLabel }}</span>
      </span>
      <span class="arrow">▼</span>
    </div>

    <div v-if="open" class="menu">
      <div
        v-for="lang in languages"
        :key="lang.value"
        class="item"
        @click.stop="select(lang.value)"
      >
        <span class="flag">{{ lang.flag }}</span>
        <span>{{ lang.label }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import { ref, computed, onMounted } from 'vue';

const { locale } = useI18n();

const open = ref(false);

const languages = [
  { value: 'en', label: 'EN', flag: '🇬🇧' },
  { value: 'cs', label: 'CS', flag: '🇨🇿' }
];

const toggle = () => {
  open.value = !open.value;
};

const select = (lang) => {
  locale.value = lang;
  localStorage.setItem('lang', lang);
  open.value = false;
};

const current = computed(() => {
  return languages.find(l => l.value === locale.value) || languages[0];
});

const currentLabel = computed(() => current.value.label);
const currentFlag = computed(() => current.value.flag);

onMounted(() => {
  const storedLang = localStorage.getItem('lang');
  if (storedLang) {
    locale.value = storedLang;
  }
});
</script>

<style scoped>
.dropdown {
  position: relative;
  width: 100px;      
  max-width: 100%;
}

.selected {
  padding: 8px 12px;
  background: #eee;
  border-radius: 10px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  box-sizing: border-box;
  width: 100%;
}

.left {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.menu {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  width: 100%;
  background: white;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  overflow: hidden;
  z-index: 20;
}

.item {
  padding: 8px 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
}

.item:hover {
  background: #f2f2f2;
}

.flag {
  font-size: 16px;
  flex-shrink: 0;
}

.arrow {
  font-size: 10px;
  flex-shrink: 0;
}

@media (max-width: 640px) {
  .selected,
  .menu,
  .item {
    width: 100%;
    box-sizing: border-box;
  }
}
</style>