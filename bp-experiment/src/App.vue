<script setup>
import { useI18n } from 'vue-i18n';
import { RouterLink, RouterView } from 'vue-router';
import LangSwitcher from './components/LangSwitcher.vue';

const { t } = useI18n();
</script>

<template>
  <div class="page">
    <header class="header">
      <div class="brand">
        <div class="title">{{ t('thesis_title') }}</div>
        <div class="subtitle">{{ t('thesis_subtitle') }}</div>
      </div>

      <div class="right">
        <nav class="nav">
          <RouterLink to="/">{{ t('nav_about') }}</RouterLink>
          <RouterLink to="/experiment">{{ t('nav_experiment') }}</RouterLink>
        </nav>

        <div class="langWrap">
          <LangSwitcher />
        </div>
      </div>
    </header>

    <main class="main">
      <RouterView />
    </main>

    <footer class="footer">
      <small>© {{ new Date().getFullYear() }} • FIT VUT Brno • xprova06@stud.fit.vut.cz</small>
    </footer>
  </div>
</template>

<style scoped>
.page {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #ddd;
}

.brand {
  min-width: 0;
}

.right {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}

.nav {
  display: flex;
  gap: 12px;
  align-items: center;
}

.title {
  font-weight: 700;
  font-size: 18px;
}

.subtitle {
  font-size: 13px;
  color: #666;
  margin-top: 2px;
}

.nav a {
  text-decoration: none;
  padding: 6px 10px;
  border-radius: 8px;
  color: #111;
  white-space: nowrap;
}

.nav a.router-link-active {
  background: #eee;
}

.nav a:hover {
  background: #f2f2f2;
}

.langWrap {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.main {
  padding: 20px 0;
}

.footer {
  padding-top: 16px;
  border-top: 1px solid #ddd;
  color: #666;
}

@media (max-width: 640px) {
  .page {
    padding: 16px;
  }

  .header {
    flex-direction: column;
    align-items: stretch;
    gap: 14px;
  }

  .right {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    width: 100%;
  }

  .nav {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }

  .langWrap {
    flex-shrink: 0;
  }

  .footer {
    font-size: 13px;
  }
}
</style>