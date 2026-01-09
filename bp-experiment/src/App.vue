<template>
  <div class="page">
    <header class="header">
      <div class="brand">
        <div class="title">Bakalářská práce – Experiment</div>
        <div class="subtitle">FIT VUT</div>
      </div>

      <nav class="nav">
        <RouterLink to="/">O práci</RouterLink>
        <RouterLink to="/experiment">Spustit pokus</RouterLink>
      </nav>
    </header>

    <main class="main">
      <RouterView />
    </main>

    <footer class="footer">
      <small>© {{ new Date().getFullYear() }} • FIT VUT</small>
    </footer>
  </div>
</template>

<style scoped>
.page { max-width: 900px; margin: 0 auto; padding: 24px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.header { display: flex; justify-content: space-between; align-items: end; gap: 16px; padding-bottom: 16px; border-bottom: 1px solid #ddd; }
.title { font-weight: 700; font-size: 18px; }
.subtitle { font-size: 13px; color: #666; margin-top: 4px; }
.nav { display: flex; gap: 12px; }
.nav a { text-decoration: none; padding: 6px 10px; border-radius: 8px; color: #111; }
.nav a.router-link-active { background: #eee; }
.main { padding: 20px 0; }
.footer { padding-top: 16px; border-top: 1px solid #ddd; color: #666; }
</style>
