<script setup>
import { RouterLink } from 'vue-router';
import { useI18n } from 'vue-i18n';

const { t } = useI18n();
</script>

<template>
  <section class="home">
    <h1>{{ t('home_title') }}</h1>

    <p>
      {{ t('home_description') }}
      <strong>{{ t('thesis_topic') }}</strong>
    </p>

    <h2>{{ t('experiment_how_title') }}</h2>
    <ol>
      <li>{{ t('experiment_how_step1') }}</li>
      <li>{{ t('experiment_how_step2') }}</li>
      <li>
        {{ t('experiment_how_step3') }}
        <strong>{{ t('experiment_how_step3_bold') }}</strong>
      </li>
      <li>{{ t('experiment_how_step4') }}</li>
    </ol>

    <div class="calib">
      <p>{{ t('experiment_calibration') }}</p>
      <RouterLink class="btn secondary" to="/calibration">
        {{ t('calibration_button') }}
      </RouterLink>
    </div>

    <h2>{{ t('time_required') }}</h2>
    <p>{{ t('time_required_description') }}</p>
    <p><strong>{{ t('thank_you') }}</strong></p>

    <h2>{{ t('more_info') }}</h2>
    <p>{{ t('more_info_description') }}</p>

    <ul class="links">
      <li>
        <a
          href="https://tmgc.fit.vutbr.cz/#what-is-a-malleable-glyph"
          target="_blank"
          rel="noopener"
        >
          The Malleable Glyph Challenge
        </a>
      </li>
      <li>
        <a
          href="https://github.com/adamherout/mglyph"
          target="_blank"
          rel="noopener"
        >
          {{ t('github_repository') }}
        </a>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.home {
  max-width: 720px;
  line-height: 1.65;
  color: #222;
}

h1 {
  margin: 0 0 14px;
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
}

h2 {
  margin: 28px 0 10px;
  font-size: 19px;
  font-weight: 600;
  line-height: 1.3;
}

p,
li {
  font-size: 15px;
}

p {
  margin: 0 0 14px;
}

ol,
ul {
  padding-left: 22px;
  margin: 0 0 16px;
}

li {
  margin-bottom: 8px;
}

strong {
  font-weight: 600;
}

.calib {
  margin: 22px 0;
  padding: 16px 18px;
  background: #f6fbf2;
  border: 1px solid #d8eac8;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.calib p {
  margin-bottom: 12px;
}

.btn {
  display: inline-block;
  padding: 10px 18px;
  border-radius: 12px;
  border: 1px solid #111;
  background: #111;
  color: #fff;
  text-decoration: none;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn:hover {
  background: #222;
  transform: translateY(-1px);
}

.btn:active {
  transform: translateY(0);
  background: #000;
}

.btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.15);
}

.btn.secondary {
  background: #fff;
  color: #111;
  border: 1px solid #d6d6d6;
}

.btn.secondary:hover {
  background: #f7f7f7;
}

.links a {
  color: #111;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.links a:hover {
  color: #444;
}

@media (max-width: 640px) {
  .home {
    max-width: 100%;
  }

  h1 {
    font-size: 24px;
  }

  h2 {
    font-size: 18px;
  }

  .calib {
    padding: 14px;
  }

  .btn {
    width: 100%;
    text-align: center;
    box-sizing: border-box;
  }
}
</style>