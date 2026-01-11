<script setup>
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();

// ----------------- stav -----------------
const inchPx = ref(96); // výchozí odhad (CSS inch)
const dpi = computed(() => Math.round(inchPx.value));

// ----------------- cookie helpers -----------------
function setCookie(name, value, days = 365) {
  const maxAge = days * 24 * 60 * 60;
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAge}; SameSite=Strict`;
}
function getCookie(name) {
  const m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
  return m ? decodeURIComponent(m[2]) : null;
}

// ----------------- lifecycle -----------------
onMounted(() => {
  const saved = Number(getCookie("dpi"));
  if (Number.isFinite(saved) && saved > 30 && saved < 2000) {
    inchPx.value = saved;
  }
});

// ----------------- akce -----------------
function saveCalibration() {
  setCookie("dpi", dpi.value);
  document.documentElement.style.setProperty("--glyph-size", `${dpi.value}px`);
  router.push("/experiment");
}

function reset() {
  inchPx.value = 96;
}
</script>

<template>
  <section class="wrap">
    <h1>Kalibrace velikosti (1 inch)</h1>

    <p class="lead">
      Upravte velikost čtverce tak, aby jeho strana odpovídala
      <b>1 palci (2.54&nbsp;cm)</b> na vašem pravítku.
      Díky tomu budou glyphy vykreslovány ve správné fyzické velikosti.
    </p>

    <div class="calibArea">
      <div
        class="inchSquare"
        :style="{ width: inchPx + 'px', height: inchPx + 'px' }"
      >
        1 inch<br>(2.54 cm)
      </div>

      <div class="stats">
        <div><b>Velikost:</b> {{ inchPx }} px</div>
        <div><b>DPI:</b> {{ dpi }}</div>
      </div>
    </div>

    <label class="sliderLabel">
      Nastavení velikosti:
      <input
        type="range"
        min="50"
        max="600"
        step="1"
        v-model.number="inchPx"
      />
    </label>

    <div class="actions">
      <button class="btn secondary" @click="reset">Reset</button>
      <button class="btn" @click="saveCalibration">Uložit a pokračovat</button>
    </div>
  </section>
</template>

<style scoped>
.wrap {
  max-width: 720px;
  margin: 0 auto;
  padding: 16px;
}

.lead {
  margin: 10px 0 16px;
  line-height: 1.5;
}

.calibArea {
  display: flex;
  gap: 20px;
  align-items: center;
  flex-wrap: wrap;
  margin: 16px 0;
}

.inchSquare {
  border: 2px solid #111;
  background: #f3f3f3;
  display: grid;
  place-items: center;
  font-weight: 700;
  text-align: center;
  border-radius: 12px;
  user-select: none;
}

.stats {
  font-size: 14px;
}

.sliderLabel {
  display: block;
  margin: 14px 0 18px;
}

input[type="range"] {
  width: 100%;
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.btn {
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid #111;
  background: #111;
  color: #fff;
  cursor: pointer;
}

.btn.secondary {
  background: #fff;
  color: #111;
}
</style>
