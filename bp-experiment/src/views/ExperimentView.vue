<script setup>
import { computed, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { submitResult } from "../services/api";
import { useI18n } from "vue-i18n";

const { t } = useI18n();

// --------------- Nastavení experimentu ---------------
const GLYPH_TYPES = [
  "sun",
  "tree_growth",
  "flower",
  "circular_progressbar",
  "beer",
  "candle",
  "ripple_wave",
];

const REPEATS_PER_GLYPH = 20;
const X_MIN = 10;
const X_MAX = 50;

const BINS = [
  [1, 33],
  [34, 66],
  [67, 100],
];

const PREFETCH_RADIUS = 10;

// --------------- Stav ---------------
const sessionId = ref("");
const trials = ref([]);
const trialIndex = ref(0);
const sizeB = ref(1);

const isSubmitting = ref(false);
const errorMsg = ref("");
const done = ref(false);

// --------------- Helpers ---------------
function clampInt(v, lo, hi) {
  v = Math.round(Number(v));
  if (Number.isNaN(v)) v = lo;
  return Math.max(lo, Math.min(hi, v));
}

function glyphUrl(type, value) {
  return `glyphs/${type}/${value}.png`;
}

function prefetch(type, value) {
  if (!type) return;
  const v = clampInt(value, 1, 100);
  const img = new Image();
  img.src = glyphUrl(type, v);
}

function prefetchWindow(type, center, radius = PREFETCH_RADIUS) {
  const c = clampInt(center, 1, 100);
  const from = Math.max(1, c - radius);
  const to = Math.min(100, c + radius);
  for (let v = from; v <= to; v++) prefetch(type, v);
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function makeBalancedBinOrder(n) {
  const bins = [];
  const base = Math.floor(n / BINS.length);
  const rem = n % BINS.length;

  for (let i = 0; i < BINS.length; i++) {
    const count = base + (i < rem ? 1 : 0);
    for (let k = 0; k < count; k++) bins.push(i);
  }
  return shuffleInPlace(bins);
}

function generatePairForBin(binIdx) {
  const [lo, hi] = BINS[binIdx];
  const anchorIsMin = Math.random() < 0.5;
  const anchor = lo + Math.floor(Math.random() * (hi - lo + 1));

  const maxX = anchorIsMin
    ? Math.min(X_MAX, 100 - anchor)
    : Math.min(X_MAX, anchor - 1);

  const minX = Math.min(X_MIN, maxX);

  if (maxX < 1) return null;

  const x = minX + Math.floor(Math.random() * (maxX - minX + 1));
  const other = anchorIsMin ? anchor + x : anchor - x;

  let a = anchorIsMin ? anchor : other;
  let c = anchorIsMin ? other : anchor;

  if (Math.random() < 0.5) [a, c] = [c, a];

  return { sizeA: a, sizeC: c };
}

function generateTrials() {
  const order = [];
  for (const g of GLYPH_TYPES) {
    for (let i = 0; i < REPEATS_PER_GLYPH; i++) order.push(g);
  }
  shuffleInPlace(order);

  const n = order.length;
  const binOrder = makeBalancedBinOrder(n);

  const out = [];
  for (let i = 0; i < n; i++) {
    const glyphType = order[i];
    const binIdx = binOrder[i];

    let pair = null;
    for (let tries = 0; tries < 50; tries++) {
      pair = generatePairForBin(binIdx);
      if (pair) break;
    }

    if (!pair) {
      const a = 1 + Math.floor(Math.random() * 100);
      let c = 1 + Math.floor(Math.random() * 100);
      if (c === a) c = clampInt(c + 1, 1, 100);
      pair = { sizeA: a, sizeC: c };
    }

    out.push({ glyphType, ...pair });
  }
  return out;
}

function getCookie(name) {
  const m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
  return m ? decodeURIComponent(m[2]) : null;
}

function makeSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();

  if (window.crypto?.getRandomValues) {
    const buf = new Uint8Array(16);
    window.crypto.getRandomValues(buf);
    return [...buf].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

const dpi = ref(96);

// --------------- Computed ---------------
const totalTrials = computed(() => trials.value.length);
const current = computed(() => trials.value[trialIndex.value] || null);

const counterText = computed(() => {
  if (!totalTrials.value) return "";
  return `${Math.min(trialIndex.value + 1, totalTrials.value)} / ${totalTrials.value}`;
});

const imgA = computed(() =>
  current.value ? glyphUrl(current.value.glyphType, current.value.sizeA) : ""
);
const imgB = computed(() =>
  current.value ? glyphUrl(current.value.glyphType, sizeB.value) : ""
);
const imgC = computed(() =>
  current.value ? glyphUrl(current.value.glyphType, current.value.sizeC) : ""
);

// --------------- Lifecycle ---------------
onMounted(() => {
  try {
    sessionId.value = makeSessionId();

    trials.value = generateTrials();
    trialIndex.value = 0;
    sizeB.value = 1;

    if (current.value) {
      prefetch(current.value.glyphType, current.value.sizeA);
      prefetch(current.value.glyphType, current.value.sizeC);
      prefetchWindow(current.value.glyphType, sizeB.value);
    }

    const saved = Number(getCookie("dpi"));
    dpi.value =
      Number.isFinite(saved) && saved > 30 && saved < 2000
        ? Math.round(saved)
        : 96;

    document.documentElement.style.setProperty("--glyph-size", `${dpi.value}px`);
  } catch (e) {
    errorMsg.value = e?.message || String(e);
  }
});

// --------------- UI Actions ---------------
const dragging = ref(false);
let startY = 0;
let startValue = 1;

const PX_PER_STEP = 12;

function onPointerDownB(e) {
  if (e.pointerType === "mouse" && e.button !== 0) return;

  dragging.value = true;
  startY = e.clientY;
  startValue = sizeB.value;

  e.currentTarget?.setPointerCapture?.(e.pointerId);
  e.preventDefault();
}

function onPointerMoveB(e) {
  if (!dragging.value) return;

  const dy = startY - e.clientY;
  const steps = Math.round(dy / PX_PER_STEP);

  sizeB.value = clampInt(startValue + steps, 1, 100);

  if (current.value) prefetchWindow(current.value.glyphType, sizeB.value);
  e.preventDefault();
}

function onPointerUpB(e) {
  dragging.value = false;
  e.preventDefault();
}

function onWheelB(e) {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 1 : -1;
  sizeB.value = clampInt(sizeB.value + delta, 1, 100);
  if (current.value) prefetchWindow(current.value.glyphType, sizeB.value);
}

async function onNext() {
  errorMsg.value = "";
  if (!current.value) return;

  const payload = {
    session_id: sessionId.value,
    index: trialIndex.value + 1,
    glyph_type: current.value.glyphType,
    sizeA: current.value.sizeA,
    sizeB: sizeB.value,
    sizeC: current.value.sizeC,
  };

  isSubmitting.value = true;
  try {
    await submitResult(payload);

    trialIndex.value += 1;

    if (trialIndex.value >= trials.value.length) {
      done.value = true;
      return;
    }

    sizeB.value = 1;

    if (current.value) {
      prefetch(current.value.glyphType, current.value.sizeA);
      prefetch(current.value.glyphType, current.value.sizeC);
      prefetchWindow(current.value.glyphType, sizeB.value);
    }
  } catch (err) {
    errorMsg.value = err?.message || String(err);
  } finally {
    isSubmitting.value = false;
  }
}

function onExit() {
  if (confirm(t('exit_confirm'))) {
    done.value = true;
  }
}
</script>

<template>
  <section class="experiment">
    <h1>Experiment</h1>

    <div v-if="done" class="done">
      <h2>{{ t('done') }}</h2>
      <p>{{ t('experiment_done_message') }}</p>
      <p><strong>Session ID:</strong> <code>{{ sessionId }}</code></p>
    </div>

    <template v-else>
      <p v-if="dpi === 96" class="warn">
        {{ t('calibration_warning') }}
        <RouterLink to="/calibration">{{ t('calibration_link') }}</RouterLink>.
      </p>

      <div class="counter">{{ counterText }}</div>

      <div v-if="current" class="row">
        <div class="glyphCol">
          <div class="label">A</div>
          <img class="glyph" :src="imgA" alt="Glyph A" />
        </div>

        <div class="glyphCol">
          <div class="label">B</div>
          <img
            class="glyph glyphB"
            :class="{ dragging }"
            :src="imgB"
            alt="Glyph B"
            @wheel="onWheelB"
            @pointerdown="onPointerDownB"
            @pointermove="onPointerMoveB"
            @pointerup="onPointerUpB"
            @pointercancel="onPointerUpB"
          />
          <div class="hint">{{ t('glyphB_hint') }}</div>
        </div>

        <div class="glyphCol">
          <div class="label">C</div>
          <img class="glyph" :src="imgC" alt="Glyph C" />
        </div>
      </div>

      <div class="actions">
        <button class="btn secondary" @click="onExit">
          {{ t('exit') }}
        </button>

        <button class="btn" :disabled="isSubmitting || !current" @click="onNext">
          {{ isSubmitting ? t('submitting') : t('next') }}
        </button>
      </div>

      <p v-if="errorMsg" class="error">
        <strong>{{ t('error') }}</strong> {{ errorMsg }}
      </p>
    </template>
  </section>
</template>

<style scoped>
.experiment {
  max-width: 720px;
  margin: 0 auto;
  line-height: 1.65;
  color: #222;
}

h1 {
  margin: 0 0 16px;
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
}

h2 {
  margin: 0 0 10px;
  font-size: 20px;
  font-weight: 600;
  line-height: 1.3;
}

p {
  margin: 0 0 14px;
  font-size: 15px;
}

.counter {
  margin: 18px 0 20px;
  text-align: center;
  font-size: 15px;
  font-weight: 700;
  color: #444;
}

.row {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 24px;
  flex-wrap: nowrap;
  margin-bottom: 24px;
}

.glyphCol {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  min-width: 180px;
}

.label {
  font-size: 15px;
  font-weight: 700;
  color: #111;
}

.glyph {
  width: var(--glyph-size, 96px);
  height: var(--glyph-size, 96px);
  border: 1px solid #ddd;
  border-radius: 14px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.glyphB {
  cursor: ns-resize;
  touch-action: none;
  user-select: none;
  -webkit-user-drag: none;
  transition: transform 0.15s ease, box-shadow 0.15s ease, outline-color 0.15s ease;
}

.glyphB:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.glyphB.dragging {
  outline: 2px solid #111;
  outline-offset: 3px;
}

.hint {
  font-size: 12px;
  color: #666;
  text-align: center;
  max-width: 140px;
  line-height: 1.4;
}

.actions {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.btn {
  display: inline-block;
  padding: 10px 18px;
  border-radius: 12px;
  border: 1px solid #111;
  background: #111;
  color: #fff;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn:hover:not(:disabled) {
  background: #222;
  transform: translateY(-1px);
}

.btn:active:not(:disabled) {
  transform: translateY(0);
  background: #000;
}

.btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.15);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.btn.secondary {
  background: #fff;
  color: #111;
  border: 1px solid #d6d6d6;
}

.btn.secondary:hover:not(:disabled) {
  background: #f7f7f7;
}

.warn {
  margin: 10px 0 18px;
  padding: 14px 16px;
  background: #fff8db;
  border: 1px solid #f0d36b;
  border-radius: 14px;
  font-size: 14px;
  color: #5f4b00;
  text-align: center;
}

.warn a {
  color: #111;
  font-weight: 600;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.error {
  margin-top: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: #fff1f3;
  border: 1px solid #f3c7cf;
  color: #b00020;
  text-align: center;
  font-size: 14px;
}

.done {
  margin-top: 8px;
  padding: 18px;
  border: 1px solid #dcdcdc;
  border-radius: 14px;
  background: #fafafa;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

code {
  background: #f3f3f3;
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 13px;
}

@media (max-width: 640px) {
  .experiment {
    max-width: 100%;
  }

  h1 {
    font-size: 24px;
  }

  h2 {
    font-size: 18px;
  }

.row {
    flex-direction: column;
    align-items: center;
  }

  .glyphCol {
    min-width: 140px;
  }

  .actions {
    flex-direction: column-reverse;
    align-items: stretch;
  }

  .btn {
    width: 100%;
    text-align: center;
  }
}
</style>