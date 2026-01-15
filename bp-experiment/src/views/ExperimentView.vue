<script setup>
import { computed, onMounted, ref } from "vue";
import { submitResult } from "../services/api";

// --------------- Nastavení experimentu ---------------
// UPRAV SI podle toho, co budeš mít v public/glyphs/<name>/
const GLYPH_TYPES = [
  // příklady – nahraď svými složkami:
  "sun",
  "tree_growth",
  "flower",
  "circular_progressbar",
  "beer",
  "candle",
  "ripple_wave",
  // + další simple:
  //"line",
  //"square",
  //"circle",
  //"star",
  //"polygon",
];

// kolikrát opakovat každý glyph
const REPEATS_PER_GLYPH = 20;

// rozdíl mezi A a C
const X_MIN = 10;
const X_MAX = 50;

// biny pro "kotvu" (aby byly rovnoměrně)
const BINS = [
  [1, 33],
  [34, 66],
  [67, 100],
];

// prefetch okno pro plynulé kolečko
const PREFETCH_RADIUS = 10;

// --------------- Stav ---------------
const sessionId = ref("");

const trials = ref([]); // { glyphType, sizeA, sizeC }
const trialIndex = ref(0); // 0-based
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

// rovnoměrné rozložení binů: vytvoříme seznam bin indexů stejné délky jako trials
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

/**
 * Vygeneruje jednu dvojici (A,C) tak, aby:
 * - kotva (min nebo max) byla rovnoměrně v určeném binu
 * - rozdíl x byl v [X_MIN, X_MAX] a druhá hodnota zůstala v [1,100]
 */
function generatePairForBin(binIdx) {
  const [lo, hi] = BINS[binIdx];

  // 50/50: kotva bude minimum nebo maximum z (A,C)
  const anchorIsMin = Math.random() < 0.5;

  // anchor rovnoměrně z binu
  const anchor = lo + Math.floor(Math.random() * (hi - lo + 1));

  // povolený rozsah x podle toho, jestli jdeme nahoru nebo dolů
  const maxX = anchorIsMin ? Math.min(X_MAX, 100 - anchor) : Math.min(X_MAX, anchor - 1);
  const minX = Math.min(X_MIN, maxX); // když je maxX menší než X_MIN (blízko hranic), zmenšíme min

  // když by maxX bylo < 1, je to špatně (bin je moc u kraje); zkusíme znovu (v praxi to nastane málokdy)
  if (maxX < 1) return null;

  const x = minX + Math.floor(Math.random() * (maxX - minX + 1));
  const other = anchorIsMin ? anchor + x : anchor - x;

  let a = anchorIsMin ? anchor : other;
  let c = anchorIsMin ? other : anchor;

  // náhodně prohodit A/C, aby nebylo vždy A menší a C větší
  if (Math.random() < 0.5) [a, c] = [c, a];

  return { sizeA: a, sizeC: c };
}

function generateTrials() {
  // order glyphů: každý N×
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

    // robustní generace: zkusíme párkrát, než vzdáme
    let pair = null;
    for (let tries = 0; tries < 50; tries++) {
      pair = generatePairForBin(binIdx);
      if (pair) break;
    }
    if (!pair) {
      // fallback (extrémně vzácné)
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

const dpi = ref(96);

// --------------- Computed ---------------
const totalTrials = computed(() => trials.value.length);
const current = computed(() => trials.value[trialIndex.value] || null);

const counterText = computed(() => {
  if (!totalTrials.value) return "";
  return `${Math.min(trialIndex.value + 1, totalTrials.value)} / ${totalTrials.value}`;
});

const imgA = computed(() => (current.value ? glyphUrl(current.value.glyphType, current.value.sizeA) : ""));
const imgB = computed(() => (current.value ? glyphUrl(current.value.glyphType, sizeB.value) : ""));
const imgC = computed(() => (current.value ? glyphUrl(current.value.glyphType, current.value.sizeC) : ""));

function makeSessionId() {
  // 1) moderní prohlížeče (secure contexts)
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();

  // 2) fallback: crypto.getRandomValues (většinou funguje i na http)
  if (window.crypto?.getRandomValues) {
    const buf = new Uint8Array(16);
    window.crypto.getRandomValues(buf);
    return [...buf].map(b => b.toString(16).padStart(2, "0")).join("");
  }

  // 3) poslední fallback
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

// --------------- Lifecycle ---------------
onMounted(() => {
  try
  {
    sessionId.value = makeSessionId();

    trials.value = generateTrials();
    trialIndex.value = 0;
    sizeB.value = 1;

    // Prefetch prvního trialu
    if (current.value) {
      prefetch(current.value.glyphType, current.value.sizeA);
      prefetch(current.value.glyphType, current.value.sizeC);
      prefetchWindow(current.value.glyphType, sizeB.value);
    }

    const saved = Number(getCookie("dpi"));
    dpi.value = (Number.isFinite(saved) && saved > 30 && saved < 2000) ? Math.round(saved) : 96;
    document.documentElement.style.setProperty("--glyph-size", `${dpi.value}px`);
  }
  catch (e) {
    errorMsg.value = e?.message || String(e);
  }
});


// --------------- UI Actions ---------------
// --- Touch / Pointer ovládání pro B ---
const dragging = ref(false);
let startY = 0;
let startValue = 1;

const PX_PER_STEP = 12; // citlivost: kolik px = 1 krok (uprav dle pocitu)

function onPointerDownB(e) {
  // jen levé tlačítko myši; touch/stylus tím neblokuj
  if (e.pointerType === "mouse" && e.button !== 0) return;

  dragging.value = true;
  startY = e.clientY;
  startValue = sizeB.value;

  // capture = i když ujedeš mimo obrázek, pořád to táhne
  e.currentTarget?.setPointerCapture?.(e.pointerId);

  // zabrání scrollu / selection
  e.preventDefault();
}

function onPointerMoveB(e) {
  if (!dragging.value) return;

  const dy = startY - e.clientY; // nahoru = plus
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
  const delta = e.deltaY < 0 ? 1 : -1; // wheel up = +1
  sizeB.value = clampInt(sizeB.value + delta, 1, 100);
  if (current.value) prefetchWindow(current.value.glyphType, sizeB.value);
}

async function onNext() {
  errorMsg.value = "";
  if (!current.value) return;

  const payload = {
    session_id: sessionId.value,
    index: trialIndex.value + 1, // 1-based
    glyph_type: current.value.glyphType,
    sizeA: current.value.sizeA,
    sizeB: sizeB.value,
    sizeC: current.value.sizeC,
  };

  isSubmitting.value = true;
  try {
    await submitResult(payload);

    // posun na další trial
    trialIndex.value += 1;

    if (trialIndex.value >= trials.value.length) {
      done.value = true;
      return;
    }

    // reset B
    sizeB.value = 1;

    // prefetch pro nový trial
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
  if (confirm("Opravdu chcete ukončit experiment?")) {
    done.value = true;
  }
}
</script>

<template>
    <section>
        <h1>Experiment</h1>

        <div v-if="done" class="done">
        <h2>Hotovo ✅</h2>
        <p>
            Děkuji! Výsledky se ukládají do CSV souboru pro tuto session.
        </p>
        <p>
            <strong>Session ID:</strong> <code>{{ sessionId }}</code>
        </p>
        </div>

        <template v-else>
        <p v-if="dpi === 96" class="warn">
          Pro přesnou fyzickou velikost glyphů (1x1 inch) doporučuji provést
          <RouterLink to="/calibration">kalibraci monitoru</RouterLink>.
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
              title="Kolečko myši / táhni prstem nahoru-dolů"
            />
            </div>

            <div class="glyphCol">
            <div class="label">C</div>
            <img class="glyph" :src="imgC" alt="Glyph C" />
            </div>
        </div>

        <div class="actions">
            <button class="btn secondary" @click="onExit">
                Ukončit experiment
            </button>

            <button class="btn" :disabled="isSubmitting || !current" @click="onNext">
            {{ isSubmitting ? "Ukládám..." : "Next" }}
            </button>
        </div>

        <p v-if="errorMsg" class="error">
            <strong>Chyba:</strong> {{ errorMsg }}
        </p>
        </template>
    </section>
</template>

<style scoped>
.counter {
    text-align: center;
    font-weight: 700;
    margin: 10px 0 14px;
}

.row {
    display: flex;
    justify-content: center;
    gap: 18px;
    align-items: start;
    flex-wrap: wrap;
}

.glyphCol {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    min-width: 220px;
}

.label {
    font-weight: 700;
}

.glyph {
    width: var(--glyph-size, 96px);
    height: var(--glyph-size, 96px);
    border: 1px solid #ddd;
    border-radius: 14px;
    background: #fff;
}

.glyphB {
  cursor: ns-resize;
  touch-action: none;      /* klíčové: zabrání scrollování při dragu */
  user-select: none;
  -webkit-user-drag: none;
}

.glyphB.dragging {
  outline: 2px solid #111;
  outline-offset: 2px;
}

.hint {
    font-size: 12px;
    color: #666;
}

.actions {
    display: flex;
    justify-content: center;
    margin-top: 14px;
}

.btn {
    padding: 10px 16px;
    border-radius: 12px;
    border: 1px solid #111;
    background: #111;
    color: #fff;
    cursor: pointer;
}
.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.btn.secondary {
    background: #fff;
    color: #111;
    margin-right: 1px;
}
.error {
    margin-top: 12px;
    color: #b00020;
    text-align: center;
}

.done {
    padding: 14px;
    border: 1px solid #ddd;
    border-radius: 12px;
}
code {
    background: #f3f3f3;
    padding: 2px 6px;
    border-radius: 8px;
}

.warn{
  text-align:center;
  background:#fff8db;
  border:1px solid #f2c200;
  padding:10px 12px;
  border-radius:12px;
  margin: 10px auto 12px;
  max-width: 680px;
}
</style>
