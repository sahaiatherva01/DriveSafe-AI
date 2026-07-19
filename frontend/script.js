/* ══════════════════════════════════════════════════════════════════════════════
   SentinelAI — script.js
   Live monitoring page (real-time) + Session Report page (post-session analysis)
   ══════════════════════════════════════════════════════════════════════════════ */

'use strict';

/* ── Config ──────────────────────────────────────────────────────────────────── */
const API_BASE      = 'http://localhost:5000';
const POLL_MS       = 300;    // status poll when running
const CHART_MS      = 3000;   // chart / analytics refresh
const SYS_MS        = 2000;   // system metrics (CPU/RAM)
const HEALTH_MS     = 3000;   // health check when stopped
const AGENT_MS      = 5000;   // AI agent refresh
const MAX_CHART_PTS = 200;    // max data points per chart (full session)

/* ── State ───────────────────────────────────────────────────────────────────── */
let isRunning      = false;
let pollTimer      = null;
let chartTimer     = null;
let sysTimer       = null;
let healthTimer    = null;
let agentTimer     = null;
let sessionStart   = null;
let sessionDate    = null;
let sessionId      = null;

// Session accumulators for the report
let earAccum       = [];
let marAccum       = [];
let attAccum       = [];
let fatAccum       = [];
let fpsAccum       = [];
let latencyAccum   = [];
let confAccum      = [];
let pitchAccum     = [];
let yawAccum       = [];
let rollAccum      = [];
let stabilityAccum = [];
let maxClosure     = 0;
let maxYawnDur     = 0;
let sessionEnd     = null;
let sessionMeta    = null;   // from /api/session_meta (frames processed etc.)

// Latest known snapshots (used both live + when building the report)
let lastStatusData = null;
let lastAgentData  = null;

// Timeline events ring buffer
const TL_MAX       = 40;
const tlEvents     = [];
let   lastStatus   = null;   // for edge detection (status change events)

/* ── DOM helpers ─────────────────────────────────────────────────────────────── */
const $  = id  => document.getElementById(id);
const el = sel => document.querySelector(sel);

/* ── Chart.js defaults ───────────────────────────────────────────────────────── */
Chart.defaults.color          = '#475569';
Chart.defaults.font.family    = "'Inter', sans-serif";
Chart.defaults.font.size      = 11;
Chart.defaults.plugins.legend.display = false;

const CHART_GRID = { color: 'rgba(255,255,255,0.05)', drawBorder: false };

function makeChart(id, datasets, yMin, yMax, labelFn) {
  const ctx = $(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets },
    options: {
      responsive: true,
      animation:  { duration: 200 },
      interaction: { intersect: false, mode: 'index' },
      scales: {
        x: { grid: CHART_GRID, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: {
          grid: CHART_GRID,
          min:  yMin, max: yMax,
          ticks: { callback: labelFn || (v => v) },
        },
      },
      plugins: {
        tooltip: {
          backgroundColor: 'rgba(12,16,32,0.95)',
          borderColor: 'rgba(0,212,255,0.25)',
          borderWidth: 1,
          padding: 8,
        },
      },
      elements: {
        point:  { radius: 0, hoverRadius: 4 },
        line:   { tension: 0.35, borderWidth: 2 },
      },
    },
  });
}

function chartLine(color, fill = false) {
  return {
    data: [],
    borderColor: color,
    backgroundColor: fill ? color.replace(')', ', 0.10)').replace('rgb', 'rgba') : 'transparent',
    fill,
  };
}

/* ── Initialise charts (live in DOM only inside report view) ──────────────────── */
const charts = {
  ear:    makeChart('chartEAR',    [chartLine('#00d4ff')], 0, 0.5),
  mar:    makeChart('chartMAR',    [chartLine('#f59e0b')], 0, 0.9),
  attfat: makeChart('chartAttFat',
    [
      { ...chartLine('#22c55e', true), label: 'Attention' },
      { ...chartLine('#ef4444', true), label: 'Fatigue'   },
    ], 0, 1, v => `${Math.round(v * 100)}%`),
  risk:   makeChart('chartRisk',   [chartLine('#7c3aed', true)], 0, 3,
            v => ['NORMAL','LOW','MED','HIGH'][Math.round(v)] || ''),
  blink:  makeChart('chartBlink',  [chartLine('#84cc16')], 0, 40),
  pose:   makeChart('chartPose',
    [
      { ...chartLine('#00d4ff'), label: 'Pitch' },
      { ...chartLine('#f59e0b'), label: 'Yaw'   },
    ], -60, 60),
};

// Re-enable legend only for multi-dataset charts
[charts.attfat, charts.pose].forEach(c => {
  c.options.plugins.legend.display = true;
  c.options.plugins.legend.labels  = { color: '#94a3b8', boxWidth: 10, padding: 12 };
  c.update();
});

function pushChart(chart, label, ...values) {
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > MAX_CHART_PTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(d => d.data.shift());
  }
  chart.update('none');
}

function resetCharts() {
  Object.values(charts).forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(ds => ds.data = []);
    c.update('none');
  });
}

/* ── Utilities ───────────────────────────────────────────────────────────────── */
function fmtTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function riskClass(risk) {
  return { NORMAL:'good', LOW:'good', MEDIUM:'warn', HIGH:'danger' }[risk] || '';
}

function setBar(id, pct, cls = '') {
  const node = $(id);
  if (!node) return;
  node.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  node.className   = `bar-fill${cls ? ' ' + cls : ''}`;
}

async function apiPost(ep) {
  const r = await fetch(`${API_BASE}${ep}`, { method: 'POST' });
  return r.json();
}
async function apiGet(ep) {
  const r = await fetch(`${API_BASE}${ep}`);
  return r.json();
}

/* ── Connection indicator ────────────────────────────────────────────────────── */
function setConn(state) {   // 'online' | 'offline' | 'error'
  const dot   = $('pillDot');
  const label = $('pillLabel');
  dot.className   = `pill-dot ${state}`;
  label.textContent = state === 'online' ? 'Connected'
                    : state === 'error'  ? 'Error'
                    : 'Offline';
}

/* ── View switching (Live / Report) ──────────────────────────────────────────── */
function setView(name) {
  $('viewLive').classList.toggle('hidden',   name !== 'live');
  $('viewReport').classList.toggle('hidden', name !== 'report');
  $('navLive').classList.toggle('active',   name === 'live');
  $('navReport').classList.toggle('active', name === 'report');
}

$('navLive').addEventListener('click', () => setView('live'));
$('navReport').addEventListener('click', () => {
  if (!$('navReport').disabled) setView('report');
});

/* ── Status ring + driver card ───────────────────────────────────────────────── */
const STATUS_ICONS = {
  awake:   `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>`,
  drowsy:  `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`,
  yawning: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 3 4 3 4-3 4-3"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>`,
  noface:  `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  offline: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
};

const STATUS_HINTS = {
  offline: 'Start detection to monitor',
  awake:   'Driver is alert and attentive',
  drowsy:  '⚠ Eyes closing — take a break!',
  yawning: 'Yawning detected — stay alert',
  noface:  'No face detected in frame',
};

function applyStatus(data) {
  lastStatusData = data;

  const raw  = (data.status || 'OFFLINE').toUpperCase();
  const key  = !data.running       ? 'offline'
             : raw === 'AWAKE'     ? 'awake'
             : raw === 'DROWSY'    ? 'drowsy'
             : raw === 'YAWNING'   ? 'yawning'
             : 'noface';

  // Status ring
  $('statusRing').className = `status-ring ${key === 'noface' ? 'noface' : key}`;

  // Orb
  const orb = $('statusOrb');
  orb.className = `status-orb s-${key}`;
  orb.innerHTML = STATUS_ICONS[key] || STATUS_ICONS.offline;

  // Label + hint
  const labels = { offline:'OFFLINE', awake:'AWAKE', drowsy:'DROWSY', yawning:'YAWNING', noface:'NO FACE' };
  $('driverStatusVal').textContent = labels[key];
  $('driverStatusVal').className   = `driver-status-val s-${key}`;
  $('driverStatusHint').textContent = STATUS_HINTS[key] || '';

  // Live badge
  const badge = $('liveBadge');
  if (key === 'awake' || key === 'drowsy' || key === 'yawning') {
    badge.textContent = 'LIVE';
    badge.className   = 'section-badge active';
  } else {
    badge.textContent = 'OFFLINE';
    badge.className   = 'section-badge';
  }

  // Alert card
  applyAlert(key);

  // Timeline event on status change
  if (data.running && key !== lastStatus) {
    addTlEvent(key, raw);
    lastStatus = key;
  }

  // Session timer chip
  const secs = data.session_seconds || 0;
  $('sessionTimerChip').textContent = fmtTime(secs);

  if (!data.running) return;

  // ── EAR ──────────────────────────────────────────────────────────────────
  const ear = data.ear || 0;
  $('valEAR').textContent = ear.toFixed(3);
  const earCls = ear < 0.20 ? 'danger' : ear < 0.25 ? 'warn' : 'good';
  $('valEAR').className   = `metric-big ${earCls}`;
  setBar('barEAR', (ear / 0.45) * 100, earCls === 'good' ? '' : earCls);
  earAccum.push(ear);

  // ── MAR ──────────────────────────────────────────────────────────────────
  const mar = data.mar || 0;
  $('valMAR').textContent = mar.toFixed(3);
  const marCls = mar > 0.65 ? 'danger' : mar > 0.55 ? 'warn' : 'good';
  $('valMAR').className   = `metric-big ${marCls}`;
  setBar('barMAR', (mar / 0.9) * 100, marCls === 'good' ? '' : marCls);
  marAccum.push(mar);

  // ── Attention ─────────────────────────────────────────────────────────────
  const att = data.attention_index || 0;
  $('valAttention').textContent = `${Math.round(att * 100)}%`;
  $('valAttention').className   = `metric-big ${att < 0.4 ? 'danger' : att < 0.6 ? 'warn' : 'good'}`;
  setBar('barAttention', att * 100, att < 0.4 ? 'danger' : att < 0.6 ? 'warn' : '');
  attAccum.push(att);

  // ── Fatigue ───────────────────────────────────────────────────────────────
  const fat = data.fatigue_index || 0;
  $('valFatigue').textContent = `${Math.round(fat * 100)}%`;
  $('valFatigue').className   = `metric-big ${fat > 0.6 ? 'danger' : fat > 0.35 ? 'warn' : 'good'}`;
  setBar('barFatigue', fat * 100, fat > 0.6 ? 'danger' : fat > 0.35 ? 'warn' : '');
  fatAccum.push(fat);

  // ── Blink rate ────────────────────────────────────────────────────────────
  const br = data.blink_rate || 0;
  $('valBlinkRate').textContent = `${br.toFixed(1)}/m`;
  $('valBlinkRate').className   = `metric-big ${br < 8 || br > 28 ? 'warn' : 'good'}`;

  // ── Eye state badge ───────────────────────────────────────────────────────
  const es = data.eye_state || 'N/A';
  $('badgeEye').textContent = es;
  $('badgeEye').className   = `metric-badge-lg ${es === 'Open' ? 'open' : es === 'Closed' ? 'closed' : ''}`;

  // ── Head pose chips ───────────────────────────────────────────────────────
  const pitch = data.pitch || 0, yaw = data.yaw || 0, roll = data.roll || 0;
  const poseCls = (v, t) => Math.abs(v) > t * 1.5 ? 'danger' : Math.abs(v) > t ? 'warn' : '';
  $('posePitch').textContent = `P ${pitch > 0 ? '+' : ''}${pitch.toFixed(0)}°`;
  $('posePitch').className   = `pose-chip ${poseCls(pitch, 15)}`;
  $('poseYaw').textContent   = `Y ${yaw   > 0 ? '+' : ''}${yaw.toFixed(0)}°`;
  $('poseYaw').className     = `pose-chip ${poseCls(yaw, 20)}`;
  $('poseRoll').textContent  = `R ${roll  > 0 ? '+' : ''}${roll.toFixed(0)}°`;
  $('poseRoll').className    = `pose-chip ${poseCls(roll, 20)}`;
  pitchAccum.push(pitch);
  yawAccum.push(yaw);
  rollAccum.push(roll);

  // ── Blink / yawn / microsleep counters ──────────────────────────────────
  $('cntBlinks').textContent  = data.blink_count || 0;
  $('cntYawnLive').textContent = data.yawn_count || data.yawn_events || 0;
  $('cntMicro').textContent   = data.microsleep_count || 0;

  // ── AI sources ────────────────────────────────────────────────────────────
  applyAISrc('aiRuleRisk',  'aiRuleBar',  data.rule_risk,  data.rule_confidence);
  applyAISrc('aiModelRisk', 'aiModelBar', data.model_risk, data.model_confidence);
  const tsVal = data.temporal_smoothed || 0;
  const tsRisk = ['NORMAL','LOW','MEDIUM','HIGH'][Math.min(3, Math.round(tsVal * 3))];
  applyAISrc('aiTempRisk',  'aiTempBar',  tsRisk, tsVal);

  const fr  = data.final_risk || 'NORMAL';
  $('aiFinalRisk').textContent = fr;
  $('aiFinalRisk').className   = `ai-final-risk ${riskClass(fr)}`;

  $('badgePT').className = `ai-badge${data.model_available  ? ' active' : ''}`;
  $('badgeTF').className = `ai-badge${data.tf_available     ? ' active' : ''}`;

  // Quick stats
  $('qFPS').textContent  = `${(data.fps || 0).toFixed(0)}`;
  $('qConf').textContent = `${Math.round((data.final_confidence || 0) * 100)}%`;
  $('qRisk').textContent = fr;
  $('qRisk').className   = `qstat-val ${riskClass(fr)}`;
  if (lastAgentData) {
    $('qSafety').textContent = `${lastAgentData.grade_score ?? 0}`;
  }

  // Longest closure / yawn tracking
  const lc = data.last_closure_dur || 0;
  if (lc > maxClosure) maxClosure = lc;
  const lyd = data.last_yawn_dur || 0;
  if (lyd > maxYawnDur) maxYawnDur = lyd;

  // Performance accumulators
  fpsAccum.push(data.fps || 0);
  latencyAccum.push(data.proc_ms || 0);
  confAccum.push(data.detection_confidence || 0);
  stabilityAccum.push(data.face_stability || 0);
}

function applyAISrc(riskId, barId, risk, conf) {
  const node = $(riskId);
  node.textContent = risk || '—';
  node.className   = `ai-src-risk ${riskClass(risk || 'NORMAL')}`;
  const pct = { NORMAL:10, LOW:35, MEDIUM:65, HIGH:95 }[risk] || 0;
  setBar(barId, pct, riskClass(risk || 'NORMAL'));
}

/* ── Alert card ──────────────────────────────────────────────────────────────── */
function applyAlert(key) {
  const card = $('alertCard');
  const text = $('alertText');
  const msgs = {
    drowsy:  '⚠ Drowsiness detected — please pull over and rest immediately!',
    yawning: 'Yawning detected — consider taking a short break.',
    noface:  'No face detected — please position yourself in frame.',
  };
  if (msgs[key]) {
    card.className   = `alert-card alert-${key}`;
    text.textContent = msgs[key];
  } else {
    card.className = 'alert-card hidden';
  }
}

/* ── Timeline ────────────────────────────────────────────────────────────────── */
function addTlEvent(key, label) {
  const now   = sessionStart ? Math.floor((Date.now() - sessionStart) / 1000) : 0;
  const msgs  = {
    awake:   'Driver alert — resumed normal state',
    drowsy:  '⚠ Drowsiness detected',
    yawning: 'Yawning detected',
    noface:  'Face lost — no detection',
    start:   'Detection session started',
    end:     'Session ended',
  };
  tlEvents.push({ key, label: msgs[key] || label, time: fmtTime(now) });
  if (tlEvents.length > TL_MAX) tlEvents.shift();
  renderTimeline();
}

function renderTimeline() {
  const ul = $('timeline');
  if (!tlEvents.length) {
    ul.innerHTML = '<li class="tl-empty">No events yet — start a session</li>';
    return;
  }
  ul.innerHTML = tlEvents.map(e => `
    <li class="tl-item ev-${e.key}">
      <span class="tl-time">${e.time}</span>
      <span>${e.label}</span>
    </li>
  `).join('');
}

/* ── System metrics (fetched, used for performance averages only) ─────────────── */
async function pollSystem() {
  try { await apiGet('/api/system'); } catch (_) {}
}

/* ── Charts poll ─────────────────────────────────────────────────────────────── */
async function pollCharts() {
  if (!isRunning) return;
  try {
    const d = await apiGet('/api/analytics');
    if (!d.running || !d.timeline?.length) return;

    const pts = d.timeline.slice(-MAX_CHART_PTS);
    resetCharts();

    pts.forEach(p => {
      const lbl = `${p.t.toFixed(0)}s`;
      pushChart(charts.ear,    lbl, p.ear);
      pushChart(charts.mar,    lbl, p.mar);
      pushChart(charts.attfat, lbl, p.attention, p.fatigue);
      pushChart(charts.risk,   lbl, p.risk_index);
      pushChart(charts.blink,  lbl, p.blink_rate);
      pushChart(charts.pose,   lbl, p.pitch, p.yaw);
    });
  } catch (_) {}
}

/* ── AI Agent ────────────────────────────────────────────────────────────────── */
async function pollAgent() {
  try {
    const d = await apiGet('/api/agent');
    lastAgentData = d;
  } catch (_) {}
}

function renderList(id, items) {
  const ul = $(id);
  if (!items.length) {
    ul.innerHTML = '<li class="agent-list-empty">No data yet</li>';
    return;
  }
  ul.innerHTML = items.map(s => `<li>${s}</li>`).join('');
}

/* ── Main status poll ────────────────────────────────────────────────────────── */
async function poll() {
  try {
    const data = await apiGet('/api/status');
    data.running = isRunning;
    applyStatus(data);
    setConn('online');
  } catch (_) {
    setConn('error');
  }
}

/* ── Health check (when stopped) ─────────────────────────────────────────────── */
async function healthCheck() {
  try {
    const d = await apiGet('/api/health');
    setConn(d.status === 'ok' ? 'online' : 'offline');
    if (d.running && !isRunning) autoResume();
  } catch (_) {
    setConn('offline');
  }
}

function autoResume() {
  isRunning    = true;
  sessionStart = Date.now();
  sessionDate  = new Date();
  sessionId    = makeSessionId();
  $('btnStop').disabled  = false;
  $('btnStart').disabled = true;
  startVideoFeed();
  startPolling();
  stopHealthCheck();
}

/* ── Timers ──────────────────────────────────────────────────────────────────── */
function startPolling() {
  if (pollTimer)  return;
  pollTimer  = setInterval(poll,        POLL_MS);
  chartTimer = setInterval(pollCharts,  CHART_MS);
  sysTimer   = setInterval(pollSystem,  SYS_MS);
  agentTimer = setInterval(pollAgent,   AGENT_MS);
}
function stopPolling() {
  [pollTimer, chartTimer, sysTimer, agentTimer].forEach(clearInterval);
  pollTimer = chartTimer = sysTimer = agentTimer = null;
}
function startHealthCheck() {
  if (healthTimer) return;
  healthCheck();
  healthTimer = setInterval(healthCheck, HEALTH_MS);
}
function stopHealthCheck() {
  clearInterval(healthTimer);
  healthTimer = null;
}

/* ── Camera feed ─────────────────────────────────────────────────────────────── */
function startVideoFeed() {
  const img = $('videoFeed');
  img.src = `${API_BASE}/video_feed?t=${Date.now()}`;
  img.onload = () => {
    img.classList.add('visible');
    $('camPlaceholder').classList.add('hidden');
  };
  img.onerror = () => img.classList.remove('visible');
}
function stopVideoFeed() {
  const img = $('videoFeed');
  img.src   = '';
  img.classList.remove('visible');
  $('camPlaceholder').classList.remove('hidden');
  $('statusRing').className = 'status-ring';
}

/* ── Session helpers ──────────────────────────────────────────────────────────── */
function makeSessionId() {
  const n = Math.floor(100000 + Math.random() * 900000);
  return `SESSION-${n}`;
}

function resetAccumulators() {
  earAccum = []; marAccum = []; attAccum = []; fatAccum = [];
  fpsAccum = []; latencyAccum = []; confAccum = [];
  pitchAccum = []; yawAccum = []; rollAccum = []; stabilityAccum = [];
  maxClosure = 0; maxYawnDur = 0; sessionEnd = null; sessionMeta = null;
}

/* ── Processing → Report transition ──────────────────────────────────────────── */
const PROCESSING_STEPS = [
  'Analyzing Session...',
  'Processing Frames...',
  'Calculating Statistics...',
  'Generating AI Summary...',
  'Preparing Charts...',
  'Finalizing Session...',
];

async function runProcessingSequence() {
  const overlay  = $('processingOverlay');
  const textEl   = $('processingText');
  const barEl    = $('processingBarFill');

  overlay.classList.remove('hidden');
  barEl.style.width = '0%';

  sessionEnd = new Date();

  // Make sure we have a final agent read + session meta before building the report
  const agentFetch = apiGet('/api/agent').then(d => { lastAgentData = d; }).catch(() => {});
  const metaFetch  = apiGet('/api/session_meta').then(d => { sessionMeta = d; }).catch(() => {});

  const stepMs = 480;
  for (let i = 0; i < PROCESSING_STEPS.length; i++) {
    textEl.textContent = PROCESSING_STEPS[i];
    barEl.style.width = `${Math.round(((i + 1) / PROCESSING_STEPS.length) * 100)}%`;
    await new Promise(res => setTimeout(res, stepMs));
  }

  await agentFetch;
  await metaFetch;
  buildReport();

  overlay.classList.add('hidden');
  $('navReport').disabled = false;
  setView('report');
}

/* ── Build the Session Report from accumulated + agent data ───────────────────── */
function buildReport() {
  const data  = lastStatusData || {};
  const agent = lastAgentData  || {};
  const secs  = data.session_seconds || 0;

  // ── Summary header ──────────────────────────────────────────────────────────
  $('repSessionId').textContent = sessionId || makeSessionId();
  $('repDate').textContent      = (sessionDate || new Date()).toLocaleDateString();
  $('repStartTime').textContent = sessionDate ? sessionDate.toLocaleTimeString() : '—';
  $('repEndTime').textContent   = sessionEnd  ? sessionEnd.toLocaleTimeString()  : '—';
  $('repDuration').textContent  = fmtTime(secs);

  const finalRisk = data.final_risk || 'NORMAL';
  const driverState = lastStatus === 'drowsy' ? 'Drowsy detected'
                     : lastStatus === 'yawning' ? 'Yawning detected'
                     : lastStatus === 'noface'  ? 'Face lost intermittently'
                     : 'Attentive';
  $('repDriverState').textContent = driverState;

  const grade = agent.driver_grade || '—';
  const score = agent.grade_score  ?? 0;
  $('repGradeBadge').textContent = grade;
  $('repGradeBadge').className   = `rs-grade-badge grade-${String(grade).toLowerCase()}`;
  $('repSafetyScore').textContent = `${score}/100`;
  $('repOverallRisk').textContent = finalRisk;
  $('repOverallRisk').className   = `rs-stat-val ${riskClass(finalRisk)}`;

  // Safety score breakdown — qualitative factor summary from real session data
  $('repSafetyBreakdown').innerHTML =
    `Based on: <b>eye closure</b> (${data.microsleep_count || 0} microsleep${(data.microsleep_count || 0) === 1 ? '' : 's'}), ` +
    `<b>blink behaviour</b> (${data.long_blink_count || 0} long blink${(data.long_blink_count || 0) === 1 ? '' : 's'}), ` +
    `<b>yawning</b> (${data.yawn_count || data.yawn_events || 0} event${(data.yawn_count || data.yawn_events || 0) === 1 ? '' : 's'}), ` +
    `<b>head stability</b> (${stabilityAccum.length ? Math.round(avg(stabilityAccum) * 100) : 0}%), ` +
    `<b>attention</b> (avg ${attAccum.length ? Math.round(avg(attAccum) * 100) : 0}%), and ` +
    `<b>risk trend</b> (${computeRiskTrend()}).`;

  const recs = agent.recommendations || [];
  $('repHeadlineRec').textContent = recs.length
    ? recs[0]
    : (finalRisk === 'NORMAL' ? 'No concerns detected — continue driving safely.' : 'Review the recommendations below before your next trip.');

  // ── AI Summary ───────────────────────────────────────────────────────────────
  $('repAiSummary').textContent = agent.summary
    || 'Not enough session data was collected to generate a summary.';

  // ── Recommendations list ────────────────────────────────────────────────────
  renderList('repRecs', recs);

  // ── Categorized metrics ─────────────────────────────────────────────────────
  $('catAvgEAR').textContent      = earAccum.length ? avg(earAccum).toFixed(3) : '—';
  $('catMinEAR').textContent      = earAccum.length ? Math.min(...earAccum).toFixed(3) : '—';
  $('catBlinks').textContent      = data.blink_count || 0;
  $('catLongBlinks').textContent  = data.long_blink_count || 0;
  $('catMicro').textContent       = data.microsleep_count || 0;
  $('catLongClosure').textContent = maxClosure > 0 ? `${maxClosure.toFixed(2)}s` : '—';

  $('catAvgMAR').textContent      = marAccum.length ? avg(marAccum).toFixed(3) : '—';
  $('catYawns').textContent       = data.yawn_count || data.yawn_events || 0;
  $('catLongYawn').textContent    = maxYawnDur > 0 ? `${maxYawnDur.toFixed(2)}s` : '—';

  $('catAvgPitch').textContent    = pitchAccum.length ? `${avg(pitchAccum).toFixed(1)}°` : '—';
  $('catAvgYaw').textContent      = yawAccum.length   ? `${avg(yawAccum).toFixed(1)}°`   : '—';
  const allTilts = [...pitchAccum.map(Math.abs), ...yawAccum.map(Math.abs), ...rollAccum.map(Math.abs)];
  $('catMaxTilt').textContent     = allTilts.length ? `${Math.max(...allTilts).toFixed(0)}°` : '—';
  $('catStability').textContent   = stabilityAccum.length ? `${Math.round(avg(stabilityAccum) * 100)}%` : '—';

  $('catAvgFPS').textContent      = fpsAccum.length     ? avg(fpsAccum).toFixed(0)          : '—';
  $('catAvgLatency').textContent  = latencyAccum.length ? `${avg(latencyAccum).toFixed(0)} ms` : '—';
  $('catAvgConf').textContent     = confAccum.length    ? `${Math.round(avg(confAccum) * 100)}%` : '—';

  $('catDuration').textContent    = fmtTime(secs);
  $('catFrames').textContent      = sessionMeta?.frames ? sessionMeta.frames.toLocaleString() : '—';
  $('catDrowsy').textContent      = data.drowsy_events || 0;
  $('catAttention').textContent   = attAccum.length ? `${Math.round(avg(attAccum) * 100)}%` : '—';
  $('catFatigue').textContent     = fatAccum.length ? `${Math.round(avg(fatAccum) * 100)}%` : '—';

  // Final end-of-session timeline marker
  addTlEvent('end', 'Session ended');

  // Charts already hold the full session (200-point buffer); leave as-is for report view.
}

/* Compares average risk_index in the first vs second half of the session's
   chart data to describe whether risk trended up, down, or stayed flat. */
function computeRiskTrend() {
  const riskData = charts.risk.data.datasets[0].data;
  if (riskData.length < 6) return 'stable';
  const mid = Math.floor(riskData.length / 2);
  const firstHalf  = avg(riskData.slice(0, mid));
  const secondHalf = avg(riskData.slice(mid));
  const delta = secondHalf - firstHalf;
  if (delta > 0.4)  return 'increasing';
  if (delta < -0.4) return 'decreasing';
  return 'stable';
}

/* ── Controls ────────────────────────────────────────────────────────────────── */
$('btnStart').addEventListener('click', async () => {
  $('btnStart').disabled = true;
  try {
    const d = await apiPost('/api/start');
    if (d.success) {
      isRunning    = true;
      sessionStart = Date.now();
      sessionDate  = new Date();
      sessionId    = makeSessionId();
      lastStatus   = null;
      resetAccumulators();
      resetCharts();
      tlEvents.length = 0;
      renderTimeline();
      addTlEvent('start', 'SESSION STARTED');
      $('btnStop').disabled = false;
      $('navReport').disabled = true;
      setConn('online');
      setView('live');
      startVideoFeed();
      startPolling();
      stopHealthCheck();
      pollSystem();
      pollAgent();
    } else {
      alert(`Could not start: ${d.message}`);
      $('btnStart').disabled = false;
    }
  } catch (_) {
    alert('Backend unreachable. Make sure Flask is running on port 5000.');
    $('btnStart').disabled = false;
    setConn('error');
  }
});

$('btnStop').addEventListener('click', async () => {
  $('btnStop').disabled = true;
  try { await apiPost('/api/stop'); } catch (_) {}
  isRunning = false;
  stopPolling();
  stopVideoFeed();
  $('btnStart').disabled = false;
  setConn('offline');
  startHealthCheck();

  // Transition into the processing → report flow
  await runProcessingSequence();
});

$('btnReset').addEventListener('click', async () => {
  try { await apiPost('/api/reset'); } catch (_) {}
  resetAccumulators();
  sessionStart = Date.now();
  sessionDate  = new Date();
  sessionId    = makeSessionId();
  lastStatus   = null;
  tlEvents.length = 0;
  renderTimeline();
  resetCharts();
  $('navReport').disabled = true;
  setView('live');
});

$('btnNewSession').addEventListener('click', () => {
  setView('live');
});

/* ── Init ─────────────────────────────────────────────────────────────────────── */
applyStatus({ running: false });
startHealthCheck();
pollSystem();
setView('live');
