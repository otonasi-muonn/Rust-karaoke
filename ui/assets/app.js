/**
 * Rust Karaoke - Frontend Application
 * Canvas-based pitch visualization with Tauri event listeners
 * Plan2 Phase 6: Event-driven architecture using listen()
 */

// ---- Tauri API ----
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { getCurrentWindow } = window.__TAURI__.window;

// ---- DOM Elements ----
const $ = (sel) => document.querySelector(sel);
const dom = {
  url: $('#youtube-url'),
  btnAnalyze: $('#btn-analyze'),
  btnStart: $('#btn-start'),
  btnStop: $('#btn-stop'),
  btnBack: $('#btn-back'),
  btnPip: $('#btn-pip'),
  btnBorderless: $('#btn-borderless'),
  btnMinimize: $('#btn-minimize'),
  btnMaximize: $('#btn-maximize'),
  btnClose: $('#btn-close'),
  inputSection: $('#input-section'),
  karaokeSection: $('#karaoke-section'),
  progressContainer: $('#progress-bar-container'),
  progressFill: $('#progress-fill'),
  progressText: $('#progress-text'),
  canvas: $('#pitch-canvas'),
  currentNote: $('#current-note'),
  pitchHz: $('#pitch-hz'),
  scoreValue: $('#score-value'),
  judgementText: $('#judgement-text'),
  perfectCount: $('#perfect-count'),
  greatCount: $('#great-count'),
  goodCount: $('#good-count'),
  missCount: $('#miss-count'),
  loadingOverlay: $('#loading-overlay'),
  loadingTitle: $('#loading-title'),
  loadingMessage: $('#loading-message'),
  loadingProgressFill: $('#loading-progress-fill'),
};

// ---- State ----
let state = {
  isPlaying: false,
  isPip: false,
  isBorderless: true,  // starts borderless (decorations: false)
  referencePitches: [],
  durationMs: 0,
  animationFrame: null,
  userPitchHistory: [], // { time_ms, pitch_hz, judgement }
  latestScore: null,     // Latest score snapshot from backend event
  playbackStartTime: 0,  // performance.now() when playback started
  unlistenPitch: null,    // Unlisten handle for pitch-update
  unlistenScore: null,    // Unlisten handle for score-update
  pipSizeBeforeRestore: null, // Original size before PiP
};

// ---- Canvas Renderer ----
class PitchRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.width = rect.width;
    this.height = rect.height;
  }

  // Convert Hz to Y position (logarithmic scale)
  hzToY(hz) {
    if (hz <= 0) return this.height;
    const minHz = 80;   // C2
    const maxHz = 1200;  // D6
    const minLog = Math.log2(minHz);
    const maxLog = Math.log2(maxHz);
    const logHz = Math.log2(hz);
    const normalized = (logHz - minLog) / (maxLog - minLog);
    return this.height - (normalized * (this.height - 40)) - 20;
  }

  // Convert time to X position
  timeToX(timeMs, currentTimeMs, windowMs = 8000) {
    const halfWindow = windowMs / 2;
    const normalized = (timeMs - currentTimeMs + halfWindow) / windowMs;
    return normalized * this.width;
  }

  draw(currentTimeMs) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const windowMs = 8000; // 8 second window

    // Clear
    ctx.fillStyle = '#0d0d15';
    ctx.fillRect(0, 0, w, h);

    // Draw grid lines for musical notes
    this.drawGrid(ctx, w, h);

    // Draw center line (current time)
    const centerX = w / 2;
    ctx.strokeStyle = 'rgba(108, 92, 231, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw reference pitches (upcoming notes)
    this.drawReferencePitches(ctx, currentTimeMs, windowMs);

    // Draw user pitch trail
    this.drawUserPitch(ctx, currentTimeMs, windowMs);
  }

  drawGrid(ctx, w, h) {
    const notes = [
      { hz: 130.81, name: 'C3' },
      { hz: 196.00, name: 'G3' },
      { hz: 261.63, name: 'C4' },
      { hz: 392.00, name: 'G4' },
      { hz: 523.25, name: 'C5' },
      { hz: 783.99, name: 'G5' },
    ];

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    ctx.font = '10px Segoe UI';
    ctx.fillStyle = 'rgba(255,255,255,0.15)';

    for (const note of notes) {
      const y = this.hzToY(note.hz);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillText(note.name, 4, y - 3);
    }
  }

  drawReferencePitches(ctx, currentTimeMs, windowMs) {
    const halfW = windowMs / 2;
    const startTime = currentTimeMs - halfW;
    const endTime = currentTimeMs + halfW;

    // Phase 6: Draw reference pitches as scrolling rectangular bars (fillRect)
    // Each voiced segment is rendered as a filled bar at the corresponding pitch level
    const barHeight = 6; // pixels tall per pitch bar
    const refs = state.referencePitches;
    const len = refs.length;

    // Pre-compute time range for binary search optimization
    let startIdx = 0;
    for (let j = 0; j < len; j++) {
      if (refs[j].time_ms >= startTime) { startIdx = j; break; }
    }

    let prevVoiced = false;
    let segStartX = 0;
    let segY = 0;

    for (let j = startIdx; j < len; j++) {
      const p = refs[j];
      if (p.time_ms > endTime) break;

      if (!p.is_voiced || p.pitch_hz <= 0) {
        // End current segment
        if (prevVoiced) {
          const segEndX = this.timeToX(refs[j - 1].time_ms, currentTimeMs, windowMs);
          const w = Math.max(segEndX - segStartX, 2);
          // Draw filled bar with glow
          ctx.fillStyle = 'rgba(108, 92, 231, 0.35)';
          ctx.fillRect(segStartX, segY - barHeight / 2, w, barHeight);
          // Brighter border  
          ctx.strokeStyle = 'rgba(108, 92, 231, 0.7)';
          ctx.lineWidth = 1;
          ctx.strokeRect(segStartX, segY - barHeight / 2, w, barHeight);
        }
        prevVoiced = false;
        continue;
      }

      const x = this.timeToX(p.time_ms, currentTimeMs, windowMs);
      const y = this.hzToY(p.pitch_hz);

      if (!prevVoiced) {
        // Start new segment
        segStartX = x;
        segY = y;
        prevVoiced = true;
      } else {
        // Continue segment — if pitch changed significantly, flush and start new
        const yDiff = Math.abs(y - segY);
        if (yDiff > barHeight * 2) {
          // Flush current bar
          const segEndX = this.timeToX(refs[j - 1].time_ms, currentTimeMs, windowMs);
          const w = Math.max(segEndX - segStartX, 2);
          ctx.fillStyle = 'rgba(108, 92, 231, 0.35)';
          ctx.fillRect(segStartX, segY - barHeight / 2, w, barHeight);
          ctx.strokeStyle = 'rgba(108, 92, 231, 0.7)';
          ctx.lineWidth = 1;
          ctx.strokeRect(segStartX, segY - barHeight / 2, w, barHeight);
          // Start new segment
          segStartX = x;
          segY = y;
        }
      }
    }

    // Flush last segment
    if (prevVoiced && len > 0) {
      const lastIdx = Math.min(len - 1, startIdx + len);
      const lastRef = refs[lastIdx];
      if (lastRef) {
        const segEndX = this.timeToX(lastRef.time_ms, currentTimeMs, windowMs);
        const w = Math.max(segEndX - segStartX, 2);
        ctx.fillStyle = 'rgba(108, 92, 231, 0.35)';
        ctx.fillRect(segStartX, segY - barHeight / 2, w, barHeight);
        ctx.strokeStyle = 'rgba(108, 92, 231, 0.7)';
        ctx.lineWidth = 1;
        ctx.strokeRect(segStartX, segY - barHeight / 2, w, barHeight);
      }
    }
  }

  drawUserPitch(ctx, currentTimeMs, windowMs) {
    const halfW = windowMs / 2;
    const startTime = currentTimeMs - halfW;

    // Phase 6: Efficient batched lineTo drawing with judgement-based coloring
    const colorMap = {
      'Perfect': '#00e676',
      'Great': '#64dd17',
      'Good': '#ffab00',
      'Miss': '#ff1744',
    };

    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let prevX = null, prevY = null;
    let currentColor = null;

    for (const p of state.userPitchHistory) {
      if (p.time_ms < startTime) continue;
      if (p.pitch_hz <= 0) {
        prevX = null;
        currentColor = null;
        continue;
      }

      const x = this.timeToX(p.time_ms, currentTimeMs, windowMs);
      const y = this.hzToY(p.pitch_hz);
      const color = colorMap[p.judgement] || '#ff1744';

      // Draw connecting line
      if (prevX !== null) {
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();
      }

      // Draw glow dot at current position
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      prevX = x;
      prevY = y;
      currentColor = color;
    }

    // Draw a larger pulsing dot at the latest position (cursor indicator)
    if (prevX !== null && currentColor) {
      ctx.fillStyle = currentColor;
      ctx.shadowColor = currentColor;
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.arc(prevX, prevY, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }
}

// ---- Pitch Renderer Instance ----
let renderer;

// ---- Helpers ----
function hzToNoteName(hz) {
  if (hz <= 0) return '---';
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const midi = 69 + 12 * Math.log2(hz / 440);
  const noteNum = Math.round(midi) % 12;
  const octave = Math.floor(Math.round(midi) / 12) - 1;
  return noteNames[(noteNum + 12) % 12] + octave;
}

function showSection(section) {
  dom.inputSection.classList.toggle('hidden', section !== 'input');
  dom.karaokeSection.classList.toggle('hidden', section !== 'karaoke');
  if (section === 'karaoke') {
    dom.karaokeSection.classList.add('animate-in');
    renderer.resize();
  }
}

const judgementLabels = {
  'Perfect': '完璧',
  'Great': '素晴らしい',
  'Good': '良い',
  'Miss': 'ミス',
};

function updateJudgement(judgement) {
  dom.judgementText.textContent = judgementLabels[judgement] || judgement;
  dom.judgementText.className = 'judgement-' + judgement.toLowerCase();
}

// ---- Progress Polling ----
let progressInterval;

const stageOrder = ['download', 'decode', 'separation', 'pitch', 'done'];
const stageMessages = {
  download: 'YouTubeから楽曲をダウンロードしています...',
  decode: '音声データをデコードしています...',
  separation: 'AIがボーカルを分離しています...',
  pitch: 'リファレンスピッチを解析しています...',
  done: '解析が完了しました！',
};

function updateLoadingSteps(currentStage) {
  const currentIdx = stageOrder.indexOf(currentStage);
  stageOrder.forEach((stage, idx) => {
    const el = document.getElementById('step-' + stage);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (idx < currentIdx) {
      el.classList.add('done');
    } else if (idx === currentIdx) {
      el.classList.add('active');
    }
  });
}

async function pollProgress() {
  try {
    const progress = await invoke('get_analysis_progress');
    // Update old progress bar (hidden)
    dom.progressFill.style.width = (progress.progress * 100) + '%';
    dom.progressText.textContent = progress.message;

    // Update loading overlay
    dom.loadingProgressFill.style.width = (progress.progress * 100) + '%';
    dom.loadingMessage.textContent = stageMessages[progress.stage] || progress.message;
    updateLoadingSteps(progress.stage);

    if (progress.stage === 'done') {
      clearInterval(progressInterval);
    }
  } catch (e) {
    console.error('Progress poll error:', e);
  }
}

// ---- Score Polling (removed: now event-driven) ----

// ---- Tauri Event Listeners ----
async function setupEventListeners() {
  // Listen for real-time pitch updates from the inference thread
  state.unlistenPitch = await listen('pitch-update', (event) => {
    const p = event.payload;

    // Add to user pitch history for Canvas rendering
    if (p.is_voiced && p.user_pitch_hz > 0) {
      state.userPitchHistory.push({
        time_ms: p.time_ms,
        pitch_hz: p.user_pitch_hz,
        judgement: p.judgement,
      });
      // Trim old entries (keep last 30 seconds)
      const cutoffMs = p.time_ms - 30000;
      while (state.userPitchHistory.length > 0 && state.userPitchHistory[0].time_ms < cutoffMs) {
        state.userPitchHistory.shift();
      }
    }

    // Update current note display
    dom.currentNote.textContent = hzToNoteName(p.user_pitch_hz);
    dom.pitchHz.textContent = p.user_pitch_hz > 0 ? p.user_pitch_hz.toFixed(1) + ' Hz' : '---';

    // Update judgement display
    updateJudgement(p.judgement);
  });

  // Listen for score snapshot updates (~every 200ms)
  state.unlistenScore = await listen('score-update', (event) => {
    const s = event.payload;
    state.latestScore = s;

    dom.scoreValue.textContent = s.current_score.toFixed(1);
    dom.perfectCount.textContent = s.perfect_count;
    dom.greatCount.textContent = s.great_count;
    dom.goodCount.textContent = s.good_count;
    dom.missCount.textContent = s.miss_count;
  });
}

function teardownEventListeners() {
  if (state.unlistenPitch) {
    state.unlistenPitch();
    state.unlistenPitch = null;
  }
  if (state.unlistenScore) {
    state.unlistenScore();
    state.unlistenScore = null;
  }
}

// ---- Animation Loop ----
let startTime = 0;

function animationLoop(timestamp) {
  if (!state.isPlaying) return;

  // Use backend-synced time: the latest pitch event time_ms is the true playback position
  let currentTimeMs;
  if (state.userPitchHistory.length > 0) {
    currentTimeMs = state.userPitchHistory[state.userPitchHistory.length - 1].time_ms;
  } else {
    // Fallback to client-side estimate
    currentTimeMs = timestamp - startTime;
  }

  renderer.draw(currentTimeMs);
  state.animationFrame = requestAnimationFrame(animationLoop);
}

// ---- Event Handlers ----

// Analyze YouTube URL
dom.btnAnalyze.addEventListener('click', async () => {
  const url = dom.url.value.trim();
  if (!url) return;

  dom.btnAnalyze.disabled = true;

  // Show loading overlay
  dom.loadingOverlay.classList.remove('hidden');
  dom.loadingProgressFill.style.width = '0%';
  dom.loadingTitle.textContent = '楽曲を解析中...';
  dom.loadingMessage.textContent = 'AIが音源分離・ピッチ解析を行っています';
  updateLoadingSteps('');

  // Also update old progress bar (backup)
  dom.progressContainer.classList.remove('hidden');
  dom.progressFill.style.width = '0%';
  dom.progressText.textContent = '解析を開始しています...';

  progressInterval = setInterval(pollProgress, 500);

  try {
    const data = await invoke('analyze_youtube_url', { url });
    state.referencePitches = data.reference_pitches;
    state.durationMs = data.duration_ms;

    clearInterval(progressInterval);
    dom.progressFill.style.width = '100%';
    dom.progressText.textContent = '解析完了！カラオケの準備ができました';
    dom.loadingProgressFill.style.width = '100%';
    dom.loadingTitle.textContent = '解析完了！';
    dom.loadingMessage.textContent = 'カラオケの準備ができました 🎉';
    updateLoadingSteps('done');

    setTimeout(() => {
      dom.loadingOverlay.classList.add('hidden');
      showSection('karaoke');
    }, 800);
  } catch (e) {
    clearInterval(progressInterval);
    dom.progressText.textContent = '解析エラー: ' + e;
    dom.progressFill.style.width = '0%';
    dom.loadingTitle.textContent = 'エラーが発生しました';
    dom.loadingMessage.textContent = e;
    dom.loadingProgressFill.style.width = '0%';
    // Hide loading after 3 seconds on error
    setTimeout(() => {
      dom.loadingOverlay.classList.add('hidden');
    }, 3000);
    console.error('Analysis error:', e);
  } finally {
    dom.btnAnalyze.disabled = false;
  }
});

// Enter key for URL input
dom.url.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') dom.btnAnalyze.click();
});

// Start karaoke
dom.btnStart.addEventListener('click', async () => {
  try {
    // Set up event listeners BEFORE starting karaoke
    await setupEventListeners();

    await invoke('start_karaoke');
    state.isPlaying = true;
    state.userPitchHistory = [];
    state.latestScore = null;
    dom.btnStart.classList.add('hidden');
    dom.btnStop.classList.remove('hidden');

    startTime = performance.now();
    state.animationFrame = requestAnimationFrame(animationLoop);
  } catch (e) {
    console.error('Start error:', e);
    teardownEventListeners();
    alert('カラオケの開始に失敗しました: ' + e);
  }
});

// Stop karaoke
dom.btnStop.addEventListener('click', async () => {
  try {
    await invoke('stop_karaoke');
  } catch (e) {
    console.error('Stop error:', e);
  }
  state.isPlaying = false;
  cancelAnimationFrame(state.animationFrame);
  teardownEventListeners();
  dom.btnStop.classList.add('hidden');
  dom.btnStart.classList.remove('hidden');
});

// Back button
dom.btnBack.addEventListener('click', () => {
  if (state.isPlaying) {
    dom.btnStop.click();
  }
  showSection('input');
});

// PiP mode
dom.btnPip.addEventListener('click', async () => {
  const win = getCurrentWindow();
  state.isPip = !state.isPip;

  if (state.isPip) {
    // Enter PiP: small always-on-top window
    document.body.classList.add('pip-mode');
    await win.setAlwaysOnTop(true);
    await win.setSize(new window.__TAURI__.window.LogicalSize(400, 300));
    dom.btnPip.textContent = '⊟';
    dom.btnPip.classList.add('active');
  } else {
    // Exit PiP: restore normal size
    document.body.classList.remove('pip-mode');
    await win.setAlwaysOnTop(false);
    await win.setSize(new window.__TAURI__.window.LogicalSize(1200, 800));
    dom.btnPip.textContent = '⊞';
    dom.btnPip.classList.remove('active');
  }

  // Resize canvas after size change
  if (renderer) {
    setTimeout(() => renderer.resize(), 100);
  }
});

// Borderless window toggle
dom.btnBorderless.addEventListener('click', async () => {
  const win = getCurrentWindow();
  state.isBorderless = !state.isBorderless;

  await win.setDecorations(!state.isBorderless);

  if (state.isBorderless) {
    // Borderless: show custom titlebar
    document.getElementById('titlebar').style.display = '';
    document.getElementById('app').style.height = 'calc(100% - 36px)';
    dom.btnBorderless.classList.add('active');
    dom.btnBorderless.title = 'ボーダーレス (ON)';
  } else {
    // Decorated: hide custom titlebar (OS handles it)
    document.getElementById('titlebar').style.display = 'none';
    document.getElementById('app').style.height = '100%';
    dom.btnBorderless.classList.remove('active');
    dom.btnBorderless.title = 'ボーダーレス (OFF)';
  }
});

// Window controls
dom.btnMinimize.addEventListener('click', async () => {
  const win = getCurrentWindow();
  await win.minimize();
});

dom.btnMaximize.addEventListener('click', async () => {
  const win = getCurrentWindow();
  const maximized = await win.isMaximized();
  if (maximized) {
    await win.unmaximize();
  } else {
    await win.maximize();
  }
});

dom.btnClose.addEventListener('click', async () => {
  const win = getCurrentWindow();
  await win.close();
});

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
  renderer = new PitchRenderer(dom.canvas);
  // Draw initial empty state
  renderer.draw(0);

  // Initial borderless state (starts as decorations: false)
  dom.btnBorderless.classList.add('active');
  dom.btnBorderless.title = 'ボーダーレス (ON)';
});
