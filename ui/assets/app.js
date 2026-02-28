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
  // Settings
  btnSettings: $('#btn-settings'),
  settingsPanel: $('#settings-panel'),
  btnSettingsClose: $('#btn-settings-close'),
  micSelect: $('#mic-select'),
  btnMicRefresh: $('#btn-mic-refresh'),
  micGain: $('#mic-gain'),
  micGainValue: $('#mic-gain-value'),
  btnMicTest: $('#btn-mic-test'),
  micLevelFill: $('#mic-level-fill'),
  micTestStatus: $('#mic-test-status'),
  bgmVolume: $('#bgm-volume'),
  bgmVolumeValue: $('#bgm-volume-value'),
  // Zoom
  uiZoom: $('#ui-zoom'),
  uiZoomValue: $('#ui-zoom-value'),
  btnZoomIn: $('#btn-zoom-in'),
  btnZoomOut: $('#btn-zoom-out'),
  btnZoomReset: $('#btn-zoom-reset'),
};

// ---- State ----
let state = {
  isPlaying: false,
  isPip: false,
  isBorderless: true,  // starts borderless (decorations: false)
  referencePitches: [],
  noteSegments: [],
  durationMs: 0,
  animationFrame: null,
  userPitchHistory: [], // { time_ms, pitch_hz, judgement }
  latestScore: null,     // Latest score snapshot from backend event
  playbackStartTime: 0,  // performance.now() when playback started
  unlistenPitch: null,    // Unlisten handle for pitch-update
  unlistenScore: null,    // Unlisten handle for score-update
  unlistenMicTest: null,  // Unlisten handle for mic-test-level
  micTestActive: false,   // Whether persistent mic test is running
  pipSizeBeforeRestore: null, // Original size before PiP
};

// ---- Canvas Renderer ----
class PitchRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    // Dynamic vertical range (smoothed)
    this.viewMinHz = 150;
    this.viewMaxHz = 600;
    this.targetMinHz = 150;
    this.targetMaxHz = 600;
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

  // Convert Hz to Y position (logarithmic scale, dynamic range)
  hzToY(hz) {
    if (hz <= 0) return this.height;
    const minLog = Math.log2(this.viewMinHz);
    const maxLog = Math.log2(this.viewMaxHz);
    const logHz = Math.log2(hz);
    const normalized = (logHz - minLog) / (maxLog - minLog);
    return this.height - (normalized * (this.height - 40)) - 20;
  }

  // Update dynamic vertical range based on visible notes
  updateViewRange(currentTimeMs, windowMs) {
    const halfW = windowMs / 2;
    const lookAhead = currentTimeMs + halfW * 1.5; // Look slightly further ahead
    const lookBehind = currentTimeMs - halfW;

    let minHz = Infinity;
    let maxHz = -Infinity;
    let hasNotes = false;

    // Scan note segments for visible pitch range
    const segs = state.noteSegments;
    if (segs && segs.length > 0) {
      for (const seg of segs) {
        if (seg.endMs < lookBehind) continue;
        if (seg.startMs > lookAhead) break;
        if (seg.hz > 0) {
          minHz = Math.min(minHz, seg.hz);
          maxHz = Math.max(maxHz, seg.hz);
          hasNotes = true;
        }
      }
    }

    // Also consider user pitch
    for (const p of state.userPitchHistory) {
      if (p.time_ms < lookBehind) continue;
      if (p.pitch_hz > 0) {
        minHz = Math.min(minHz, p.pitch_hz);
        maxHz = Math.max(maxHz, p.pitch_hz);
        hasNotes = true;
      }
    }

    if (hasNotes && isFinite(minHz) && isFinite(maxHz)) {
      // Add padding: 1 octave below and above, minimum 2 octave range
      const centerLog = (Math.log2(minHz) + Math.log2(maxHz)) / 2;
      const rangeLog = Math.log2(maxHz) - Math.log2(minHz);
      const minRange = 2.0; // minimum 2 octaves displayed
      const paddedRange = Math.max(rangeLog + 1.0, minRange); // +1 octave padding total

      this.targetMinHz = Math.pow(2, centerLog - paddedRange / 2);
      this.targetMaxHz = Math.pow(2, centerLog + paddedRange / 2);

      // Clamp to reasonable bounds
      this.targetMinHz = Math.max(this.targetMinHz, 60);
      this.targetMaxHz = Math.min(this.targetMaxHz, 2000);
    }

    // Smooth transition (exponential interpolation in log space)
    const smoothing = 0.04;
    const curMinLog = Math.log2(this.viewMinHz);
    const curMaxLog = Math.log2(this.viewMaxHz);
    const tgtMinLog = Math.log2(this.targetMinHz);
    const tgtMaxLog = Math.log2(this.targetMaxHz);

    this.viewMinHz = Math.pow(2, curMinLog + (tgtMinLog - curMinLog) * smoothing);
    this.viewMaxHz = Math.pow(2, curMaxLog + (tgtMaxLog - curMaxLog) * smoothing);
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

    // Update dynamic vertical range
    this.updateViewRange(currentTimeMs, windowMs);

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
    // Dynamically generate grid lines based on current view range
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const minMidi = Math.floor(69 + 12 * Math.log2(this.viewMinHz / 440));
    const maxMidi = Math.ceil(69 + 12 * Math.log2(this.viewMaxHz / 440));

    // Determine grid density — show every note, every 3rd, etc.
    const midiRange = maxMidi - minMidi;
    let step = 1;
    if (midiRange > 36) step = 6;      // Show every 6 semitones
    else if (midiRange > 24) step = 3;  // Show every minor 3rd
    else if (midiRange > 18) step = 2;  // Show every whole tone

    ctx.lineWidth = 1;
    ctx.font = '10px Segoe UI';

    for (let midi = minMidi; midi <= maxMidi; midi++) {
      if (midi % step !== 0) continue;
      const hz = 440 * Math.pow(2, (midi - 69) / 12);
      const y = this.hzToY(hz);
      if (y < 5 || y > h - 5) continue;

      const noteIdx = ((midi % 12) + 12) % 12;
      const octave = Math.floor(midi / 12) - 1;
      const name = noteNames[noteIdx] + octave;
      const isC = noteIdx === 0;

      ctx.strokeStyle = isC ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.03)';
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      ctx.fillStyle = isC ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.12)';
      ctx.fillText(name, 4, y - 3);
    }
  }

  drawReferencePitches(ctx, currentTimeMs, windowMs) {
    const halfW = windowMs / 2;
    const startTime = currentTimeMs - halfW;
    const endTime = currentTimeMs + halfW;

    const segs = state.noteSegments;
    if (!segs || segs.length === 0) return;

    const barHeight = 14;
    const radius = 4;

    // Binary search for first visible segment
    let lo = 0, hi = segs.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (segs[mid].endMs < startTime) lo = mid + 1;
      else hi = mid;
    }

    for (let i = lo; i < segs.length; i++) {
      const seg = segs[i];
      if (seg.startMs > endTime) break;

      const x1 = this.timeToX(seg.startMs, currentTimeMs, windowMs);
      const x2 = this.timeToX(seg.endMs, currentTimeMs, windowMs);
      const y = this.hzToY(seg.hz);
      const w = Math.max(x2 - x1, 4);

      // Determine if note is past (already sung), current, or upcoming
      const isPast = seg.endMs < currentTimeMs;
      const isCurrent = seg.startMs <= currentTimeMs && seg.endMs >= currentTimeMs;

      // Bar fill — gradient
      let alpha = isPast ? 0.2 : (isCurrent ? 0.6 : 0.4);
      let borderAlpha = isPast ? 0.3 : (isCurrent ? 0.9 : 0.65);
      const baseR = 108, baseG = 92, baseB = 231;  // Purple theme

      // Draw glow for current note
      if (isCurrent) {
        ctx.shadowColor = `rgba(${baseR}, ${baseG}, ${baseB}, 0.5)`;
        ctx.shadowBlur = 12;
      }

      // Rounded rectangle bar
      const bx = x1;
      const by = y - barHeight / 2;

      ctx.beginPath();
      if (w > radius * 2) {
        ctx.moveTo(bx + radius, by);
        ctx.lineTo(bx + w - radius, by);
        ctx.arcTo(bx + w, by, bx + w, by + radius, radius);
        ctx.lineTo(bx + w, by + barHeight - radius);
        ctx.arcTo(bx + w, by + barHeight, bx + w - radius, by + barHeight, radius);
        ctx.lineTo(bx + radius, by + barHeight);
        ctx.arcTo(bx, by + barHeight, bx, by + barHeight - radius, radius);
        ctx.lineTo(bx, by + radius);
        ctx.arcTo(bx, by, bx + radius, by, radius);
      } else {
        ctx.rect(bx, by, w, barHeight);
      }
      ctx.closePath();

      // Gradient fill
      const grad = ctx.createLinearGradient(bx, by, bx, by + barHeight);
      grad.addColorStop(0, `rgba(${baseR + 40}, ${baseG + 40}, ${baseB}, ${alpha})`);
      grad.addColorStop(0.5, `rgba(${baseR}, ${baseG}, ${baseB}, ${alpha + 0.1})`);
      grad.addColorStop(1, `rgba(${baseR - 20}, ${baseG - 20}, ${baseB - 30}, ${alpha})`);
      ctx.fillStyle = grad;
      ctx.fill();

      // Border
      ctx.strokeStyle = `rgba(${baseR + 20}, ${baseG + 20}, ${baseB}, ${borderAlpha})`;
      ctx.lineWidth = isCurrent ? 1.5 : 1;
      ctx.stroke();

      ctx.shadowBlur = 0;
      ctx.shadowColor = 'transparent';

      // Note name label — only on bars wide enough
      if (w > 28) {
        ctx.font = '9px Segoe UI';
        ctx.fillStyle = `rgba(255, 255, 255, ${isPast ? 0.25 : 0.55})`;
        ctx.textBaseline = 'middle';
        ctx.fillText(seg.noteName, bx + 4, y + 1);
      }
    }

    // Draw connecting lines between consecutive segments
    ctx.strokeStyle = 'rgba(108, 92, 231, 0.15)';
    ctx.lineWidth = 1;
    for (let i = lo; i < segs.length - 1; i++) {
      const seg = segs[i];
      const next = segs[i + 1];
      if (seg.startMs > endTime) break;
      if (next.startMs > endTime) break;

      const gap = next.startMs - seg.endMs;
      if (gap > 0 && gap < 300 && seg.midiNote !== next.midiNote) {
        const x1 = this.timeToX(seg.endMs, currentTimeMs, windowMs);
        const x2 = this.timeToX(next.startMs, currentTimeMs, windowMs);
        const y1 = this.hzToY(seg.hz);
        const y2 = this.hzToY(next.hz);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
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

// ---- Note Segmentation (frame-level pitches → karaoke-style note bars) ----
function buildNoteSegments(referencePitches) {
  if (!referencePitches || referencePitches.length === 0) return [];

  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  function hzToMidi(hz) {
    return Math.round(69 + 12 * Math.log2(hz / 440));
  }
  function midiToHz(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
  }
  function midiToName(midi) {
    const n = ((midi % 12) + 12) % 12;
    const oct = Math.floor(midi / 12) - 1;
    return noteNames[n] + oct;
  }

  // Step 1: Quantize each voiced frame to MIDI, allowing ±1 semitone wobble
  // Use a 5-frame majority vote window to stabilize notes
  const VOTE_WINDOW = 5;
  const voiced = referencePitches.filter(p => p.is_voiced && p.pitch_hz > 0);
  if (voiced.length === 0) return [];

  const midiValues = voiced.map(p => hzToMidi(p.pitch_hz));
  const stableMidi = midiValues.map((m, idx) => {
    const start = Math.max(0, idx - Math.floor(VOTE_WINDOW / 2));
    const end = Math.min(midiValues.length, idx + Math.floor(VOTE_WINDOW / 2) + 1);
    const counts = {};
    for (let i = start; i < end; i++) {
      const v = midiValues[i];
      counts[v] = (counts[v] || 0) + 1;
    }
    let bestNote = m, bestCount = 0;
    for (const [note, count] of Object.entries(counts)) {
      if (count > bestCount) { bestCount = count; bestNote = parseInt(note); }
    }
    return bestNote;
  });

  // Step 2: Build raw segments from stabilized MIDI values
  const segments = [];
  let curMidi = stableMidi[0];
  let curStart = voiced[0].time_ms;
  let curEnd = voiced[0].time_ms;

  for (let i = 1; i < voiced.length; i++) {
    const midi = stableMidi[i];
    const timeMs = voiced[i].time_ms;
    const gap = timeMs - curEnd;

    if (midi === curMidi && gap < 200) {
      // Same note, extend (allow gaps up to 200ms within a note)
      curEnd = timeMs;
    } else {
      // Flush previous segment
      segments.push({
        startMs: curStart, endMs: curEnd, midiNote: curMidi,
        hz: midiToHz(curMidi), noteName: midiToName(curMidi),
      });
      curMidi = midi;
      curStart = timeMs;
      curEnd = timeMs;
    }
  }
  // Flush last
  segments.push({
    startMs: curStart, endMs: curEnd, midiNote: curMidi,
    hz: midiToHz(curMidi), noteName: midiToName(curMidi),
  });

  // Step 3: Absorb very short segments (< 60ms) into neighbors if within ±1 semitone
  for (let i = 1; i < segments.length - 1; i++) {
    const seg = segments[i];
    const dur = seg.endMs - seg.startMs;
    if (dur >= 60) continue;

    const prev = segments[i - 1];
    const next = segments[i + 1];
    const diffPrev = Math.abs(seg.midiNote - prev.midiNote);
    const diffNext = Math.abs(seg.midiNote - next.midiNote);

    // Absorb into whichever neighbor is closer in pitch
    if (diffPrev <= 1 && (seg.startMs - prev.endMs) < 200) {
      prev.endMs = seg.endMs;
      segments.splice(i, 1);
      i--;
    } else if (diffNext <= 1 && (next.startMs - seg.endMs) < 200) {
      next.startMs = seg.startMs;
      segments.splice(i, 1);
      i--;
    }
  }

  // Step 4: Merge same-note segments with gaps < 300ms
  const merged = [];
  for (const seg of segments) {
    if (merged.length > 0) {
      const prev = merged[merged.length - 1];
      if (prev.midiNote === seg.midiNote && (seg.startMs - prev.endMs) < 300) {
        prev.endMs = seg.endMs;
        continue;
      }
    }
    merged.push({ ...seg });
  }

  // Step 5: Extend each segment by a small amount to fill visual gaps
  const EXTEND_MS = 15; // add 15ms padding to each end
  for (const seg of merged) {
    seg.startMs = Math.max(0, seg.startMs - EXTEND_MS);
    seg.endMs += EXTEND_MS;
  }

  // Step 6: Filter out very short segments (<30ms after all merging)
  const result = merged.filter(s => (s.endMs - s.startMs) >= 30);
  console.log(`buildNoteSegments: ${referencePitches.length} frames -> ${voiced.length} voiced -> ${result.length} bars`);
  return result;
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
  '---': '',
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
        judgement: p.judgement === '---' ? 'Good' : p.judgement,
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

  // Always use wall-clock time for smooth continuous scrolling
  const currentTimeMs = performance.now() - state.playbackStartTime;

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
    state.noteSegments = buildNoteSegments(data.reference_pitches);
    state.durationMs = data.duration_ms;
    console.log(`Note segments: ${state.noteSegments.length} bars built from ${data.reference_pitches.length} frames`);

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

    state.playbackStartTime = performance.now();
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

// ---- Settings & Persistence ----
const STORAGE_KEY = 'rust-karaoke-settings';

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {}
  return {};
}

function saveSettings(partial) {
  const current = loadSettings();
  const merged = { ...current, ...partial };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
}

async function applyStoredSettings() {
  const s = loadSettings();

  // Apply mic device
  if (s.micDevice) {
    await invoke('set_mic_device', { name: s.micDevice }).catch(() => {});
  }

  // Apply mic gain
  if (s.micGain !== undefined) {
    dom.micGain.value = s.micGain;
    dom.micGainValue.textContent = s.micGain + '%';
    await invoke('set_mic_volume', { gain: s.micGain / 100 }).catch(() => {});
  }

  // Apply BGM volume
  if (s.bgmVolume !== undefined) {
    dom.bgmVolume.value = s.bgmVolume;
    dom.bgmVolumeValue.textContent = s.bgmVolume + '%';
    await invoke('set_accompaniment_volume', { volume: s.bgmVolume / 100 }).catch(() => {});
  }

  // Apply UI zoom
  if (s.uiZoom !== undefined) {
    applyZoom(s.uiZoom);
  }
}

async function loadMicDevices() {
  try {
    const devices = await invoke('list_mic_devices');
    dom.micSelect.innerHTML = '';

    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = '(既定のデバイス)';
    dom.micSelect.appendChild(defaultOpt);

    for (const name of devices) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      dom.micSelect.appendChild(opt);
    }

    // Restore saved device
    const s = loadSettings();
    if (s.micDevice) {
      dom.micSelect.value = s.micDevice;
    }
  } catch (e) {
    console.error('Failed to list mic devices:', e);
    dom.micSelect.innerHTML = '<option value="">デバイスの取得に失敗</option>';
  }
}

// Settings panel open/close
dom.btnSettings.addEventListener('click', async () => {
  dom.settingsPanel.classList.toggle('hidden');
  if (!dom.settingsPanel.classList.contains('hidden')) {
    await loadMicDevices();
  }
});

dom.btnSettingsClose.addEventListener('click', () => {
  dom.settingsPanel.classList.add('hidden');
  if (state.micTestActive) stopMicTest();
});

// Click outside settings to close
dom.settingsPanel.addEventListener('click', (e) => {
  if (e.target === dom.settingsPanel) {
    dom.settingsPanel.classList.add('hidden');
    if (state.micTestActive) stopMicTest();
  }
});

// Mic device selection
dom.micSelect.addEventListener('change', async () => {
  const name = dom.micSelect.value || null;
  await invoke('set_mic_device', { name }).catch(console.error);
  saveSettings({ micDevice: dom.micSelect.value });
});

// Refresh mic devices
dom.btnMicRefresh.addEventListener('click', () => loadMicDevices());

// Mic gain slider
dom.micGain.addEventListener('input', async () => {
  const val = parseInt(dom.micGain.value, 10);
  dom.micGainValue.textContent = val + '%';
  await invoke('set_mic_volume', { gain: val / 100 }).catch(console.error);
  saveSettings({ micGain: val });
});

// Persistent mic test (toggle on/off)
async function startMicTest() {
  try {
    await invoke('start_mic_test');
    state.micTestActive = true;
    dom.btnMicTest.textContent = '⏹ テスト停止';
    dom.btnMicTest.classList.add('active');
    dom.micTestStatus.textContent = 'テスト中...';
    dom.micTestStatus.style.color = '';

    // Listen for level events
    state.unlistenMicTest = await listen('mic-test-level', (event) => {
      const level = event.payload;
      const pct = Math.round(level * 100);
      dom.micLevelFill.style.width = pct + '%';
      if (pct > 5) {
        dom.micTestStatus.textContent = '検出OK (' + pct + '%)';
        dom.micTestStatus.style.color = 'var(--perfect)';
      } else {
        dom.micTestStatus.textContent = '音が小さい (' + pct + '%)';
        dom.micTestStatus.style.color = 'var(--good)';
      }
    });
  } catch (e) {
    dom.micTestStatus.textContent = 'エラー';
    dom.micTestStatus.style.color = 'var(--miss)';
    console.error('Mic test start error:', e);
  }
}

async function stopMicTest() {
  try {
    await invoke('stop_mic_test');
  } catch (e) {
    console.error('Mic test stop error:', e);
  }
  state.micTestActive = false;
  dom.btnMicTest.textContent = '🎙 テスト';
  dom.btnMicTest.classList.remove('active');
  if (state.unlistenMicTest) {
    state.unlistenMicTest();
    state.unlistenMicTest = null;
  }
}

dom.btnMicTest.addEventListener('click', async () => {
  if (state.micTestActive) {
    await stopMicTest();
  } else {
    await startMicTest();
  }
});

// BGM volume slider
dom.bgmVolume.addEventListener('input', async () => {
  const val = parseInt(dom.bgmVolume.value, 10);
  dom.bgmVolumeValue.textContent = val + '%';
  await invoke('set_accompaniment_volume', { volume: val / 100 }).catch(console.error);
  saveSettings({ bgmVolume: val });
});

// ---- UI Zoom ----
function applyZoom(pct) {
  const scale = pct / 100;
  document.body.style.zoom = scale;
  dom.uiZoom.value = pct;
  dom.uiZoomValue.textContent = pct + '%';
  saveSettings({ uiZoom: pct });
  // Resize canvas after zoom change
  if (renderer) {
    setTimeout(() => renderer.resize(), 50);
  }
}

dom.uiZoom.addEventListener('input', () => {
  applyZoom(parseInt(dom.uiZoom.value, 10));
});

dom.btnZoomIn.addEventListener('click', () => {
  const cur = parseInt(dom.uiZoom.value, 10);
  applyZoom(Math.min(cur + 10, 200));
});

dom.btnZoomOut.addEventListener('click', () => {
  const cur = parseInt(dom.uiZoom.value, 10);
  applyZoom(Math.max(cur - 10, 50));
});

dom.btnZoomReset.addEventListener('click', () => {
  applyZoom(100);
});

// ---- Init ----
document.addEventListener('DOMContentLoaded', async () => {
  renderer = new PitchRenderer(dom.canvas);
  // Draw initial empty state
  renderer.draw(0);

  // Initial borderless state (starts as decorations: false)
  dom.btnBorderless.classList.add('active');
  dom.btnBorderless.title = 'ボーダーレス (ON)';

  // Load and apply saved settings
  await applyStoredSettings();
});
