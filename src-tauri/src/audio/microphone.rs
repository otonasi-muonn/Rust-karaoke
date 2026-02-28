/// Microphone capture module
/// Uses cpal for low-latency audio input with lock-free ring buffer
use crate::state::AppState;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static MIC_RUNNING: AtomicBool = AtomicBool::new(false);

use std::sync::Mutex as StdMutex;

/// Wrapper to make Stream sendable via Mutex
/// Safety: cpal::Stream is not Send, but we only access it from one thread at a time
/// through the Mutex. The stream callbacks run on OS audio threads independently.
#[allow(dead_code)]
struct SendStream(cpal::Stream);
unsafe impl Send for SendStream {}

static MIC_STREAM_HOLDER: once_cell::sync::Lazy<StdMutex<Option<SendStream>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(None));

/// Microphone volume gain (0.0 - 3.0)
static MIC_GAIN: once_cell::sync::Lazy<std::sync::atomic::AtomicU32> =
    once_cell::sync::Lazy::new(|| std::sync::atomic::AtomicU32::new(f32::to_bits(1.0)));

/// Selected microphone device name
static SELECTED_MIC: once_cell::sync::Lazy<StdMutex<Option<String>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(None));

/// Set microphone input gain
pub fn set_mic_gain(gain: f32) {
    MIC_GAIN.store(gain.to_bits(), Ordering::Relaxed);
}

/// Get current microphone input gain
pub fn get_mic_gain() -> f32 {
    f32::from_bits(MIC_GAIN.load(Ordering::Relaxed))
}

/// Set the selected microphone device name
pub fn set_selected_device(name: Option<String>) {
    let mut sel = SELECTED_MIC.lock().unwrap();
    *sel = name;
}

/// List available input devices
pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// Find the input device to use (selected or default)
fn get_input_device() -> Result<cpal::Device> {
    let host = cpal::default_host();
    let selected = SELECTED_MIC.lock().unwrap().clone();

    if let Some(ref name) = selected {
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(dname) = device.name() {
                    if dname == *name {
                        return Ok(device);
                    }
                }
            }
        }
        log::warn!("Selected mic '{}' not found, falling back to default", name);
    }

    host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("マイクデバイスが見つかりません"))
}

/// Start capturing audio from the selected (or default) input device
pub fn start_capture(state: &Arc<AppState>) -> Result<()> {
    if MIC_RUNNING.load(Ordering::Relaxed) {
        log::warn!("Microphone already running");
        return Ok(());
    }

    let device = get_input_device()?;
    log::info!("Using input device: {}", device.name()?);

    // Try 16kHz mono first; fall back to device default if unsupported
    let (config, native_channels, native_rate) = {
        let preferred = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16000),
            buffer_size: cpal::BufferSize::Fixed(512),
        };
        // Check if device supports preferred config
        let supported = device.supported_input_configs()
            .ok()
            .and_then(|mut cfgs| cfgs.find(|c| {
                c.channels() == 1
                    && c.min_sample_rate().0 <= 16000
                    && c.max_sample_rate().0 >= 16000
            }));
        if supported.is_some() {
            (preferred, 1u16, 16000u32)
        } else {
            let def = device.default_input_config()
                .map_err(|e| anyhow::anyhow!("デフォルト入力設定の取得に失敗: {}", e))?;
            let ch = def.channels();
            let sr = def.sample_rate().0;
            log::info!("Falling back to device default: {} ch, {} Hz", ch, sr);
            let cfg = cpal::StreamConfig {
                channels: ch,
                sample_rate: cpal::SampleRate(sr),
                buffer_size: cpal::BufferSize::Default,
            };
            (cfg, ch, sr)
        }
    };

    let sender = state
        .get_mic_sender()
        .ok_or_else(|| anyhow::anyhow!("Mic sender not available"))?;

    // Ring buffer for lock-free producer/consumer pattern
    let ring = ringbuf::HeapRb::<f32>::new(32000); // 2 seconds buffer
    let (mut producer, mut consumer) = ring.split();

    let state_clone = Arc::clone(state);
    MIC_RUNNING.store(true, Ordering::Relaxed);

    let ch = native_channels as usize;
    let need_resample = native_rate != 16000;
    let resample_ratio = if need_resample { 16000.0 / native_rate as f64 } else { 1.0 };

    // Audio callback thread - producer
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let gain = get_mic_gain();
            if !need_resample && ch == 1 {
                // Fast path: already 16kHz mono
                for &sample in data {
                    let _ = producer.try_push(sample * gain);
                }
            } else {
                // Extract mono from first channel
                let mono: Vec<f32> = data.chunks(ch)
                    .filter_map(|c| c.first().copied())
                    .map(|s| s * gain)
                    .collect();
                if need_resample {
                    // Simple linear resampling to 16kHz
                    let out_len = (mono.len() as f64 * resample_ratio) as usize;
                    for i in 0..out_len {
                        let src_idx = i as f64 / resample_ratio;
                        let idx0 = src_idx as usize;
                        let frac = (src_idx - idx0 as f64) as f32;
                        let s0 = mono.get(idx0).copied().unwrap_or(0.0);
                        let s1 = mono.get(idx0 + 1).copied().unwrap_or(s0);
                        let _ = producer.try_push(s0 + (s1 - s0) * frac);
                    }
                } else {
                    for &s in &mono {
                        let _ = producer.try_push(s);
                    }
                }
            }
        },
        move |err| {
            log::error!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    // Store stream to keep it alive
    *MIC_STREAM_HOLDER.lock().unwrap() = Some(SendStream(stream));
    state.set_mic_active(true);

    // Inference thread - consumer with sliding window for better pitch context
    std::thread::spawn(move || {
        let window_target: usize = 16000; // 1 second at 16kHz for rich context
        let hop_size: usize = 640; // 40ms hop - new samples per iteration
        let mut window: Vec<f32> = Vec::with_capacity(window_target);
        let mut hop_buf = vec![0.0f32; hop_size];
        let mut event_counter = 0u64;
        let mut prev_pitch_hz = 0.0f64; // For exponential smoothing
        let smooth_alpha = 0.6; // 0=full smooth, 1=no smooth

        while MIC_RUNNING.load(Ordering::Relaxed) && state_clone.is_playing() {
            // Collect hop_size new samples
            let available = consumer.occupied_len();
            if available < hop_size {
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }

            let count = consumer.pop_slice(&mut hop_buf);
            if count < hop_size {
                continue;
            }

            // Append new samples to sliding window
            window.extend_from_slice(&hop_buf[..hop_size]);
            if window.len() > window_target {
                let excess = window.len() - window_target;
                window.drain(..excess);
            }

            // Need at least 200ms (3200 samples) for reasonable extraction
            if window.len() < 3200 {
                continue;
            }

            // Run pitch extraction on full window, take last (most recent) frame
            match crate::inference::fcpe::extract_pitch_batch(&window) {
                Ok(frames) if !frames.is_empty() => {
                    let (raw_pitch_hz, is_voiced) = frames[frames.len() - 1];

                    // Exponential smoothing to reduce pitch jitter
                    let pitch_hz = if is_voiced && raw_pitch_hz > 0.0 {
                        if prev_pitch_hz > 0.0 {
                            // Check if pitch jumped too much (octave error)
                            let cents_diff = (1200.0 * (raw_pitch_hz / prev_pitch_hz).log2()).abs();
                            let adj_cents = cents_diff % 1200.0;
                            let adj_cents = if adj_cents > 600.0 { 1200.0 - adj_cents } else { adj_cents };

                            if adj_cents < 400.0 {
                                // Smooth: blend with previous
                                let smoothed = prev_pitch_hz.ln() * (1.0 - smooth_alpha) + raw_pitch_hz.ln() * smooth_alpha;
                                let result = smoothed.exp();
                                prev_pitch_hz = result;
                                result
                            } else {
                                // Large jump: accept new pitch immediately (note change)
                                prev_pitch_hz = raw_pitch_hz;
                                raw_pitch_hz
                            }
                        } else {
                            prev_pitch_hz = raw_pitch_hz;
                            raw_pitch_hz
                        }
                    } else {
                        prev_pitch_hz = 0.0;
                        raw_pitch_hz
                    };

                    let elapsed = if let Some(start) = state_clone.get_playback_start() {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64()
                            * 1000.0;
                        now - start
                    } else {
                        0.0
                    };

                    let data = crate::state::MicPitchData {
                        time_ms: elapsed,
                        pitch_hz,
                        is_voiced,
                    };

                    // Score this frame and emit events to frontend
                    let ref_pitches = state_clone.get_reference_pitches();
                    let (judgement, diff, ref_hz) = if !ref_pitches.is_empty() && is_voiced {
                        let (j, d, r) =
                            crate::scoring::score_frame(elapsed, pitch_hz, &ref_pitches);
                        // "---" means no reference at this time (rest) — don't count it
                        if j != "---" {
                            state_clone.update_score(&j, pitch_hz, r, d);
                        }
                        (j, d, r)
                    } else {
                        ("---".to_string(), f64::MAX, 0.0)
                    };

                    // Emit pitch-update event to frontend (every frame)
                    state_clone.emit_pitch_update(crate::state::PitchUpdateEvent {
                        time_ms: elapsed,
                        user_pitch_hz: pitch_hz,
                        ref_pitch_hz: ref_hz,
                        is_voiced,
                        judgement: judgement.clone(),
                        diff_cents: if diff == f64::MAX { -1.0 } else { diff },
                    });

                    // Emit score-update event every 5 frames (~200ms) to avoid flooding
                    event_counter += 1;
                    if event_counter % 5 == 0 {
                        state_clone.emit_score_update();
                    }

                    let _ = sender.try_send(data);
                }
                Ok(_) => {}
                Err(e) => {
                    log::debug!("Pitch extraction error: {}", e);
                }
            }
        }

        // Emit final score update
        state_clone.emit_score_update();
        log::info!("Inference thread stopped");
    });

    Ok(())
}

/// Stop microphone capture
pub fn stop_capture(state: &Arc<AppState>) {
    MIC_RUNNING.store(false, Ordering::Relaxed);
    // Drop the stream to release audio device
    *MIC_STREAM_HOLDER.lock().unwrap() = None;
    state.set_mic_active(false);
    log::info!("Microphone capture stopped");
}

// ---- Persistent Mic Test ----
static MIC_TEST_RUNNING: AtomicBool = AtomicBool::new(false);
static MIC_TEST_STREAM_HOLDER: once_cell::sync::Lazy<StdMutex<Option<SendStream>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(None));

/// Start persistent microphone test - streams level data via events until stopped
pub fn start_mic_test(state: &Arc<AppState>) -> Result<()> {
    // Stop any existing test first
    stop_mic_test();

    let device = get_input_device()?;
    let default_config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("デフォルト入力設定の取得に失敗: {}", e))?;

    let sample_format = default_config.sample_format();
    let config: cpal::StreamConfig = default_config.into();
    let channels = config.channels as usize;

    log::info!(
        "Mic test start: {} ch, {} Hz, {:?}",
        channels,
        config.sample_rate.0,
        sample_format
    );

    let level_buf = Arc::new(StdMutex::new(Vec::<f32>::new()));
    let buf_f32 = Arc::clone(&level_buf);
    let buf_i16 = Arc::clone(&level_buf);
    let buf_u16 = Arc::clone(&level_buf);

    MIC_TEST_RUNNING.store(true, Ordering::Relaxed);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !MIC_TEST_RUNNING.load(Ordering::Relaxed) {
                    return;
                }
                let gain = get_mic_gain();
                let mut s = buf_f32.lock().unwrap();
                for chunk in data.chunks(channels) {
                    if let Some(&sample) = chunk.first() {
                        s.push(sample * gain);
                    }
                }
            },
            move |err| log::error!("Mic test error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if !MIC_TEST_RUNNING.load(Ordering::Relaxed) {
                    return;
                }
                let gain = get_mic_gain();
                let mut s = buf_i16.lock().unwrap();
                for chunk in data.chunks(channels) {
                    if let Some(&sample) = chunk.first() {
                        s.push(sample as f32 / i16::MAX as f32 * gain);
                    }
                }
            },
            move |err| log::error!("Mic test error: {}", err),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                if !MIC_TEST_RUNNING.load(Ordering::Relaxed) {
                    return;
                }
                let gain = get_mic_gain();
                let mut s = buf_u16.lock().unwrap();
                for chunk in data.chunks(channels) {
                    if let Some(&sample) = chunk.first() {
                        s.push(((sample as f32 / u16::MAX as f32) * 2.0 - 1.0) * gain);
                    }
                }
            },
            move |err| log::error!("Mic test error: {}", err),
            None,
        )?,
        _ => {
            MIC_TEST_RUNNING.store(false, Ordering::Relaxed);
            return Err(anyhow::anyhow!(
                "サポートされていないサンプル形式: {:?}",
                sample_format
            ));
        }
    };

    stream.play()?;
    *MIC_TEST_STREAM_HOLDER.lock().unwrap() = Some(SendStream(stream));

    // Spawn level reporting thread (20 updates/sec)
    let state_clone = Arc::clone(state);
    std::thread::spawn(move || {
        while MIC_TEST_RUNNING.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(50));

            let samples: Vec<f32> = {
                let mut buf = level_buf.lock().unwrap();
                let v = std::mem::take(&mut *buf);
                v
            };

            if samples.is_empty() {
                continue;
            }

            let rms = (samples
                .iter()
                .map(|&s| (s as f64) * (s as f64))
                .sum::<f64>()
                / samples.len() as f64)
                .sqrt() as f32;
            let level = (rms * 10.0).min(1.0);
            state_clone.emit_mic_test_level(level);
        }
        log::info!("Mic test level thread stopped");
    });

    Ok(())
}

/// Stop persistent microphone test
pub fn stop_mic_test() {
    MIC_TEST_RUNNING.store(false, Ordering::Relaxed);
    *MIC_TEST_STREAM_HOLDER.lock().unwrap() = None;
}

/// Check if mic test is currently running
pub fn is_mic_test_running() -> bool {
    MIC_TEST_RUNNING.load(Ordering::Relaxed)
}
