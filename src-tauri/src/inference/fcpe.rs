/// FCPE (Fast Context-based Pitch Estimator) inference
/// Lightweight pitch extraction model (~10MB) with RTF ~0.0062
///
/// Model inputs:
///   - mel: [batch=1, n_mels=128, n_frames] mel spectrogram
///   - threshold: [1] voicing threshold (default: 0.006)
/// Model output:
///   - pitchf: [batch=1, n_frames] pitch in Hz (0 = unvoiced)
///
/// Mel spectrogram parameters (standard torchfcpe):
///   sr=16000, n_fft=1024, hop=160, win=1024, n_mels=128, fmin=0, fmax=8000
use anyhow::Result;
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::types::PitchFrame;

// Mel spectrogram constants matching torchfcpe
const MEL_SR: u32 = 16000;
const MEL_N_FFT: usize = 1024;
const MEL_HOP: usize = 160;
const MEL_WIN: usize = 1024;
const MEL_N_MELS: usize = 128;
const MEL_FMIN: f64 = 0.0;
const MEL_FMAX: f64 = 8000.0;
const VOICING_THRESHOLD: f32 = 0.02;
/// Model expects exactly 128 mel frames
const FCPE_N_FRAMES: usize = 128;
/// Audio samples needed for 128 mel frames: (128-1)*160 = 20320
const FCPE_CHUNK_SAMPLES: usize = (FCPE_N_FRAMES - 1) * MEL_HOP;

/// FCPE model session (loaded once, reused)
static FCPE_SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));

/// Pre-computed mel filterbank [n_mels, n_fft/2+1]
static MEL_FILTERBANK: Lazy<Vec<Vec<f64>>> = Lazy::new(|| {
    create_mel_filterbank(MEL_SR, MEL_N_FFT, MEL_N_MELS, MEL_FMIN, MEL_FMAX)
});

/// Convert Hz to mel scale
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Create an [n_mels, n_fft/2+1] mel filterbank (librosa-compatible)
fn create_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 evenly spaced mel points
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices (fractional)
    let fft_bins: Vec<f64> = hz_points
        .iter()
        .map(|&h| h * n_fft as f64 / sr as f64)
        .collect();

    let mut filterbank = vec![vec![0.0f64; n_freqs]; n_mels];

    for m in 0..n_mels {
        let f_left = fft_bins[m];
        let f_center = fft_bins[m + 1];
        let f_right = fft_bins[m + 2];

        for k in 0..n_freqs {
            let freq = k as f64;
            if freq >= f_left && freq <= f_center && f_center > f_left {
                filterbank[m][k] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                filterbank[m][k] = (f_right - freq) / (f_right - f_center);
            }
        }

        // Slaney normalization (librosa default)
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);
        for k in 0..n_freqs {
            filterbank[m][k] *= enorm;
        }
    }

    filterbank
}

/// Compute mel spectrogram from 16kHz mono PCM, always producing exactly `target_frames` frames.
/// Audio is zero-padded or truncated as needed.
/// Returns: flattened [n_mels, target_frames] in row-major order
fn compute_mel_spectrogram_fixed(audio: &[f32], target_frames: usize) -> Vec<f32> {
    let n_freqs = MEL_N_FFT / 2 + 1;
    let filterbank = &*MEL_FILTERBANK;

    // Pad audio to center frames (like librosa pad_mode='reflect')
    let pad = MEL_N_FFT / 2;
    let padded_len = audio.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    // Reflect padding at start
    for i in 0..pad {
        let src_idx = if audio.is_empty() { 0 } else { (pad - 1 - i) % audio.len() };
        padded[i] = audio[src_idx.min(audio.len().saturating_sub(1))];
    }
    if !audio.is_empty() {
        padded[pad..pad + audio.len()].copy_from_slice(audio);
    }
    // Reflect padding at end
    for i in 0..pad {
        let src_idx = if audio.is_empty() { 0 } else { (audio.len().saturating_sub(1).saturating_sub(i)) % audio.len() };
        padded[pad + audio.len() + i] = audio[src_idx.min(audio.len().saturating_sub(1))];
    }

    // FFT planner
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(MEL_N_FFT);

    // Hann window
    let window: Vec<f64> = (0..MEL_WIN)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / MEL_WIN as f64).cos()))
        .collect();

    // Output: exactly target_frames frames, zero-padded if audio is short
    let mut mel_spec = vec![0.0f32; MEL_N_MELS * target_frames];
    // Minimum log-mel value for padding frames
    let pad_val = (1e-10_f64).ln() as f32;

    let actual_frames = if padded_len >= MEL_WIN {
        (padded_len - MEL_WIN) / MEL_HOP + 1
    } else {
        0
    };
    let compute_frames = actual_frames.min(target_frames);

    for frame_idx in 0..compute_frames {
        let start = frame_idx * MEL_HOP;
        if start + MEL_WIN > padded_len {
            break;
        }

        // Windowed frame
        let mut fft_buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); MEL_N_FFT];
        for i in 0..MEL_WIN {
            fft_buf[i] = Complex::new(padded[start + i] as f64 * window[i], 0.0);
        }

        // FFT
        fft.process(&mut fft_buf);

        // Power spectrum (magnitude squared)
        let power: Vec<f64> = fft_buf[..n_freqs]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Apply mel filterbank and log scale
        for mel_idx in 0..MEL_N_MELS {
            let mut energy = 0.0f64;
            for k in 0..n_freqs {
                energy += filterbank[mel_idx][k] * power[k];
            }
            let log_mel = (energy.max(1e-10)).ln() as f32;
            mel_spec[mel_idx * target_frames + frame_idx] = log_mel;
        }
    }

    // Fill remaining frames with silence value
    for frame_idx in compute_frames..target_frames {
        for mel_idx in 0..MEL_N_MELS {
            mel_spec[mel_idx * target_frames + frame_idx] = pad_val;
        }
    }

    mel_spec
}

/// Get or initialize the FCPE ONNX session
fn get_session() -> Result<()> {
    let mut session = FCPE_SESSION.lock().unwrap();
    if session.is_none() {
        let model_path = get_model_path("fcpe.onnx");
        if !model_path.exists() {
            log::warn!(
                "FCPE model not found at {:?}. Using fallback pitch estimation.",
                model_path
            );
            return Ok(());
        }

        log::info!("Loading FCPE model from {:?}", model_path);
        let s = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;

        // Log model I/O for debugging shape mismatches
        for input in s.inputs().iter() {
            log::info!("  FCPE input: '{}'", input.name());
        }
        for output in s.outputs().iter() {
            log::info!("  FCPE output: '{}'", output.name());
        }

        *session = Some(s);
        log::info!("FCPE model loaded successfully");
    }
    Ok(())
}

/// Extract pitch from a single audio chunk (16kHz mono)
/// Returns (frequency_hz, is_voiced)
pub fn extract_pitch(audio: &[f32], _sample_rate: u32) -> Result<(f64, bool)> {
    let _ = get_session();
    let mut session_guard = FCPE_SESSION.lock().unwrap();

    if let Some(ref mut sess) = *session_guard {
        // Compute mel spectrogram with exactly FCPE_N_FRAMES frames
        let mel_data = compute_mel_spectrogram_fixed(audio, FCPE_N_FRAMES);

        // Build mel tensor: [batch=1, n_mels=128, 128]
        let mel_tensor = Tensor::from_array(([1usize, MEL_N_MELS, FCPE_N_FRAMES], mel_data))?;
        let threshold_tensor = Tensor::from_array(([1usize], vec![VOICING_THRESHOLD]))?;

        let outputs = sess.run(ort::inputs!["mel" => mel_tensor, "threshold" => threshold_tensor])?;

        // Parse output: pitchf [1, 128] — Hz values (0 = unvoiced)
        // Take the last frame as the most recent pitch
        let pitch_view = outputs[0].try_extract_array::<f32>()?;
        let pitch_values: Vec<f32> = pitch_view.iter().copied().collect();
        // Find last non-zero pitch from valid frames
        let actual_frames = (audio.len() / MEL_HOP).min(FCPE_N_FRAMES);
        let pitch_hz = pitch_values.iter().take(actual_frames.max(1)).rev()
            .find(|&&p| p > 50.0)
            .copied()
            .unwrap_or(0.0) as f64;
        let is_voiced = pitch_hz > 50.0;

        Ok((pitch_hz, is_voiced))
    } else {
        fallback_pitch_detection(audio, 16000)
    }
}

/// Batch pitch extraction: process longer audio and return PitchFrame vector
/// This maps FCPE output directly to the types::PitchFrame struct
/// as required by Phase 3 step 2 of the plan.
///
/// `vocal_pcm`: mono f32 at `sample_rate` Hz
/// `hop_ms`: analysis hop in milliseconds (e.g. 10.0)
/// `frame_ms`: analysis frame in milliseconds (e.g. 40.0)
pub fn extract_pitch_frames(
    vocal_pcm: &[f32],
    sample_rate: u32,
    hop_ms: f64,
    frame_ms: f64,
) -> Result<Vec<PitchFrame>> {
    let hop_samples = ((hop_ms / 1000.0) * sample_rate as f64) as usize;
    let frame_samples = ((frame_ms / 1000.0) * sample_rate as f64) as usize;
    let total_frames = vocal_pcm.len().saturating_sub(frame_samples) / hop_samples;

    let mut frames = Vec::with_capacity(total_frames);

    for i in 0..total_frames {
        let start = i * hop_samples;
        let end = (start + frame_samples).min(vocal_pcm.len());
        if end - start < frame_samples / 2 {
            break;
        }

        let chunk = &vocal_pcm[start..end];
        let time_ms = (start as f64 / sample_rate as f64) * 1000.0;

        match extract_pitch(chunk, sample_rate) {
            Ok((pitch_hz, is_voiced)) => {
                frames.push(PitchFrame {
                    time_ms,
                    pitch_hz,
                    is_voiced,
                });
            }
            Err(e) => {
                log::debug!("Pitch extraction error at {:.1}ms: {}", time_ms, e);
                frames.push(PitchFrame {
                    time_ms,
                    pitch_hz: 0.0,
                    is_voiced: false,
                });
            }
        }
    }

    Ok(frames)
}

/// Batch extraction: process a long audio buffer and return ALL output frames
/// Processes in fixed 128-frame chunks (FCPE model requires exactly 128 mel frames)
/// Uses 50% overlap between chunks for smooth results
/// Input: mono f32 PCM at 16kHz
/// Returns: Vec of (frequency_hz, is_voiced) for each hop frame
pub fn extract_pitch_batch(audio_16k: &[f32]) -> Result<Vec<(f64, bool)>> {
    let _ = get_session();
    let mut session_guard = FCPE_SESSION.lock().unwrap();

    if let Some(ref mut sess) = *session_guard {
        // Process in chunks of FCPE_CHUNK_SAMPLES with 50% overlap
        let chunk_size = FCPE_CHUNK_SAMPLES; // 20320 samples = ~1.27s
        let chunk_step = chunk_size / 2; // 50% overlap
        let total_frames_needed = audio_16k.len() / MEL_HOP; // approximate total output frames

        let mut all_pitches = vec![0.0f32; total_frames_needed + FCPE_N_FRAMES];
        let mut all_weights = vec![0.0f32; total_frames_needed + FCPE_N_FRAMES];

        let mut pos = 0usize;
        let mut chunk_idx = 0usize;
        let threshold_tensor = Tensor::from_array(([1usize], vec![VOICING_THRESHOLD]))?;

        while pos < audio_16k.len() {
            let end = (pos + chunk_size).min(audio_16k.len());
            let chunk = &audio_16k[pos..end];

            // Compute mel with exactly 128 frames (zero-padded if chunk is short)
            let mel_data = compute_mel_spectrogram_fixed(chunk, FCPE_N_FRAMES);
            let mel_tensor = Tensor::from_array(([1usize, MEL_N_MELS, FCPE_N_FRAMES], mel_data))?;
            let threshold_copy = Tensor::from_array(([1usize], vec![VOICING_THRESHOLD]))?;

            let outputs = sess.run(ort::inputs!["mel" => mel_tensor, "threshold" => threshold_copy])?;

            let pitch_view = outputs[0].try_extract_array::<f32>()?;
            let pitch_values: Vec<f32> = pitch_view.iter().copied().collect();

            if chunk_idx == 0 {
                let pitch_shape = pitch_view.shape().to_vec();
                log::info!(
                    "FCPE: chunk_size={}, n_frames={}, output shape {:?}, voiced={}",
                    chunk_size, FCPE_N_FRAMES, pitch_shape,
                    pitch_values.iter().filter(|&&p| p > 50.0).count()
                );
            }

            // Map chunk frames back to global frame indices
            let actual_audio_frames = ((end - pos) / MEL_HOP).min(FCPE_N_FRAMES);
            let global_frame_offset = pos / MEL_HOP;

            for i in 0..actual_audio_frames.min(pitch_values.len()) {
                let global_idx = global_frame_offset + i;
                if global_idx < all_pitches.len() {
                    // Use triangular window for overlap-add blending
                    let weight = if i < actual_audio_frames / 4 {
                        i as f32 / (actual_audio_frames as f32 / 4.0)
                    } else if i > actual_audio_frames * 3 / 4 {
                        (actual_audio_frames - i) as f32 / (actual_audio_frames as f32 / 4.0)
                    } else {
                        1.0
                    };
                    all_pitches[global_idx] += pitch_values[i] * weight;
                    all_weights[global_idx] += weight;
                }
            }

            chunk_idx += 1;
            if pos + chunk_size >= audio_16k.len() {
                break;
            }
            pos += chunk_step;
        }

        // Normalize overlapping regions and build result
        let output_frames = total_frames_needed;
        let mut result = Vec::with_capacity(output_frames);
        let mut voiced_count = 0;
        for i in 0..output_frames {
            let hz = if all_weights[i] > 0.0 {
                (all_pitches[i] / all_weights[i]) as f64
            } else {
                0.0
            };
            let voiced = hz > 50.0;
            if voiced { voiced_count += 1; }
            result.push((hz, voiced));
        }

        log::info!(
            "FCPE batch: {} samples -> {} chunks -> {} frames ({} voiced)",
            audio_16k.len(), chunk_idx, output_frames, voiced_count
        );

        Ok(result)
    } else {
        // Fallback: frame-by-frame autocorrelation
        let hop = 160;
        let frame_size = 640;
        let total = audio_16k.len().saturating_sub(frame_size) / hop;
        let mut result = Vec::with_capacity(total);
        for i in 0..total {
            let start = i * hop;
            let end = (start + frame_size).min(audio_16k.len());
            if end - start < frame_size / 2 {
                break;
            }
            match fallback_pitch_detection(&audio_16k[start..end], 16000) {
                Ok((hz, voiced)) => result.push((hz, voiced)),
                Err(_) => result.push((0.0, false)),
            }
        }
        Ok(result)
    }
}

/// Fallback pitch detection using the YIN algorithm
/// High-accuracy pitch detection without ML model
/// Reference: de Cheveigné & Kawahara (2002) "YIN, a fundamental frequency estimator"
fn fallback_pitch_detection(audio: &[f32], sample_rate: u32) -> Result<(f64, bool)> {
    let n = audio.len();
    if n < 128 {
        return Ok((0.0, false));
    }

    // RMS voicing gate
    let rms: f64 = (audio.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / n as f64).sqrt();
    if rms < 0.005 {
        return Ok((0.0, false));
    }

    let min_freq = 60.0; // Hz, lowest pitch we detect
    let max_freq = 1100.0; // Hz, highest pitch we detect
    let min_lag = (sample_rate as f64 / max_freq) as usize;
    let max_lag = ((sample_rate as f64 / min_freq) as usize).min(n / 2);

    if min_lag >= max_lag || max_lag < 2 {
        return Ok((0.0, false));
    }

    // Step 1: Compute the difference function d(tau)
    let w = max_lag; // analysis window size
    let mut d = vec![0.0f64; max_lag + 1];
    d[0] = 0.0;

    // Efficient cumulative computation
    for tau in 1..=max_lag {
        let mut sum = 0.0f64;
        let limit = w.min(n - tau);
        for j in 0..limit {
            let diff = audio[j] as f64 - audio[j + tau] as f64;
            sum += diff * diff;
        }
        d[tau] = sum;
    }

    // Step 2: Cumulative mean normalized difference function d'(tau)
    let mut d_prime = vec![1.0f64; max_lag + 1];
    d_prime[0] = 1.0;
    let mut running_sum = 0.0f64;
    for tau in 1..=max_lag {
        running_sum += d[tau];
        if running_sum > 0.0 {
            d_prime[tau] = d[tau] * tau as f64 / running_sum;
        } else {
            d_prime[tau] = 1.0;
        }
    }

    // Step 3: Absolute threshold — find first dip below threshold
    let yin_threshold = 0.15; // Lower = stricter voicing
    let mut best_tau = 0usize;

    // Search for first minimum below threshold (skip tau < min_lag)
    let mut tau = min_lag;
    while tau < max_lag {
        if d_prime[tau] < yin_threshold {
            // Find the local minimum in this valley
            while tau + 1 < max_lag && d_prime[tau + 1] < d_prime[tau] {
                tau += 1;
            }
            best_tau = tau;
            break;
        }
        tau += 1;
    }

    // If no dip below threshold, find global minimum as fallback
    if best_tau == 0 {
        let mut min_val = f64::MAX;
        for tau in min_lag..max_lag {
            if d_prime[tau] < min_val {
                min_val = d_prime[tau];
                best_tau = tau;
            }
        }
        // Require reasonably low value
        if min_val > 0.5 {
            return Ok((0.0, false));
        }
    }

    if best_tau == 0 {
        return Ok((0.0, false));
    }

    // Step 4: Parabolic interpolation for sub-sample accuracy
    let tau_f = if best_tau > min_lag && best_tau + 1 < max_lag {
        let s0 = d_prime[best_tau - 1];
        let s1 = d_prime[best_tau];
        let s2 = d_prime[best_tau + 1];
        let denom = 2.0 * s1 - s2 - s0;
        if denom.abs() > 1e-12 {
            best_tau as f64 + (s0 - s2) / (2.0 * denom)
        } else {
            best_tau as f64
        }
    } else {
        best_tau as f64
    };

    let pitch_hz = sample_rate as f64 / tau_f;

    if pitch_hz < min_freq || pitch_hz > max_freq {
        return Ok((0.0, false));
    }

    let confidence = 1.0 - d_prime[best_tau];
    Ok((pitch_hz, confidence > 0.5))
}

/// Get the model file path - searches multiple locations for release/dev builds
fn get_model_path(model_name: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    // Check multiple locations including workspace root for release builds
    let candidates = vec![
        exe_dir.join("models").join(model_name),
        exe_dir.join("..").join("..").join("models").join(model_name), // target/release/../../models = src-tauri/models
        exe_dir.join("..").join("..").join("..").join("models").join(model_name), // workspace/models
        exe_dir.join("..").join("..").join("..").join("src-tauri").join("models").join(model_name), // workspace/src-tauri/models
        PathBuf::from("models").join(model_name),
        PathBuf::from("src-tauri/models").join(model_name),
        PathBuf::from("src-tauri").join("models").join(model_name),
    ];

    for path in &candidates {
        if let Ok(canonical) = path.canonicalize() {
            log::info!("Found model at: {:?}", canonical);
            return canonical;
        }
    }

    log::warn!("Model '{}' not found in any candidate paths:", model_name);
    for path in &candidates {
        log::warn!("  tried: {:?} (exists={})", path, path.exists());
    }

    // Return first candidate (may not exist — triggers fallback)
    candidates[0].clone()
}
