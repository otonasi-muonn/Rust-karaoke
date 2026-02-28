/// FCPE (Fast Context-based Pitch Estimator) inference
/// Lightweight pitch extraction model (~10MB) with RTF ~0.0062
///
/// Input tensor shape: [batch_size=1, sequence_length]  (f32 PCM at 16kHz)
/// Output tensors:
///   - pitch: [batch_size, num_frames] (Hz values)
///   - voicing: [batch_size, num_frames] (probability 0..1)
use anyhow::Result;
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::types::PitchFrame;

/// FCPE model session (loaded once, reused)
static FCPE_SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));

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

/// Extract pitch from a single audio chunk
/// Returns (frequency_hz, is_voiced)
pub fn extract_pitch(audio: &[f32], sample_rate: u32) -> Result<(f64, bool)> {
    // Ensure session is initialized
    let _ = get_session();

    let mut session_guard = FCPE_SESSION.lock().unwrap();

    if let Some(ref mut sess) = *session_guard {
        // Determine input name from model metadata
        let input_name = sess
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "audio".to_string());

        // Build tensor: [batch=1, sequence_length]
        let len = audio.len();
        let input_tensor = Tensor::from_array(([1usize, len], audio.to_vec()))?;

        // Run inference
        let outputs = sess.run(ort::inputs![&input_name => input_tensor])?;

        // Parse output: pitch (Hz) array
        let pitch_view = outputs[0].try_extract_array::<f32>()?;
        let pitch_hz = pitch_view.iter().next().copied().unwrap_or(0.0) as f64;

        // Parse voicing probability if available
        let is_voiced = if outputs.len() > 1 {
            let voicing_view = outputs[1].try_extract_array::<f32>()?;
            let voicing_prob = voicing_view.iter().next().copied().unwrap_or(0.5);
            voicing_prob > 0.5 && pitch_hz > 50.0
        } else {
            pitch_hz > 50.0
        };

        Ok((pitch_hz, is_voiced))
    } else {
        // Fallback: simple autocorrelation-based pitch detection
        fallback_pitch_detection(audio, sample_rate)
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
/// Much more accurate than frame-by-frame because FCPE gets full temporal context
/// Input: mono f32 PCM at 16kHz
/// Returns: Vec of (frequency_hz, is_voiced) for each model hop frame
pub fn extract_pitch_batch(audio_16k: &[f32]) -> Result<Vec<(f64, bool)>> {
    let _ = get_session();
    let mut session_guard = FCPE_SESSION.lock().unwrap();

    if let Some(ref mut sess) = *session_guard {
        let input_name = sess
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "audio".to_string());

        let len = audio_16k.len();
        let input_tensor = Tensor::from_array(([1usize, len], audio_16k.to_vec()))?;
        let outputs = sess.run(ort::inputs![&input_name => input_tensor])?;

        // Collect ALL pitch values from model output [1, num_frames]
        let pitch_view = outputs[0].try_extract_array::<f32>()?;
        let pitch_values: Vec<f32> = pitch_view.iter().copied().collect();

        let voicing_values: Vec<f32> = if outputs.len() > 1 {
            outputs[1]
                .try_extract_array::<f32>()?
                .iter()
                .copied()
                .collect()
        } else {
            // No voicing output; use pitch threshold
            pitch_values
                .iter()
                .map(|&p| if p > 50.0 { 1.0 } else { 0.0 })
                .collect()
        };

        let mut result = Vec::with_capacity(pitch_values.len());
        for (&pitch, &voicing) in pitch_values.iter().zip(voicing_values.iter()) {
            let hz = pitch as f64;
            let voiced = voicing > 0.5 && hz > 50.0;
            result.push((hz, voiced));
        }

        log::debug!(
            "Batch extraction: {} input samples -> {} output frames",
            len,
            result.len()
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
        exe_dir.join("..").join("..").join("..").join("models").join(model_name), // target/release -> src-tauri/models
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
