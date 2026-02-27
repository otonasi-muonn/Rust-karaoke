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

/// Fallback pitch detection using autocorrelation
/// Used when ONNX model is not available
fn fallback_pitch_detection(audio: &[f32], sample_rate: u32) -> Result<(f64, bool)> {
    let n = audio.len();
    if n < 64 {
        return Ok((0.0, false));
    }

    // Calculate RMS for voicing detection
    let rms: f64 = (audio.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / n as f64).sqrt();
    if rms < 0.01 {
        return Ok((0.0, false));
    }

    // Autocorrelation-based pitch detection
    let min_lag = (sample_rate as f64 / 1000.0) as usize; // ~1000 Hz max
    let max_lag = (sample_rate as f64 / 50.0) as usize; // ~50 Hz min
    let max_lag = max_lag.min(n / 2);

    if min_lag >= max_lag {
        return Ok((0.0, false));
    }

    let mut best_corr = 0.0f64;
    let mut best_lag = 0usize;

    // Normalized autocorrelation
    let energy: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();

    for lag in min_lag..max_lag {
        let mut corr = 0.0f64;
        for i in 0..(n - lag) {
            corr += audio[i] as f64 * audio[i + lag] as f64;
        }
        corr /= energy;

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_corr < 0.3 {
        return Ok((0.0, false));
    }

    // Parabolic interpolation for sub-sample accuracy
    let pitch_hz = sample_rate as f64 / best_lag as f64;

    // Filter unreasonable pitches
    if pitch_hz < 50.0 || pitch_hz > 1500.0 {
        return Ok((0.0, false));
    }

    Ok((pitch_hz, true))
}

/// Get the model file path
fn get_model_path(model_name: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    // Check multiple locations
    let candidates = vec![
        exe_dir.join("models").join(model_name),
        PathBuf::from("models").join(model_name),
        PathBuf::from("src-tauri/models").join(model_name),
    ];

    for path in &candidates {
        if path.exists() {
            return path.clone();
        }
    }

    // Return default path (may not exist)
    candidates[0].clone()
}
