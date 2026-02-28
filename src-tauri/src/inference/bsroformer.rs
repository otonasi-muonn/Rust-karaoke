/// BS-RoFormer inference for vocal separation
/// Separates audio into vocal and accompaniment tracks
///
/// Input tensor shape: [batch_size=1, channels=1, sequence_length]  (f32)
/// Output tensor shape: [batch_size=1, sources=2, sequence_length]  (f32)
///   - source 0 = vocal, source 1 = accompaniment
use anyhow::Result;
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;

/// BS-RoFormer session (loaded once)
static BSROFORMER_SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));

/// Initialize the BS-RoFormer ONNX session
fn get_session() -> Result<()> {
    let mut session = BSROFORMER_SESSION.lock().unwrap();
    if session.is_none() {
        let model_path = get_model_path("bsroformer.onnx");
        if !model_path.exists() {
            log::warn!(
                "BS-RoFormer model not found at {:?}. Using fallback separation.",
                model_path
            );
            return Ok(());
        }

        log::info!("Loading BS-RoFormer model from {:?}", model_path);
        let s = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;

        // Log model input/output shapes for debugging
        for input in s.inputs().iter() {
            log::info!("  Model input: '{}'", input.name());
        }
        for output in s.outputs().iter() {
            log::info!("  Model output: '{}'", output.name());
        }

        *session = Some(s);
        log::info!("BS-RoFormer model loaded successfully");
    }
    Ok(())
}

/// Separate vocals from accompaniment
/// Input: mono PCM f32 at 44100Hz
/// Output: (vocal, accompaniment) as Vec<f32>
pub fn separate(pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, Vec<f32>)> {
    let _ = get_session();
    let mut session_guard = BSROFORMER_SESSION.lock().unwrap();

    if let Some(ref mut sess) = *session_guard {
        // Determine input name from model metadata
        let input_name = sess
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "mix".to_string());

        // Process in chunks of ~10 seconds with 1s overlap for crossfade
        let chunk_size = sample_rate as usize * 10;
        let overlap = sample_rate as usize; // 1s overlap
        let total_len = pcm.len();

        let mut vocal = vec![0.0f32; total_len];
        let mut accompaniment = vec![0.0f32; total_len];

        let mut pos = 0usize;
        let mut chunk_idx = 0usize;
        while pos < total_len {
            let end = (pos + chunk_size).min(total_len);
            let chunk = &pcm[pos..end];
            let chunk_len = chunk.len();

            // Build tensor: [batch=1, channels=1, samples]
            let input_tensor = Tensor::from_array(([1usize, 1, chunk_len], chunk.to_vec()))?;

            let outputs = sess.run(ort::inputs![&input_name => input_tensor])?;

            // Output: [batch=1, sources=2, samples]
            let output_view = outputs[0].try_extract_array::<f32>()?;

            // Copy results with crossfade in overlap regions
            for i in 0..chunk_len {
                let global_idx = pos + i;
                if global_idx >= total_len {
                    break;
                }

                // Linear crossfade for overlapping regions (smooth transition)
                let fade = if pos > 0 && i < overlap {
                    i as f32 / overlap as f32
                } else {
                    1.0
                };

                // Access via ndarray indexing: [batch=0, source, sample]
                let vocal_val = output_view
                    .get([0, 0, i])
                    .copied()
                    .unwrap_or(0.0);
                let acc_val = output_view
                    .get([0, 1, i])
                    .copied()
                    .unwrap_or(0.0);

                vocal[global_idx] =
                    vocal[global_idx] * (1.0 - fade) + vocal_val * fade;
                accompaniment[global_idx] =
                    accompaniment[global_idx] * (1.0 - fade) + acc_val * fade;
            }

            chunk_idx += 1;
            if chunk_idx % 5 == 0 {
                log::info!(
                    "BS-RoFormer progress: {:.1}%",
                    (pos as f64 / total_len as f64) * 100.0
                );
            }

            pos += chunk_size - overlap;
        }

        log::info!("BS-RoFormer separation complete: {} samples processed", total_len);
        Ok((vocal, accompaniment))
    } else {
        // Fallback: apply bandpass filter to isolate vocal frequencies
        // Without BS-RoFormer, we at least remove bass and high-frequency instruments
        log::warn!("Using fallback vocal separation (no model loaded) — applying bandpass filter");
        let vocal = bandpass_filter_vocal(pcm, sample_rate);
        Ok((vocal, pcm.to_vec()))
    }
}

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
            log::info!("Found BS-RoFormer model at: {:?}", canonical);
            return canonical;
        }
    }

    log::warn!("BS-RoFormer model '{}' not found in any candidate paths:", model_name);
    for path in &candidates {
        log::warn!("  tried: {:?} (exists={})", path, path.exists());
    }

    // Return first candidate (may not exist — triggers fallback)
    candidates[0].clone()
}

/// Fallback vocal isolation using cascaded IIR bandpass filter
/// Removes sub-bass (<150Hz) and high-frequency instruments (>4kHz)
/// This is a crude approximation but much better than no separation
fn bandpass_filter_vocal(pcm: &[f32], sample_rate: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }

    let len = pcm.len();

    // High-pass filter at 150Hz (remove bass guitar, kick drum)
    let hp_cutoff = 150.0;
    let rc_hp = 1.0 / (2.0 * std::f64::consts::PI * hp_cutoff);
    let dt = 1.0 / sample_rate as f64;
    let alpha_hp = rc_hp / (rc_hp + dt);

    let mut hp1 = vec![0.0f64; len];
    hp1[0] = pcm[0] as f64;
    for i in 1..len {
        hp1[i] = alpha_hp * (hp1[i - 1] + pcm[i] as f64 - pcm[i - 1] as f64);
    }

    // Second pass for steeper roll-off
    let mut hp2 = vec![0.0f64; len];
    hp2[0] = hp1[0];
    for i in 1..len {
        hp2[i] = alpha_hp * (hp2[i - 1] + hp1[i] - hp1[i - 1]);
    }

    // Low-pass filter at 4kHz (remove cymbals, hi-hats, high harmonics)
    let lp_cutoff = 4000.0;
    let rc_lp = 1.0 / (2.0 * std::f64::consts::PI * lp_cutoff);
    let alpha_lp = dt / (rc_lp + dt);

    let mut lp1 = vec![0.0f64; len];
    lp1[0] = hp2[0];
    for i in 1..len {
        lp1[i] = lp1[i - 1] + alpha_lp * (hp2[i] - lp1[i - 1]);
    }

    // Second pass
    let mut result = vec![0.0f32; len];
    result[0] = lp1[0] as f32;
    let mut prev = lp1[0];
    for i in 1..len {
        prev = prev + alpha_lp * (lp1[i] - prev);
        result[i] = prev as f32;
    }

    log::info!(
        "Fallback bandpass filter applied: {}Hz-{}Hz ({} samples)",
        hp_cutoff,
        lp_cutoff,
        len
    );
    result
}
