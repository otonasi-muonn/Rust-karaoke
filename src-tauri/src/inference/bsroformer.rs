/// BS-RoFormer inference for vocal separation
/// Separates audio into vocal and accompaniment tracks
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
        // Process in chunks of ~10 seconds with overlap
        let chunk_size = sample_rate as usize * 10;
        let overlap = sample_rate as usize; // 1s overlap
        let total_len = pcm.len();

        let mut vocal = vec![0.0f32; total_len];
        let mut accompaniment = vec![0.0f32; total_len];

        let mut pos = 0usize;
        while pos < total_len {
            let end = (pos + chunk_size).min(total_len);
            let chunk = &pcm[pos..end];
            let chunk_len = chunk.len();

            // Prepare input: [batch=1, channels=1, samples]
            let input_tensor = Tensor::from_array(([1i64, 1, chunk_len as i64], chunk.to_vec()))?;

            let outputs = sess.run(ort::inputs!["mix" => input_tensor])?;

            // Output: [batch=1, sources=2, samples]
            let output_view = outputs[0].try_extract_array::<f32>()?;

            // Copy results with crossfade in overlap regions
            for i in 0..chunk_len {
                let global_idx = pos + i;
                if global_idx >= total_len {
                    break;
                }

                // Simple crossfade for overlapping regions
                let fade = if pos > 0 && i < overlap {
                    i as f32 / overlap as f32
                } else {
                    1.0
                };

                // Access flattened: source 0 = vocal, source 1 = accompaniment
                let vocal_val = output_view.get([0, 0, i]).copied().unwrap_or(0.0);
                let acc_val = output_view.get([0, 1, i]).copied().unwrap_or(0.0);

                vocal[global_idx] =
                    vocal[global_idx] * (1.0 - fade) + vocal_val * fade;
                accompaniment[global_idx] =
                    accompaniment[global_idx] * (1.0 - fade) + acc_val * fade;
            }

            pos += chunk_size - overlap;
        }

        Ok((vocal, accompaniment))
    } else {
        // Fallback: simple spectral subtraction approximation
        // Return original as both vocal and accompaniment (no separation)
        log::warn!("Using fallback vocal separation (no model loaded)");
        Ok((pcm.to_vec(), pcm.to_vec()))
    }
}

fn get_model_path(model_name: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

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

    candidates[0].clone()
}
