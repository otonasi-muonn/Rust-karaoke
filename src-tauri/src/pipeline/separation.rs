/// Vocal separation pipeline step
/// Uses BS-RoFormer to separate vocals from accompaniment
use crate::state::AppState;
use anyhow::Result;
use std::sync::Arc;

/// Separate vocals from the mixed audio
/// Returns (vocal_pcm, accompaniment_pcm) at 44100Hz mono
pub async fn separate_vocals(
    pcm: &[f32],
    state: &Arc<AppState>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    state.set_progress("separation", 0.5, "BS-RoFormerでボーカル分離中...");

    let pcm_owned = pcm.to_vec();

    // Run separation in a blocking thread pool (ONNX is sync)
    let result = tokio::task::spawn_blocking(move || {
        crate::inference::bsroformer::separate(&pcm_owned, 44100)
    })
    .await??;

    state.set_progress("separation", 0.75, "ボーカル分離完了");

    Ok(result)
}
