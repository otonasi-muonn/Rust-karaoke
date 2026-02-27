/// Reference pitch extraction pipeline step
/// Uses FCPE to extract pitch map from vocal track
use crate::state::PitchPoint;
use anyhow::Result;

/// Extract reference pitch data from the vocal track
/// Returns a vector of PitchPoints with time, frequency, and voicing info
pub fn extract_reference_pitches(vocal_pcm: &[f32]) -> Result<Vec<PitchPoint>> {
    let sample_rate = 44100u32;
    let _hop_size = 441; // 10ms hop at 44100Hz
    let _frame_size = 1024; // ~23ms window

    // Resample to 16kHz for FCPE
    let resampled = resample_to_16k(vocal_pcm, sample_rate);
    let analysis_sr = 16000u32;
    let analysis_hop = 160; // 10ms hop at 16kHz
    let analysis_frame = 640; // 40ms window at 16kHz

    let mut pitches = Vec::new();
    let total_frames = resampled.len().saturating_sub(analysis_frame) / analysis_hop;

    log::info!(
        "Extracting reference pitches: {} frames from {} samples",
        total_frames,
        resampled.len()
    );

    for i in 0..total_frames {
        let start = i * analysis_hop;
        let end = (start + analysis_frame).min(resampled.len());

        if end - start < analysis_frame / 2 {
            break;
        }

        let chunk = &resampled[start..end];
        let time_ms = (start as f64 / analysis_sr as f64) * 1000.0;

        match crate::inference::fcpe::extract_pitch(chunk, analysis_sr) {
            Ok((pitch_hz, is_voiced)) => {
                pitches.push(PitchPoint {
                    time_ms,
                    pitch_hz,
                    is_voiced,
                });
            }
            Err(e) => {
                log::debug!("Pitch extraction error at {:.1}ms: {}", time_ms, e);
                pitches.push(PitchPoint {
                    time_ms,
                    pitch_hz: 0.0,
                    is_voiced: false,
                });
            }
        }

        // Log progress periodically
        if i % 1000 == 0 && total_frames > 0 {
            let pct = i as f64 / total_frames as f64 * 100.0;
            log::debug!("Pitch extraction: {:.1}%", pct);
        }
    }

    log::info!(
        "Extracted {} pitch points ({} voiced)",
        pitches.len(),
        pitches.iter().filter(|p| p.is_voiced).count()
    );

    Ok(pitches)
}

/// Resample from source rate to 16kHz using linear interpolation
fn resample_to_16k(input: &[f32], from_rate: u32) -> Vec<f32> {
    let to_rate = 16000u32;
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx + 1 < input.len() {
            output.push(input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac);
        } else if src_idx < input.len() {
            output.push(input[src_idx]);
        }
    }

    output
}
