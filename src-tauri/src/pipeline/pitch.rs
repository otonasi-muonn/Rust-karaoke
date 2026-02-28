/// Reference pitch extraction pipeline step
/// Uses FCPE batch processing for high-accuracy pitch extraction
use crate::state::PitchPoint;
use anyhow::Result;

/// Extract reference pitch data from the vocal track
/// Processes in 30-second segments with 2-second overlap for optimal accuracy
/// This leverages FCPE's temporal context for much better results than frame-by-frame
pub fn extract_reference_pitches(vocal_pcm: &[f32]) -> Result<Vec<PitchPoint>> {
    let resampled = resample_to_16k(vocal_pcm, 44100);
    let sr = 16000u32;
    let total_s = resampled.len() as f64 / sr as f64;

    log::info!(
        "Extracting reference pitches: {} samples ({:.1}s)",
        resampled.len(),
        total_s
    );

    // Process in 30-second segments with 2-second overlap
    let seg_samples = sr as usize * 30;
    let overlap = sr as usize * 2;
    let step = seg_samples.saturating_sub(overlap).max(sr as usize);

    let mut all_pitches = Vec::new();
    let mut offset = 0usize;
    let mut segment_idx = 0;

    loop {
        let end = (offset + seg_samples).min(resampled.len());
        let segment = &resampled[offset..end];

        // Skip segments shorter than 0.5 seconds
        if segment.len() < sr as usize / 2 {
            break;
        }

        let pct = (offset as f64 / resampled.len() as f64 * 100.0).min(100.0);
        log::info!(
            "  Segment {} at {:.1}s - {:.1}s ({:.0}%)",
            segment_idx,
            offset as f64 / sr as f64,
            end as f64 / sr as f64,
            pct
        );

        match crate::inference::fcpe::extract_pitch_batch(segment) {
            Ok(frames) if !frames.is_empty() => {
                // Infer hop size from output
                let hop = segment.len() as f64 / frames.len() as f64;

                // Skip overlap region with previous segment
                let skip = if offset > 0 {
                    ((overlap as f64 / 2.0) / hop).ceil() as usize
                } else {
                    0
                };

                for (i, &(hz, voiced)) in frames.iter().enumerate().skip(skip) {
                    let sample_pos = offset as f64 + i as f64 * hop;
                    let time_ms = sample_pos / sr as f64 * 1000.0;
                    all_pitches.push(PitchPoint {
                        time_ms,
                        pitch_hz: hz,
                        is_voiced: voiced,
                    });
                }

                log::info!(
                    "    {} frames (hop={:.1}ms)",
                    frames.len(),
                    hop / sr as f64 * 1000.0
                );
            }
            Ok(_) => {
                log::warn!("  Segment {} produced no frames", segment_idx);
            }
            Err(e) => {
                log::warn!("  Segment {} extraction error: {}", segment_idx, e);
            }
        }

        if end >= resampled.len() {
            break;
        }
        offset += step;
        segment_idx += 1;
    }

    log::info!(
        "Extracted {} pitch points ({} voiced)",
        all_pitches.len(),
        all_pitches.iter().filter(|p| p.is_voiced).count()
    );

    // ---- Post-processing pipeline ----
    // 1. Remove isolated outlier pitches (spike removal)
    remove_pitch_outliers(&mut all_pitches);

    // 2. Median filter (3-frame window) to smooth pitch contour
    median_filter_pitch(&mut all_pitches, 3);

    // 3. Fill short gaps (interpolate between voiced segments)
    fill_pitch_gaps(&mut all_pitches, 5);

    let final_voiced = all_pitches.iter().filter(|p| p.is_voiced).count();
    log::info!(
        "After post-processing: {} points ({} voiced)",
        all_pitches.len(),
        final_voiced
    );

    Ok(all_pitches)
}

/// Remove isolated outlier frames whose pitch jumps > 400 cents from both neighbors
fn remove_pitch_outliers(pitches: &mut [PitchPoint]) {
    if pitches.len() < 3 {
        return;
    }
    let cents = |hz: f64| -> f64 {
        if hz <= 0.0 {
            0.0
        } else {
            1200.0 * (hz / 440.0).log2() + 6900.0
        }
    };

    let len = pitches.len();
    let mut kill = vec![false; len];

    for i in 1..len - 1 {
        if !pitches[i].is_voiced {
            continue;
        }
        let c = cents(pitches[i].pitch_hz);

        let left_ok = pitches[i - 1].is_voiced && {
            let mut diff = (c - cents(pitches[i - 1].pitch_hz)).abs();
            diff = diff % 1200.0;
            if diff > 600.0 {
                diff = 1200.0 - diff;
            }
            diff < 400.0
        };
        let right_ok = pitches[i + 1].is_voiced && {
            let mut diff = (c - cents(pitches[i + 1].pitch_hz)).abs();
            diff = diff % 1200.0;
            if diff > 600.0 {
                diff = 1200.0 - diff;
            }
            diff < 400.0
        };

        if !left_ok && !right_ok {
            kill[i] = true;
        }
    }

    let removed: usize = kill.iter().filter(|&&k| k).count();
    for i in 0..len {
        if kill[i] {
            pitches[i].is_voiced = false;
            pitches[i].pitch_hz = 0.0;
        }
    }
    if removed > 0 {
        log::debug!("Removed {} outlier frames", removed);
    }
}

/// Median filter on pitch values (only affects voiced frames)
fn median_filter_pitch(pitches: &mut [PitchPoint], window: usize) {
    if pitches.len() < window {
        return;
    }
    let half = window / 2;
    let orig: Vec<f64> = pitches.iter().map(|p| p.pitch_hz).collect();

    for i in half..pitches.len().saturating_sub(half) {
        if !pitches[i].is_voiced {
            continue;
        }
        let mut vals: Vec<f64> = Vec::new();
        for j in (i.saturating_sub(half))..=(i + half).min(pitches.len() - 1) {
            if pitches[j].is_voiced && orig[j] > 0.0 {
                vals.push(orig[j]);
            }
        }
        if vals.len() >= 2 {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            pitches[i].pitch_hz = vals[vals.len() / 2];
        }
    }
}

/// Fill short unvoiced gaps between voiced segments by interpolating
fn fill_pitch_gaps(pitches: &mut Vec<PitchPoint>, max_gap: usize) {
    let len = pitches.len();
    if len < 3 {
        return;
    }

    let mut i = 0;
    let mut filled = 0usize;
    while i < len {
        if !pitches[i].is_voiced {
            let gap_start = i;
            while i < len && !pitches[i].is_voiced {
                i += 1;
            }
            let gap_end = i;
            let gap_len = gap_end - gap_start;

            if gap_len <= max_gap && gap_start > 0 && gap_end < len {
                let left_hz = pitches[gap_start - 1].pitch_hz;
                let right_hz = pitches[gap_end].pitch_hz;

                if left_hz > 0.0 && right_hz > 0.0 {
                    let cents_diff =
                        ((1200.0 * (left_hz / right_hz).log2()).abs()) % 1200.0;
                    let cents_diff = if cents_diff > 600.0 {
                        1200.0 - cents_diff
                    } else {
                        cents_diff
                    };

                    if cents_diff < 500.0 {
                        for g in 0..gap_len {
                            let t = (g + 1) as f64 / (gap_len + 1) as f64;
                            let log_left = left_hz.ln();
                            let log_right = right_hz.ln();
                            let interp_hz = (log_left + t * (log_right - log_left)).exp();
                            pitches[gap_start + g].pitch_hz = interp_hz;
                            pitches[gap_start + g].is_voiced = true;
                            filled += 1;
                        }
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    if filled > 0 {
        log::debug!("Filled {} gap frames", filled);
    }
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
