/// Reference pitch extraction pipeline step
/// Uses FCPE batch processing for high-accuracy pitch extraction
use crate::state::PitchPoint;
use anyhow::Result;

/// Extract reference pitch data from the vocal track
/// Processes in 30-second segments with 2-second overlap for optimal accuracy
/// This leverages FCPE's temporal context for much better results than frame-by-frame
pub fn extract_reference_pitches(vocal_pcm: &[f32]) -> Result<Vec<PitchPoint>> {
    // Compute RMS to check if vocal track has audible content
    let rms: f64 = (vocal_pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / vocal_pcm.len().max(1) as f64).sqrt();
    log::info!("Vocal PCM: {} samples, RMS={:.6}", vocal_pcm.len(), rms);

    // Pre-process: Apply bandpass filter (100-3000Hz) to remove instrument leakage
    // Bass instruments and high-frequency harmonics confuse pitch detection
    let filtered = apply_bandpass_filter(vocal_pcm, 44100, 100.0, 3000.0);
    let filtered_rms: f64 = (filtered.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / filtered.len().max(1) as f64).sqrt();
    log::info!("After bandpass (100-3000Hz): RMS={:.6}", filtered_rms);

    // Pre-process: RMS-based gate — zero out segments below threshold
    // This removes quiet instrument leakage between vocal phrases
    let gated = rms_gate(&filtered, 44100, 0.02, 50);
    let gated_rms: f64 = (gated.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / gated.len().max(1) as f64).sqrt();
    log::info!("After RMS gate (threshold=0.02): RMS={:.6}", gated_rms);

    let resampled = resample_to_16k(&gated, 44100);
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
    let v_count = |p: &[PitchPoint]| p.iter().filter(|p| p.is_voiced).count();

    // 0. Filter by vocal frequency range (remove bass/instrument leakage)
    let before = v_count(&all_pitches);
    filter_vocal_range(&mut all_pitches, 65.0, 1500.0);
    log::info!("  vocal_range filter: {} -> {} voiced", before, v_count(&all_pitches));

    // 1. Octave correction — fix harmonic confusion (2x/0.5x jumps)
    let before = v_count(&all_pitches);
    correct_octave_jumps(&mut all_pitches);
    log::info!("  octave_correction: {} -> {} voiced", before, v_count(&all_pitches));

    // 2. Running median rejection — remove frames far from local pitch trend
    let before = v_count(&all_pitches);
    running_median_reject(&mut all_pitches, 30, 500.0);
    log::info!("  median_reject: {} -> {} voiced", before, v_count(&all_pitches));

    // 3. Remove isolated outlier pitches (spike removal)
    let before = v_count(&all_pitches);
    remove_pitch_outliers(&mut all_pitches);
    log::info!("  outlier filter: {} -> {} voiced", before, v_count(&all_pitches));

    // 4. Remove very short voiced segments (<30ms) — likely instrument transients
    let before = v_count(&all_pitches);
    remove_short_segments(&mut all_pitches, 30.0);
    log::info!("  short_seg filter: {} -> {} voiced", before, v_count(&all_pitches));

    // 5. Median filter (5-frame window) to smooth pitch contour
    median_filter_pitch(&mut all_pitches, 5);

    // 6. Fill short gaps (interpolate between voiced segments)
    let before = v_count(&all_pitches);
    fill_pitch_gaps(&mut all_pitches, 10);
    log::info!("  gap_fill: {} -> {} voiced", before, v_count(&all_pitches));

    let final_voiced = all_pitches.iter().filter(|p| p.is_voiced).count();
    log::info!(
        "After post-processing: {} points ({} voiced)",
        all_pitches.len(),
        final_voiced
    );

    Ok(all_pitches)
}

/// Correct octave jumps caused by harmonic confusion
/// If a frame is ~1200 cents (1 octave) away from its neighbors, snap it to the correct octave
fn correct_octave_jumps(pitches: &mut [PitchPoint]) {
    let len = pitches.len();
    if len < 3 {
        return;
    }

    let hz_to_cents = |hz: f64| -> f64 {
        if hz <= 0.0 { 0.0 } else { 1200.0 * (hz / 440.0).log2() + 6900.0 }
    };

    let mut corrected = 0usize;

    for i in 1..len - 1 {
        if !pitches[i].is_voiced {
            continue;
        }

        // Find nearest voiced neighbors (within ±10 frames)
        let left_hz = (1..=10.min(i))
            .find_map(|d| {
                if pitches[i - d].is_voiced { Some(pitches[i - d].pitch_hz) } else { None }
            });
        let right_hz = ((i + 1)..len.min(i + 11))
            .find_map(|j| {
                if pitches[j].is_voiced { Some(pitches[j].pitch_hz) } else { None }
            });

        // Get reference pitch (average of neighbors if both exist)
        let ref_hz = match (left_hz, right_hz) {
            (Some(l), Some(r)) => (l * r).sqrt(), // geometric mean
            (Some(l), None) => l,
            (None, Some(r)) => r,
            _ => continue,
        };

        let current = pitches[i].pitch_hz;
        let ref_cents = hz_to_cents(ref_hz);
        let cur_cents = hz_to_cents(current);
        let diff = (cur_cents - ref_cents).abs();

        // Check for octave jumps: ~1200, ~2400 cents difference
        if diff > 1000.0 && diff < 1400.0 {
            // 1 octave jump — snap to closer octave
            let down = current / 2.0;
            let up = current * 2.0;
            let diff_down = (hz_to_cents(down) - ref_cents).abs();
            let diff_up = (hz_to_cents(up) - ref_cents).abs();
            if diff_down < diff_up && diff_down < 300.0 {
                pitches[i].pitch_hz = down;
                corrected += 1;
            } else if diff_up < 300.0 {
                pitches[i].pitch_hz = up;
                corrected += 1;
            }
        } else if diff > 2200.0 && diff < 2600.0 {
            // 2 octave jump
            let down = current / 4.0;
            let up = current * 4.0;
            let diff_down = (hz_to_cents(down) - ref_cents).abs();
            let diff_up = (hz_to_cents(up) - ref_cents).abs();
            if diff_down < diff_up && diff_down < 300.0 {
                pitches[i].pitch_hz = down;
                corrected += 1;
            } else if diff_up < 300.0 {
                pitches[i].pitch_hz = up;
                corrected += 1;
            }
        }
    }

    if corrected > 0 {
        log::info!("Octave correction: fixed {} frames", corrected);
    }
}

/// Reject voiced frames whose pitch deviates too far from the local median
/// Uses a sliding window to establish the expected pitch contour
fn running_median_reject(pitches: &mut [PitchPoint], window_half: usize, max_diff_cents: f64) {
    let len = pitches.len();
    if len < 5 {
        return;
    }

    let hz_to_cents = |hz: f64| -> f64 {
        if hz <= 0.0 { 0.0 } else { 1200.0 * (hz / 440.0).log2() + 6900.0 }
    };

    let mut kill = vec![false; len];

    for i in 0..len {
        if !pitches[i].is_voiced {
            continue;
        }

        // Collect voiced pitches in the window
        let start = i.saturating_sub(window_half);
        let end = (i + window_half + 1).min(len);
        let mut window_cents: Vec<f64> = Vec::new();
        for j in start..end {
            if pitches[j].is_voiced && !kill[j] {
                window_cents.push(hz_to_cents(pitches[j].pitch_hz));
            }
        }

        if window_cents.len() < 3 {
            continue; // Not enough context
        }

        window_cents.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = window_cents[window_cents.len() / 2];
        let cur = hz_to_cents(pitches[i].pitch_hz);
        let mut diff = (cur - median).abs();
        // Allow octave equivalence
        diff = diff % 1200.0;
        if diff > 600.0 {
            diff = 1200.0 - diff;
        }

        if diff > max_diff_cents {
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
        log::info!("Running median reject: removed {} frames (window={}, threshold={}cents)", removed, window_half * 2 + 1, max_diff_cents);
    }
}

/// Remove isolated outlier frames whose pitch jumps > 600 cents from both neighbors
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
            diff < 600.0
        };
        let right_ok = pitches[i + 1].is_voiced && {
            let mut diff = (c - cents(pitches[i + 1].pitch_hz)).abs();
            diff = diff % 1200.0;
            if diff > 600.0 {
                diff = 1200.0 - diff;
            }
            diff < 600.0
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

/// Remove voiced segments shorter than min_duration_ms
/// Short bursts are typically instrument transients, not vocal notes
fn remove_short_segments(pitches: &mut [PitchPoint], min_duration_ms: f64) {
    let len = pitches.len();
    if len < 2 {
        return;
    }

    let mut removed = 0usize;
    let mut i = 0;
    while i < len {
        if pitches[i].is_voiced {
            let seg_start = i;
            while i < len && pitches[i].is_voiced {
                i += 1;
            }
            let seg_end = i; // exclusive

            let duration = if seg_end > seg_start + 1 {
                pitches[seg_end - 1].time_ms - pitches[seg_start].time_ms
            } else {
                0.0
            };

            if duration < min_duration_ms {
                for j in seg_start..seg_end {
                    pitches[j].is_voiced = false;
                    pitches[j].pitch_hz = 0.0;
                    removed += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    if removed > 0 {
        log::info!("Removed {} frames from short segments (<{}ms)", removed, min_duration_ms);
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

/// Apply a 1st-order IIR high-pass filter to remove sub-bass leakage from instruments
/// y[n] = alpha * (y[n-1] + x[n] - x[n-1])
fn apply_highpass_filter(input: &[f32], sample_rate: u32, cutoff_hz: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate as f64;
    let alpha = rc / (rc + dt);

    let mut output = vec![0.0f32; input.len()];
    output[0] = input[0];

    for i in 1..input.len() {
        output[i] = (alpha * (output[i - 1] as f64 + input[i] as f64 - input[i - 1] as f64)) as f32;
    }

    // Apply a second pass for steeper roll-off (2nd order)
    let mut output2 = vec![0.0f32; input.len()];
    output2[0] = output[0];
    for i in 1..output.len() {
        output2[i] = (alpha * (output2[i - 1] as f64 + output[i] as f64 - output[i - 1] as f64)) as f32;
    }

    log::debug!(
        "Applied 2nd-order high-pass filter at {}Hz (alpha={:.4})",
        cutoff_hz,
        alpha
    );
    output2
}

/// Filter pitches to a typical vocal fundamental frequency range
/// Marks frames outside the range as unvoiced
fn filter_vocal_range(pitches: &mut [PitchPoint], min_hz: f64, max_hz: f64) {
    let mut filtered = 0usize;
    for p in pitches.iter_mut() {
        if p.is_voiced && (p.pitch_hz < min_hz || p.pitch_hz > max_hz) {
            p.is_voiced = false;
            p.pitch_hz = 0.0;
            filtered += 1;
        }
    }
    if filtered > 0 {
        log::info!(
            "Vocal range filter: removed {} frames outside {:.0}-{:.0}Hz",
            filtered,
            min_hz,
            max_hz
        );
    }
}

/// Apply bandpass filter (cascaded high-pass + low-pass) to isolate vocal frequency range
/// Uses 4th-order IIR (two 2nd-order passes) for effective instrument leakage removal
fn apply_bandpass_filter(input: &[f32], sample_rate: u32, low_hz: f64, high_hz: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let len = input.len();
    let dt = 1.0 / sample_rate as f64;

    // High-pass filter (2 passes for 4th order)
    let rc_hp = 1.0 / (2.0 * std::f64::consts::PI * low_hz);
    let alpha_hp = rc_hp / (rc_hp + dt);

    let mut hp = vec![0.0f64; len];
    hp[0] = input[0] as f64;
    for i in 1..len {
        hp[i] = alpha_hp * (hp[i - 1] + input[i] as f64 - input[i - 1] as f64);
    }
    // 2nd pass
    let mut hp2 = vec![0.0f64; len];
    hp2[0] = hp[0];
    for i in 1..len {
        hp2[i] = alpha_hp * (hp2[i - 1] + hp[i] - hp[i - 1]);
    }

    // Low-pass filter (2 passes for 4th order)
    let rc_lp = 1.0 / (2.0 * std::f64::consts::PI * high_hz);
    let alpha_lp = dt / (rc_lp + dt);

    let mut lp = vec![0.0f64; len];
    lp[0] = hp2[0];
    for i in 1..len {
        lp[i] = lp[i - 1] + alpha_lp * (hp2[i] - lp[i - 1]);
    }
    // 2nd pass
    let mut result = vec![0.0f32; len];
    let mut prev = lp[0];
    for i in 0..len {
        prev = prev + alpha_lp * (lp[i] - prev);
        result[i] = prev as f32;
    }

    result
}

/// RMS-based noise gate: zero out frames where local RMS is below threshold
/// window_ms: analysis window size in milliseconds
fn rms_gate(input: &[f32], sample_rate: u32, threshold: f32, window_ms: usize) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let window_samples = (sample_rate as usize * window_ms) / 1000;
    let half_window = window_samples / 2;
    let len = input.len();
    let mut output = input.to_vec();

    // Compute running RMS and gate
    let mut gated_samples = 0usize;
    let step = window_samples / 2; // hop by half window
    let mut gate_mask = vec![false; len]; // true = pass through

    let mut pos = 0;
    while pos < len {
        let start = pos.saturating_sub(half_window);
        let end = (pos + half_window).min(len);
        let segment = &input[start..end];
        let rms: f32 = (segment.iter().map(|&s| s * s).sum::<f32>() / segment.len().max(1) as f32).sqrt();

        if rms >= threshold {
            // Mark this window region as active
            for j in start..end {
                gate_mask[j] = true;
            }
        }
        pos += step;
    }

    for i in 0..len {
        if !gate_mask[i] {
            output[i] = 0.0;
            gated_samples += 1;
        }
    }

    let pct = gated_samples as f64 / len as f64 * 100.0;
    log::info!("RMS gate: zeroed {}/{} samples ({:.1}%)", gated_samples, len, pct);

    output
}
