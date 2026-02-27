/// Scoring algorithm module
/// Converts physical Hz to musical cents and evaluates pitch accuracy
///
/// Key formula: MIDI note p = 69 + 12 * log2(f / 440)
/// Cents = 100 * p (100 cents = 1 semitone)

use crate::state::PitchPoint;

/// Scoring thresholds in cents
const PERFECT_THRESHOLD: f64 = 50.0; // Within 50 cents = Perfect
const GREAT_THRESHOLD: f64 = 100.0; // Within 100 cents = Great
const GOOD_THRESHOLD: f64 = 200.0; // Within 200 cents = Good
// Beyond 200 cents = Miss

/// Time tolerance for matching reference pitches (ms)
const TIME_TOLERANCE_MS: f64 = 50.0;

/// Convert frequency (Hz) to cents (relative to A4=440Hz)
/// cents = 1200 * log2(f / 440) + 6900
pub fn hz_to_cents(freq_hz: f64) -> f64 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    1200.0 * (freq_hz / 440.0).log2() + 6900.0
}

/// Convert frequency to MIDI note number
/// p = 69 + 12 * log2(f / 440)
#[allow(dead_code)]
pub fn hz_to_midi(freq_hz: f64) -> f64 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (freq_hz / 440.0).log2()
}

/// Calculate pitch difference in cents with octave tolerance
/// Removes octave differences (1200 cents = 1 octave) using modular arithmetic
pub fn pitch_diff_cents(user_hz: f64, reference_hz: f64) -> f64 {
    if user_hz <= 0.0 || reference_hz <= 0.0 {
        return f64::MAX;
    }

    let user_cents = hz_to_cents(user_hz);
    let ref_cents = hz_to_cents(reference_hz);

    // Raw difference
    let mut diff = (user_cents - ref_cents).abs();

    // Octave tolerance: reduce by multiples of 1200 cents
    diff = diff % 1200.0;
    if diff > 600.0 {
        diff = 1200.0 - diff;
    }

    diff
}

/// Score a single frame against reference pitches
/// Returns (judgement, diff_cents, matched_reference_hz)
pub fn score_frame(
    time_ms: f64,
    user_pitch_hz: f64,
    reference: &[PitchPoint],
) -> (String, f64, f64) {
    if user_pitch_hz <= 0.0 {
        return ("Miss".to_string(), f64::MAX, 0.0);
    }

    // Find the closest reference pitch within time tolerance
    let mut best_diff = f64::MAX;
    let mut best_ref_hz = 0.0;

    for ref_point in reference {
        // Skip unvoiced reference points
        if !ref_point.is_voiced || ref_point.pitch_hz <= 0.0 {
            continue;
        }

        // Check time proximity (±50ms tolerance for timing variation)
        let time_diff = (time_ms - ref_point.time_ms).abs();
        if time_diff > TIME_TOLERANCE_MS {
            continue;
        }

        // Calculate pitch difference with octave tolerance
        let diff = pitch_diff_cents(user_pitch_hz, ref_point.pitch_hz);
        if diff < best_diff {
            best_diff = diff;
            best_ref_hz = ref_point.pitch_hz;
        }
    }

    // If no reference found within time window, it might be a rest
    if best_ref_hz <= 0.0 {
        return ("Miss".to_string(), f64::MAX, 0.0);
    }

    // Determine judgement based on cents difference
    let judgement = if best_diff <= PERFECT_THRESHOLD {
        "Perfect"
    } else if best_diff <= GREAT_THRESHOLD {
        "Great"
    } else if best_diff <= GOOD_THRESHOLD {
        "Good"
    } else {
        "Miss"
    };

    (judgement.to_string(), best_diff, best_ref_hz)
}

/// Calculate the note name from Hz
#[allow(dead_code)]
pub fn hz_to_note_name(freq_hz: f64) -> String {
    if freq_hz <= 0.0 {
        return "---".to_string();
    }

    let midi = hz_to_midi(freq_hz);
    let note_num = (midi.round() as i32) % 12;
    let octave = (midi.round() as i32) / 12 - 1;

    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];

    if note_num >= 0 && (note_num as usize) < note_names.len() {
        format!("{}{}", note_names[note_num as usize], octave)
    } else {
        "---".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_midi() {
        // A4 = 440Hz should be MIDI 69
        assert!((hz_to_midi(440.0) - 69.0).abs() < 0.01);
        // C4 ≈ 261.63Hz should be MIDI 60
        assert!((hz_to_midi(261.63) - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_hz_to_cents() {
        // A4 = 440Hz → 6900 cents
        assert!((hz_to_cents(440.0) - 6900.0).abs() < 0.01);
    }

    #[test]
    fn test_pitch_diff_octave_tolerance() {
        // Same note, one octave apart (880Hz vs 440Hz)
        let diff = pitch_diff_cents(880.0, 440.0);
        assert!(diff < 1.0, "Octave difference should be near 0: {}", diff);
    }

    #[test]
    fn test_perfect_score() {
        // Exact match
        let diff = pitch_diff_cents(440.0, 440.0);
        assert!(diff <= PERFECT_THRESHOLD);
    }

    #[test]
    fn test_note_names() {
        assert_eq!(hz_to_note_name(440.0), "A4");
        assert_eq!(hz_to_note_name(261.63), "C4");
    }
}
