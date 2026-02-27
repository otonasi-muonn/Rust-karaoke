/// Shared type definitions
/// Used by both the Tauri commands (main.rs) and the library (lib.rs)
use serde::{Deserialize, Serialize};

/// Score snapshot returned to the frontend
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScoreResult {
    pub current_score: f64,
    pub judgement: String,
    pub pitch_diff_cents: f64,
    pub user_pitch_hz: f64,
    pub ref_pitch_hz: f64,
    pub total_frames: u64,
    pub perfect_count: u64,
    pub great_count: u64,
    pub good_count: u64,
    pub miss_count: u64,
}

/// Analysis progress info
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnalysisProgress {
    pub stage: String,
    pub progress: f64,
    pub message: String,
}

/// A single pitch data frame for IPC
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PitchFrame {
    pub time_ms: f64,
    pub pitch_hz: f64,
    pub is_voiced: bool,
}

/// Karaoke analysis result
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KaraokeData {
    pub reference_pitches: Vec<PitchFrame>,
    pub duration_ms: f64,
}
