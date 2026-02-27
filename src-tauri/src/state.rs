/// Application state management
/// Thread-safe shared state for the karaoke system
use crate::commands::ScoreResult;
use std::sync::Mutex;

/// Represents a single pitch data point from the reference track
#[derive(Clone, Debug)]
pub struct PitchPoint {
    pub time_ms: f64,
    pub pitch_hz: f64,
    pub is_voiced: bool,
}

/// Main application state shared across threads
pub struct AppState {
    /// Analysis progress tracking
    progress: Mutex<(String, f64, String)>,
    /// Reference pitch data from analyzed song
    reference_pitches: Mutex<Vec<PitchPoint>>,
    /// Accompaniment PCM data for playback (f32 samples at 44100Hz)
    accompaniment: Mutex<Vec<f32>>,
    /// Whether karaoke session is active
    is_playing: Mutex<bool>,
    /// Current scoring state
    score_state: Mutex<ScoreState>,
    /// Microphone capture active flag
    mic_active: Mutex<bool>,
    /// Playback start timestamp in milliseconds
    playback_start_ms: Mutex<Option<f64>>,
    /// Ring buffer consumer for mic audio (shared with inference thread)
    mic_pitch_receiver: Mutex<Option<crossbeam_channel::Receiver<MicPitchData>>>,
    /// Sender for mic pitch data
    mic_pitch_sender: Mutex<Option<crossbeam_channel::Sender<MicPitchData>>>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MicPitchData {
    pub time_ms: f64,
    pub pitch_hz: f64,
    pub is_voiced: bool,
}

#[derive(Clone, Debug)]
pub struct ScoreState {
    pub total_frames: u64,
    pub perfect_count: u64,
    pub great_count: u64,
    pub good_count: u64,
    pub miss_count: u64,
    pub last_user_pitch: f64,
    pub last_ref_pitch: f64,
    pub last_diff_cents: f64,
    pub last_judgement: String,
}

impl Default for ScoreState {
    fn default() -> Self {
        Self {
            total_frames: 0,
            perfect_count: 0,
            great_count: 0,
            good_count: 0,
            miss_count: 0,
            last_user_pitch: 0.0,
            last_ref_pitch: 0.0,
            last_diff_cents: 0.0,
            last_judgement: "---".to_string(),
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        Self {
            progress: Mutex::new(("idle".to_string(), 0.0, "待機中".to_string())),
            reference_pitches: Mutex::new(Vec::new()),
            accompaniment: Mutex::new(Vec::new()),
            is_playing: Mutex::new(false),
            score_state: Mutex::new(ScoreState::default()),
            mic_active: Mutex::new(false),
            playback_start_ms: Mutex::new(None),
            mic_pitch_receiver: Mutex::new(Some(rx)),
            mic_pitch_sender: Mutex::new(Some(tx)),
        }
    }

    pub fn set_progress(&self, stage: &str, progress: f64, message: &str) {
        let mut p = self.progress.lock().unwrap();
        *p = (stage.to_string(), progress, message.to_string());
    }

    pub fn get_progress(&self) -> (String, f64, String) {
        self.progress.lock().unwrap().clone()
    }

    pub fn set_reference_pitches(&self, pitches: Vec<PitchPoint>) {
        let mut rp = self.reference_pitches.lock().unwrap();
        *rp = pitches;
    }

    pub fn get_reference_pitches(&self) -> Vec<PitchPoint> {
        self.reference_pitches.lock().unwrap().clone()
    }

    pub fn set_accompaniment(&self, data: Vec<f32>) {
        let mut acc = self.accompaniment.lock().unwrap();
        *acc = data;
    }

    pub fn get_accompaniment(&self) -> Vec<f32> {
        self.accompaniment.lock().unwrap().clone()
    }

    pub fn set_playing(&self, playing: bool) {
        let mut p = self.is_playing.lock().unwrap();
        *p = playing;
    }

    pub fn is_playing(&self) -> bool {
        *self.is_playing.lock().unwrap()
    }

    pub fn set_mic_active(&self, active: bool) {
        let mut m = self.mic_active.lock().unwrap();
        *m = active;
    }

    #[allow(dead_code)]
    pub fn is_mic_active(&self) -> bool {
        *self.mic_active.lock().unwrap()
    }

    pub fn set_playback_start(&self, time_ms: f64) {
        let mut t = self.playback_start_ms.lock().unwrap();
        *t = Some(time_ms);
    }

    pub fn get_playback_start(&self) -> Option<f64> {
        *self.playback_start_ms.lock().unwrap()
    }

    pub fn get_mic_sender(&self) -> Option<crossbeam_channel::Sender<MicPitchData>> {
        self.mic_pitch_sender.lock().unwrap().clone()
    }

    #[allow(dead_code)]
    pub fn get_mic_receiver(&self) -> Option<crossbeam_channel::Receiver<MicPitchData>> {
        self.mic_pitch_receiver.lock().unwrap().clone()
    }

    pub fn update_score(&self, judgement: &str, user_pitch: f64, ref_pitch: f64, diff_cents: f64) {
        let mut s = self.score_state.lock().unwrap();
        s.total_frames += 1;
        match judgement {
            "Perfect" => s.perfect_count += 1,
            "Great" => s.great_count += 1,
            "Good" => s.good_count += 1,
            _ => s.miss_count += 1,
        }
        s.last_user_pitch = user_pitch;
        s.last_ref_pitch = ref_pitch;
        s.last_diff_cents = diff_cents;
        s.last_judgement = judgement.to_string();
    }

    pub fn get_current_score(&self) -> ScoreResult {
        let s = self.score_state.lock().unwrap();
        let total = s.total_frames.max(1) as f64;
        let score = (s.perfect_count as f64 * 100.0
            + s.great_count as f64 * 75.0
            + s.good_count as f64 * 50.0)
            / (total * 100.0)
            * 100.0;

        ScoreResult {
            current_score: score,
            judgement: s.last_judgement.clone(),
            pitch_diff_cents: s.last_diff_cents,
            user_pitch_hz: s.last_user_pitch,
            ref_pitch_hz: s.last_ref_pitch,
            total_frames: s.total_frames,
            perfect_count: s.perfect_count,
            great_count: s.great_count,
            good_count: s.good_count,
            miss_count: s.miss_count,
        }
    }

    pub fn reset_score(&self) {
        let mut s = self.score_state.lock().unwrap();
        *s = ScoreState::default();
    }
}
