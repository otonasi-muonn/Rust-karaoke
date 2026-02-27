// Rust Karaoke - AI-powered karaoke scoring system
// Main entry point for the Tauri application

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod inference;
mod pipeline;
mod scoring;
mod state;

use state::AppState;
use std::sync::Arc;
use tauri::Manager;

fn main() {
    env_logger::init();

    let app_state = Arc::new(AppState::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::analyze_youtube_url,
            commands::start_karaoke,
            commands::stop_karaoke,
            commands::get_score,
            commands::start_microphone,
            commands::stop_microphone,
            commands::set_pip_mode,
            commands::get_analysis_progress,
        ])
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            // Allow drag on the custom title bar
            #[cfg(debug_assertions)]
            window.open_devtools();
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Tauri command handlers - IPC bridge between frontend and backend
mod commands {
    use crate::state::AppState;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tauri::State;

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct AnalysisProgress {
        pub stage: String,
        pub progress: f64,
        pub message: String,
    }

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

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct PitchFrame {
        pub time_ms: f64,
        pub pitch_hz: f64,
        pub is_voiced: bool,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct KaraokeData {
        pub reference_pitches: Vec<PitchFrame>,
        pub duration_ms: f64,
    }

    #[tauri::command]
    pub async fn analyze_youtube_url(
        url: String,
        state: State<'_, Arc<AppState>>,
    ) -> Result<KaraokeData, String> {
        log::info!("Analyzing YouTube URL: {}", url);
        state.set_progress("download", 0.0, "ダウンロード開始...");

        // Step 1: Download audio
        let audio_path = crate::pipeline::download::download_audio(&url, &state)
            .await
            .map_err(|e| format!("ダウンロードエラー: {}", e))?;

        // Step 2: Decode to PCM
        state.set_progress("decode", 0.3, "デコード中...");
        let pcm_data = crate::pipeline::decode::decode_to_pcm(&audio_path)
            .map_err(|e| format!("デコードエラー: {}", e))?;

        // Step 3: Vocal separation with BS-RoFormer
        state.set_progress("separation", 0.5, "ボーカル分離中...");
        let (vocal, accompaniment) =
            crate::pipeline::separation::separate_vocals(&pcm_data, &state)
                .await
                .map_err(|e| format!("音源分離エラー: {}", e))?;

        // Step 4: Pitch extraction with FCPE
        state.set_progress("pitch", 0.8, "ピッチ解析中...");
        let reference_pitches = crate::pipeline::pitch::extract_reference_pitches(&vocal)
            .map_err(|e| format!("ピッチ抽出エラー: {}", e))?;

        // Store accompaniment for playback
        let duration_ms = (accompaniment.len() as f64 / 44100.0) * 1000.0;
        state.set_accompaniment(accompaniment);
        state.set_reference_pitches(reference_pitches.clone());

        state.set_progress("done", 1.0, "解析完了");

        let pitch_frames: Vec<PitchFrame> = reference_pitches
            .iter()
            .map(|p| PitchFrame {
                time_ms: p.time_ms,
                pitch_hz: p.pitch_hz,
                is_voiced: p.is_voiced,
            })
            .collect();

        Ok(KaraokeData {
            reference_pitches: pitch_frames,
            duration_ms,
        })
    }

    #[tauri::command]
    pub async fn start_karaoke(state: State<'_, Arc<AppState>>) -> Result<(), String> {
        log::info!("Starting karaoke session");
        crate::audio::playback::start_playback(&state)
            .map_err(|e| format!("再生開始エラー: {}", e))?;
        crate::audio::microphone::start_capture(&state)
            .map_err(|e| format!("マイク開始エラー: {}", e))?;
        state.set_playing(true);
        Ok(())
    }

    #[tauri::command]
    pub async fn stop_karaoke(state: State<'_, Arc<AppState>>) -> Result<(), String> {
        log::info!("Stopping karaoke session");
        state.set_playing(false);
        crate::audio::playback::stop_playback(&state);
        crate::audio::microphone::stop_capture(&state);
        Ok(())
    }

    #[tauri::command]
    pub async fn get_score(state: State<'_, Arc<AppState>>) -> Result<ScoreResult, String> {
        let score = state.get_current_score();
        Ok(score)
    }

    #[tauri::command]
    pub async fn start_microphone(state: State<'_, Arc<AppState>>) -> Result<(), String> {
        crate::audio::microphone::start_capture(&state)
            .map_err(|e| format!("マイクエラー: {}", e))
    }

    #[tauri::command]
    pub async fn stop_microphone(state: State<'_, Arc<AppState>>) -> Result<(), String> {
        crate::audio::microphone::stop_capture(&state);
        Ok(())
    }

    #[tauri::command]
    pub async fn set_pip_mode(
        enabled: bool,
        window: tauri::WebviewWindow,
    ) -> Result<(), String> {
        if enabled {
            window.set_always_on_top(true).map_err(|e| e.to_string())?;
            window
                .set_size(tauri::Size::Logical(tauri::LogicalSize {
                    width: 400.0,
                    height: 300.0,
                }))
                .map_err(|e| e.to_string())?;
            window
                .set_ignore_cursor_events(true)
                .map_err(|e| e.to_string())?;
        } else {
            window
                .set_always_on_top(false)
                .map_err(|e| e.to_string())?;
            window
                .set_size(tauri::Size::Logical(tauri::LogicalSize {
                    width: 1200.0,
                    height: 800.0,
                }))
                .map_err(|e| e.to_string())?;
            window
                .set_ignore_cursor_events(false)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    #[tauri::command]
    pub async fn get_analysis_progress(
        state: State<'_, Arc<AppState>>,
    ) -> Result<AnalysisProgress, String> {
        let progress = state.get_progress();
        Ok(AnalysisProgress {
            stage: progress.0,
            progress: progress.1,
            message: progress.2,
        })
    }
}
