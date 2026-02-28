// Rust Karaoke - AI-powered karaoke scoring system
// Main entry point for the Tauri application

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rust_karaoke::state::AppState;
use rust_karaoke::types::{AnalysisProgress, KaraokeData, PitchFrame, ScoreResult};
use std::sync::Arc;
use tauri::Manager;

fn main() {
    env_logger::init();

    let app_state = Arc::new(AppState::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            analyze_youtube_url,
            start_karaoke,
            stop_karaoke,
            get_score,
            start_microphone,
            stop_microphone,
            set_pip_mode,
            get_analysis_progress,
            list_mic_devices,
            set_mic_device,
            start_mic_test,
            stop_mic_test,
            set_mic_volume,
            set_accompaniment_volume,
        ])
        .setup(|app| {
            // Store AppHandle in state for event emission from worker threads
            let state: &Arc<AppState> = app.state::<Arc<AppState>>().inner();
            state.set_app_handle(app.handle().clone());

            let window = app.get_webview_window("main").unwrap();
            // Allow drag on the custom title bar
            #[cfg(debug_assertions)]
            window.open_devtools();
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ---- Tauri IPC command handlers ----

#[tauri::command]
async fn analyze_youtube_url(
    url: String,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<KaraokeData, String> {
    log::info!("Analyzing YouTube URL: {}", url);
    state.set_progress("download", 0.0, "ダウンロード開始...");

    // Step 1: Download audio
    let audio_path = rust_karaoke::pipeline::download::download_audio(&url, &state)
        .await
        .map_err(|e| format!("ダウンロードエラー: {}", e))?;

    // Step 2: Decode to PCM
    state.set_progress("decode", 0.3, "デコード中...");
    let pcm_data = rust_karaoke::pipeline::decode::decode_to_pcm(&audio_path)
        .map_err(|e| format!("デコードエラー: {}", e))?;

    // Step 3: Vocal separation with BS-RoFormer
    state.set_progress("separation", 0.5, "ボーカル分離中...");
    let (vocal, accompaniment) =
        rust_karaoke::pipeline::separation::separate_vocals(&pcm_data, &state)
            .await
            .map_err(|e| format!("音源分離エラー: {}", e))?;

    // Step 4: Pitch extraction with FCPE
    state.set_progress("pitch", 0.8, "ピッチ解析中...");
    let reference_pitches = rust_karaoke::pipeline::pitch::extract_reference_pitches(&vocal)
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
async fn start_karaoke(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    log::info!("Starting karaoke session");
    rust_karaoke::audio::playback::start_playback(&state)
        .map_err(|e| format!("再生開始エラー: {}", e))?;
    rust_karaoke::audio::microphone::start_capture(&state)
        .map_err(|e| format!("マイク開始エラー: {}", e))?;
    state.set_playing(true);
    Ok(())
}

#[tauri::command]
async fn stop_karaoke(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    log::info!("Stopping karaoke session");
    state.set_playing(false);
    rust_karaoke::audio::playback::stop_playback(&state);
    rust_karaoke::audio::microphone::stop_capture(&state);
    Ok(())
}

#[tauri::command]
async fn get_score(state: tauri::State<'_, Arc<AppState>>) -> Result<ScoreResult, String> {
    let score = state.get_current_score();
    Ok(score)
}

#[tauri::command]
async fn start_microphone(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    rust_karaoke::audio::microphone::start_capture(&state)
        .map_err(|e| format!("マイクエラー: {}", e))
}

#[tauri::command]
async fn stop_microphone(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    rust_karaoke::audio::microphone::stop_capture(&state);
    Ok(())
}

#[tauri::command]
async fn set_pip_mode(
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
async fn get_analysis_progress(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<AnalysisProgress, String> {
    let progress = state.get_progress();
    Ok(AnalysisProgress {
        stage: progress.0,
        progress: progress.1,
        message: progress.2,
    })
}

#[tauri::command]
async fn list_mic_devices() -> Result<Vec<String>, String> {
    rust_karaoke::audio::microphone::list_input_devices()
        .map_err(|e| format!("デバイス列挙エラー: {}", e))
}

#[tauri::command]
async fn set_mic_device(name: Option<String>) -> Result<(), String> {
    rust_karaoke::audio::microphone::set_selected_device(name);
    Ok(())
}

#[tauri::command]
async fn start_mic_test(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    let state = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        rust_karaoke::audio::microphone::start_mic_test(&state)
            .map_err(|e| format!("マイクテストエラー: {}", e))
    })
    .await
    .map_err(|e| format!("タスクエラー: {}", e))?
}

#[tauri::command]
async fn stop_mic_test() -> Result<(), String> {
    rust_karaoke::audio::microphone::stop_mic_test();
    Ok(())
}

#[tauri::command]
async fn set_mic_volume(gain: f32) -> Result<(), String> {
    rust_karaoke::audio::microphone::set_mic_gain(gain.clamp(0.0, 3.0));
    Ok(())
}

#[tauri::command]
async fn set_accompaniment_volume(volume: f32) -> Result<(), String> {
    let vol = volume.clamp(0.0, 2.0);
    rust_karaoke::audio::playback::set_volume(vol);
    rust_karaoke::audio::playback::update_sink_volume();
    Ok(())
}

