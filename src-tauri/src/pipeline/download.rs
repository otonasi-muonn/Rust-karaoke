/// YouTube audio download module
/// Uses yt-dlp CLI wrapper for reliable YouTube audio extraction
use crate::state::AppState;
use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

/// Download audio from a YouTube URL
/// Returns the path to the downloaded audio file
pub async fn download_audio(url: &str, state: &Arc<AppState>) -> Result<PathBuf> {
    let output_dir = get_cache_dir()?;
    let output_path = output_dir.join("downloaded_audio.m4a");

    // Try yt-dlp first (more reliable)
    if try_ytdlp(url, &output_path, state).await.is_ok() {
        return Ok(output_path);
    }

    // Fallback to rusty_ytdl
    try_rusty_ytdl(url, &output_path, state).await?;
    Ok(output_path)
}

/// Download using yt-dlp CLI
async fn try_ytdlp(url: &str, output_path: &PathBuf, state: &Arc<AppState>) -> Result<()> {
    state.set_progress("download", 0.1, "yt-dlpでダウンロード中...");

    let output = tokio::process::Command::new("yt-dlp")
        .args([
            "-f",
            "bestaudio",
            "-o",
            output_path.to_str().unwrap(),
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "m4a",
            "--audio-quality",
            "0",
            url,
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("yt-dlp failed: {}", stderr));
    }

    state.set_progress("download", 0.25, "ダウンロード完了");
    Ok(())
}

/// Download using rusty_ytdl library
async fn try_rusty_ytdl(url: &str, output_path: &PathBuf, state: &Arc<AppState>) -> Result<()> {
    use rusty_ytdl::Video;

    state.set_progress("download", 0.1, "rusty_ytdlでダウンロード中...");

    let video = Video::new(url)?;
    let video_info = video.get_info().await?;

    log::info!(
        "Downloading: {}",
        video_info.video_details.title
    );

    state.set_progress("download", 0.15, &format!("ダウンロード中: {}", video_info.video_details.title));

    // Download to file
    let path = output_path.to_str().unwrap().to_string();
    video.download(&path).await?;

    state.set_progress("download", 0.25, "ダウンロード完了");
    Ok(())
}

/// Get/create cache directory for downloaded files
fn get_cache_dir() -> Result<PathBuf> {
    let cache_dir = dirs_next().join("rust-karaoke").join("cache");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get a reasonable temp/data directory
fn dirs_next() -> PathBuf {
    if let Some(data_dir) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(data_dir)
    } else if let Some(home) = std::env::var_os("USERPROFILE") {
        PathBuf::from(home).join("AppData").join("Local")
    } else {
        std::env::temp_dir()
    }
}
