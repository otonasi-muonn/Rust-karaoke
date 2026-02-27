/// CLI Pipeline Test
/// Phase 2 requirement: verify download + decode pipeline from CLI
///
/// Usage: cargo run --example pipeline_test -- <youtube_url_or_file_path>
use anyhow::Result;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <youtube_url_or_audio_file_path>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} https://www.youtube.com/watch?v=xxxxx", args[0]);
        eprintln!("  {} ./test_audio.mp3", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let audio_path: PathBuf;

    // Check if input is a local file or YouTube URL
    if std::path::Path::new(input).exists() {
        println!("[1/3] ローカルファイルを使用: {}", input);
        audio_path = PathBuf::from(input);
    } else if input.contains("youtube.com") || input.contains("youtu.be") {
        println!("[1/3] YouTube URLからダウンロード中: {}", input);

        // Create a temporary AppState for download progress tracking
        let state = std::sync::Arc::new(rust_karaoke::state::AppState::new());
        audio_path = rust_karaoke::pipeline::download::download_audio(input, &state).await?;
        println!("  -> ダウンロード完了: {:?}", audio_path);
    } else {
        eprintln!("Error: '{}' はファイルパスでもYouTube URLでもありません", input);
        std::process::exit(1);
    }

    // Step 2: Decode to PCM
    println!("[2/3] PCMデコード中...");
    let pcm = rust_karaoke::pipeline::decode::decode_to_pcm(&audio_path)?;
    let duration_secs = pcm.len() as f64 / 44100.0;
    println!("  -> PCMサンプル数: {}", pcm.len());
    println!("  -> 再生時間: {:.2} 秒", duration_secs);
    println!("  -> サンプルレート: 44100 Hz");

    // Step 3: Quick pitch extraction test (first 5 seconds)
    println!("[3/3] ピッチ抽出テスト (最初5秒)...");
    let test_samples = pcm.len().min(44100 * 5);
    let test_pcm = &pcm[..test_samples];
    let pitches = rust_karaoke::pipeline::pitch::extract_reference_pitches(test_pcm)?;

    let voiced_count = pitches.iter().filter(|p| p.is_voiced).count();
    let avg_pitch: f64 = {
        let voiced: Vec<f64> = pitches
            .iter()
            .filter(|p| p.is_voiced && p.pitch_hz > 0.0)
            .map(|p| p.pitch_hz)
            .collect();
        if voiced.is_empty() {
            0.0
        } else {
            voiced.iter().sum::<f64>() / voiced.len() as f64
        }
    };

    println!("  -> 抽出フレーム数: {}", pitches.len());
    println!("  -> 有声フレーム数: {} ({:.1}%)", voiced_count,
        if pitches.is_empty() { 0.0 } else { voiced_count as f64 / pitches.len() as f64 * 100.0 });
    println!("  -> 平均ピッチ: {:.1} Hz", avg_pitch);

    println!();
    println!("✓ パイプラインテスト完了");

    Ok(())
}
