/// CLI Pipeline Test
/// Phase 2 + Phase 3 requirement: verify full pipeline from CLI
///
/// Usage:
///   cargo run --example pipeline_test -- <youtube_url_or_file_path>
///   cargo run --example pipeline_test -- <file_path> --full   (includes BS-RoFormer separation)
use anyhow::Result;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <youtube_url_or_audio_file_path> [--full]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --full    Run full pipeline including BS-RoFormer vocal separation");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} https://www.youtube.com/watch?v=xxxxx", args[0]);
        eprintln!("  {} ./test_audio.mp3", args[0]);
        eprintln!("  {} ./test_audio.mp3 --full", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let full_mode = args.iter().any(|a| a == "--full");
    let audio_path: PathBuf;

    // =========================================================
    // Step 1: Download / locate audio (Phase 2)
    // =========================================================
    if std::path::Path::new(input).exists() {
        println!("[1/{}] ローカルファイルを使用: {}", if full_mode { 5 } else { 3 }, input);
        audio_path = PathBuf::from(input);
    } else if input.contains("youtube.com") || input.contains("youtu.be") {
        println!("[1/{}] YouTube URLからダウンロード中: {}", if full_mode { 5 } else { 3 }, input);
        let state = std::sync::Arc::new(rust_karaoke::state::AppState::new());
        audio_path = rust_karaoke::pipeline::download::download_audio(input, &state).await?;
        println!("  -> ダウンロード完了: {:?}", audio_path);
    } else {
        eprintln!("Error: '{}' はファイルパスでもYouTube URLでもありません", input);
        std::process::exit(1);
    }

    // =========================================================
    // Step 2: Decode to PCM (Phase 2)
    // =========================================================
    println!("[2/{}] PCMデコード中...", if full_mode { 5 } else { 3 });
    let pcm = rust_karaoke::pipeline::decode::decode_to_pcm(&audio_path)?;
    let duration_secs = pcm.len() as f64 / 44100.0;
    println!("  -> PCMサンプル数: {}", pcm.len());
    println!("  -> 再生時間: {:.2} 秒", duration_secs);
    println!("  -> サンプルレート: 44100 Hz");

    // Determine which PCM to use for pitch extraction
    let vocal_pcm: Vec<f32>;

    if full_mode {
        // =========================================================
        // Step 3: BS-RoFormer vocal separation (Phase 3)
        // =========================================================
        println!("[3/5] BS-RoFormerでボーカル分離中...");
        let start = std::time::Instant::now();
        let (vocal, accompaniment) = rust_karaoke::inference::bsroformer::separate(&pcm, 44100)?;
        let elapsed = start.elapsed();

        let vocal_rms = rms(&vocal);
        let acc_rms = rms(&accompaniment);
        println!("  -> 処理時間: {:.2} 秒", elapsed.as_secs_f64());
        println!("  -> ボーカルRMS: {:.6}", vocal_rms);
        println!("  -> 伴奏RMS:   {:.6}", acc_rms);
        println!("  -> ボーカルサンプル数: {}", vocal.len());

        vocal_pcm = vocal;

        // =========================================================
        // Step 4: FCPE pitch extraction via batch API (Phase 3)
        // =========================================================
        println!("[4/5] FCPE一括ピッチ抽出中 (最初5秒)...");
    } else {
        vocal_pcm = pcm.clone();
        println!("[3/3] ピッチ抽出テスト (最初5秒)...");
    }

    // =========================================================
    // Pitch extraction (works in both modes)
    // =========================================================
    let test_samples = vocal_pcm.len().min(44100 * 5);
    let test_pcm = &vocal_pcm[..test_samples];

    let start = std::time::Instant::now();
    let pitches = rust_karaoke::pipeline::pitch::extract_reference_pitches(test_pcm)?;
    let elapsed = start.elapsed();

    let voiced_count = pitches.iter().filter(|p| p.is_voiced).count();
    let voiced_pitches: Vec<f64> = pitches
        .iter()
        .filter(|p| p.is_voiced && p.pitch_hz > 0.0)
        .map(|p| p.pitch_hz)
        .collect();

    let avg_pitch = if voiced_pitches.is_empty() {
        0.0
    } else {
        voiced_pitches.iter().sum::<f64>() / voiced_pitches.len() as f64
    };

    let min_pitch = voiced_pitches.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_pitch = voiced_pitches.iter().cloned().fold(0.0f64, f64::max);

    println!("  -> 処理時間: {:.2} 秒", elapsed.as_secs_f64());
    println!("  -> 抽出フレーム数: {}", pitches.len());
    println!(
        "  -> 有声フレーム数: {} ({:.1}%)",
        voiced_count,
        if pitches.is_empty() {
            0.0
        } else {
            voiced_count as f64 / pitches.len() as f64 * 100.0
        }
    );
    println!("  -> 平均ピッチ: {:.1} Hz", avg_pitch);
    if !voiced_pitches.is_empty() {
        println!("  -> ピッチ範囲: {:.1} - {:.1} Hz", min_pitch, max_pitch);
        // Convert to musical note names for human verification
        println!(
            "  -> 音域: {} - {}",
            rust_karaoke::scoring::hz_to_note_name(min_pitch),
            rust_karaoke::scoring::hz_to_note_name(max_pitch)
        );
    }

    if full_mode {
        // =========================================================
        // Step 5: Scoring sanity check (Phase 5)
        // =========================================================
        println!("[5/5] 採点ロジック検証...");

        // Simulate scoring the reference against itself (should be all Perfect)
        let mut perfect = 0u64;
        let mut great = 0u64;
        let mut good = 0u64;
        let mut miss = 0u64;

        for p in pitches.iter().filter(|p| p.is_voiced && p.pitch_hz > 0.0) {
            let (judgement, _diff, _ref_hz) =
                rust_karaoke::scoring::score_frame(p.time_ms, p.pitch_hz, &pitches);
            match judgement.as_str() {
                "Perfect" => perfect += 1,
                "Great" => great += 1,
                "Good" => good += 1,
                _ => miss += 1,
            }
        }

        let total = (perfect + great + good + miss).max(1);
        println!("  -> 自己採点結果 (全てPerfectが正常):");
        println!(
            "     Perfect: {} ({:.1}%), Great: {}, Good: {}, Miss: {}",
            perfect,
            perfect as f64 / total as f64 * 100.0,
            great,
            good,
            miss
        );
    }

    println!();
    println!("✓ パイプラインテスト完了");

    Ok(())
}

/// Calculate RMS of audio signal
fn rms(audio: &[f32]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / audio.len() as f64).sqrt()
}
