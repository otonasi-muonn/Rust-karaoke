/// Audio decoding module
/// Uses symphonia to decode compressed audio to raw PCM f32
use anyhow::Result;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Target sample rate for the pipeline
const TARGET_SAMPLE_RATE: u32 = 44100;

/// Decode an audio file to mono PCM f32 at 44100Hz
pub fn decode_to_pcm(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("オーディオトラックが見つかりません"))?;

    let track_id = track.id;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2);
    let source_sample_rate = track.codec_params.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let duration = decoded.capacity();

                let mut sample_buf = SampleBuffer::<f32>::new(duration as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let samples = sample_buf.samples();

                // Convert to mono by averaging channels
                if channels > 1 {
                    for chunk in samples.chunks(channels) {
                        let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                        all_samples.push(mono);
                    }
                } else {
                    all_samples.extend_from_slice(samples);
                }
            }
            Err(e) => {
                log::warn!("Decode error (skipping packet): {}", e);
                continue;
            }
        }
    }

    // Resample if necessary
    if source_sample_rate != TARGET_SAMPLE_RATE {
        all_samples = resample(&all_samples, source_sample_rate, TARGET_SAMPLE_RATE);
    }

    log::info!(
        "Decoded {} samples ({:.1}s) at {}Hz",
        all_samples.len(),
        all_samples.len() as f64 / TARGET_SAMPLE_RATE as f64,
        TARGET_SAMPLE_RATE
    );

    Ok(all_samples)
}

/// Simple linear interpolation resampler
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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
            let sample = input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac;
            output.push(sample);
        } else if src_idx < input.len() {
            output.push(input[src_idx]);
        }
    }

    output
}
