/// Microphone capture module
/// Uses cpal for low-latency audio input with lock-free ring buffer
use crate::state::AppState;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static MIC_RUNNING: AtomicBool = AtomicBool::new(false);

use std::sync::Mutex as StdMutex;

/// Wrapper to make Stream sendable via Mutex
/// Safety: cpal::Stream is not Send, but we only access it from one thread at a time
/// through the Mutex. The stream callbacks run on OS audio threads independently.
#[allow(dead_code)]
struct SendStream(cpal::Stream);
unsafe impl Send for SendStream {}

static MIC_STREAM_HOLDER: once_cell::sync::Lazy<StdMutex<Option<SendStream>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(None));

/// Start capturing audio from the default input device
pub fn start_capture(state: &Arc<AppState>) -> Result<()> {
    if MIC_RUNNING.load(Ordering::Relaxed) {
        log::warn!("Microphone already running");
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("マイクデバイスが見つかりません"))?;

    log::info!("Using input device: {}", device.name()?);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000), // 16kHz for FCPE
        buffer_size: cpal::BufferSize::Fixed(512),
    };

    let sender = state
        .get_mic_sender()
        .ok_or_else(|| anyhow::anyhow!("Mic sender not available"))?;

    // Ring buffer for lock-free producer/consumer pattern
    let ring = ringbuf::HeapRb::<f32>::new(16000); // 1 second buffer
    let (mut producer, mut consumer) = ring.split();

    let state_clone = Arc::clone(state);
    MIC_RUNNING.store(true, Ordering::Relaxed);

    // Audio callback thread - producer
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Lock-free write to ring buffer
            for &sample in data {
                let _ = producer.try_push(sample);
            }
        },
        move |err| {
            log::error!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    // Store stream to keep it alive
    *MIC_STREAM_HOLDER.lock().unwrap() = Some(SendStream(stream));
    state.set_mic_active(true);

    // Inference thread - consumer
    std::thread::spawn(move || {
        let chunk_size = 640; // 40ms at 16kHz
        let mut buffer = vec![0.0f32; chunk_size];
        let mut time_offset = 0.0f64;
        let hop_ms = 40.0; // 40ms hop

        while MIC_RUNNING.load(Ordering::Relaxed) && state_clone.is_playing() {
            // Collect enough samples
            let available = consumer.occupied_len();
            if available < chunk_size {
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }

            // Zero-copy read from ring buffer
            let count = consumer.pop_slice(&mut buffer);
            if count < chunk_size {
                continue;
            }

            // Run FCPE inference for pitch extraction
            match crate::inference::fcpe::extract_pitch(&buffer, 16000) {
                Ok((pitch_hz, is_voiced)) => {
                    let elapsed = if let Some(start) = state_clone.get_playback_start() {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64()
                            * 1000.0;
                        now - start
                    } else {
                        time_offset
                    };

                    let data = crate::state::MicPitchData {
                        time_ms: elapsed,
                        pitch_hz,
                        is_voiced,
                    };

                    // Score this frame
                    let ref_pitches = state_clone.get_reference_pitches();
                    if !ref_pitches.is_empty() && is_voiced {
                        let (judgement, diff, ref_hz) =
                            crate::scoring::score_frame(elapsed, pitch_hz, &ref_pitches);
                        state_clone.update_score(&judgement, pitch_hz, ref_hz, diff);
                    }

                    let _ = sender.try_send(data);
                }
                Err(e) => {
                    log::debug!("Pitch extraction error: {}", e);
                }
            }

            time_offset += hop_ms;
        }

        log::info!("Inference thread stopped");
    });

    Ok(())
}

/// Stop microphone capture
pub fn stop_capture(state: &Arc<AppState>) {
    MIC_RUNNING.store(false, Ordering::Relaxed);
    // Drop the stream to release audio device
    *MIC_STREAM_HOLDER.lock().unwrap() = None;
    state.set_mic_active(false);
    log::info!("Microphone capture stopped");
}
