/// Audio playback module
/// Uses rodio for accompaniment playback with timing synchronization
use crate::state::AppState;
use anyhow::Result;
use rodio::{OutputStream, Sink, Source};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

static PLAYBACK_RUNNING: AtomicBool = AtomicBool::new(false);

/// Custom PCM source for rodio
struct PcmSource {
    data: Vec<f32>,
    position: usize,
    sample_rate: u32,
}

impl Iterator for PcmSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.position >= self.data.len() {
            return None;
        }
        let sample = self.data[self.position];
        self.position += 1;
        Some(sample)
    }
}

impl Source for PcmSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.data.len() - self.position)
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn total_duration(&self) -> Option<Duration> {
        let secs = self.data.len() as f64 / self.sample_rate as f64;
        Some(Duration::from_secs_f64(secs))
    }
}

/// Start playing the accompaniment track
pub fn start_playback(state: &Arc<AppState>) -> Result<()> {
    if PLAYBACK_RUNNING.load(Ordering::Relaxed) {
        log::warn!("Playback already running");
        return Ok(());
    }

    let accompaniment = state.get_accompaniment();
    if accompaniment.is_empty() {
        return Err(anyhow::anyhow!("伴奏データがありません"));
    }

    state.reset_score();
    PLAYBACK_RUNNING.store(true, Ordering::Relaxed);

    let state_clone = Arc::clone(state);

    std::thread::spawn(move || {
        let (_stream, stream_handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to open audio output: {}", e);
                return;
            }
        };

        let sink = Sink::try_new(&stream_handle).unwrap();

        let source = PcmSource {
            data: accompaniment,
            position: 0,
            sample_rate: 44100,
        };

        // Record playback start time for synchronization
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            * 1000.0;
        state_clone.set_playback_start(start_time);

        sink.append(source);

        // Wait for playback to finish or stop signal
        while !sink.empty() && PLAYBACK_RUNNING.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }

        sink.stop();
        state_clone.set_playing(false);
        PLAYBACK_RUNNING.store(false, Ordering::Relaxed);
        log::info!("Playback finished");
    });

    Ok(())
}

/// Stop playback
pub fn stop_playback(state: &Arc<AppState>) {
    PLAYBACK_RUNNING.store(false, Ordering::Relaxed);
    state.set_playing(false);
    log::info!("Playback stopped");
}
