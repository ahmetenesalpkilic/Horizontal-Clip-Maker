import os
import shutil
import json
import time
import logging
from pathlib import Path

import numpy as np
import librosa
import cv2

from moviepy.editor import VideoFileClip, concatenate_videoclips


# =========================
# LOGGING
# =========================
LOG_FILE = "highlight_generator.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# =========================
# PATHS
# =========================
INPUT_DIR = Path("input_videos")
OUTPUT_DIR = Path("output_clips")
SUMMARY_DIR = Path("summary_videos")
PROCESSED_DIR = Path("processed_videos")
FAILED_DIR = Path("failed_videos")

for d in [INPUT_DIR, OUTPUT_DIR, SUMMARY_DIR, PROCESSED_DIR, FAILED_DIR]:
    d.mkdir(exist_ok=True)

# =========================
# CONFIG
# =========================
CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "pre_min": 12,
    "pre_max": 20,
    "post_min": 20,
    "post_max": 30,
    "min_clips": 2,
    "max_clips": 7,
    "audio_cluster_sec": 2.0,
    "motion_threshold": 15.0,
    "audio_sensitivity": 1.8
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    logging.info("Varsayılan config.json oluşturuldu")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)


# =========================
# AUDIO SPIKE DETECTION
# =========================
def detect_audio_spikes(path):
    y, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    median = np.median(rms)
    threshold = median * CFG["audio_sensitivity"]

    raw_spikes = times[rms > threshold]

    # cluster spikes
    clustered = []
    for t in raw_spikes:
        if not clustered or t - clustered[-1] > CFG["audio_cluster_sec"]:
            clustered.append(t)

    return clustered


# =========================
# MOTION CHECK
# =========================
def has_motion(video_path, t, duration=1.0):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return False

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    diffs = []

    for _ in range(int(fps * duration)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diffs.append(np.mean(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray

    cap.release()
    return np.mean(diffs) > CFG["motion_threshold"] if diffs else False


# =========================
# BUILD CLIPS
# =========================
def build_clip_ranges(spikes, video_duration):
    ranges = []
    for t in spikes:
        pre = np.random.uniform(CFG["pre_min"], CFG["pre_max"])
        post = np.random.uniform(CFG["post_min"], CFG["post_max"])

        start = max(0, t - pre)
        end = min(video_duration, t + post)

        if end - start >= 20:
            ranges.append((start, end))

    return ranges[:CFG["max_clips"]]


# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path):
    start_time = time.time()
    logging.info(f"🎮 {video_path.name} başladı")

    try:
        video = VideoFileClip(str(video_path))
        duration = video.duration

        spikes = detect_audio_spikes(str(video_path))

        valid_spikes = []
        for t in spikes:
            if has_motion(video_path, t):
                valid_spikes.append(t)

        if not valid_spikes:
            logging.warning("Spike yok / motion düşük → fallback kullanılıyor")
            valid_spikes = [duration / 2]

        ranges = build_clip_ranges(valid_spikes, duration)

        if len(ranges) < CFG["min_clips"]:
            logging.warning("Yeterli klip bulunamadı")
            raise RuntimeError("Klip bulunamadı")

        clips = []
        clip_paths = []

        for i, (s, e) in enumerate(ranges):
            clip = video.subclip(s, e).without_audio()
            out = OUTPUT_DIR / f"{video_path.stem}_clip_{i+1}.mp4"

            clip.write_videofile(
                str(out),
                codec="libx264",
                audio=False,
                fps=30,
                preset="ultrafast",
                threads=1,
                logger=None
            )

            clip.close()
            clips.append(VideoFileClip(str(out)))
            clip_paths.append(out)

        # SUMMARY VIDEO
        summary = concatenate_videoclips(clips, method="compose").without_audio()
        summary_out = SUMMARY_DIR / f"{video_path.stem}_SUMMARY.mp4"

        summary.write_videofile(
            str(summary_out),
            codec="libx264",
            audio=False,
            fps=30,
            preset="ultrafast",
            threads=1,
            logger=None
        )

        summary.close()
        for c in clips:
            c.close()

        video.close()
        shutil.move(str(video_path), PROCESSED_DIR / video_path.name)

        logging.info(
            f"✅ Tamamlandı | Klip: {len(ranges)} | Süre: {round(time.time()-start_time,1)}s"
        )

    except Exception as e:
        logging.error(f"{video_path.name} hata: {e}", exc_info=True)
        try:
            video.close()
        except:
            pass
        shutil.move(str(video_path), FAILED_DIR / video_path.name)


# =========================
# MAIN
# =========================
def main():
    videos = list(INPUT_DIR.glob("*.mp4"))
    if not videos:
        logging.info("İşlenecek video yok")
        return

    success = 0
    for v in videos:
        process_video(v)
        success += 1

    logging.info(f"🏁 Bitti | İşlenen video: {success}")


if __name__ == "__main__":
    main()
