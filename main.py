import os
import re
import warnings
import whisper
from transformers import pipeline
import shutil
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import librosa
import cv2

# suppress common librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout


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
PROCESSED_DIR = Path("processed_videos")
FAILED_DIR = Path("failed_videos")

OUTPUT_BASE_DIR = Path("output")
OUTPUT_DIR = OUTPUT_BASE_DIR / "clips"
SUMMARY_DIR = OUTPUT_BASE_DIR / "summary"

for d in [INPUT_DIR, OUTPUT_BASE_DIR, OUTPUT_DIR, SUMMARY_DIR, PROCESSED_DIR, FAILED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# PROFILE SYSTEM
# =========================
PROFILES = {
    "default": {},
    "sports": {
        "pre_min": 5,
        "post_min": 15,
        "motion_threshold": 25,
    },
    "interview": {
        "pre_min": 2,
        "post_min": 10,
        "audio_sensitivity": 2.5,
    },
    "gaming": {
        "pre_min": 8,
        "post_min": 20,
        "audio_cluster_sec": 1.5,
    }
}

# =========================
# CONFIG
# =========================
CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "profile": "default",
    "pre_min": 12,
    "pre_max": 20,
    "post_min": 20,
    "post_max": 30,
    "min_clips": 2,
    "max_clips": 7,
    "audio_cluster_sec": 2.0,
    "motion_threshold": 15.0,
    "audio_sensitivity": 1.8,
    "use_transitions": True,
    "transition_duration": 0.3,
    "parallel_processing": True
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    # Apply profile overrides
    profile_name = config.get("profile", "default")
    if profile_name in PROFILES:
        config.update(PROFILES[profile_name])

    return config


CFG = load_config()

# =========================
# AUDIO SPIKE DETECTION
# =========================
def detect_audio_spikes(path):
    try:
        y, sr = librosa.load(path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

        median = np.median(rms)
        threshold = median * CFG["audio_sensitivity"]

        raw_spikes = times[rms > threshold]

        clustered = []
        for t in raw_spikes:
            if not clustered or t - clustered[-1] > CFG["audio_cluster_sec"]:
                clustered.append(t)

        return clustered
    except Exception as e:
        logging.error(f"Audio error: {e}")
        return []

# =========================
# MOTION CHECK
# =========================
def has_motion(video_path, t, duration=1.0):
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))

        ret, prev = cap.read()
        if not ret:
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
    except:
        return False

# =========================
# BUILD CLIPS
# =========================
def make_safe_filename(text: str, max_len: int = 40) -> str:
    """Return a filesystem-safe version of *text* suitable for use in a file name."""
    # replace whitespace with underscore, remove characters that are invalid in Windows filenames
    safe = re.sub(r"[\\/:*?\"<>|]", "", text)
    safe = "_".join(safe.split())
    return safe[:max_len]


def build_clip_ranges(spikes, duration):
    ranges = []
    for t in spikes:
        pre = np.random.uniform(CFG["pre_min"], CFG["pre_max"])
        post = np.random.uniform(CFG["post_min"], CFG["post_max"])

        start = max(0, t - pre)
        end = min(duration, t + post)

        if end - start >= 15:
            ranges.append((start, end))

    # Eğer yeterli klip yoksa, videoyu eşit parçalara bölerek min_clips kadar klip oluştur
    min_clips = CFG.get("min_clips", 2)
    max_clips = CFG.get("max_clips", 7)
    if len(ranges) < min_clips:
        # Eksik klip sayısı kadar ek klip oluştur
        needed = min_clips - len(ranges)
        # Videoyu min_clips kadar eşit parçaya böl
        segment_length = duration / min_clips
        for i in range(min_clips):
            start = i * segment_length
            end = min(duration, start + segment_length)
            # Eğer bu aralık zaten eklenmişse atla
            overlap = any(abs(start - r[0]) < 1 and abs(end - r[1]) < 1 for r in ranges)
            if not overlap and end - start >= 15:
                ranges.append((start, end))
            if len(ranges) >= min_clips:
                break

    return ranges[:max_clips]

# =========================
# SMART COMPILATION
# =========================
def create_smart_compilation(clips):
    if CFG["use_transitions"]:
        processed = []
        for c in clips:
            c = fadein(c, CFG["transition_duration"])
            c = fadeout(c, CFG["transition_duration"])
            processed.append(c)
        return concatenate_videoclips(processed, method="compose")
    else:
        return concatenate_videoclips(clips, method="compose")

# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path):
    try:
        video = VideoFileClip(str(video_path))
        duration = video.duration

        spikes = detect_audio_spikes(str(video_path))
        valid_spikes = [t for t in spikes if has_motion(video_path, t)]

        if not valid_spikes:
            valid_spikes = [duration / 2]

        ranges = build_clip_ranges(valid_spikes, duration)

        clips = []
        # Speech-to-text ve başlık üretimi için modelleri yükle
        whisper_model = None
        summarizer = None
        try:
            whisper_model = whisper.load_model("base")
        except Exception as e:
            logging.error(f"Whisper model load error: {e}")

        def _load_summarizer():
            try:
                return pipeline("summarization", model="facebook/bart-large-cnn")
            except Exception as e1:
                logging.warning(f"Summarization pipeline failed: {e1}")
                try:
                    return pipeline("text2text-generation", model="t5-small")
                except Exception as e2:
                    logging.error(f"Fallback summarizer load error: {e2}")
                    return None

        summarizer = _load_summarizer()

        def overlay_text(base_clip, text):
            if not text or base_clip.duration <= 0:
                return base_clip
            try:
                txt = (
                    TextClip(text, fontsize=24, color="white", bg_color="black", size=(base_clip.w, None), method="caption")
                    .set_duration(min(5, base_clip.duration))
                    .set_position(("center", "top"))
                )
                return CompositeVideoClip([base_clip, txt])
            except Exception as e:
                logging.error(f"Text overlay error: {e}")
                return base_clip

        for i, (s, e) in enumerate(ranges):
            clip = video.subclip(s, e)
            temp_audio_path = OUTPUT_DIR / f"{video_path.stem}_clip_{i+1}_audio.wav"
            # Klipten sesi çıkar
            clip.audio.write_audiofile(str(temp_audio_path), logger=None)
            # Konuşma analizi ve başlık üretimi
            transcript = ""
            title = f"Clip {i+1}"
            if whisper_model:
                try:
                    result = whisper_model.transcribe(str(temp_audio_path))
                    transcript = result.get("text", "").strip()
                except Exception as e:
                    logging.error(f"Whisper error: {e}")
            if summarizer and transcript:
                try:
                    if hasattr(summarizer, "task") and summarizer.task == "text2text-generation":
                        out = summarizer(f"summarize: {transcript}", max_length=60, min_length=10, do_sample=False)
                        title = out[0].get("generated_text", title)
                    else:
                        out = summarizer(transcript, max_length=60, min_length=10, do_sample=False)
                        title = out[0].get("summary_text", title)
                except Exception as e:
                    logging.error(f"Summarizer error: {e}")
            logging.info(f"Clip {i+1} title: {title}")
            # ---------- overlay title on clip ----------
            safe_title = make_safe_filename(title) or f"clip_{i+1}"
            clip_with_title = overlay_text(clip, title)

            out = OUTPUT_DIR / f"{video_path.stem}_clip_{i+1}_{safe_title}.mp4"
            # write the clip (with embedded title if available)
            clip_with_title.write_videofile(str(out), codec="libx264", fps=30, logger=None)
            clips.append(VideoFileClip(str(out)))
            # Geçici ses dosyasını sil
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        if clips:
            summary = create_smart_compilation(clips)
            # add a simple overlay to the summary video
            try:
                summary = overlay_text(summary, f"Summary of {video_path.stem}")
            except Exception as e:
                logging.error(f"Summary overlay error: {e}")
            summary_title = make_safe_filename(f"{video_path.stem}_SUMMARY")
            summary_out = SUMMARY_DIR / f"{summary_title}.mp4"
            summary.write_videofile(str(summary_out), codec="libx264", fps=30, logger=None)
            summary.close()

        video.close()
        shutil.move(str(video_path), PROCESSED_DIR / video_path.name)

        return True

    except Exception as e:
        logging.error(f"Processing error: {e}")
        shutil.move(str(video_path), FAILED_DIR / video_path.name)
        return False

# =========================
# MAIN
# =========================
def main():
    videos = list(INPUT_DIR.glob("*.mp4"))
    if not videos:
        logging.info("No videos found.")
        return

    if CFG["parallel_processing"]:
        cpu_count = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            list(executor.map(process_video, videos))
    else:
        for v in videos:
            process_video(v)


if __name__ == "__main__":
    main()
