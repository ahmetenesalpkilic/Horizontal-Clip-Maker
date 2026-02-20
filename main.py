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
        logging.info(f"Processing video: {video_path.name}")

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
            
            # Extract audio
            logging.info(f"Extracting audio for clip {i+1}...")
            clip.audio.write_audiofile(str(temp_audio_path), logger=None, verbose=False)
            
            # Transcribe using Whisper
            transcript = ""
            title = f"Clip {i+1}"
            
            if GLOBAL_WHISPER_MODEL:
                try:
                    logging.info(f"Transcribing audio for clip {i+1}...")
                    result = GLOBAL_WHISPER_MODEL.transcribe(str(temp_audio_path))
                    transcript = result.get("text", "").strip()
                    if transcript:
                        logging.info(f"Transcript (clip {i+1}): {transcript[:100]}...")
                    else:
                        logging.warning(f"Clip {i+1}: No speech detected in audio.")
                except Exception as e:
                    logging.error(f"Whisper transcription failed for clip {i+1}: {e}")
            else:
                logging.warning(f"Whisper model not available for clip {i+1}")
            
            # Generate title from transcript
            if GLOBAL_SUMMARIZER and transcript:
                try:
                    logging.info(f"Summarizing transcript for clip {i+1}...")
                    if hasattr(GLOBAL_SUMMARIZER, "task") and GLOBAL_SUMMARIZER.task == "text2text-generation":
                        out = GLOBAL_SUMMARIZER(f"summarize: {transcript}", max_length=60, min_length=10, do_sample=False)
                        title = out[0].get("generated_text", title)
                    else:
                        out = GLOBAL_SUMMARIZER(transcript, max_length=60, min_length=10, do_sample=False)
                        title = out[0].get("summary_text", title)
                    logging.info(f"Generated title for clip {i+1}: {title}")
                except Exception as e:
                    logging.error(f"Summarization failed for clip {i+1}: {e}")
                    # Fallback: use first 50 chars of transcript
                    if transcript:
                        title = transcript[:50].strip()
                        logging.info(f"Using transcript excerpt as title for clip {i+1}: {title}")
            elif transcript:
                # No summarizer, but we have transcript - use it as is (first ~50 chars)
                title = transcript[:50].strip()
                logging.info(f"No summarizer available; using transcript excerpt for clip {i+1}: {title}")
            else:
                # No transcript and no summarizer - use default
                logging.warning(f"No transcript and no summarizer; using default title for clip {i+1}")
            
            # Ensure title is not empty
            if not title or title.strip() == "":
                title = f"Clip {i+1}"
                logging.warning(f"Title was empty; using default: {title}")
            
            # Create safe filename
            safe_title = make_safe_filename(title) or f"clip_{i+1}"
            logging.info(f"Safe filename for clip {i+1}: {safe_title}")
            
            # Overlay title on clip
            clip_with_title = overlay_text(clip, title)

            out = OUTPUT_DIR / f"{video_path.stem}_clip_{i+1}_{safe_title}.mp4"
            logging.info(f"Writing clip {i+1} to: {out.name}")
            clip_with_title.write_videofile(str(out), codec="libx264", fps=30, logger=None, verbose=False, audio_codec="aac")
            clips.append(VideoFileClip(str(out)))
            
            # Clean up temp audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logging.info(f"Deleted temp audio: {temp_audio_path.name}")

        if clips:
            logging.info(f"Creating summary video from {len(clips)} clips...")
            summary = create_smart_compilation(clips)
            # add overlay to the summary video
            try:
                summary = overlay_text(summary, f"Summary of {video_path.stem}")
            except Exception as e:
                logging.error(f"Summary overlay error: {e}")
            summary_title = make_safe_filename(f"{video_path.stem}_SUMMARY")
            summary_out = SUMMARY_DIR / f"{summary_title}.mp4"
            logging.info(f"Writing summary video to: {summary_out.name}")
            summary.write_videofile(str(summary_out), codec="libx264", fps=30, logger=None, verbose=False, audio_codec="aac")
            summary.close()
            logging.info(f"Summary video created successfully.")

        video.close()
        shutil.move(str(video_path), PROCESSED_DIR / video_path.name)
        logging.info(f"Video {video_path.name} processed and moved to {PROCESSED_DIR.name}/")

        return True

    except Exception as e:
        logging.error(f"Processing error: {e}")
        shutil.move(str(video_path), FAILED_DIR / video_path.name)
        return False

# =========================
# MAIN
# =========================

# Global models - loaded once at startup
GLOBAL_WHISPER_MODEL = None
GLOBAL_SUMMARIZER = None

def load_models():
    """Load Whisper and summarizer models globally."""
    global GLOBAL_WHISPER_MODEL, GLOBAL_SUMMARIZER
    
    try:
        logging.info("Loading Whisper model...")
        GLOBAL_WHISPER_MODEL = whisper.load_model("base")
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        GLOBAL_WHISPER_MODEL = None
    
    def _load_summarizer():
        try:
            logging.info("Loading summarizer (BART)...")
            model = pipeline("summarization", model="facebook/bart-large-cnn")
            logging.info("Summarizer (BART) loaded successfully.")
            return model
        except Exception as e1:
            logging.warning(f"BART summarizer failed: {e1}, trying T5 fallback...")
            try:
                logging.info("Loading T5-small summarizer...")
                model = pipeline("text2text-generation", model="t5-small")
                logging.info("T5-small summarizer loaded successfully.")
                return model
            except Exception as e2:
                logging.error(f"Failed to load fallback summarizer: {e2}")
                return None
    
    GLOBAL_SUMMARIZER = _load_summarizer()

def main():
    # Load models at startup
    load_models()
    
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
