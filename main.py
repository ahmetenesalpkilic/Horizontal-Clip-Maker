import os
import json
import time
import random
import logging
import traceback
import gc
import psutil
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
import cv2
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip, concatenate_videoclips

# ==========================================================
# LOGGING
# ==========================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("highlight_generator.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("HIGHLIGHT_ENGINE")

logger = setup_logging()

# ==========================================================
# CONFIG
# ==========================================================

DEFAULT_CONFIG = {
    "INPUT_DIR": "input_videos",
    "CLIP_DIR": "output/clips",
    "SUMMARY_DIR": "output/summary",

    "PRE_MIN": 12,
    "PRE_MAX": 20,
    "POST_MIN": 20,
    "POST_MAX": 30,

    "MIN_CLIPS": 2,
    "MAX_CLIPS": 7,

    "SPIKE_CLUSTER_WINDOW": 2.0,
    "MIN_EVENT_GAP": 25,
    "MOTION_THRESHOLD": 18.0,

    "VIDEO_FORMATS": [".mp4", ".mkv", ".avi", ".mov"],

    "FALLBACK_ENABLED": True,
    "FALLBACK_INTERVAL": 60,

    "MAX_MEMORY_PERCENT": 80,
    "PARALLEL": True,
    "MAX_WORKERS": 4
}

def load_config(path="config.json") -> Dict:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.info("Varsayılan config.json oluşturuldu")
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)

    return cfg

CFG = load_config()

# ==========================================================
# SETUP
# ==========================================================

for d in [CFG["INPUT_DIR"], CFG["CLIP_DIR"], CFG["SUMMARY_DIR"]]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# UTILS
# ==========================================================

def check_memory():
    p = psutil.Process(os.getpid())
    mem = p.memory_percent()
    if mem > CFG["MAX_MEMORY_PERCENT"]:
        logger.warning(f"Yüksek bellek kullanımı: %{mem:.1f}")
        gc.collect()

def is_valid_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext not in CFG["VIDEO_FORMATS"]:
        return False
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    return ok

# ==========================================================
# AUDIO
# ==========================================================

def detect_audio_spikes(path: str) -> List[Tuple[float, float]]:
    y, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    med = np.median(rms)
    mad = np.median(np.abs(rms - med))
    threshold = med + 3 * mad

    peaks, props = find_peaks(rms, height=threshold)
    return list(zip(times[peaks], props["peak_heights"]))

def cluster_spikes(spikes: List[Tuple[float, float]]) -> List[float]:
    clusters, current = [], []
    for t, v in spikes:
        if not current or t - current[-1][0] <= CFG["SPIKE_CLUSTER_WINDOW"]:
            current.append((t, v))
        else:
            clusters.append(max(current, key=lambda x: x[1]))
            current = [(t, v)]
    if current:
        clusters.append(max(current, key=lambda x: x[1]))
    return [c[0] for c in clusters]

# ==========================================================
# MOTION
# ==========================================================

def motion_score(path: str, time_sec: float, window=1.0) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((time_sec - window) * fps)))

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return 0

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    diffs = []

    for _ in range(int(window * fps * 2)):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diffs.append(np.mean(cv2.absdiff(prev_g, g)))
        prev_g = g

    cap.release()
    return float(np.mean(diffs)) if diffs else 0

# ==========================================================
# CLIPS
# ==========================================================

def build_clips(events: List[float], path: str, dur: float) -> List[Dict]:
    clips = []
    for t in events:
        m = motion_score(path, t)
        if m < CFG["MOTION_THRESHOLD"]:
            continue

        pre = random.randint(CFG["PRE_MIN"], CFG["PRE_MAX"])
        post = random.randint(CFG["POST_MIN"], CFG["POST_MAX"])

        clips.append({
            "time": t,
            "start": max(0, t - pre),
            "end": min(dur, t + post),
            "score": m + post
        })
    return clips

def fallback_clips(dur: float) -> List[Dict]:
    clips = []
    step = CFG["FALLBACK_INTERVAL"]
    for i in range(int(dur // step)):
        t = (i + 1) * step
        pre = random.randint(CFG["PRE_MIN"], CFG["PRE_MAX"])
        post = random.randint(CFG["POST_MIN"], CFG["POST_MAX"])
        clips.append({
            "time": t,
            "start": max(0, t - pre),
            "end": min(dur, t + post),
            "score": 50
        })
    return clips

def select_clips(cands: List[Dict]) -> List[Dict]:
    cands.sort(key=lambda x: x["score"], reverse=True)
    sel = []
    for c in cands:
        if all(abs(c["time"] - s["time"]) > CFG["MIN_EVENT_GAP"] for s in sel):
            sel.append(c)
        if len(sel) >= CFG["MAX_CLIPS"]:
            break
    return sorted(sel[:max(CFG["MIN_CLIPS"], len(sel))], key=lambda x: x["start"])

# ==========================================================
# VIDEO PIPELINE
# ==========================================================

def process_video(path: str) -> bool:
    name = os.path.basename(path)
    logger.info(f"🎮 {name} başlandı")

    video = None
    try:
        video = VideoFileClip(path)
        dur = video.duration

        spikes = cluster_spikes(detect_audio_spikes(path))
        candidates = build_clips(spikes, path, dur) if spikes else fallback_clips(dur)
        selected = select_clips(candidates)

        if not selected:
            logger.warning("Klip yok")
            return False

        outputs = []
        for i, c in enumerate(selected, 1):
            clip = video.subclip(c["start"], c["end"])
            out = f"{CFG['CLIP_DIR']}/{name[:-4]}_{i}.mp4"
            clip.write_videofile(out, codec="libx264", audio_codec="aac",
                                 verbose=False, logger=None)
            clip.close()
            outputs.append(out)
            check_memory()

        clips = [VideoFileClip(p) for p in outputs]
        summary = concatenate_videoclips(clips)
        summary.write_videofile(
            f"{CFG['SUMMARY_DIR']}/{name[:-4]}_HIGHLIGHTS.mp4",
            codec="libx264", audio_codec="aac",
            verbose=False, logger=None
        )

        summary.close()
        for c in clips:
            c.close()

        logger.info(f"✅ {name} tamamlandı")
        return True

    except Exception as e:
        logger.error(f"{name} hata: {e}")
        traceback.print_exc()
        return False

    finally:
        if video:
            video.close()
        gc.collect()

# ==========================================================
# MAIN
# ==========================================================

def main():
    start = time.time()
    files = [
        os.path.join(CFG["INPUT_DIR"], f)
        for f in os.listdir(CFG["INPUT_DIR"])
        if is_valid_video(os.path.join(CFG["INPUT_DIR"], f))
    ]

    if not files:
        logger.warning("Geçerli video yok")
        return

    ok = 0
    if CFG["PARALLEL"] and len(files) > 1:
        with ThreadPoolExecutor(max_workers=CFG["MAX_WORKERS"]) as ex:
            for fut in as_completed([ex.submit(process_video, f) for f in files]):
                ok += int(fut.result())
    else:
        for f in files:
            ok += int(process_video(f))

    logger.info(f"🏁 Bitti | Başarılı: {ok}/{len(files)} | Süre: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
