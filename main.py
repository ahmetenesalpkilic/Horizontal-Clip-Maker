import os
import random
import numpy as np
import librosa
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips

# ================== AYARLAR ==================

INPUT_DIR = "input_videos"
CLIP_DIR = "output/clips"
SUMMARY_DIR = "output/summary"

os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

PRE_MIN, PRE_MAX = 12, 20
POST_MIN, POST_MAX = 20, 30

MIN_CLIPS = 2
MAX_CLIPS = 7

MIN_EVENT_GAP = 25  # saniye (aynı olayı tekrar alma)

# ================== SES ANALİZİ ==================

def detect_audio_spikes(video_path, threshold_mult=2.5):
    y, sr = librosa.load(video_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    threshold = np.mean(rms) * threshold_mult
    spikes = times[rms > threshold]

    return spikes.tolist()

# ================== KLİP OLUŞTUR ==================

def build_clip_ranges(spikes, video_duration):
    clips = []

    for t in spikes:
        pre = random.randint(PRE_MIN, PRE_MAX)
        post = random.randint(POST_MIN, POST_MAX)

        start = max(0, t - pre)
        end = min(video_duration, t + post)

        clips.append({
            "time": t,
            "start": start,
            "end": end,
            "score": post + pre  # basit skor
        })

    return clips

# ================== FİLTRELE ==================

def filter_clips(candidates):
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []

    for c in candidates:
        if all(abs(c["time"] - s["time"]) > MIN_EVENT_GAP for s in selected):
            selected.append(c)
        if len(selected) >= MAX_CLIPS:
            break

    if len(selected) < MIN_CLIPS:
        selected = candidates[:MIN_CLIPS]

    return sorted(selected, key=lambda x: x["start"])

# ================== ANA AKIŞ ==================

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".mp4"):
        continue

    video_path = os.path.join(INPUT_DIR, file)
    print(f"\n🎮 İşleniyor: {file}")

    video = VideoFileClip(video_path)

    print("🔊 Ses spike'ları aranıyor...")
    spikes = detect_audio_spikes(video_path)

    if not spikes:
        print("❌ Spike bulunamadı, video atlandı.")
        continue

    candidates = build_clip_ranges(spikes, video.duration)
    selected = filter_clips(candidates)

    clip_paths = []

    print(f"✂ {len(selected)} klip çıkarılıyor...")

    for i, c in enumerate(selected, 1):
        clip = video.subclip(c["start"], c["end"])
        out_path = os.path.join(CLIP_DIR, f"{file[:-4]}_clip_{i}.mp4")

        clip.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )

        clip_paths.append(out_path)
        print(f"   ✅ clip_{i} ({int(c['end'] - c['start'])} sn)")

    # ================== ÖZET VİDEO ==================

    print("🎬 Özet video oluşturuluyor...")

    clips = [VideoFileClip(p) for p in clip_paths]
    summary = concatenate_videoclips(clips, method="compose")

    summary_path = os.path.join(
        SUMMARY_DIR,
        f"{file[:-4]}_HIGHLIGHTS.mp4"
    )

    summary.write_videofile(
        summary_path,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None
    )

    print(f"🏁 Özet hazır: {summary_path}")
