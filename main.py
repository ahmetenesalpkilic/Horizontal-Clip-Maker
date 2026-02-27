import os
import re
import warnings
import logging
import shutil
from pathlib import Path

import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip, concatenate_videoclips

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

# =========================
# PATHS
# =========================
INPUT_DIR = Path("input_videos")
PROCESSED_DIR = Path("processed_videos")
FAILED_DIR = Path("failed_videos")
OUTPUT_DIR = Path("output/clips")
SUMMARY_DIR = Path("output/summary")
TEMP_DIR = Path("output/temp")

for d in [INPUT_DIR, PROCESSED_DIR, FAILED_DIR, OUTPUT_DIR, SUMMARY_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================
CFG = {
    "whisper_model_size": "base",  # tiny yaparsan daha hızlı olur
}

# =========================
# MODEL MANAGER
# =========================
class ModelManager:
    whisper_model = None
    title_model = None

    @classmethod
    def load(cls):
        if cls.whisper_model is None:
            logging.info("Whisper yükleniyor...")
            cls.whisper_model = whisper.load_model(CFG["whisper_model_size"])

        if cls.title_model is None:
            logging.info("Başlık modeli yükleniyor...")
            cls.title_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base"
            )

ModelManager.load()

# =========================
# UTILS
# =========================
def clean_transcript(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_safe_filename(text):
    text = re.sub(r'[\\/:*?"<>|]', '', text)
    text = re.sub(r'[._]', '', text)  # _ ve . kaldır
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:80]

def unique_path(path: Path):
    counter = 1
    new_path = path
    while new_path.exists():
        new_path = path.with_stem(f"{path.stem}{counter}")
        counter += 1
    return new_path

# =========================
# TRANSCRIPTION
# =========================
def transcribe_audio(audio_path: Path):
    model = ModelManager.whisper_model
    result = model.transcribe(str(audio_path), fp16=False)
    return clean_transcript(result.get("text", ""))

# =========================
# TITLE GENERATION
# =========================
def generate_title(text, fallback):
    if not text or len(text) < 10:
        return fallback

    prompt = f"""
    Aşağıdaki konuşmadan kısa, dikkat çekici ve anlamlı bir YouTube başlığı üret.
    Sadece başlığı yaz. Açıklama yazma.

    Metin:
    {text[:800]}
    """

    try:
        result = ModelManager.title_model(
            prompt,
            max_length=40,
            do_sample=False
        )

        title = result[0]["generated_text"]
        title = make_safe_filename(title)

        if len(title) < 5:
            return fallback

        return title

    except Exception as e:
        logging.error(f"Başlık hatası: {e}")
        return fallback

# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path: Path):
    try:
        video = VideoFileClip(str(video_path))
        duration = video.duration

        # Basit ortadan clip alma (örnek)
        start = max(0, duration/2 - 10)
        end = min(duration, duration/2 + 15)

        clip = video.subclip(start, end)

        temp_audio = TEMP_DIR / "temp.wav"
        clip.audio.write_audiofile(str(temp_audio), logger=None)

        transcript = transcribe_audio(temp_audio)
        title = generate_title(transcript, "Video")

        temp_audio.unlink(missing_ok=True)

        safe_title = make_safe_filename(title)
        out_path = unique_path(OUTPUT_DIR / f"{safe_title}.mp4")

        clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        video.close()
        shutil.move(str(video_path), PROCESSED_DIR / video_path.name)

        logging.info(f"Oluşturuldu: {safe_title}")
        return True

    except Exception as e:
        logging.error(f"Hata: {e}")
        shutil.move(str(video_path), FAILED_DIR / video_path.name)
        return False

# =========================
# MAIN
# =========================
def main():
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    for v in videos:
        process_video(v)

if __name__ == "__main__":
    main()