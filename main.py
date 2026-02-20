import os
import re
import warnings
import json
import time
import logging
import shutil
from pathlib import Path

import numpy as np
import librosa
import cv2
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout

# suppress common librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

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
INPUT_DIR      = Path("input_videos")
PROCESSED_DIR  = Path("processed_videos")
FAILED_DIR     = Path("failed_videos")
OUTPUT_BASE_DIR = Path("output")
OUTPUT_DIR     = OUTPUT_BASE_DIR / "clips"
SUMMARY_DIR    = OUTPUT_BASE_DIR / "summary"
TEMP_DIR       = OUTPUT_BASE_DIR / "temp"

for d in [INPUT_DIR, OUTPUT_BASE_DIR, OUTPUT_DIR, SUMMARY_DIR,
          PROCESSED_DIR, FAILED_DIR, TEMP_DIR]:
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
    "profile":             "default",
    "pre_min":             12,
    "pre_max":             20,
    "post_min":            20,
    "post_max":            30,
    "min_clips":           2,
    "max_clips":           7,
    "audio_cluster_sec":   2.0,
    "motion_threshold":    15.0,
    "audio_sensitivity":   1.8,
    "use_transitions":     True,
    "transition_duration": 0.3,
    # parallel_processing kaldırıldı — model paylaşımı ile uyumsuz
    "whisper_model_size":  "base",   # tiny | base | small | medium | large
    "overlay_font_size":   24,
    "overlay_duration":    5,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return dict(DEFAULT_CONFIG)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # fill missing keys with defaults
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    # apply profile overrides
    profile_name = config.get("profile", "default")
    if profile_name in PROFILES:
        config.update(PROFILES[profile_name])

    return config


CFG: dict = load_config()

# =========================
# MODEL MANAGER  (singleton)
# =========================
class ModelManager:
    """
    Modelleri yalnızca bir kez yükler.
    İlk erişimde lazy-load yapılır; sonraki çağrılarda cache'ten döner.
    """
    _whisper_model = None
    _summarizer    = None
    _loaded        = False

    @classmethod
    def load(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True   # hata olsa bile tekrar denememek için önce işaretle

        # --- Whisper ---
        try:
            size = CFG.get("whisper_model_size", "base")
            logging.info(f"Whisper modeli yükleniyor: '{size}' ...")
            cls._whisper_model = whisper.load_model(size)
            logging.info("Whisper modeli başarıyla yüklendi.")
        except Exception as exc:
            logging.error(f"Whisper yüklenemedi: {exc}")
            cls._whisper_model = None

        # --- Summarizer (BART → T5 fallback) ---
        try:
            logging.info("Özetleyici yükleniyor (BART) ...")
            cls._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            logging.info("BART özetleyici yüklendi.")
        except Exception as e1:
            logging.warning(f"BART yüklenemedi: {e1}  →  T5-small deneniyor ...")
            try:
                cls._summarizer = pipeline("text2text-generation", model="t5-small")
                logging.info("T5-small özetleyici yüklendi.")
            except Exception as e2:
                logging.error(f"Özetleyici de yüklenemedi: {e2}")
                cls._summarizer = None

    @classmethod
    def whisper(cls):
        if not cls._loaded:
            cls.load()
        return cls._whisper_model

    @classmethod
    def summarizer(cls):
        if not cls._loaded:
            cls.load()
        return cls._summarizer


# =========================
# HELPERS
# =========================
def make_safe_filename(text: str, max_len: int = 40) -> str:
    """Metin → dosya sistemine uygun güvenli isim."""
    safe = re.sub(r'[\\/:*?"<>|]', "", text)
    safe = "_".join(safe.split())
    return safe[:max_len]


# =========================
# AUDIO SPIKE DETECTION
# =========================
def detect_audio_spikes(path: str) -> list[float]:
    """Seste ani yükselmeler olan zaman noktalarını döndürür."""
    try:
        y, sr = librosa.load(path, sr=None)
        rms   = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

        threshold   = np.median(rms) * CFG["audio_sensitivity"]
        raw_spikes  = times[rms > threshold]

        clustered: list[float] = []
        for t in raw_spikes:
            if not clustered or t - clustered[-1] > CFG["audio_cluster_sec"]:
                clustered.append(float(t))

        return clustered
    except Exception as exc:
        logging.error(f"Ses spike tespiti hatası: {exc}")
        return []


# =========================
# MOTION CHECK
# =========================
def has_motion(video_path: Path, t: float, duration: float = 1.0) -> bool:
    """t anında yeterli hareket var mı?"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))

        ret, prev = cap.read()
        if not ret:
            cap.release()
            return False

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diffs: list[float] = []

        for _ in range(int(fps * duration)):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray

        cap.release()
        return bool(np.mean(diffs) > CFG["motion_threshold"]) if diffs else False
    except Exception:
        return False


# =========================
# TRANSCRIPTION
# =========================
def transcribe_audio(audio_path: Path) -> str:
    """
    Whisper ile sesi metne çevirir.
    Döndürülen metin temizlenmiş haldedir.
    """
    model = ModelManager.whisper()
    if model is None:
        logging.warning("Whisper modeli mevcut değil; transkripsiyon atlanıyor.")
        return ""
    try:
        logging.info(f"Transkripsiyon başlatılıyor: {audio_path.name}")
        result = model.transcribe(str(audio_path), fp16=False)
        text   = result.get("text", "").strip()
        if text:
            logging.info(f"Transkript ({audio_path.name}): {text[:120]} ...")
        else:
            logging.warning(f"Transkript bulunamadı: {audio_path.name}")
        return text
    except Exception as exc:
        logging.error(f"Transkripsiyon hatası ({audio_path.name}): {exc}")
        return ""


# =========================
# TITLE GENERATION
# =========================
def generate_title(transcript: str, fallback: str) -> str:
    """
    Transkriptten kısa bir başlık üretir.

    Öncelik sırası:
      1. Özetleyici (BART veya T5)
      2. İlk 50 karakter transkript
      3. Fallback (varsayılan isim)
    """
    if not transcript:
        return fallback

    summarizer = ModelManager.summarizer()

    if summarizer is not None:
        try:
            # T5 pipeline task adı "text2text-generation"
            is_t5 = getattr(summarizer, "task", "") == "text2text-generation"
            prompt = f"summarize: {transcript}" if is_t5 else transcript
            out    = summarizer(prompt, max_length=60, min_length=8, do_sample=False)
            key    = "generated_text" if is_t5 else "summary_text"
            title  = out[0].get(key, "").strip()
            if title:
                logging.info(f"Üretilen başlık: {title}")
                return title
        except Exception as exc:
            logging.error(f"Başlık üretme hatası: {exc}")

    # Özetleyici yoksa ya da hata verdiyse: transkriptin ilk 50 karakteri
    short = transcript[:50].strip()
    logging.info(f"Kısa transkript başlık: {short}")
    return short or fallback


# =========================
# TEXT OVERLAY
# =========================
def overlay_text(base_clip, text: str):
    """Klibin üstüne yarı-saydam başlık ekler."""
    if not text or base_clip.duration <= 0:
        return base_clip
    try:
        duration = min(CFG["overlay_duration"], base_clip.duration)
        txt = (
            TextClip(
                text,
                fontsize=CFG["overlay_font_size"],
                color="white",
                bg_color="rgba(0,0,0,128)",  # yarı saydam arka plan
                size=(base_clip.w, None),
                method="caption",
            )
            .set_duration(duration)
            .set_position(("center", "top"))
        )
        return CompositeVideoClip([base_clip, txt])
    except Exception as exc:
        logging.error(f"Metin kaplama hatası: {exc}")
        return base_clip


# =========================
# BUILD CLIP RANGES
# =========================
def build_clip_ranges(spikes: list[float], duration: float) -> list[tuple[float, float]]:
    """Spike noktalarından klip başlangıç-bitiş aralıkları üretir."""
    ranges: list[tuple[float, float]] = []

    for t in spikes:
        pre   = np.random.uniform(CFG["pre_min"],  CFG["pre_max"])
        post  = np.random.uniform(CFG["post_min"], CFG["post_max"])
        start = max(0.0, t - pre)
        end   = min(duration, t + post)
        if end - start >= 15:
            ranges.append((start, end))

    # Yeterli klip yoksa videoyu eşit parçalara böl
    min_clips = CFG.get("min_clips", 2)
    max_clips = CFG.get("max_clips", 7)

    if len(ranges) < min_clips:
        segment_length = duration / min_clips
        for i in range(min_clips):
            start = i * segment_length
            end   = min(duration, start + segment_length)
            overlap = any(abs(start - r[0]) < 1 and abs(end - r[1]) < 1 for r in ranges)
            if not overlap and end - start >= 15:
                ranges.append((start, end))
            if len(ranges) >= min_clips:
                break

    return ranges[:max_clips]


# =========================
# SMART COMPILATION
# =========================
def create_smart_compilation(clips: list):
    """Klipleri geçişlerle birleştirir."""
    if CFG["use_transitions"]:
        dur = CFG["transition_duration"]
        processed = [fadeout(fadein(c, dur), dur) for c in clips]
        return concatenate_videoclips(processed, method="compose")
    return concatenate_videoclips(clips, method="compose")


# =========================
# PROCESS SINGLE VIDEO
# =========================
def process_video(video_path: Path) -> bool:
    """
    Tek bir videoyu işler:
      1. Ses spike'larını tespit et
      2. Hareket kontrolü yap
      3. Klipler oluştur
      4. Her klip için: ses çıkar → transkripsiyon → başlık üret → overlay ekle
      5. Özet video oluştur
      6. Videoyu 'processed' klasörüne taşı
    """
    logging.info(f"{'='*60}")
    logging.info(f"Video işleniyor: {video_path.name}")

    video = None
    clips_for_summary: list = []

    try:
        video    = VideoFileClip(str(video_path))
        duration = video.duration

        # --- Spike & hareket tespiti ---
        spikes       = detect_audio_spikes(str(video_path))
        valid_spikes = [t for t in spikes if has_motion(video_path, t)]
        logging.info(f"  Toplam spike: {len(spikes)}  |  Hareketle spike: {len(valid_spikes)}")

        if not valid_spikes:
            logging.warning("  Hareketle spike yok; videonun ortası kullanılıyor.")
            valid_spikes = [duration / 2]

        ranges = build_clip_ranges(valid_spikes, duration)
        logging.info(f"  Oluşturulacak klip sayısı: {len(ranges)}")

        for i, (start, end) in enumerate(ranges, start=1):
            clip_label    = f"{video_path.stem}_clip_{i}"
            temp_audio    = TEMP_DIR / f"{clip_label}_audio.wav"
            clip          = video.subclip(start, end)

            # --- Ses çıkar ---
            try:
                clip.audio.write_audiofile(
                    str(temp_audio), logger=None, verbose=False
                )
            except Exception as exc:
                logging.error(f"  Klip {i} ses çıkarma hatası: {exc}")
                temp_audio = None

            # --- Transkripsiyon ---
            transcript = transcribe_audio(temp_audio) if temp_audio else ""

            # --- Başlık üret ---
            title = generate_title(transcript, fallback=f"Clip {i}")

            # --- Geçici ses dosyasını sil ---
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()
                logging.info(f"  Geçici ses silindi: {temp_audio.name}")

            # --- Overlay ekle ---
            clip_with_title = overlay_text(clip, title)

            # --- Klip dosyasını yaz ---
            safe_title = make_safe_filename(title) or f"clip_{i}"
            out_path   = OUTPUT_DIR / f"{clip_label}_{safe_title}.mp4"
            logging.info(f"  Klip {i} yazılıyor → {out_path.name}")
            clip_with_title.write_videofile(
                str(out_path),
                codec="libx264",
                fps=30,
                audio_codec="aac",
                logger=None,
                verbose=False,
            )
            clips_for_summary.append(VideoFileClip(str(out_path)))
            logging.info(f"  Klip {i} tamamlandı: '{title}'")

        # --- Özet video ---
        if clips_for_summary:
            logging.info(f"Özet video oluşturuluyor ({len(clips_for_summary)} klip) ...")
            summary = create_smart_compilation(clips_for_summary)
            summary = overlay_text(summary, f"Summary — {video_path.stem}")

            summary_name = make_safe_filename(f"{video_path.stem}_SUMMARY")
            summary_out  = SUMMARY_DIR / f"{summary_name}.mp4"
            summary.write_videofile(
                str(summary_out),
                codec="libx264",
                fps=30,
                audio_codec="aac",
                logger=None,
                verbose=False,
            )
            summary.close()
            logging.info(f"Özet video yazıldı: {summary_out.name}")

        # --- Kaynak temizle ---
        for c in clips_for_summary:
            try:
                c.close()
            except Exception:
                pass
        video.close()

        # --- Videoyu 'processed' klasörüne taşı ---
        dest = PROCESSED_DIR / video_path.name
        shutil.move(str(video_path), str(dest))
        logging.info(f"Video taşındı → {PROCESSED_DIR.name}/{video_path.name}")
        return True

    except Exception as exc:
        logging.error(f"İşleme hatası [{video_path.name}]: {exc}", exc_info=True)
        if video:
            try:
                video.close()
            except Exception:
                pass
        dest = FAILED_DIR / video_path.name
        shutil.move(str(video_path), str(dest))
        logging.warning(f"Video başarısız klasörüne taşındı: {dest}")
        return False


# =========================
# MAIN
# =========================
def main():
    logging.info("Uygulama başlatılıyor ...")

    # Modelleri ana süreçte bir kez yükle
    ModelManager.load()

    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        logging.info("input_videos/ klasöründe .mp4 bulunamadı. Çıkılıyor.")
        return

    logging.info(f"Toplam {len(videos)} video işlenecek.")
    results = {"success": 0, "fail": 0}

    for video_path in videos:
        ok = process_video(video_path)
        if ok:
            results["success"] += 1
        else:
            results["fail"] += 1

    logging.info(
        f"Tüm videolar tamamlandı — "
        f"Başarılı: {results['success']} | Başarısız: {results['fail']}"
    )


if __name__ == "__main__":
    main()