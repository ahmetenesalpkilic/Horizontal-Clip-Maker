# 🎯 Horizontal Clip Maker (AI-Powered)

Bu proje, uzun YouTube videolarını yapay zeka kullanarak analiz eden ve en dikkat çekici anları otomatik olarak **yatay (16:9)** formatta kliplere dönüştüren bir otomasyon aracıdır. 

Özellikle oyun videoları, podcastler ve eğitim içerikleri için "Highlights" (Öne Çıkanlar) oluşturmak amacıyla tasarlanmıştır.

## 🧠 Nasıl Çalışır?

Sistem temel olarak şu akışı takip eder:

1.  **Audio-to-Text (ASR):** `OpenAI Whisper` ile videonun sesi metne dönüştürülür.
2.  **Smart Scoring:** Belirlenen anahtar kelimeler ve soru kalıpları üzerinden metin analiz edilerek "ilgi çekici" segmentler belirlenir.
3.  **Video Processing:** `MoviePy` kütüphanesi ile belirlenen zaman damgaları (timestamps) üzerinden video kayıpsız bir şekilde kesilir.
4.  **Auto-Title:** Her segment için içeriğe uygun otomatik başlık önerileri sunulur.

## 🛠️ Kullanılan Teknolojiler

* **Python:** Ana programlama dili.
* **OpenAI Whisper:** Ses tanıma ve transkript çıkarma.
* **MoviePy:** Video düzenleme ve işleme.
* **NLP Logic:** Kelime bazlı skorlama ve segment seçimi.

## 🚀 Kurulum

Projeyi yerelinizde çalıştırmak için:

```bash
git clone [https://github.com/ahmetenesalpkilic/Horizontal-Clip-Maker.git](https://github.com/ahmetenesalpkilic/Horizontal-Clip-Maker.git)
cd Horizontal-Clip-Maker
pip install -r requirements.txt
