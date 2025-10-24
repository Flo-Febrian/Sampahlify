
# Klasifikasi Sampah (MobileNetV2)

Aplikasi Gradio untuk klasifikasi sampah Organik/Anorganik.
- Model: Transfer Learning MobileNetV2 (sigmoid 1 neuron)
- Input: gambar .jpg/.png
- Output: label + probabilitas

## Cara deploy di Hugging Face Spaces
1. Buat Space baru (type: **Gradio**, hardware: **CPU**).
2. Upload file berikut ke root:
   - `app.py`
   - `requirements.txt`
   - `model_klasifikasi_sampah.h5`  ← file model dari Colab
   - `label_map.json`               ← mapping kelas dari Colab
3. Tunggu build sampai selesai. URL permanen akan aktif.

> Alternatif: simpan model di repo Hub terpisah. Set `USE_HF_HUB=True` di `app.py`, isi `HF_REPO_ID`, dan pastikan dua file (`model_klasifikasi_sampah.h5` & `label_map.json`) tersedia di repo tersebut.
