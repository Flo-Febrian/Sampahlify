
import os, json
import numpy as np
from PIL import Image
import gradio as gr

# ====== KONFIGURASI ======
IMG_SIZE = (224, 224)
MODEL_LOCAL_PATH = "model_klasifikasi_sampah.h5"   # letakkan file ini di root Space
LABEL_MAP_LOCAL_PATH = "label_map.json"            # letakkan di root Space
USE_HF_HUB = False                                 # set True jika mau download dari repo model terpisah
HF_REPO_ID = "username/repo-model-anda"            # ganti jika USE_HF_HUB=True
HF_MODEL_FILENAME = "model_klasifikasi_sampah.h5"
HF_LABELMAP_FILENAME = "label_map.json"

# ====== LOAD MODEL & LABEL MAP ======
def ensure_assets():
    model_path = MODEL_LOCAL_PATH
    labelmap_path = LABEL_MAP_LOCAL_PATH

    if USE_HF_HUB:
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
            labelmap_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_LABELMAP_FILENAME)
        except Exception as e:
            raise RuntimeError(f"Gagal unduh aset dari Hub: {e}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
    if not os.path.exists(labelmap_path):
        raise FileNotFoundError(f"File label_map tidak ditemukan: {labelmap_path}")

    return model_path, labelmap_path

model_path, labelmap_path = ensure_assets()

# Import TensorFlow setelah aset siap (mempercepat start-up jika gagal)
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(model_path)

with open(labelmap_path, "r") as f:
    label_map = json.load(f)  # contoh: {"anorganik": 0, "organik": 1}
inv_label_map = {v: k for k, v in label_map.items()}
pretty = {"anorganik": "Anorganik", "organik": "Organik"}

def preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(pil_img: Image.Image):
    if pil_img is None:
        return "Unggah gambar dulu.", {}
    x = preprocess(pil_img)
    # Output sigmoid 1 neuron => probabilitas kelas indeks 1
    prob_class1 = float(model.predict(x, verbose=0)[0][0])

    idx_organik = label_map.get("organik", 1)
    if idx_organik == 1:
        p_org = prob_class1
        p_anon = 1.0 - prob_class1
    else:
        p_anon = prob_class1
        p_org = 1.0 - prob_class1

    if p_org >= p_anon:
        label = pretty["organik"]; conf = p_org
    else:
        label = pretty["anorganik"]; conf = p_anon

    probs = {"Anorganik": round(p_anon, 4), "Organik": round(p_org, 4)}
    return f"{label} — {conf*100:.2f}%", probs

with gr.Blocks(title="Klasifikasi Sampah (MobileNetV2)") as demo:
    gr.Markdown(
        '''
        # Klasifikasi Sampah (MobileNetV2)
        Unggah foto sampah. Model memprediksi **Organik** atau **Anorganik**.
        '''
    )
    inp = gr.Image(type="pil", label="Unggah Gambar", sources=["upload", "clipboard"])
    out_text = gr.Textbox(label="Prediksi", interactive=False)
    out_probs = gr.Label(label="Probabilitas")
    btn = gr.Button("Prediksi")

    btn.click(fn=predict, inputs=inp, outputs=[out_text, out_probs])
    inp.change(fn=predict, inputs=inp, outputs=[out_text, out_probs])

demo.launch()
