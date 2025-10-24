# app.py — Sampahlify (Streamlit)
import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# ====== KONFIGURASI ======
IMG_SIZE = (224, 224)
MODEL_LOCAL_PATH = "model_klasifikasi_sampah.h5"   # taruh di root repo/app
LABEL_MAP_LOCAL_PATH = "label_map.json"            # taruh di root repo/app

# Jika ingin ambil aset dari Hugging Face Hub, set True dan isi repo_id
USE_HF_HUB = False
HF_REPO_ID = "username/repo-model-anda"            # ex: "luciphella/sampahlify-model"
HF_MODEL_FILENAME = "model_klasifikasi_sampah.h5"
HF_LABELMAP_FILENAME = "label_map.json"

# ====== UTIL ======
def ensure_assets() -> tuple[str, str]:
    """Pastikan file model & label_map tersedia, opsional unduh dari Hub."""
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

# ====== LOAD ASET (cache agar cepat) ======
@st.cache_resource(show_spinner=False)
def load_assets():
    model_path, labelmap_path = ensure_assets()

    # Import TF di dalam cache agar tidak reload berulang
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    with open(labelmap_path, "r") as f:
        label_map = json.load(f)   # contoh: {"anorganik": 0, "organik": 1}
    inv_label_map = {v: k for k, v in label_map.items()}
    pretty = {"anorganik": "Anorganik", "organik": "Organik"}
    return model, label_map, inv_label_map, pretty

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

def predict_image(model, label_map, inv_label_map, pretty, pil_img: Image.Image):
    x = preprocess(pil_img)
    # Output sigmoid 1 neuron → probabilitas kelas indeks 1
    prob_class1 = float(model.predict(x, verbose=0)[0][0])

    idx_organik = label_map.get("organik", 1)
    if idx_organik == 1:
        p_org = prob_class1
        p_anon = 1.0 - prob_class1
    else:
        p_anon = prob_class1
        p_org = 1.0 - prob_class1

    if p_org >= p_anon:
        label_idx = 1
        conf = p_org
    else:
        label_idx = 0
        conf = p_anon

    label_raw = inv_label_map.get(label_idx, "organik" if label_idx == 1 else "anorganik")
    label_pretty = pretty.get(label_raw, label_raw.capitalize())
    probs = {"Anorganik": round(p_anon, 4), "Organik": round(p_org, 4)}
    return label_pretty, conf, probs

# ====== UI ======
st.set_page_config(page_title="Sampahlify", page_icon="♻️", layout="centered")

st.title("♻️ Sampahlify")
st.caption("Klasifikasi sampah **Organik** vs **Anorganik** (MobileNetV2, Transfer Learning)")

with st.expander("Cara pakai", expanded=False):
    st.markdown(
        "- Unggah gambar .jpg/.png\n"
        "- Aplikasi akan menampilkan label prediksi dan confidence\n"
        "- Model & mapping kelas dibaca dari file di root repo"
    )

# Sidebar
st.sidebar.header("Pengaturan")
st.sidebar.write(f"Ukuran input model: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")
st.sidebar.write("Mode aset:", "**Hugging Face Hub**" if USE_HF_HUB else "**Local files**")

# Load model & label map
with st.status("Memuat model…", expanded=False):
    model, label_map, inv_label_map, pretty = load_assets()
st.success("Model siap dipakai.")

# Upload
uploaded = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Gambar diunggah", use_container_width=True)

    with st.spinner("Memproses…"):
        label, conf, probs = predict_image(model, label_map, inv_label_map, pretty, pil)

    st.subheader(f"Hasil: **{label}** — {conf*100:.2f}%")
    st.progress(conf)

    # Tampilkan probabilitas
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Organik", f"{probs['Organik']*100:.2f}%")
    with c2:
        st.metric("Anorganik", f"{probs['Anorganik']*100:.2f}%")

    st.caption("Tip: gunakan foto jelas dengan objek dominan untuk hasil terbaik.")
else:
    st.info("Unggah gambar untuk mulai prediksi.")
