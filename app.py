# ============================================
# app.py — Full PPE Detection App (Cloud-safe)
# ============================================

import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── Lazy imports (inside functions to avoid top-level crash) ──
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error("❌ Fichier best.pt introuvable. Placez-le dans le même dossier que app.py")
        st.stop()
    return YOLO(model_path)

def draw_boxes_pil(image_path, results):
    """Draw bounding boxes using PIL instead of OpenCV (no libGL needed)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    helmet_count   = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 0:       # helmet → VERT
            color = (0, 200, 0)
            label = f"Helmet {conf:.0%}"
            helmet_count += 1
        elif cls_id == 1:     # head (no helmet) → ROUGE
            color = (220, 0, 0)
            label = f"No Helmet {conf:.0%}"
            no_helmet_count += 1
        else:
            continue

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background + text
        text_w = len(label) * 7
        draw.rectangle([x1, y1 - 20, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - 18), label, fill=(255, 255, 255))

    total      = helmet_count + no_helmet_count
    compliance = (helmet_count / total * 100) if total > 0 else 0

    return img, helmet_count, no_helmet_count, compliance


def analyze_image(image_path):
    model   = load_model()
    results = model(image_path)[0]
    return draw_boxes_pil(image_path, results)


# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(page_title="PPE Monitor", layout="wide")
st.title("🦺 Surveillance Automatisée des EPI")

mode = st.sidebar.radio("Mode", ["🔍 Contrôle Unitaire", "📁 Audit Chantier"])

# ─────────────────────────────────────────
# MODE 1 : Image unique
# ─────────────────────────────────────────
if mode == "🔍 Contrôle Unitaire":
    st.header("Contrôle d'Accès - Image Unique")
    uploaded = st.file_uploader("Charger une photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyse en cours..."):
            img_result, helmets, no_helmets, compliance = analyze_image(tmp_path)

        col1, col2 = st.columns(2)
        col1.image(Image.open(tmp_path), caption="Original", use_column_width=True)
        col2.image(img_result,           caption="Résultat Détection", use_column_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Avec casque",         helmets)
        c2.metric("🚨 Sans casque",          no_helmets)
        c3.metric("📊 Taux de conformité",   f"{compliance:.1f}%")

        if no_helmets > 0:
            st.error(f"⚠️ ALERTE : {no_helmets} travailleur(s) sans casque détecté(s)!")
        else:
            st.success("✅ Tous les travailleurs portent un casque!")

        os.unlink(tmp_path)

# ─────────────────────────────────────────
# MODE 2 : Audit par lot
# ─────────────────────────────────────────
elif mode == "📁 Audit Chantier":
    st.header("Audit Chantier - Traitement par Lot")
    uploaded_files = st.file_uploader(
        "Charger plusieurs photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Lancer l'Audit"):
        report   = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            _, helmets, no_helmets, compliance = analyze_image(tmp_path)
            total = helmets + no_helmets
            report.append({
                "Photo":           f.name,
                "Total ouvriers":  total,
                "Avec casque":     helmets,
                "Sans casque":     no_helmets,
                "Conformité (%)":  round(compliance, 1)
            })
            os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(report)
        st.dataframe(df)

        # Export Excel
        excel_path = os.path.join(tempfile.gettempdir(), "rapport_audit.xlsx")
        df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button(
                "📥 Télécharger Rapport Excel", f,
                file_name="rapport_audit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
