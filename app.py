# ============================================
# app.py — Full PPE Detection App
# ============================================

import cv2
from ultralytics import YOLO
import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image
import numpy as np

# ── Load model (put best.pt in the same folder as app.py) ──
model = YOLO("best.pt")

# ── This is your analyze function ──
def analyze_image(image_path):
    results = model(image_path)[0]
    img = cv2.imread(image_path)

    helmet_count = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 0:  # helmet → GREEN
            color = (0, 255, 0)
            label = f"Helmet {conf:.0%}"
            helmet_count += 1
        elif cls_id == 1:  # head (no helmet) → RED
            color = (0, 0, 255)
            label = f"No Helmet {conf:.0%}"
            no_helmet_count += 1
        else:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total = helmet_count + no_helmet_count
    compliance = (helmet_count / total * 100) if total > 0 else 0

    return img, helmet_count, no_helmet_count, compliance


# ── Streamlit UI starts here ──
st.set_page_config(page_title="PPE Monitor", layout="wide")
st.title("🦺 Surveillance Automatisée des EPI")

mode = st.sidebar.radio("Mode", ["🔍 Contrôle Unitaire", "📁 Audit Chantier"])

# ─────────────────────────────────────────
# MODE 1: Single image
# ─────────────────────────────────────────
if mode == "🔍 Contrôle Unitaire":
    st.header("Contrôle d'Accès - Image Unique")
    uploaded = st.file_uploader("Charger une photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        img_result, helmets, no_helmets, compliance = analyze_image(tmp_path)
        img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        col1.image(Image.open(tmp_path), caption="Original")
        col2.image(img_rgb, caption="Résultat Détection")

        st.metric("✅ Avec casque", helmets)
        st.metric("🚨 Sans casque", no_helmets)
        st.metric("📊 Taux de conformité", f"{compliance:.1f}%")

        if no_helmets > 0:
            st.error(f"⚠️ ALERTE : {no_helmets} travailleur(s) sans casque détecté(s)!")
        else:
            st.success("✅ Tous les travailleurs portent un casque!")

# ─────────────────────────────────────────
# MODE 2: Batch audit
# ─────────────────────────────────────────
elif mode == "📁 Audit Chantier":
    st.header("Audit Chantier - Traitement par Lot")
    uploaded_files = st.file_uploader("Charger plusieurs photos",
                                      type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)

    if uploaded_files and st.button("Lancer l'Audit"):
        report = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            _, helmets, no_helmets, compliance = analyze_image(tmp_path)
            total = helmets + no_helmets
            report.append({
                "Photo": f.name,
                "Total ouvriers": total,
                "Avec casque": helmets,
                "Sans casque": no_helmets,
                "Conformité (%)": round(compliance, 1)
            })
            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(report)
        st.dataframe(df)

        excel_path = "/tmp/rapport_audit.xlsx"
        df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("📥 Télécharger Rapport Excel", f,
                               file_name="rapport_audit.xlsx")