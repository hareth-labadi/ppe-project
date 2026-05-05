# ============================================
# app.py — PPE Detection App (Cloud-safe)
# No top-level cv2 or ultralytics imports
# Bounding boxes via PIL (no libGL needed)
# ============================================

import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np


# ── Model loader — lazy + cached ──────────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO  # imported here, not at top level
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    if not os.path.exists(model_path):
        st.error("❌ Fichier 'best.pt' introuvable. Placez-le dans le même dossier que app.py.")
        st.stop()
    return YOLO(model_path)


# ── Draw bounding boxes with PIL (no cv2 / no libGL) ──────────────────────
def draw_boxes(image_path, results):
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    helmet_count    = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 0:            # helmet → VERT
            color = (34, 197, 94)
            label = f"Helmet {conf:.0%}"
            helmet_count += 1
        elif cls_id == 1:          # head (no helmet) → ROUGE
            color = (239, 68, 68)
            label = f"No Helmet {conf:.0%}"
            no_helmet_count += 1
        else:
            continue

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        text_w = len(label) * 7 + 6
        text_h = 18
        draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w, y1], fill=color)

        # Label text
        draw.text((x1 + 3, max(0, y1 - text_h + 2)), label, fill=(255, 255, 255))

    total      = helmet_count + no_helmet_count
    compliance = (helmet_count / total * 100) if total > 0 else 0

    return img, helmet_count, no_helmet_count, compliance


# ── Main analyze function ──────────────────────────────────────────────────
def analyze_image(image_path):
    model   = load_model()
    results = model(image_path)[0]
    return draw_boxes(image_path, results)


# ══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "PPE Monitor — Détection Casques",
    page_icon  = "🦺",
    layout     = "wide"
)

st.title("🦺 Surveillance Automatisée des EPI")
st.caption("Détection automatique du port du casque de sécurité — YOLOv8")

mode = st.sidebar.radio(
    "Mode d'utilisation",
    ["🔍 Contrôle Unitaire", "📁 Audit Chantier"],
    help="Choisissez le mode selon votre besoin"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Code couleur :**")
st.sidebar.markdown("🟢 Vert = casque détecté")
st.sidebar.markdown("🔴 Rouge = sans casque")


# ─────────────────────────────────────────
# MODE 1 — Image unique
# ─────────────────────────────────────────
if mode == "🔍 Contrôle Unitaire":
    st.header("Contrôle d'Accès — Image Unique")

    uploaded = st.file_uploader(
        "Charger une photo de chantier",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Analyse en cours..."):
            img_result, helmets, no_helmets, compliance = analyze_image(tmp_path)

        col1, col2 = st.columns(2)
        col1.image(Image.open(tmp_path), caption="📷 Image originale",   use_column_width=True)
        col2.image(img_result,           caption="🎯 Résultat détection", use_column_width=True)

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Avec casque",       helmets)
        c2.metric("🚨 Sans casque",        no_helmets)
        c3.metric("📊 Taux de conformité", f"{compliance:.1f}%")

        st.markdown("---")
        if no_helmets > 0:
            st.error(f"⚠️ ALERTE : {no_helmets} travailleur(s) sans casque détecté(s) !")
        elif helmets == 0:
            st.warning("⚠️ Aucune personne détectée sur cette image.")
        else:
            st.success("✅ Tous les travailleurs portent un casque !")

        os.unlink(tmp_path)


# ─────────────────────────────────────────
# MODE 2 — Audit par lot
# ─────────────────────────────────────────
elif mode == "📁 Audit Chantier":
    st.header("Audit Chantier — Traitement par Lot")

    uploaded_files = st.file_uploader(
        "Charger plusieurs photos de chantier",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📂 {len(uploaded_files)} photo(s) chargée(s). Cliquez sur 'Lancer l'Audit' pour démarrer.")

    if uploaded_files and st.button("🚀 Lancer l'Audit", type="primary"):
        report   = []
        progress = st.progress(0, text="Analyse en cours...")
        preview_cols = st.columns(min(len(uploaded_files), 4))

        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            img_result, helmets, no_helmets, compliance = analyze_image(tmp_path)

            if i < 4:
                with preview_cols[i]:
                    st.image(img_result, caption=f.name, use_column_width=True)

            total = helmets + no_helmets  
            report.append({
                "Photo":          f.name,
                "Total ouvriers": helmets + no_helmets,
                "Avec casque":    helmets,
                "Sans casque":    no_helmets,
                "Conformité (%)": round(compliance, 1),
                "Statut":         "✅ Conforme" if no_helmets == 0 else "🚨 Non conforme"
            })

            os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files),
                               text=f"Analyse : {f.name} ({i+1}/{len(uploaded_files)})")

        progress.empty()
        st.markdown("---")
        st.subheader("📊 Rapport d'Audit")

        df = pd.DataFrame(report)
        st.dataframe(df, use_container_width=True)

        total_workers    = df["Total ouvriers"].sum()
        total_helmets    = df["Avec casque"].sum()
        total_no_helmet  = df["Sans casque"].sum()
        global_compliance = (total_helmets / total_workers * 100) if total_workers > 0 else 0

        st.markdown("---")
        st.subheader("📈 Statistiques Globales")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📸 Photos analysées",  len(uploaded_files))
        s2.metric("👷 Total ouvriers",     total_workers)
        s3.metric("✅ Avec casque",        total_helmets)
        s4.metric("📊 Conformité globale", f"{global_compliance:.1f}%")

        st.markdown("---")
        excel_path = os.path.join(tempfile.gettempdir(), "rapport_audit_epi.xlsx")
        df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button(
                label     = "📥 Télécharger le Rapport Excel",
                data      = f,
                file_name = "rapport_audit_epi.xlsx",
                mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type      = "primary"
            )