"""Smart Supermarket web app.

Lab 4 equivalent: the ModelArts Real-Time Service "Prediction" page.
Lab 3 equivalent: the OCR General Table app served on port 8501.
Run: streamlit run app.py   ->  http://localhost:8501
"""
import io
import os

import pandas as pd
import streamlit as st
from PIL import Image

from src import ocr
from src.model import GroceryRecognizer

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
DOCS = os.path.join(ROOT, "docs")

st.set_page_config(page_title="Smart Supermarket", page_icon="🛒", layout="wide")

# Streamlit Cloud secrets -> environment variables, so src/ocr.py works the same everywhere
try:
    for k in ocr.KEYS:
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = st.secrets[k]
except Exception:
    pass


@st.cache_resource(show_spinner="Loading the model (first run downloads MobileNetV2, about 14 MB)...")
def load_model():
    return GroceryRecognizer()


st.title("🛒 Smart Supermarket")
st.caption("Grocery product recognition and table OCR · HCCDA-AI project applying Huawei Cloud Labs 1–4")

with st.sidebar:
    st.header("Lab steps, applied here")
    st.markdown("""
| Huawei lab step | In this app |
|---|---|
| OBS bucket | `models/`, `samples/` folders |
| ModelArts notebook | `notebooks/` (Jupyter or Colab) |
| Model Management | `models/grocery_classifier.joblib` |
| Real-Time Service | **Prediction** tab |
| OCR General Table | **OCR** tab |
| Port 8501 rule | Streamlit runs on 8501 |
""")
    st.markdown("**Model:** MobileNetV2 + logistic regression, 81 products")
    st.markdown("**Test accuracy:** 75.6% exact · 85.5% type · 97.0% top-5")

tab_pred, tab_ocr, tab_data = st.tabs(["Prediction", "OCR General Table", "Dataset insights"])

# ---------------- Lab 4, Task 5: upload a test image for prediction ----------------
with tab_pred:
    st.subheader("Upload a product photo")
    left, right = st.columns([1, 1])
    with left:
        upload = st.file_uploader("Photo of a grocery product (JPG or PNG)", type=["jpg", "jpeg", "png"])
        sample_files = sorted(f for f in os.listdir(SAMPLES) if f.endswith(".jpg"))
        choice = st.selectbox("...or use a test photo", ["(none)"] + sample_files)
        image = None
        if upload is not None:
            image = Image.open(upload)
        elif choice != "(none)":
            image = Image.open(os.path.join(SAMPLES, choice))
        if image is not None:
            st.image(image, width="stretch")
    with right:
        if image is None:
            st.info("Upload a photo or pick a test photo to see the prediction.")
        else:
            results, ms = load_model().predict(image)
            best = results[0]
            st.metric("Prediction", best["product"], f"{best['confidence']:.0%} confidence", delta_color="off")
            st.caption(f"Answered in {ms:.0f} ms")
            df = pd.DataFrame(results)
            df["confidence"] = (df.confidence * 100).round(1)
            st.bar_chart(df.set_index("product")["confidence"], horizontal=True, y_label="confidence %")
            with st.expander("Response JSON (same idea as the ModelArts service response)"):
                st.json({"predicted_label": best["product"], "scores": results})
    st.caption("The model knows the 81 products in the Grocery Store Dataset (fruit, vegetables, juices, milks, yoghurts). Other products get the closest match.")

# ---------------- Lab 3: OCR General Table ----------------
with tab_ocr:
    st.subheader("Read a table from an image")
    table_up = st.file_uploader("Table image (PNG, JPG, BMP, TIFF)", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="ocr")
    default_table = os.path.join(DOCS, "results_table.png")
    if table_up is not None:
        img_bytes = table_up.getvalue()
    else:
        with open(default_table, "rb") as f:
            img_bytes = f.read()
        st.caption("Using the project's own results table. Upload your own above.")
    st.image(img_bytes, width=480)

    missing = ocr.missing_keys()
    if missing:
        st.warning("Huawei OCR keys aren't set, so this tab can't call the API. Add "
                   + ", ".join(f"`{k}`" for k in missing)
                   + " as environment variables (or Streamlit secrets), and subscribe to General Table in AP-Singapore.")
    elif st.button("Recognize table", type="primary"):
        with st.spinner("Calling Huawei OCR General Table..."):
            try:
                result = ocr.recognize_general_table(img_bytes)
            except Exception as e:
                st.error(f"OCR call failed: {e}")
                result = None
        if result:
            table = ocr.to_dataframe(result)
            st.dataframe(table, width="stretch")
            xlsx = ocr.excel_bytes(result)
            if xlsx:
                st.download_button("Download as Excel", xlsx, "ocr_table.xlsx")

# ---------------- Labs 1 and 2: dataset analysis ----------------
with tab_data:
    st.subheader("What the model learned from")
    c1, c2, c3 = st.columns(3)
    c1.metric("Photos", "5,421")
    c2.metric("Products", "81")
    c3.metric("Test photos", "2,485")
    st.image(os.path.join(DOCS, "eda.png"), width="stretch")
    st.markdown("**Sample predictions on unseen test photos**")
    st.image(os.path.join(DOCS, "predictions.png"), width="stretch")
    st.markdown("""
**Where it struggles:** look-alike products (Satsumas vs Orange, vanilla vs blueberry yoghurt), and the
7 products with fewer than 20 training photos (65.3% accuracy vs 76.1% for the rest).
Full analysis in `notebooks/Smart_Supermarket_HCCDA_AI_Project.ipynb`.

Dataset: Klasson et al., *A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels*, WACV 2019 (MIT license).
""")
