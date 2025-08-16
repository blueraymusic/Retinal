import os
import streamlit as st
import gdown

DOWNLOAD_DIR = "models"
MODELS = {
    "v4_less_strict.pth": "1OkuZxACzkVlLVE3hKWAO78GELk72j-Xo",
    "v4.pth": "1V5yGuEqpqlQI6gZkfz_nkEjT1G4hZUUb",
    "v4_d_changed.pth": "192nnSzP7TNufsbW2PVSr46a7TuUG_Hgr",
    "v4_best.pth": "1-Cx3QxZLayBlEQZzb5AEPBY2SENlTIp9",
}

def download_model(file_name, file_id):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    output_path = os.path.join(DOWNLOAD_DIR, file_name)
    st.info(f"Downloading {file_name}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    st.success(f"Saved {file_name} to {DOWNLOAD_DIR}")

def check_models():
    """Check if any models exist. If not, show download UI."""
    models_exist = os.path.exists(DOWNLOAD_DIR) and any(f.endswith(".pth") for f in os.listdir(DOWNLOAD_DIR))

    if not models_exist:
        st.set_page_config(page_title="Download Models", layout="centered")
        st.title("No Models Found")
        st.warning("There are currently no model files in the 'models/' directory.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Default Model"):
                download_model("v4_best.pth", MODELS["v4_best.pth"])
                st.rerun()
        with col2:
            if st.button("Download All Models"):
                for name, fid in MODELS.items():
                    download_model(name, fid)
                st.rerun()
        st.stop()  # Stop the app until models are available
