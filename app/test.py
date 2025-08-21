import streamlit as st
import pandas as pd
from PIL import Image

# -------- Page Config --------
st.set_page_config(
    page_title="Retinal Report",
    layout="wide",
    page_icon="visualization/icon.png"
)

# ====== HEADER ======
st.markdown(
    """
    <style>
    .report-container {
        font-family: Arial, sans-serif;
        color: #333;
        line-height: 1.6;
        padding: 20px;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #0056b3;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 20px;
    }
    .divider {
        margin: 20px 0;
        border-bottom: 2px solid #ddd;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #0056b3;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 5px solid #007bff;
        padding-left: 10px;
    }
    .summary-box {
        background-color: #f0f8ff;
        border-left: 5px solid #007bff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .condition-list-item {
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 5px;
        font-size: 1em;
    }
    .high-conf {
        background-color: #e6f7ff;
        border-left: 4px solid #007bff;
    }
    .low-conf {
        background-color: #fffbe6;
        border-left: 4px solid #ffc107;
    }
    .interpretation-box {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 25px;
        border: 1px solid #ddd;
        font-style: italic;
    }
    .disclaimer {
        text-align: center;
        font-size: 0.8em;
        color: #888;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#st.markdown("<div class='report-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>AI-Powered Retinal Scan Report</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Comprehensive Analysis & Clinical Findings</p>", unsafe_allow_html=True)

# --- Check Data ---
if "report_data" not in st.session_state:
    st.error("⚠️ No report available. Please analyze an image first.")
    st.stop()

data = st.session_state["report_data"]
sorted_results = data.get("results", [])
explanation_text = data.get("explanation", "No explanation available.")

# Images passed from analysis page
original_img = st.session_state.get("original_img", None)
gradcam_img = st.session_state.get("gradcam_img", None)

# ====== IMAGE SECTION ======
if original_img and gradcam_img:
    st.markdown("### Retinal Scan & AI Heatmap")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div style="padding: 10px; background: #FDFEFE; border-radius: 12px; 
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color:#2E4053;">Original Image</h4>
            </div>
            """, unsafe_allow_html=True
        )
        st.image(original_img, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div style="padding: 10px; background: #FDFEFE; border-radius: 12px; 
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color:#2E4053;">Attention Heatmap (Grad-CAM)</h4>
            </div>
            """, unsafe_allow_html=True
        )
        st.image(gradcam_img, use_container_width=True)

st.markdown("---")

# ====== SUMMARY CARD ======
st.markdown("### Summary")
if sorted_results:
    top_condition, top_prob = sorted_results[0]
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #E8F6F3, #D6EAF8);
            color: #154360;
            font-size:16px;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;">
            <strong>Most Likely Condition:</strong> {top_condition}<br>
            <strong>Confidence:</strong> {top_prob:.1%}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("No conditions detected.")

# ====== CONFIDENCE CHART ======
st.markdown("### Confidence Levels")
if sorted_results:
    df_results = pd.DataFrame(sorted_results, columns=["Condition", "Confidence"])
    st.bar_chart(df_results.set_index("Condition"))
else:
    st.info("No data available to plot.")

st.markdown("---")

# ====== DETAILED FINDINGS ======
st.markdown("### Detailed Findings")
high_conf = [(c, p) for c, p in sorted_results if p >= 0.5]
low_conf = [(c, p) for c, p in sorted_results if 0.1 <= p < 0.5]

if high_conf:
    st.markdown("**High Confidence Conditions**")
    for cond, conf in high_conf:
        st.markdown(
            f"<div style='padding:12px; border-left: 6px solid #1ABC9C; "
            f"background:#EBF5FB; margin:6px 0; border-radius:6px; color: #154360;'>"
            f"<b>{cond}</b> — {conf:.2%}</div>",
            unsafe_allow_html=True
        )

if low_conf:
    st.markdown("**Other Possible Conditions**")
    for cond, conf in low_conf:
        st.markdown(
            f"<div style='padding:12px; border-left: 6px solid #F39C12; "
            f"background:#FEF5E7; margin:6px 0; border-radius:6px; color: #154360;'>"
            f"{cond} — {conf:.1%}</div>",
            unsafe_allow_html=True
        )

if not high_conf and not low_conf:
    st.info("No notable conditions detected.")

st.markdown("---")

# ====== INTERPRETATION ======
st.markdown("### Interpretation & Clinical Notes")
st.markdown(
    f"""
    <div style="padding: 20px; background: #FDFEFE; border-radius: 12px; 
    border:1px solid #E5E7E9; font-size:15px; line-height:1.6; 
    color: #2C3E50; box-shadow: 0px 3px 8px rgba(0,0,0,0.05);">
    {explanation_text}
    </div>
    """,
    unsafe_allow_html=True
)

# ====== FOOTER ======
st.markdown("---")
if st.button("Back to Analysis"):
    st.switch_page("../Analyze_Scan.py")

st.caption("Model & data © Blueray / Company")
