import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration for a centered layout and a title
st.set_page_config(page_title="Retinal Report", layout="centered")

# Set the main title of the page
st.title("Retinal Report")

# --- Sample Data (for demonstration) ---
# This dictionary simulates the data that would be passed to the report page.
# In a real app, this would be generated from the analysis of the image.
# We include this here to make the code runnable and demonstrate the new features.
sample_report_data = {
    "results": [
        ("Diabetic Retinopathy", 0.85),
        ("Glaucoma", 0.62),
        ("Cataracts", 0.25),
        ("Hypertensive Retinopathy", 0.15),
        ("Normal", 0.05)
    ],
    "explanation": (
        "This report provides an analysis of the retinal scan. "
        "The model has identified several possible conditions with "
        "varying levels of confidence based on the visual markers found. "
        "The most likely condition is Diabetic Retinopathy, with other "
        "possible conditions noted below. This is an automated assessment "
        "and should not be used as a substitute for professional medical advice."
    )
}

# Check if the report data exists in the session state.
# For this example, we'll use the sample data if the session state is empty.
if "report_data" not in st.session_state:
    # In a real app, you would want to `st.stop()` here.
    # For this demonstration, we'll load the sample data.
    st.session_state["report_data"] = sample_report_data

# Retrieve the data from the session state
data = st.session_state["report_data"]
sorted_results = data["results"]
explanation_text = data["explanation"]

# --- Top Summary ---
st.markdown("### Summary")
if sorted_results:
    top_condition, top_prob = sorted_results[0]
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
            color: #000;
            font-size:16px;
        ">
        <strong>Most Likely Condition:</strong> {top_condition} <br>
        <strong>Confidence:</strong> {top_prob:.1%}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("No conditions detected.")
st.markdown("---")

# --- Visual Summary with Chart ---
st.markdown("### Visual Summary")
if sorted_results:
    # Convert the list of tuples into a pandas DataFrame for easy plotting
    df_results = pd.DataFrame(sorted_results, columns=["Condition", "Confidence"])
    
    # Create a bar chart to visualize the confidence levels
    st.bar_chart(df_results.set_index("Condition"))
    st.caption("This chart shows the confidence level for each detected condition.")
else:
    st.info("No data available to plot.")

st.markdown("---")

# --- Detailed Findings ---
st.markdown("### Detailed Findings")
high_conf = [(c, p) for c, p in sorted_results if p >= 0.5]
low_conf = [(c, p) for c, p in sorted_results if 0.1 <= p < 0.5]

if high_conf:
    st.markdown("**High Confidence Conditions**")
    for cond, conf in high_conf:
        st.markdown(
            f"<div style='padding:10px; border-left: 4px solid #2E86C1; margin-bottom:5px;     color: rgb(255 255 255);'>"
            f"<strong>{cond}</strong> — {conf:.2%}</div>", 
            unsafe_allow_html=True
        )
else:
    st.info("No high-confidence conditions detected.")

if low_conf:
    st.markdown("")
    st.markdown("**Other Possible Conditions**")
    for cond, conf in low_conf:
        st.markdown(
            f"<div style='padding:10px; border-left: 4px solid #F39C12; margin-bottom:5px;  color: rgb(255 255 255);'>"
            f"{cond} — {conf:.1%}</div>", 
            unsafe_allow_html=True
        )
else:
    st.info("No low-confidence conditions detected.")

st.markdown("---")


# --- Interpretation ---
st.markdown("### Interpretation")
st.markdown(
        f"<div style='padding: 20px;background: rgb(255 255 255 / 0%);border-radius: 8px;font-family: monospace;color: rgb(255 255 255);''>"
        f"{explanation_text}</div>",
        unsafe_allow_html=True
    )
st.markdown("---")

# Back button
if st.button("To Analyse"):
    # In a real app, this would switch to the previous page.
    st.switch_page("Analyze_Scan.py")