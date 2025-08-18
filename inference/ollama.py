import os
import sys
import ollama

# ========== Path Accessibility ==========
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Ollama client

def explain_prediction(results):
    top_anomalies = [(label, prob) for label, prob in results if prob >= 0.5]
    low_anomalies = [(label, prob) for label, prob in results if 0 < prob < 0.5]

    top_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in top_anomalies])
    low_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in low_anomalies])

    prompt = (
        f"We analyzed a retinal image and detected the following conditions with high confidence: {top_str}.\n"
        f"There are also signs of these conditions with lower probability: {low_str}.\n"
        f"Now, state what we suspect with the confidence rates then explain that it's likely something.\n"
        f"Explain in 3 simple sentences what these anomalies mean and why they are important for vision and health."
    )


    messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]

    response = ollama.chat(model="gemma:2b", messages=messages)


    return response["message"]["content"].strip().replace("*", "")

# ===== Test with random results =====
results = [
    ("Diabetic Retinopathy", 0.78),
    ("Glaucoma", 0.42),
    ("Macular Degeneration", 0.15),
    ("Normal", 0.05)
]

print(explain_prediction(results))
