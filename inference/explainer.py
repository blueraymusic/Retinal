from openai import OpenAI
import os
import sys

# ========== Path Accessibility ==========
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


client = OpenAI()

"""
    Transforms prediction results into a plain-language explanation using GPT-4o.
    High-confidence results are listed first, followed by low-confidence ones.
    
"""

def explain_prediction(results):
    top_anomalies = [(label, prob) for label, prob in results if prob >= 0.5]
    low_anomalies = [(label, prob) for label, prob in results if 0 < prob < 0.5]

    top_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in top_anomalies])
    low_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in low_anomalies])

    prompt = (
        f"We analyzed a retinal image and detected the following conditions with high confidence: {top_str}.\n"
        f"There are also signs of these conditions with lower probability: {low_str}.\n\n"
        f"The probability indicates how likely it is present in this patient. "
        f"Evidences present on the scan such as veins and etc...\n\n"
        f"Keep it short, maximum 10 sentences\n\n"
        f"Use bullet points for readability!\n\n"
        f"For example at the end says: The Grad-CAM heatmap highlights areas of concern around the optic disc and macula. Early intervention is recommended to prevent progression to proliferative diabetic retinopathy\n\n"
        f"No more than 6 bullets\n\n"
    )


    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
    return response.choices[0].message.content.strip()

"""
    except:
        
        Extra Precaution to ensure an explanation is provided
        - Note: As long as the API is valid this would not run

        from app import explain_prediction
        return str(explain_prediction(results, thresold=0.5))
"""