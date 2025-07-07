## Retinal Anomaly Detection
 
**Explainable AI for Retinal Disease Detection using Deep Learning and GPT-4**

---

## Problem Statement
Early diagnosis of retinal diseases—such as **diabetic retinopathy**, **glaucoma**, and **macular degeneration**—is vital to prevent irreversible vision loss. Traditional manual diagnosis:
- Is time-consuming  
- Requires skilled ophthalmologists and if not diagnosed early cand correctly, could lead to irreversible vision loss.
- Can suffer from inter-observer variability
  

This project builds an **automated and explainable AI system** to assist healthcare professionals in detecting retinal anomalies accurately and interpretably.

---

## My Objective
To build an end-to-end AI pipeline that:
- Detects multiple retinal diseases from fundus images  
- Explains its predictions in **plain English** using **GPT-4**

The aim is to make diagnosis faster, more transparent, and educational—for both patients and practitioners.

---

## Project Overview

### 1. Multi-Label Retinal Disease Classifier
- **Model**: Transferring learning from  `ResNet50` (changed the 4th layer to train on the data, so as to keep the other trained features)
- **Dataset**: Fundus images from Kaggle + other public sources  
- **Labels**: Multi-label output for 7 retinal conditions  


### 2. Inference & Upload Interface
- Upload a retinal image via a CLI or web interface  
- The model:
  - Verifies image validity  
  - Preprocesses and predicts diseases (With Percentages)
  - Displays top diseases with confidence scores  
  - Explains the results via **GPT-4 API**

### 3. Explainability
- **GPT-4**: Converts technical predictions into human-readable summaries  

---

## OpenAI Integration
- GPT-4 receives:
  - Predicted diseases and confidence scores  
- GPT-4 responds with:
  - A **natural language explanation** for medical users  

---

## Tech Stack

| Component         | Tools / Libraries          |
|-------------------|----------------------------|
| Model Training    | PyTorch, Torchvision       |
| Data Handling     | Numpy                      |
| NLP Explanation   | OpenAI GPT-4 API           |
| Inference Server  | Flask                      |
| Optional UI       | HTML, CSS, JavaScript      |

---

## File Structure

```
Retinal-Disease-Detection/
├── train.py           # Script to train the model
├── inference.py       # Run inference on new images
├── checker.py         # Validate uploaded image files
├── explainer.py       # Generate GPT-4 explanations for predictions
├── utils.py           # Common utilities (e.g., visualization)
├── main.py            # Entry point for the application
│
├── models/            # Directory to store trained model weights
├── data/              
│   ├── train.csv      # Label file mapping images to diseases
│   └── train/         # Retinal images used for training (.jpg files)
```

---

## Dataset

- **Source**:  
  [Kaggle: VietAI Retinal Disease Detection](https://www.kaggle.com/competitions/vietai-advance-course-retinal-disease-detection)  
- ~3,285 images:  
  - 3,210 abnormal  
  - 575 normal (Messidor + EyePACS)  

- **Diseases**:  
  - Opacity  
  - Diabetic Retinopathy  
  - Glaucoma  
  - Macular Edema  
  - Macular Degeneration  
  - Retinal Vascular Occlusion (RVO)  
  - Normal

**Example label structure**:

| filename          | opacity | dr | glaucoma | edema  | degeneration  | rvo | normal |
|------------------ |---------|----|----------|--------|---------------|-----|--------|
| `c24a1b14d253.jpg`|   0     | 0  |    0     |   0    |      0        |  1  |   0    |

> Data was also accessed via this repo (due to Kaggle auth issues):  
[zequeira/Retinal-Disease-Detection](https://github.com/zequeira/Retinal-Disease-Detection/tree/main/data)

---

## Sample Output

![0e5235b62de8](https://github.com/user-attachments/assets/d9fbc550-1b09-4509-89aa-f4c3de6303ad)



\`\`\`
Prediction:
- Diabetic Retinopathy: 99.4%
- Opacity: 87.3%
- Macular Degeneration: <1%
\`\`\`

Output: We suspect with high confidence the presence of diabetic retinopathy (100.0%), macular edema (99.4%), and opacity (87.3%) in the retinal image. These findings suggest that the patient is likely experiencing significant eye health issues. Diabetic retinopathy is a condition where high blood sugar levels damage blood vessels in the retina, potentially leading to vision loss. Macular edema involves swelling in the macula, the part of the retina responsible for sharp vision, which can result in blurred or distorted vision. Opacity, such as cataracts, causes clouding of the eye's lens, reducing clarity and brightness of vision. These conditions are crucial to address as they can lead to serious vision impairment or blindness if left untreated.

---

## Future Directions (Optional)

-  Feedback loop for post-diagnosis correction and retraining 
-  Lightweight offline mode for clinics  

---

---

## License  
**MIT License** – Free for use in research, education, and development.
