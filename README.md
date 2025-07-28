## Retinal Anomaly Detection
 
**Explainable AI for Retinal Disease Detection using Deep Learning**

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
- Explains its predictions in **plain English** using **GPT-4** as the interpreter

The aim is to make diagnosis faster, more transparent, and educational—for both patients and practitioners.

---

## Targeted Diseases
As of now, the diseases covered by the model are:
- Opacity
- Diabetic Retinopathy
- Glaucoma
- Macular Edema
- Macular Degeneration
- Retinal Vascular Occlusion
- Normal Eye

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
| Optional UI       | Web (H, C, J) or TKinter   |

---

## File Structure

```
Retinal-Disease-Detection/
├── chart/
    ├── __init__.py
    ├── glcm.py
    └── utils.py

├── data/
    ├── test/
        └── ...
    ├── train/
        ├──train/
           ├── img.jpg
           └── ...
        └── train.csv
    ├── data_analysis.md
    └── trials.md

├── data_visual/
    └── output/
        └── chart.png
    ├── data.png
    └── ...

├── models/
    ├── model_best.pth
    └── model_v1.pth

├── sub/
    ├── __init__.py
    ├── checker.py
    ├── explainer.py
    └── inference.py

├── trainOutputs/
    └── graph/
        ├── v1_loss_plot.png
        └── v2_loss_plot.png
    ├── v1.txt
    └── v2.txt

├── .gitignore 
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── train.py
└── Updates.md

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

We suspect with high confidence the presence of diabetic retinopathy (99.4%), macular edema (<1%), and opacity (87.3%) in the retinal image. These findings suggest that the patient is likely experiencing significant eye health issues. Diabetic retinopathy is a condition where high blood sugar levels damage blood vessels in the retina, potentially leading to vision loss. Macular edema involves swelling in the macula, the part of the retina responsible for sharp vision, which can result in blurred or distorted vision. Opacity, such as cataracts, causes clouding of the eye's lens, reducing clarity and brightness of vision. These conditions are crucial to address as they can lead to serious vision impairment or blindness if left untreated.

---

## Current Updates & Future Directions

-  Feedback loop for post-diagnosis correction and retraining 
-  Lightweight offline mode for clinics  

-> [View Updates](Updates.md)
Updates.md would contain & track all the experiments advancements and failures regarding the data analysis and model performance. Summarizing it in order to keep track over time.

---

---

## License  
**MIT License** – Free for use in research, education, and development.