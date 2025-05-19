# MediRAG
# 🩺 MediRAG: A Multimodal RAG-Powered Diagnostic Assistant

## 🚀 Overview

**MediRAG** is a hackathon-ready project that leverages **Multimodal Retrieval-Augmented Generation (RAG)** to assist healthcare professionals in diagnosing patients using both clinical text and medical images like X-rays.

By integrating **vision-language models**, **vector search**, and **LLMs**, MediRAG demonstrates how AI can help synthesize patient records, reference images, and guidelines into a single, actionable diagnostic report.

---

## 🎯 Problem Statement

Doctors in under-resourced areas are burdened with interpreting diverse patient data — clinical notes, lab results, and imaging — without the support of specialists. This can lead to misdiagnosis or delayed treatment.

---

## 💡 Solution

MediRAG allows a clinician to:

1. Upload a short **clinical summary** (text).
2. Upload an **X-ray image**.
3. Receive a generated **diagnostic summary** with retrieved support references from medical literature and past annotated cases.

---

## 🧠 How It Works

### 1. Input Handling

* Upload patient symptoms and notes (text).
* Upload chest X-ray (image).

### 2. Multimodal Embedding

* Text encoded using **BioBERT** or **SentenceTransformers**.
* Image encoded using **CLIP** or **BiomedCLIP**.

### 3. Retrieval Phase

* Encoded input is queried against a **vector database (FAISS)** with pre-indexed documents/images.
* Top-k relevant entries are retrieved from:

  * Annotated case studies
  * WHO and NIH documentation
  * Open-source medical image databases

### 4. Generation Phase

* A **Large Language Model (LLM)** fuses the input and retrieved content.
* It generates:

  * Diagnosis suggestion
  * Confidence score
  * Explanation and reference links

---

## 🛠 Tech Stack

| Layer            | Tools Used                          |
| ---------------- | ----------------------------------- |
| Frontend         | Flask + Bootstrap Templates         |
| Backend          | Python (FastAPI or Flask)           |
| Text Embeddings  | Sentence-BERT / BioBERT             |
| Image Embeddings | OpenAI CLIP / BiomedCLIP            |
| Vector Store     | FAISS or Weaviate                   |
| LLM              | GPT-4 / Open Source (LLaMA/Mistral) |
| Hosting          | Render / Hugging Face Spaces        |

---

## 📦 Folder Structure

```bash
mediRAG/
├── app.py                  # Flask app
├── templates/
│   └── index.html          # Basic UI template
├── static/
│   └── style.css           # Optional styling
├── documents.json          # Sample text + metadata
├── xray_samples/           # Sample annotated X-rays
├── utils.py                # Embedding, retrieval helpers
├── README.md               # Project overview
```

---

## 🧪 Sample Use Case

**Input**:

* Text: "72-year-old male with persistent cough, weight loss, and fatigue."
* Image: Chest X-ray showing mass in left upper lobe.

**Output**:

* Diagnosis: "Likely pulmonary tuberculosis or malignancy"
* Confidence: 84%
* References:

  * WHO TB Guidelines 2023
  * Annotated X-ray Case #14 (tuberculosis)

---

## 🏃‍♂️ Hackathon Timeline

| Day   | Goal                                           |
| ----- | ---------------------------------------------- |
| Day 1 | Build UI, upload forms, prepare sample dataset |
| Day 2 | Implement embeddings + FAISS retrieval         |
| Day 3 | Add LLM logic for report generation            |
| Day 4 | UI polish, test cases, and pitch deck          |

---

## 📤 Future Improvements

* Add audio modality for symptom narration
* Integration with PACS systems for real-time hospital use
* Add more sophisticated fine-tuned biomedical LLMs

---

## 🤝 Contributors

* Lead Dev: Godfrey Otieno
* Model Integration: \[Your name]
* UI/UX Designer: \[Your name]
* Domain Advisor: \[Medical advisor name]

---

## 🧪 Prototype Scaffold Components

### 🔧 `app.py`

```python
from flask import Flask, render_template, request
from utils import get_text_embedding, get_image_embedding, retrieve_similar_cases, generate_diagnostic_report
import os

app = Flask(__name__)
UPLOAD_FOLDER = "xray_samples"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis, references = None, []
    if request.method == 'POST':
        text = request.form['symptoms']
        image = request.files['xray']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        diagnosis, references = generate_diagnostic_report(text, image_path)

    return render_template('index.html', diagnosis=diagnosis, references=references)

if __name__ == '__main__':
    app.run(debug=True)
```

### 🔧 `utils.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    return model.encode(text)

def get_image_embedding(image_path):
    return np.random.rand(384)  # Placeholder for CLIP-based image embedding

def load_documents():
    with open("documents.json") as f:
        return json.load(f)

def retrieve_similar_cases(text_embedding, image_embedding):
    documents = load_documents()
    results = []
    for doc in documents:
        doc_embedding = model.encode(doc['text'])
        score = np.dot(doc_embedding, text_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(text_embedding))
        results.append((score, doc))
    top = sorted(results, reverse=True, key=lambda x: x[0])[:3]
    return [doc for _, doc in top]

def generate_diagnostic_report(text, image_path):
    text_emb = get_text_embedding(text)
    img_emb = get_image_embedding(image_path)
    references = retrieve_similar_cases(text_emb, img_emb)
    diagnosis = "Likely diagnosis based on input and retrieved references."
    return diagnosis, references
```

### 🌐 `templates/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>MediRAG Diagnostic Tool</title>
</head>
<body>
    <h1>MediRAG - Diagnostic Assistant</h1>
    <form method="POST" enctype="multipart/form-data">
        <label>Enter Symptoms/Clinical Notes:</label><br>
        <textarea name="symptoms" rows="4" cols="50"></textarea><br><br>
        <label>Upload Chest X-ray:</label>
        <input type="file" name="xray"><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if diagnosis %}
        <h3>Diagnosis Suggestion:</h3>
        <p>{{ diagnosis }}</p>
        <h4>References:</h4>
        <ul>
        {% for ref in references %}
            <li><a href="{{ ref.url }}" target="_blank">{{ ref.title }}</a>: {{ ref.text }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
```

### 📄 `documents.json`

```json
[
  {
    "title": "WHO TB Guidelines 2023",
    "url": "https://www.who.int/tb/publications",
    "text": "Tuberculosis typically affects the lungs and presents with persistent cough and weight loss."
  },
  {
    "title": "Annotated X-ray Case #14",
    "url": "https://openmed.ai/cases/14",
    "text": "X-ray image of TB shows cavitary lesions in upper lobe."
  },
  {
    "title": "NIH Lung Cancer Reference",
    "url": "https://www.cancer.gov/types/lung",
    "text": "Persistent cough and fatigue in older adults may suggest malignancy in upper lung region."
  }
]
```
