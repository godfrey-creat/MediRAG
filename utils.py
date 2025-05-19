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