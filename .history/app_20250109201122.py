from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# Inisialisasi Flask
app = Flask(__name__)

# Inisialisasi perangkat dan model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# Load FAISS index dan data
index = faiss.read_index("index_file.ivf")
data = pd.read_csv("index.csv")

# Fungsi encode kalimat
def encode_sentences(sentences, model, tokenizer, device, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="encoding sentences", unit=" batch"):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

@app.route("/")
def home():
    return "FAISS API is running!"

@app.route("/search", methods=["POST"])
def search():
    try:
        # Ambil data input dari request
        data_input = request.json
        query_sentences = data_input.get("query", [])
        k = data_input.get("top_k", 10)
        
        if not query_sentences:
            return jsonify({"error": "Query sentences are required"}), 400
        
        # Encode query
        query_embeddings = encode_sentences(query_sentences, model, tokenizer, device, batch_size=1)
        query_embeddings_normalized = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Cari di FAISS index
        distances, indices = index.search(query_embeddings_normalized.astype('float32'), k)
        
        # Ambil hasil pencarian
        results = []
        for i, idx_list in enumerate(indices):
            for rank, idx in enumerate(idx_list):
                idx = int(idx)
                if idx < len(data):
                    results.append({
                        "rank": rank + 1,
                        "sentence": data.loc[idx, 'summary'],
                        "similarity": float(distances[i][rank])
                    })
        
        return jsonify({"query": query_sentences, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
