import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.corpus import wordnet
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ui_gui import display_interface  

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# ==== LOAD MODELS ====
embedder = SentenceTransformer('all-MiniLM-L6-v2')
gen_model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# ==== INDEX FILES ====
folder_path = 'data_files'
chunk_texts, chunk_sources, embeddings_list = [], [], []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            paragraphs = content.split("\n\n")
            for idx, para in enumerate(paragraphs):
                if para.strip():
                    embedding = embedder.encode(para)
                    embeddings_list.append(embedding)
                    chunk_texts.append(para)
                    chunk_sources.append(f"{filename} (section {idx+1})")

embedding_matrix = np.vstack(embeddings_list).astype('float32')
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print(f"Indexed {len(chunk_texts)} chunks in FAISS!")

# ==== SYNONYM EXPANSION ====
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def expand_query_with_synonyms(query):
    doc = nlp(query)
    important_words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
    synonym_queries = [query]
    for word in important_words:
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            if synonym.lower() != word.lower():
                new_query = query.replace(word, synonym)
                synonym_queries.append(new_query)
    return list(set(synonym_queries))

# ==== HANDLE QUERY ====
def handle_query(query):
    expanded_queries = expand_query_with_synonyms(query)
    all_query_embeddings = [embedder.encode(q) for q in expanded_queries]

    retrieved_chunks, retrieved_sources = [], []
    similarity_threshold = 0.35

    for query_embedding in all_query_embeddings:
        D, I = index.search(np.array([query_embedding]).astype('float32'), 10)
        max_distance = np.max(D[0]) if np.max(D[0]) != 0 else 1
        similarities = 1 - (D[0] / max_distance)
        for sim, idx in zip(similarities, I[0]):
            if sim >= similarity_threshold and chunk_texts[idx] not in retrieved_chunks:
                retrieved_chunks.append(chunk_texts[idx])
                retrieved_sources.append(chunk_sources[idx])

    # Build retrieved text
    if retrieved_chunks:
        retrieved_text = "\n\n".join(f"From {src}:\n{chunk}" for src, chunk in zip(retrieved_sources, retrieved_chunks))
    else:
        retrieved_text = " No relevant information found."

    # ==== BUILD PROMPT ====
    if retrieved_chunks:
        context = "\n".join(retrieved_chunks)
        lower_q = query.lower()
        if any(kw in lower_q for kw in ["advantage", "benefit", "pro"]):
            task_instruction = "List the advantages or benefits clearly around 100 words."
        elif any(kw in lower_q for kw in ["disadvantage", "drawback", "con"]):
            task_instruction = "List the disadvantages or drawbacks clearly around 100 words.."
        elif any(kw in lower_q for kw in ["describe", "detail", "explain", "summary", "explanation"]):
            task_instruction = "Write a descriptive and detailed answer around 100 words."
        else:
            task_instruction = "Provide a concise and accurate answer."
        prompt = (
            f"You are an AI assistant.\n"
            f"{task_instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"You are an AI assistant.\n"
            f"Unfortunately no relevant information was found.\n"
            f"Provide a polite message.\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(
        **inputs,
        max_new_tokens=700,
        num_beams=4,
        temperature=0.8,
        top_p=0.95,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return retrieved_text, generated_output

# ==== START GUI ====
display_interface(handle_query)