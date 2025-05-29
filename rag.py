import sys
import os
import numpy as np
import faiss
import fitz
import nltk
import spacy
from docx import Document
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, set_seed
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import re
import shelve
import hashlib
from pathlib import Path
import traceback # Import traceback for detailed error printing

from ui_gui import display_interface

BASE_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PROJECT_PATH, "models")
NLTK_DATA_DIR = os.path.join(BASE_PROJECT_PATH, "nltk_data")
CACHE_DIR = os.path.join(BASE_PROJECT_PATH, ".cache") # Cache directory

ST_MODEL_PATH = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
ONNX_FLAN_T5_DIR_PATH = os.path.join(MODELS_DIR, "flan-t5-base_onnx")
SPACY_MODEL_PATH = os.path.join(MODELS_DIR, "en_core_web_sm")

# Create cache directory if it doesn't exist
try:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Ensured cache directory exists at: {CACHE_DIR}")
except Exception as e:
    print(f"FATAL ERROR: Could not create cache directory at {CACHE_DIR}. Error: {e}")
    sys.exit("Cache directory creation failed.")


EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache")
LLM_CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache")


set_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

NLTK_DATA_CONFIGURED = False
SPACY_MODEL_LOADED = False
EMBEDDING_MODEL_LOADED = False
LLM_LOADED = False
INDEX_BUILT = False

nlp = None
embedder = None
tokenizer = None
model = None

def configure_nltk_data():
    global NLTK_DATA_CONFIGURED
    if NLTK_DATA_CONFIGURED: return
    if not os.path.exists(NLTK_DATA_DIR):
        print(f"FATAL ERROR: NLTK data directory not found at {NLTK_DATA_DIR}")
        sys.exit("NLTK data missing. Run download_assets.py first.")
    if NLTK_DATA_DIR not in nltk.data.path: nltk.data.path.insert(0, NLTK_DATA_DIR)
    try: wordnet.ensure_loaded()
    except LookupError:
        print(f"FATAL ERROR: NLTK WordNet not found in {NLTK_DATA_DIR}")
        sys.exit("WordNet data missing. Run download_assets.py and check nltk_data folder.")
    NLTK_DATA_CONFIGURED = True
    print("NLTK data path configured.")

def load_spacy_model():
    global SPACY_MODEL_LOADED, nlp
    if SPACY_MODEL_LOADED and nlp is not None: return nlp
    if not os.path.exists(SPACY_MODEL_PATH):
        print(f"FATAL ERROR: spaCy model not found at {SPACY_MODEL_PATH}")
        sys.exit("spaCy model missing. Run download_assets.py first.")
    print(f"Loading spaCy model from {SPACY_MODEL_PATH}...")
    try: nlp = spacy.load(SPACY_MODEL_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Could not load spaCy model. Error: {e}")
        sys.exit("spaCy model loading failed.")
    SPACY_MODEL_LOADED = True
    print("SpaCy model loaded.")
    return nlp

def load_models():
    global EMBEDDING_MODEL_LOADED, LLM_LOADED, embedder, tokenizer, model
    if not EMBEDDING_MODEL_LOADED or embedder is None:
        if not os.path.exists(ST_MODEL_PATH):
            print(f"FATAL ERROR: SentenceTransformer model not found at {ST_MODEL_PATH}")
            sys.exit("SentenceTransformer model missing. Run download_assets.py first.")
        print(f"Loading embedding model from {ST_MODEL_PATH}...")
        embedder = SentenceTransformer(ST_MODEL_PATH)
        EMBEDDING_MODEL_LOADED = True
        print("Embedding model loaded.")
    if not LLM_LOADED or tokenizer is None or model is None:
        if not os.path.exists(ONNX_FLAN_T5_DIR_PATH):
            print(f"FATAL ERROR: Flan-T5 ONNX model directory not found at {ONNX_FLAN_T5_DIR_PATH}")
            sys.exit("Flan-T5 ONNX model directory missing. Run download_assets.py first.")
        print(f"Loading ONNX language model and tokenizer from {ONNX_FLAN_T5_DIR_PATH}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(ONNX_FLAN_T5_DIR_PATH, local_files_only=True)
            provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            model = ORTModelForSeq2SeqLM.from_pretrained(ONNX_FLAN_T5_DIR_PATH,
                                                         local_files_only=True,
                                                         provider=provider)
            print(f"ONNX Language model loaded using {provider}.")
        except Exception as e:
            print(f"FATAL ERROR: Could not load Flan-T5 ONNX model/tokenizer. Error: {e}")
            sys.exit("Flan-T5 ONNX loading failed.")
        LLM_LOADED = True
    return embedder, tokenizer, model

def read_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()

def read_docx_paragraphs(path): # Returns list of paragraphs
    doc = Document(path)
    paragraphs_text = []
    current_mcq_block = ""
    for para in doc.paragraphs:
        p_text = para.text.strip()
        if not p_text: continue

        if re.match(r"^\d+\.\s+", p_text) or \
           re.match(r"^[A-Z]\.\s+", p_text, re.IGNORECASE) or \
           re.match(r"^Answer:\s*", p_text, re.IGNORECASE):
            current_mcq_block += p_text + "\n"
        else:
            if current_mcq_block:
                paragraphs_text.append(current_mcq_block.strip())
                current_mcq_block = ""
            paragraphs_text.append(p_text)

    if current_mcq_block:
        paragraphs_text.append(current_mcq_block.strip())
    return paragraphs_text


def read_pdf(path): # Returns list of paragraphs (pages split by \n\n, then paragraphs by \n\n)
    full_text = ""
    try:
        doc = fitz.open(path)
        for page in doc: full_text += page.get_text("text", sort=True) + "\n\n"
        doc.close()
    except Exception as e: print(f"Error reading PDF {path}: {e}"); return []
    return full_text.strip().split("\n\n")


chunk_texts_global, chunk_sources_global = [], []
index_global = None

def build_index(folder_path='data_files'):
    global chunk_texts_global, chunk_sources_global, index_global, INDEX_BUILT
    if INDEX_BUILT: return
    configure_nltk_data(); load_spacy_model(); current_embedder, _, _ = load_models()
    temp_chunks, temp_srcs, temp_embeds = [], [], []
    if not os.path.exists(folder_path): print(f"Error: Folder '{folder_path}' not found."); return
    if not os.listdir(folder_path): print(f"Warning: Folder '{folder_path}' is empty."); return
    print(f"Starting to index files from '{folder_path}'...")

    embeddings_cache = None
    try:
        print(f"Attempting to open embeddings cache at: {EMBEDDINGS_CACHE_PATH}")
        embeddings_cache = shelve.open(EMBEDDINGS_CACHE_PATH)
        print(f"Successfully opened embeddings cache.")
    except Exception as e:
        print(f"WARNING: Could not open embeddings cache at {EMBEDDINGS_CACHE_PATH}. Embeddings will not be cached.")
        print(f"Reason: {type(e).__name__}: {e}")
        # print(traceback.format_exc()) # Uncomment for full traceback if needed

    for filename in os.listdir(folder_path):
        f_path = os.path.join(folder_path, filename); paragraphs_or_mcq_blocks = []
        print(f"Processing '{filename}'...")
        try:
            file_mtime = str(os.path.getmtime(f_path))

            if filename.lower().endswith('.txt'):
                content = read_txt(f_path)
                lines = content.splitlines()
                current_block = ""
                for line in lines:
                    stripped_line = line.strip()
                    if not stripped_line:
                        if current_block: paragraphs_or_mcq_blocks.append(current_block.strip())
                        current_block = ""
                    else:
                        current_block += stripped_line + "\n"
                if current_block: paragraphs_or_mcq_blocks.append(current_block.strip())

            elif filename.lower().endswith('.pdf'):
                paragraphs_or_mcq_blocks = read_pdf(f_path)
            elif filename.lower().endswith('.docx'):
                paragraphs_or_mcq_blocks = read_docx_paragraphs(f_path)
            else: print(f"Skipping unsupported file type: {filename}"); continue

            for i, block_text in enumerate(paragraphs_or_mcq_blocks):
                block_strip = block_text.strip()
                is_mcq_like = re.search(r"^\d+\.\s*|Answer:\s*|^[A-Z]\.\s*", block_strip, re.IGNORECASE | re.MULTILINE)
                num_words = len(block_strip.split())

                if (is_mcq_like and num_words > 2 and num_words < 400) or \
                   (not is_mcq_like and 10 < num_words < 400):
                    embedding = None
                    if embeddings_cache is not None:
                        block_hash = hashlib.md5(block_strip.encode('utf-8')).hexdigest()
                        cache_key = f"{filename}_{file_mtime}_{block_hash}"
                        if cache_key in embeddings_cache:
                            embedding = embeddings_cache[cache_key]
                            # print(f"Loaded embedding from cache for a block in '{filename}'") 

                    if embedding is None: 
                        embedding = current_embedder.encode(block_strip)
                        if embeddings_cache is not None:
                            try:
                                embeddings_cache[cache_key] = embedding
                                # print(f"Stored new embedding in cache for a block in '{filename}'") 
                            except Exception as e_store:
                                print(f"WARNING: Failed to store embedding in cache for {filename}. Key: {cache_key}")
                                print(f"Reason: {type(e_store).__name__}: {e_store}")
                    
                    temp_embeds.append(embedding)
                    temp_chunks.append(block_strip)
                    temp_srcs.append(f"{filename} (block {i+1})")
        except Exception as e: print(f"Error processing file {filename}: {e}")

    if embeddings_cache is not None:
        try:
            embeddings_cache.close()
            print(f"Successfully closed embeddings cache.")
        except Exception as e:
            print(f"WARNING: Error closing embeddings cache.")
            print(f"Reason: {type(e).__name__}: {e}")
            # print(traceback.format_exc())


    if not temp_embeds: print("No text chunks extracted."); return
    embed_matrix = np.vstack(temp_embeds).astype('float32')
    index_global = faiss.IndexFlatL2(embed_matrix.shape[1])
    index_global.add(embed_matrix)
    chunk_texts_global, chunk_sources_global = temp_chunks, temp_srcs
    INDEX_BUILT = True
    print(f"✅ Indexed {len(chunk_texts_global)} chunks in FAISS.")


def classify_question_type(query_text):
    q_lower = query_text.lower().strip().rstrip("?")
    if re.search(r"\b(advantages|benefits|pros)\b", q_lower) and not re.search(r"\b(disadvantages|cons|drawbacks)\b", q_lower): return "advantages"
    if re.search(r"\b(disadvantages|cons|drawbacks|limitations|issues)\b", q_lower) and not re.search(r"\b(advantages|benefits|pros)\b", q_lower): return "disadvantages"
    desc_patterns = [r"^(explain|describe|discuss|elaborate on)\b", r"^(tell me about|what about|how about)\b",
                     r"^(what are the characteristics of|what is the purpose of|what is the impact of)\b",
                     r"^(how does|why is|why does)\b", r"\b(compare|contrast)\b"]
    if any(re.search(p, q_lower) for p in desc_patterns): return "descriptive"
    if q_lower.endswith(" and why?"): return "descriptive"
    fact_attr_q = re.compile(r"^(what|who|where|when)\s+(is|are|was|were)(\s+the)?\s+([\w\s\-\/\.,#]+?)\s+(of|for|in|at|on)\s+([\w\s\-\/\.,#\d\(\)]+)", re.IGNORECASE)
    fact_starters = ["what is the certificate number", "what is the version", "what is the date", "what is the year", "what is the capital",
                     "what is the population", "what is the name of", "what is the id", "what is the code", "when was", "who was", "where is",
                     "list the", "how many", "how much", "what is the price of"]
    short_what_is = re.compile(r"^(what|who)\s+(is|are)\s+([\w\s\-\d\(\).]{1,35})$", re.IGNORECASE)
    if fact_attr_q.match(q_lower):
        attr_part = fact_attr_q.match(q_lower).group(4).strip()
        short_attrs = ["number", "date", "year", "name", "id", "code", "version", "capital", "population", "president", "author",
                       "director", "location", "address", "phone", "email", "price", "cost", "model", "serial", "size", "length",
                       "width", "height", "weight"]
        if attr_part in short_attrs or any(sa in attr_part for sa in short_attrs) or len(attr_part.split()) <= 2: return "factual"
    if any(q_lower.startswith(s) for s in fact_starters): return "factual"
    if short_what_is.match(q_lower):
        entity = short_what_is.match(q_lower).group(3).strip()
        if not any(term in entity for term in ["meaning", "definition", "concept", "idea", "story", "history"]): return "factual"
    if q_lower.startswith("list ") and not any(re.search(p, q_lower) for p in desc_patterns): return "factual"
    return "descriptive"

def get_synonyms(word, current_nlp_model):
    syns = set(); w_lower = word.lower()
    try:
        for syn_set in wordnet.synsets(w_lower):
            for lemma in syn_set.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name == w_lower: continue
                if w_lower == "headphones" and name == "phone": continue
                if w_lower == "earphones" and name == "phone": continue
                if len(name) <= 3 and len(w_lower) > len(name) + 2: continue
                syns.add(name)
    except LookupError: print("WordNet resource not found for synonyms.")
    except Exception as e: print(f"Error in get_synonyms for '{word}': {e}")
    return list(syns)

def expand_query_with_synonyms(query, current_nlp_model):
    doc = current_nlp_model(query)
    important_words = [{'original_text': t.text, 'lemma': t.lemma_.lower()} for t in doc if t.pos_ in ["NOUN","PROPN","VERB","ADJ"] and not t.is_stop]
    if not important_words: return [query]
    expanded_qs = {query}; MAX_NEW_QS = 1
    for wd in important_words:
        if len(expanded_qs) >= MAX_NEW_QS + 1: break
        for syn in get_synonyms(wd['lemma'], current_nlp_model)[:1]:
            if syn != wd['lemma'] and syn != wd['original_text'].lower():
                try:
                    new_q = re.sub(r'\b'+re.escape(wd['original_text'])+r'\b', syn, query, 1, re.IGNORECASE)
                    if new_q != query and new_q not in expanded_qs:
                        expanded_qs.add(new_q)
                        if len(expanded_qs) >= MAX_NEW_QS + 1: break
                except re.error as e: print(f"Regex error in synonym expansion: {e}")
        if len(expanded_qs) >= MAX_NEW_QS + 1: break
    return list(expanded_qs)

def handle_query(query_text):
    if not INDEX_BUILT or index_global is None or not chunk_texts_global:
        return ([], "Index not ready. Check 'data_files' and restart.")

    current_embedder, current_tokenizer, current_llm_model = load_models()
    current_nlp_model = load_spacy_model()
    configure_nltk_data()

    expanded_queries = expand_query_with_synonyms(query_text, current_nlp_model)
    all_query_embeddings = [current_embedder.encode(q) for q in expanded_queries]

    retrieved_items = []
    num_candidates_per_query = 15
    for i, query_embedding in enumerate(all_query_embeddings):
        try:
            distances, indices = index_global.search(np.array([query_embedding]).astype('float32'), num_candidates_per_query)
        except Exception as e: print(f"FAISS search error: {e}"); continue
        if distances.size > 0 and distances[0].size > 0:
            min_d, max_d = np.min(distances[0]), np.max(distances[0])
            sim_range = max_d - min_d
            sims = 1-((distances[0]-min_d)/(sim_range if sim_range > 1e-9 else 1e-9)) if sim_range > 1e-9 else (np.ones_like(distances[0]) if min_d < 1e-9 else np.zeros_like(distances[0]))
            for k_idx, (sim, doc_idx) in enumerate(zip(sims, indices[0])):
                if 0 <= doc_idx < len(chunk_texts_global):
                    retrieved_items.append((sim, chunk_texts_global[doc_idx], chunk_sources_global[doc_idx]))

    retrieved_items.sort(key=lambda x: x[0], reverse=True)

    question_type = classify_question_type(query_text)
    print(f"DEBUG: Query '{query_text}' classified as: {question_type}")

    query_doc = current_nlp_model(query_text)
    main_subjects_from_query = [t.lemma_.lower() for t in query_doc if t.pos_ in ["NOUN","PROPN"] and not t.is_stop and len(t.lemma_)>2]
    if not main_subjects_from_query and len(query_doc)>0: main_subjects_from_query = [t.lemma_.lower() for t in query_doc if not t.is_stop and len(t.lemma_)>2]
    if not main_subjects_from_query and query_text: main_subjects_from_query = [w.lower() for w in query_text.lower().split() if len(w)>2]

    key_entity_for_strict_filtering = None
    if question_type == "descriptive" or question_type == "factual":
        q_lower_entity = query_text.lower()
        entity_patterns = [r"(?:tell me about|what is|what's|describe|explain)\s+([\w\s\-\d\(\).]{2,40})$",
                           r"(?:advantages of|disadvantages of)\s+([\w\s\-\d\(\).]{2,40})$"]
        for p in entity_patterns:
            match = re.search(p, q_lower_entity)
            if match:
                pot_entity = match.group(1).strip().rstrip("?.!")
                if pot_entity not in ["it","this","that","the topic","the subject","something","anything","information"]:
                    key_entity_for_strict_filtering = pot_entity; break
    if key_entity_for_strict_filtering: print(f"DEBUG: Extracted key entity: '{key_entity_for_strict_filtering}'")

    final_chunks_for_llm, display_items_structured = [], []
    seen_display_texts = set()
    LLM_CONTEXT_SIZE = 4; GUI_DISPLAY_ITEMS = LLM_CONTEXT_SIZE + 3; SIM_THRESHOLD = 0.35

    data_folder_for_opening = os.path.join(BASE_PROJECT_PATH, "data_files")

    for sim, text, source_str in retrieved_items:
        if len(display_items_structured) < GUI_DISPLAY_ITEMS and text not in seen_display_texts and sim >= SIM_THRESHOLD / 2:
            actual_filename = source_str.split(" (block")[0]
            file_path_to_open = os.path.join(data_folder_for_opening, actual_filename)
            display_items_structured.append({'text':text,'source_display':source_str,'similarity':sim, 'full_file_path': file_path_to_open})
            seen_display_texts.add(text)

    temp_llm_candidates = []
    adv_dis_kws = []
    if question_type == "advantages": adv_dis_kws = ["advantage","benefit","pro","positive","good","strength","ideal","useful"]
    elif question_type == "disadvantages": adv_dis_kws = ["disadvantage","drawback","con","limitation","issue","problem","negative","weakness","prone to"]

    for sim, text, src in retrieved_items:
        if sim >= SIM_THRESHOLD:
            txt_l = text.lower(); subj_match = False
            if key_entity_for_strict_filtering: subj_match = bool(re.search(r'\b'+re.escape(key_entity_for_strict_filtering)+r'\b',txt_l,re.IGNORECASE))
            elif main_subjects_from_query: subj_match = any(s in txt_l for s in main_subjects_from_query)
            else: subj_match = True
            if subj_match:
                aspect_match = not adv_dis_kws or any(kw in txt_l for kw in adv_dis_kws)
                if aspect_match and text not in [c[0] for c in temp_llm_candidates]:
                    temp_llm_candidates.append((text, sim))

    temp_llm_candidates.sort(key=lambda x:x[1], reverse=True)
    final_chunks_for_llm = [txt for txt,sim in temp_llm_candidates[:LLM_CONTEXT_SIZE]]

    if not final_chunks_for_llm and display_items_structured:
        print(f"DEBUG: Strict context filtering yielded 0. Using top display items for LLM.")
        final_chunks_for_llm = [item['text'] for item in display_items_structured[:LLM_CONTEXT_SIZE]]

    if not final_chunks_for_llm:
        return display_items_structured, "Could not find relevant information to form an answer."

    llm_context = "\n\n---\n\n".join(final_chunks_for_llm)

    llm_cache = None
    llm_cache_key = None # Initialize llm_cache_key
    try:
        print(f"Attempting to open LLM cache at: {LLM_CACHE_PATH}")
        llm_cache = shelve.open(LLM_CACHE_PATH)
        print(f"Successfully opened LLM cache.")
        
        context_hash = hashlib.md5(llm_context.encode('utf-8')).hexdigest()
        llm_cache_key = f"{query_text}_{question_type}_{context_hash}" # Assign key here

        if llm_cache_key in llm_cache:
            cached_answer = llm_cache[llm_cache_key]
            print(f"Retrieved answer from LLM cache for query: '{query_text}'")
            llm_cache.close()
            print(f"Successfully closed LLM cache after retrieval.")
            return display_items_structured, cached_answer
    except Exception as e:
        print(f"WARNING: Could not open or access LLM cache at {LLM_CACHE_PATH}. LLM responses will not be cached for this query.")
        print(f"Reason: {type(e).__name__}: {e}")
        # print(traceback.format_exc())
        if llm_cache is not None: # If it was opened but failed during key access/check
            try:
                llm_cache.close()
            except Exception as e_close:
                print(f"WARNING: Error closing LLM cache after an issue. Reason: {type(e_close).__name__}: {e_close}")
        llm_cache = None # Ensure cache is None if there was an issue

    # --- PROMPT DEFINITIONS AND FORMATTING ---
    prompt_args = {'context':llm_context, 'query':query_text}
    current_prompt_str = ""
    gen_params = {'max_new_tokens':500, 'length_penalty':2.8, 'early_stopping':False, 'num_beams':4}

    if question_type == "factual":
        current_prompt_str = (
            "You are a precise factual answering assistant. Your task is to answer the 'Question' based *only* on the provided 'Context'.\n"
            "The 'Context' may contain multiple-choice questions (MCQs) with questions, options (e.g., A., B., C., D.), and a specific answer line (e.g., 'Answer: B. Some Answer Text' or 'Answer: Some Answer Text').\n\n"
            "Follow these instructions EXACTLY:\n"
            "1. Carefully match the user's 'Question' to a question within the 'Context'.\n"
            "2. Once a matching question is found in the 'Context', locate the line that explicitly states the answer, usually starting with 'Answer:' or being the option letter followed by the answer text.\n"
            "3. Extract ONLY the text of the answer itself (e.g., if 'Answer: B. Paris', your response should be 'Paris' or 'B. Paris'. If the answer is just 'Paris' after an option letter, use 'Paris').\n"
            "4. DO NOT include the question number, the options list (A, B, C, D labels unless part of the answer text), or any part of the question itself in your final answer.\n"
            "5. DO NOT add any conversational filler, explanations, or extra sentences. Provide only the precise answer text found.\n"
            "6. If the 'Context' does not contain a clear 'Answer:' line or a clear option choice for the matched question, or if the question cannot be matched, state 'The answer to this question was not found in the provided documents.'\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nPrecise Answer (extracting only the answer text):")
        gen_params.update({'max_new_tokens':50, 'length_penalty':0.7, 'early_stopping':True, 'num_beams':2})

    elif question_type == "advantages":
        prompt_args['key_entity'] = key_entity_for_strict_filtering if key_entity_for_strict_filtering else "the main subject"
        current_prompt_str = (
            "You are an intelligent assistant. Based *only* on the provided 'Context', your task is to identify and list the advantages of '{key_entity}' as mentioned in the 'Question'.\n"
            "Instructions:\n1. Focus ONLY on advantages of '{key_entity}'.\n2. IGNORE disadvantages, history, or other general information.\n"
            "3. Present 2-4 distinct advantages, each summarized in one clear sentence.\n4. Rephrase in your own words. DO NOT copy verbatim.\n"
            "5. If no clear advantages for '{key_entity}' are found, state that.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer (listing only advantages of '{key_entity}' in 2-4 sentences):")
        gen_params.update({'max_new_tokens':350, 'length_penalty':1.8})

    elif question_type == "disadvantages":
        prompt_args['key_entity'] = key_entity_for_strict_filtering if key_entity_for_strict_filtering else "the main subject"
        current_prompt_str = (
            "You are an intelligent assistant. Based *only* on the provided 'Context', your task is to identify and list the disadvantages of '{key_entity}' as mentioned in the 'Question'.\n"
            "Instructions:\n1. Focus ONLY on disadvantages of '{key_entity}'.\n2. IGNORE advantages, history, or other general information.\n"
            "3. Present 2-4 distinct disadvantages, each summarized in one clear sentence.\n4. Rephrase in your own words. DO NOT copy verbatim.\n"
            "5. If no clear disadvantages for '{key_entity}' are found, state that.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer (listing only disadvantages of '{key_entity}' in 2-4 sentences):")
        gen_params.update({'max_new_tokens':350, 'length_penalty':1.8})

    else: # "descriptive"
        entity_focus_instr = f"The main subject of the 'Question' is '{key_entity_for_strict_filtering}'. Your answer should focus exclusively on this subject." if key_entity_for_strict_filtering else "Understand the main subject of the 'Question'."
        prompt_args['entity_focus'] = entity_focus_instr
        current_prompt_str = (
            "You are an intelligent assistant. Your task is to answer the user's 'Question' based *only* on the information present in the 'Context' below.\n\n"
            "Follow these instructions carefully:\n"
            "1. {entity_focus}\n"
            "2. Your answer MUST be 4 to 6 sentences long.\n"
            "3. From the 'Context', identify distinct points that are DIRECTLY and EXCLUSIVELY relevant to the main subject of the 'Question'.\n"
            "4. CRITICALLY IMPORTANT: IGNORE any information in the 'Context' that is about other subjects not directly asked about in the 'Question'. For example, if the 'Question' is about 'laptops', you MUST ignore any parts of the 'Context' that discuss 'smartphones' or unrelated topics. If the 'Context' mentions multiple related items (e.g., AZ-900, DP-900), ONLY focus on the specific item from the 'Question' if one is clearly identified as the main subject.\n"
            "5. For each relevant point, explain it clearly in at least one full sentence.\n"
            "6. Synthesize these explanations into a single, cohesive paragraph.\n"
            "7. EVEN MORE CRITICAL: Rephrase all information completely in your own words. DO NOT copy sentences or long phrases verbatim from the 'Context'. Summarize and explain, do not extract.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")

    prompt = current_prompt_str.format(**prompt_args)
    inputs = current_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    outputs = current_llm_model.generate(**inputs, **gen_params, no_repeat_ngram_size=3, do_sample=False)
    answer = current_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if llm_cache is not None and llm_cache_key is not None: # llm_cache might have been set to None if opening failed
        try:
            llm_cache[llm_cache_key] = answer
            print(f"Stored answer in LLM cache for query: '{query_text}'")
        except Exception as e:
            print(f"WARNING: Could not store answer in LLM cache for query '{query_text}'.")
            print(f"Reason: {type(e).__name__}: {e}")
            # print(traceback.format_exc())
        finally: # Ensure cache is closed if it was successfully opened before this stage
            try:
                llm_cache.close()
                print(f"Successfully closed LLM cache after storage attempt.")
            except Exception as e_close:
                print(f"WARNING: Error closing LLM cache after storage attempt. Reason: {type(e_close).__name__}: {e_close}")
    elif llm_cache is not None: # Case: cache was opened but something went wrong before storing (e.g. key error)
        try:
            llm_cache.close()
            print(f"Successfully closed LLM cache (was open, no storage performed).")
        except Exception as e_close:
            print(f"WARNING: Error closing LLM cache (was open, no storage performed). Reason: {type(e_close).__name__}: {e_close}")


    return display_items_structured, answer

if __name__ == '__main__':
    print("Initializing RAG Application for OFFLINE use...")
    # Simple way to clear cache for testing if needed:
    # print("Attempting to clear caches for fresh run...")
    # for cache_file_base in [EMBEDDINGS_CACHE_PATH, LLM_CACHE_PATH]:
    #     # Shelve creates multiple files, try removing common extensions
    #     # For shelve, the filename itself might be enough (e.g., 'embeddings_cache' without extension)
    #     # or it might be 'embeddings_cache.db' etc.
    #     # This is a bit brute-force and might need adjustment based on how shelve names files on your OS.
    #     cache_path_obj = Path(cache_file_base)
    #     # Try removing the base file if it exists (shelve might just use the name as is for dbm.dumb)
    #     if cache_path_obj.exists():
    #         try:
    #             os.remove(cache_path_obj)
    #             print(f"Removed {cache_path_obj}")
    #         except OSError as e:
    #             print(f"Error removing cache file {cache_path_obj}: {e}")

    #     # Try removing common shelve/dbm extensions
    #     for ext in ['.bak', '.dat', '.dir', '.db', '.gdbm', '.pag', '.cfg', '.new']: # Added more common extensions
    #         try:
    #             file_with_ext = Path(str(cache_path_obj) + ext)
    #             if file_with_ext.exists():
    #                 os.remove(file_with_ext)
    #                 print(f"Removed {file_with_ext}")
    #         except OSError as e:
    #             print(f"Error removing cache file {file_with_ext}: {e}")
    # print("Cache clearing attempt finished.")


    build_index()
    if not INDEX_BUILT:
        print("Index building failed/no data. Check 'data_files', 'models', 'nltk_data'.")
        sys.exit("Exiting due to indexing/asset issues.")
    print("Starting GUI...")
    display_interface(handle_query)
    print("Application closed.")