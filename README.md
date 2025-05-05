📄 Secure and Private On-Device RAG Engine
An offline Retrieval-Augmented Generation (RAG) system built entirely in Python, integrating semantic search, synonym-based query expansion, and local language generation — all running fully on-device with no cloud services required. The project includes a custom Tkinter-based GUI displaying retrieved documents and AI-generated answers side-by-side for an interactive, private document question-answering experience.

✨ Key Features
✅ Offline RAG pipeline: No internet or API calls; everything runs locally.

✅ Semantic search with FAISS for fast, approximate nearest neighbor retrieval.

✅ Synonym-based query expansion using spaCy and NLTK WordNet.

✅ Context-aware answer generation via Hugging Face Flan-T5 model.

✅ Tkinter GUI with side-by-side panels for retrieved text and generated answers.

✅ Secure & private: No data leaves your machine; ideal for sensitive documents.

🗃️ Data Files
This project uses .txt files stored in the /data_files folder for testing purposes.
Feel free to replace these files with your own documents (in plain text format) to customize the retrieval and answer generation.
