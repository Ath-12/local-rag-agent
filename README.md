# Local RAG Agent 🤖

A fully local, private Retrieval-Augmented Generation (RAG) application built with Python. This system allows users to chat with their personal documents (PDFs, text files) without sending any data to the cloud.

## Architecture & Tech Stack

* **Orchestration Framework:** LlamaIndex (Handles ingestion, chunking, and retrieval)
* **LLM (The Brain):** Llama 3 (Running locally via Ollama)
* **Embeddings (The Translator):** HuggingFace `all-MiniLM-L6-v2` (Runs on CPU for fast, local vectorization)
* **Vector Store (The Memory):** In-memory indexing via LlamaIndex (ready to be upgraded to ChromaDB)
* **Frontend UI:** Streamlit (Provides a ChatGPT-like web interface)

## How it Works

1.  **Ingestion:** The system scans the `data/` directory for documents.
2.  **Embedding:** Text is chunked and converted into mathematical vectors using the HuggingFace MiniLM model.
3.  **Retrieval:** When a user asks a question, the system converts the query into a vector and retrieves the most semantically similar text chunks.
4.  **Generation:** The retrieved context is passed to the local Llama 3 model to generate a factual response based *only* on the provided documents.

## Current Status
* ✅ Core RAG pipeline established.
* ✅ Local LLM and Embedding models connected.
* ✅ Streamlit chat interface built and maintaining session state.