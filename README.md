# VeoVec – Semantic Search System for Large-Scale Legal PDF Collections

## Motivation & Overview

This project is being developed with and for a company that supports enterprises in their transition toward sustainability and ensures compliance with national and international environmental regulations. Their internal document collection contains thousands of legal PDFs, many of which are unstructured or low-quality scans. This makes searching the corpus difficult and inefficient.

VeoVec is a local full-stack retrieval and semantic search system designed to index and query large collections of legal PDF documents. The quality of the search results is currently being evaluated by in-house legal experts.

The system integrates:

- Weaviate vector database  
- Ollama for local model inference  
- nomic-embed-text for 768-dimensional embeddings  
- mistral-nemo for Retrieval-Augmented Generation (RAG)  
- Streamlit as the user interface  
- A custom ingestion and PDF–text chunking pipeline  

This repository functions as a research prototype and a portfolio project demonstrating backend engineering, vector search, model integration, and UI development. No PDF dataset is included.

---

## System Architecture

### Components

1. **Weaviate**  
   Stores chunk embeddings, performs vector and hybrid search, and integrates with Ollama for embedding and generation.

2. **Ollama**  
   Hosts nomic-embed-text for embeddings and mistral-nemo for generation.

3. **Streamlit Application**  
   Provides the search interface, collection management, text chunking and ingestion workflows, and a RAG interface. PDFs are linked through a lightweight static file server.

### Embedding Details

- Embedding model: nomic-embed-text  
- Vector dimension: 768  
- Recommended chunk size: 400–600 words  
- Recommended overlap: 50–100 words  

---

## Project Structure

```text
veo_vec/
│
├── app.py                    # Main Streamlit application
├── app_utils.py              # Search, ingestion, RAG pipeline
├── fake_OCR.py               # Dummy OCR module
├── sanity.py                 # OCR and ingestion validation
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
│
├── app_data/                 # UI assets (license-free)
├── feedback_collector/       # User feedback logs
└── pdf_data/                 # Mount point for PDFs/TXTs (empty here)
```

---

## Features

### Semantic Search

- Vector-based querying using nomic-embed-text embeddings  
- Displays similarity scores and relevant text chunks  
- Links to original PDF documents  

### Hybrid Search

- Combines vector similarity with BM25 keyword scoring  
- Adjustable alpha parameter  

### Chunk-Based Ingestion

- Splits TXT files into overlapping chunks  
- Stores relative PDF paths  
- Detects duplicates and missing PDFs  
- Provides ingestion statistics and reports  

### Retrieval-Augmented Generation (RAG)

- Retrieves relevant chunks  
- Feeds them into mistral-nemo for grounded answer generation  
- Displays:
  - Model output  
  - Context documents  
  - Supporting text  
  - User feedback widget  

### Static PDF Serving

PDFs are served through a lightweight HTTP server:

http://localhost:8000/<relative_path>


---

## Running the System

### Requirements

- Docker Desktop  
- 12–16 GB RAM recommended  
- First run downloads ~7 GB of model files  

### Quick Start

Use the provided startup script: start_app.bat


This script:

1. Builds and launches all Docker services  
2. Waits for Ollama to become ready  
3. Pulls required models  
4. Opens the UI at: http://localhost:8501  

---

## Ingesting PDFs and Text

1. Place PDFs and TXT files into: pdf_data/


2. In the Streamlit UI:
   - Open "Collections"
   - Initialize a collection
   - Choose chunk size, overlap, batch size
   - Click "Populate"

The pipeline:

- Detects duplicates  
- Splits text into chunks  
- Embeds and inserts chunks into Weaviate  
- Logs ingestion statistics and issues  

---

## Searching

1. Select a collection  
2. Choose Semantic or Hybrid mode  
3. Enter a query  
4. Explore results:
   - PDF link  
   - Similarity score  
   - Matching text chunk  

PDFs open in a new browser tab.

---

## RAG Mode

Input:

- Meta-prompt  
- Query  
- Number of chunks to retrieve  
- Timeout limit  

Output:

- Final generated answer  
- Files used as context  
- Relevant text chunks  

User feedback is stored automatically.

---

## Limitations and Outlook

### OCR

Google Document AI OCR was used for the initial dataset because other tools (Tesseract, EasyOCR, PaddleOCR, Docling) performed poorly.  
To avoid OCR costs during testing, this prototype includes a dummy OCR module that copies pre-generated TXT files into place. This allows evaluators to test search quality without running OCR.

### RAG

RAG is experimental. Large models such as mistral-nemo may time out during local inference.  
Planned improvements:

- Switch to smaller but strong models (e.g. xLSTM)  
- Move inference to cloud resources for scalability  

### Explicit filtering

For a first evaluation the system is applied to venezuelan documents only. Once we upscale explicit filtering options shall be added (language, country, further metadata). 

---

## Licensing and Privacy

- No documents or proprietary data are included.  
- All images in `app_data` are license-free.  

---

## Purpose of This Repository

This project demonstrates practical skills in:

- Vector search architecture  
- LLM integration  
- Chunking and retrieval pipelines  
- Dockerized ML infrastructure  
- Streamlit-based UI design  

Intended as a portfolio project for academic applications and professional review.


