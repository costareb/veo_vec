\# VeoVec – Semantic Search System for Large-Scale Legal PDF Collections



\## Motivation \& Overview

Veo Partners supports enterprises in their transition toward sustainability and ensures compliance with national and international environmental regulations. Their internal document collection contains thousands of legal PDFs, many of which are unstructured or low-quality scans. This makes searching the corpus difficult and inefficient.



VeoVec is a local full-stack retrieval and semantic search system designed to index and query large collections of legal PDF documents. The quality of the search results is currently being evaluated by in-house legal experts.



The system integrates:



\- Weaviate vector database  

\- Ollama for local model inference 

\- nomic-embed-text for 768-dimensional embeddings  

\- mistral-nemo for Retrieval-Augmented Generation (RAG)  

\- Streamlit as the user interface  

\- A custom ingestion and PDF–text chunking pipeline



This repository functions as a research prototype and a portfolio project demonstrating backend engineering, vector search, model integration, and UI development. No PDF dataset is included. 



\## System Architecture



\### Components

1\. \*\*Weaviate\*\*  

&nbsp;  Stores chunk embeddings, performs vector and hybrid search, and integrates with Ollama for embedding and generation.



2\. \*\*Ollama\*\*  

&nbsp;  Hosts nomic-embed-text for embeddings and mistral-nemo for generation.



3\. \*\*Streamlit Application\*\*  

&nbsp;  Provides the search interface, collection management, text chunking and ingestion workflows, and a RAG interface. PDFs are linked through a lightweight static file server.



\### Embedding Details

\- Embedding model: nomic-embed-text  

\- Vector dimension: 768  

\- Recommended chunk size: 400–600 words  

\- Recommended overlap: 50–100 words  



\## Project Structure



veo\_vec/

│

├── app.py # Main Streamlit application

├── app\_utils.py # Search, ingestion, RAG pipeline

├── fake\_OCR.py # Dummy OCR module

├── sanity.py # OCR and ingestion validation

├── Dockerfile

├── docker-compose.yml

├── entrypoint.sh

├── requirements.txt

│

├── app\_data/ # UI assets (license-free)

├── feedback\_collector/ # User feedback logs

└── pdf\_data/ # Mount point for PDFs/TXTs (empty here)





\## Features



\### Semantic Search

\- Vector-based querying using nomic-embed-text embeddings  

\- Displays similarity scores and relevant text chunks for explainability

\- Links to original PDF documents



\### Hybrid Search

\- Combines vector similarity with BM25 keyword scoring  

\- Adjustable alpha parameter



\### Chunk-Based Ingestion

\- Splits TXT files into overlapping chunks  

\- Stores relative PDF paths for access  

\- Tracks duplicates and missing PDFs  

\- Provides ingestion statistics and reports



\### RAG (Retrieval-Augmented Generation)

\- Retrieves relevant chunks  

\- Feeds them into mistral-nemo for grounded answer generation  

\- Displays:

&nbsp; - Model output  

&nbsp; - Context documents  

&nbsp; - Extracted supporting text  

&nbsp; - User feedback widget



\### Static PDF Serving

PDFs are served through a lightweight HTTP server:

http://localhost:8000/<relative\_path>





\## Running the System



\### Requirements

\- Docker Desktop  

\- Minimum 12–16 GB RAM recommended  

\- First run requires downloading ~7 GB of model files



\### Quick Start



Use the provided startup script:

start\_app.bat



This script:

1\. Builds and launches all Docker services  

2\. Waits for Ollama to become ready  

3\. Pulls required models  

4\. Opens the UI at: http://localhost:8501





\## Ingesting PDFs and Text



1\. Place PDFs and TXT files into:

pdf\_data/





2\. In the Streamlit UI:

&nbsp;  - Open “Collections”

&nbsp;  - Initialize a collection

&nbsp;  - Choose chunk size, overlap, and batch size

&nbsp;  - Click “Populate”



The pipeline:

\- Computes file hashes

\- Detects duplicates  

\- Splits text into chunks  

\- Embeds and inserts into Weaviate  

\- Logs statistics and issues.



\## Searching



1\. Select a collection  

2\. Choose Semantic or Hybrid mode  

3\. Enter a query  

4\. Explore results:

&nbsp;  - PDF link  

&nbsp;  - Similarity score  

&nbsp;  - Matching chunk text  



PDFs open in a new browser tab.



\## RAG Mode



Input:

\- A meta-prompt instructing the answer generator  

\- The RAG query  

\- Number of chunks to retrieve  

\- Timeout limit  



Output includes:

\- Final generated answer  

\- Files used as context  

\- Relevant text chunks  



Users can submit feedback, which is stored automatically.



\## Limitations:



\### OCR:

The initial dataset required reliable OCR processing. Google Document AI OCR was used because other tools such as Tesseract, EasyOCR, PaddleOCR, and Docling performed poorly on the provided legal PDFs. Since the cloud OCR service is not free, and because the goal was to evaluate search quality rather than OCR, the prototype uses a dummy OCR module. The corresponding TXT files are already included and are simply copied into place. This allows evaluators to test search functionality without incurring OCR costs.



\### RAG:

RAG is considered an experimental feature. Current LLMs may produce legally unreliable or incorrect statements, so RAG is intended mainly for summarization or translation.

Using large models such as mistral-nemo causes frequent timeouts in local execution. Planned improvements include switching to smaller but strong models such as xLSTM and migrating inference to cloud resources for better scalability.



\## Licensing and Privacy

\- No documents or proprietary data are included.  

\- All images in `app\_data` are license-free.  



\## Purpose of This Repository

This project demonstrates practical skills in:

\- Vector search architecture

\- Modern LLM integration

\- Chunking and retrieval pipelines

\- Dockerized ML infrastructures

\- Streamlit-based UI design



Intended as a portfolio project for academic applications and professional review.

















