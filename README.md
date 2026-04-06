---
title: RAG Q&A System
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# RAG-Powered Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about uploaded documents.

## How to Use
1. Go to the **Upload Documents** tab and upload a PDF or TXT file
2. Wait for the processing confirmation
3. Go to the **Chat** tab and ask questions about your document
4. Adjust the sliders to control retrieval and creativity

## Technologies Used
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers, runs locally)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **LLM**: Mistral-7B-Instruct via Hugging Face Inference API
- **UI**: Gradio

## Example Queries
- Summarize the document.
- What are the key points?
- Who is mentioned in the document?
- What conclusions are drawn?

## Setup (for HF Spaces)
Add your `HF_TOKEN` in Space Settings → Variables and Secrets.
