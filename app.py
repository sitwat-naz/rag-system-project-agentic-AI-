
import os
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384
LLM_MODEL            = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN             = os.environ.get("HF_TOKEN", "")

# ── Init models ───────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
hf_client       = InferenceClient(token=HF_TOKEN)

# ── Document classes ──────────────────────────────────────────────────────────
class DocumentLoader:
    def load_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"text": text, "source": file_path, "page": 1}]

    def load_pdf(self, file_path):
        reader = PdfReader(file_path)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text, "source": file_path, "page": i + 1})
        return pages

class DocumentChunker:
    def chunk_by_tokens(self, text, chunk_size=400, overlap=50):
        words   = text.split()
        chunks  = []
        start   = 0
        idx     = 0
        while start < len(words):
            chunk_text = " ".join(words[start:start + chunk_size])
            chunks.append({"text": chunk_text, "chunk_index": idx,
                           "word_start": start})
            start += chunk_size - overlap
            idx   += 1
        return chunks

    def chunk_documents(self, documents, strategy="tokens", **kwargs):
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_by_tokens(doc["text"], **kwargs)
            for chunk in chunks:
                chunk["source"] = doc["source"]
                chunk["page"]   = doc["page"]
                all_chunks.append(chunk)
        return all_chunks

class VectorDatabase:
    def __init__(self, embedding_model, dimension=384):
        self.embedding_model = embedding_model
        self.dimension       = dimension
        self.index           = faiss.IndexFlatL2(dimension)
        self.documents       = []

    def add_documents(self, documents):
        texts      = [d["text"] for d in documents]
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query, k=5):
        qe = np.array(self.embedding_model.encode([query])).astype("float32")
        distances, indices = self.index.search(qe, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                r          = self.documents[idx].copy()
                r["score"] = float(dist)
                results.append(r)
        return results

vector_db = VectorDatabase(embedding_model, EMBEDDING_DIMENSION)

# ── LLM helpers ───────────────────────────────────────────────────────────────
def create_rag_prompt(query, context_chunks):
    context = "\n\n".join([
        f"[Document {i+1}] (Source: {c.get('source','unknown')}, Page: {c.get('page','N/A')})"
        f"\n{c['text']}"
        for i, c in enumerate(context_chunks)
    ])
    return (f"<s>[INST] You are a helpful AI assistant. Answer ONLY from the context below.\n\n"
            f"If the context lacks enough info, say so clearly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER: [/INST]")

def generate_answer(query, context_chunks, max_new_tokens=500, temperature=0.3):
    if not context_chunks:
        return "No relevant context found. Please upload a document first."
    prompt = create_rag_prompt(query, context_chunks)
    try:
        return hf_client.text_generation(
            prompt, model=LLM_MODEL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
            do_sample=True
        ).strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# ── Gradio helpers ────────────────────────────────────────────────────────────
def process_document(file):
    if file is None:
        return "No file uploaded."
    try:
        loader  = DocumentLoader()
        chunker = DocumentChunker()
        docs    = loader.load_pdf(file.name) if file.name.endswith(".pdf") else loader.load_txt(file.name)
        chunks  = chunker.chunk_documents(docs, strategy="tokens", chunk_size=400, overlap=50)
        vector_db.add_documents(chunks)
        return (f"Processed {file.name} — "
                f"{len(docs)} page(s), {len(chunks)} chunks, "
                f"{vector_db.index.ntotal} total vectors.")
    except Exception as e:
        return f"Error: {e}"

def respond(message, chat_history, num_sources, temperature):
    if not message.strip():
        return "", chat_history, ""
    results    = vector_db.search(message, k=int(num_sources))
    answer     = generate_answer(message, results, temperature=temperature)
    sources_md = "### Sources\n\n"
    for i, r in enumerate(results, 1):
        sources_md += (f"**[{i}]** `{r.get('source','unknown')}` "
                       f"Page {r.get('page','N/A')} Score {r.get('score',0):.4f}\n\n"
                       f"> {r['text'][:250]}...\n\n---\n\n")
    chat_history.append((message, answer))
    return "", chat_history, sources_md

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Q&A System") as demo:
    gr.Markdown("# RAG-Powered Q&A System")
    gr.Markdown("Upload a document, then ask questions about it.")

    with gr.Tab("Upload Documents"):
        file_input    = gr.File(label="Upload PDF or TXT", file_types=[".pdf", ".txt"])
        upload_btn    = gr.Button("Process Document", variant="primary")
        upload_status = gr.Markdown()
        upload_btn.click(process_document, inputs=[file_input], outputs=[upload_status])

    with gr.Tab("Chat"):
        chatbot  = gr.Chatbot(height=420)
        msg_box  = gr.Textbox(label="Your Question",
                              placeholder="Ask something about your document...")
        send_btn = gr.Button("Send", variant="primary")
        with gr.Row():
            num_sources = gr.Slider(1, 10, 5, step=1, label="Sources to retrieve")
            temperature = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Temperature")
        clear_btn       = gr.Button("Clear Chat")
        sources_display = gr.Markdown()
        gr.Examples(["Summarize the document.", "What are the key points?",
                     "Who is mentioned in the document?", "What conclusions are drawn?"],
                    inputs=msg_box)
        msg_box.submit(respond, [msg_box, chatbot, num_sources, temperature],
                       [msg_box, chatbot, sources_display])
        send_btn.click(respond, [msg_box, chatbot, num_sources, temperature],
                       [msg_box, chatbot, sources_display])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, sources_display])

if __name__ == "__main__":
    demo.launch()
