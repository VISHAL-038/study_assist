# Study Assistant
> A RAG-powered study tool that lets students upload their notes and interact with them through Q&A, quizzes, and summaries — built with LLaMA 3.3, FAISS, and Streamlit.

---

## Overview

AI Study Assistant is an end-to-end Retrieval-Augmented Generation (RAG) application built as an MTech project. Students upload their lecture notes or textbook PDFs and can instantly ask questions, generate quizzes, and get structured summaries — all answers grounded strictly in their own material.

---

## Features

| Feature | Description |
|---|---|
| Chat with Notes | Conversational Q&A with full memory — follow-up questions work naturally |
| Quiz Generator | Auto-generates MCQs with 4 options, correct answers, and explanations |
| Topic Summarizer | Short (bullet points) or detailed (structured) summary of any topic |
| Full PDF Summary | Summarizes the entire document with key concepts and revision points |
| Download Summaries | Save any summary as a `.txt` file |

---

## Architecture

```
PDF Upload
    ↓
Text Extraction       (PyMuPDF)
    ↓
Chunking              (LangChain RecursiveCharacterTextSplitter, 500 tokens, 50 overlap)
    ↓
Embedding             (sentence-transformers: all-MiniLM-L6-v2, runs locally)
    ↓
Vector Store          (FAISS IndexFlatL2)
    ↓
User Query → Embed Query → Similarity Search → Top-K Chunks
    ↓
Chunks + Query + Chat History → Groq API (LLaMA 3.3 70B)
    ↓
Answer / Quiz / Summary displayed in Streamlit UI
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Frontend UI | Streamlit | Rapid prototyping, no HTML/CSS needed |
| PDF Parsing | PyMuPDF (fitz) | Fast, accurate text extraction |
| Text Splitting | LangChain TextSplitters | Overlap-aware chunking prevents context loss |
| Embeddings | sentence-transformers | Free, runs locally, no API needed |
| Vector Store | FAISS (CPU) | Lightweight, no server required, fast similarity search |
| LLM | LLaMA 3.3 70B via Groq | Free tier, extremely fast inference |
| Environment | python-dotenv | Secure API key management |

---

## Project Structure

```
study_assist/
├── app.py                  # Streamlit UI — all 3 tabs
├── rag_pipeline.py         # Core RAG logic — embeddings, retrieval, LLM calls
├── utils.py                # PDF extraction and chunking
├── .env                    # API key (not committed to git)
├── requirements.txt        # All dependencies
└── README.md               # This file
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/study_assist.git
cd study_assist
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key
- Sign up at [console.groq.com](https://console.groq.com)
- Go to API Keys → Create API Key
- Copy the key (starts with `gsk_`)

### 5. Create your `.env` file
```bash
echo 'GROQ_API_KEY=your_gsk_key_here' > .env
```

### 6. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. **Upload a PDF** — any lecture notes, textbook chapter, or study material
2. **Chat with Notes** — ask questions in natural language, follow-up questions work
3. **Quiz Generator** — enter a topic, choose number of questions, reveal answers one by one
4. **Topic Summarizer** — get a short or detailed summary of any specific topic
5. **Full PDF Summary** — get a complete overview of the entire document

---

## Key Concepts Demonstrated

**Retrieval-Augmented Generation (RAG)**
Instead of relying on the LLM's training data, RAG retrieves relevant passages from the student's own notes before generating an answer. This grounds responses in the actual material and eliminates hallucination from outside sources.

**Vector Embeddings**
Text chunks are converted into high-dimensional vectors using `all-MiniLM-L6-v2`. Semantically similar text produces similar vectors, enabling meaning-based search rather than simple keyword matching.

**Chunking Strategy**
Documents are split into 500-character chunks with 50-character overlap. The overlap ensures that concepts spanning chunk boundaries are not lost during retrieval.

**Similarity Search**
FAISS (Facebook AI Similarity Search) uses L2 distance to find the top-K most relevant chunks for any query in milliseconds, even across large documents.

**Conversation Memory**
The full chat history is sent to the LLM on every request as a structured message list. This allows the model to understand follow-up questions like "explain that further" or "give me an example."

**Prompt Engineering**
System prompts constrain the LLM to answer only from retrieved context. The quiz generator uses structured JSON output formatting with temperature=0.3 for consistent, parseable responses.

---

## Evaluation

| Metric | Method |
|---|---|
| Answer relevance | Manual evaluation — does the answer match the source material? |
| Retrieval accuracy | Check "retrieved context chunks" expander — are the right sections being fetched? |
| Quiz quality | Human evaluation — are questions meaningful and options non-trivial? |
| Summary coverage | Compare summary against source PDF sections |

---

## Future Improvements

- Multi-PDF support — upload and query across multiple documents simultaneously
- RAGAS evaluation — automated retrieval and answer quality scoring
- Reranking — use a cross-encoder to rerank retrieved chunks before sending to LLM
- Export quiz as PDF — generate printable quiz sheets
- Flashcard generator — auto-generate term/definition pairs from notes

---

## Dependencies

```
streamlit
groq
faiss-cpu
sentence-transformers
pymupdf
langchain-text-splitters
python-dotenv
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- [Groq](https://groq.com) — free, fast LLaMA 3.3 inference
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [sentence-transformers](https://www.sbert.net) — local embedding model
- [Streamlit](https://streamlit.io) — UI framework
- [LangChain](https://python.langchain.com) — text splitting utilities