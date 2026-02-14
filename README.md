# 🎓 ELA Chat - RAG Assistant

An intelligent virtual assistant powered by **Retrieval-Augmented Generation (RAG)** technology, built to assist with thesis research. It combines semantic search with advanced LLM processing to provide accurate and contextual answers.

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/1979281c-20b8-42b1-893b-9e93ef379ba2" alt="ELA Chat Interface" width="90%" style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" />
</div>


---

## 📌 What It Does

- Answers questions about my thesis documents using AI
- Searches relevant context automatically
- Maintains conversation history for better understanding
- Provides real-time feedback with animated responses

---

## 🛠 How It Works

```
1. You ask a question
   ↓
2. System retrieves relevant documents (RAG)
   ↓
3. LLM processes with context
   ↓
4. Assistant responds
```

---

## 🔧 Technologies

| Component | Technology |
|-----------|-----------|
| **Frontend** | Tkinter |
| **LLM** | Llama 3.3 70B (Groq) |
| **Embeddings** | sentence-transformers (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Framework** | LangChain |
| **Backend** | Python 3.11+ |

---

## 📁 Project Structure

```
RAG/
├── ELA-Chat.py                   # Tkinter UI (simple)
├── 1_ingestion_pipeline.py       # Document indexing
├── 2_retrieval_pipeline.py       # Retrieval system
├── requirements.txt              # Dependencies
├── .env                          # Environment variables (create)
├── db/
│   └── chroma_db/               # Vector database
└── docs/                         # Input documents
```

---

