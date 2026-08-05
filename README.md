# 🩺 Healthcare AI Chatbot

> **Production-grade AI-powered Healthcare Assistant built with FastAPI, Streamlit, Gemini 2.5 Flash, and Retrieval-Augmented Generation (RAG).**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# 📌 Overview

Healthcare AI Chatbot is a **production-ready Retrieval-Augmented Generation (RAG)** application built using **FastAPI**, **Streamlit**, **Gemini 2.5 Flash**, and **FAISS**.

The chatbot provides educational healthcare information while ensuring responses remain safe through multiple guardrails.

### Topics Supported

- 🩺 Symptoms
- 💊 Diseases
- 🥗 Nutrition
- ❤️ Preventive Healthcare
- 🚑 First Aid
- 🧘 Healthy Lifestyle

> ⚠️ **This project is intended only for educational purposes and is not a substitute for professional medical advice.**

---

# ✨ Features

## 🤖 AI Features

- 🧠 Gemini 2.5 Flash LLM
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic Search
- 📄 PDF Knowledge Base
- 💬 Multi-turn Conversation Memory
- 📑 Source Citation
- ⚡ Streaming Responses
- 📖 Context-aware Answers

---

## 🛡 Safety Features

- 🚨 Emergency Detection
- ❌ No Disease Diagnosis
- ❌ No Medication Prescription
- ⚠️ Medical Disclaimer
- 🩺 Healthcare Safety Guardrails

---

## 🎨 UI Features

- 💬 Streamlit Chat Interface
- ⚡ Real-time Streaming
- 🆕 New Chat
- 📜 Chat History
- 📊 Structured Logging

---

# Screenshots

<img width="1600" height="757" alt="Image" src="https://github.com/user-attachments/assets/81d1c980-a648-4f84-81b9-57d5e54a4d6d" />
<img width="1600" height="757" alt="Image" src="https://github.com/user-attachments/assets/d86f080d-acc0-4fc9-b880-f94fe5cae062" />
<img width="1600" height="756" alt="Image" src="https://github.com/user-attachments/assets/3f0b7203-6dff-40a7-b532-5c66740409a0" />
<img width="1600" height="769" alt="Image" src="https://github.com/user-attachments/assets/68adb252-86fa-4f00-b6e2-f1dae5ab79cf" />

---

# 🏗 Architecture

```text
                   User
                     │
                     ▼
        🎨 Streamlit Frontend
                     │
              HTTP Streaming
                     │
                     ▼
          ⚡ FastAPI Backend
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 Guardrails      Conversation      RAG
                  Memory         Retrieval
                     │
                     ▼
           Prompt Construction
                     │
                     ▼
          Gemini 2.5 Flash LLM
                     │
                     ▼
      Safe Response + Citations
```

---

# 🧰 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Gemini 2.5 Flash |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Search | FAISS |
| PDF Loader | PyMuPDF |
| ML | Sentence Transformers |

---

# 📂 Project Structure

```text
Healthcare-RAG-Chatbot/
│
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── embeddings.py
│   ├── guardrails.py
│   ├── llm.py
│   ├── memory.py
│   ├── prompts.py
│   ├── rag.py
│   └── utils.py
│
├── frontend/
│   └── streamlit_app.py
│
├── docs/
│   ├── Architecture.md
│   └── Logic.md
│
├── data/
│   ├── docs/
│   └── vectorstore/
│
├── .env.example
├── requirements.txt
├── run.py
└── README.md
```

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/devanshnegi88/Healthcare-RAG-Chatbot.git

cd Healthcare-RAG-Chatbot
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Copy the example file

```bash
cp .env.example .env
```

Update

```env
GEMINI_API_KEY=your_api_key_here
```

---

# ⚙ Environment Variables

| Variable | Description |
|-----------|-------------|
| GEMINI_API_KEY | Gemini API Key |
| GEMINI_MODEL | Gemini Model |
| EMBEDDING_MODEL_NAME | Embedding Model |
| CHUNK_SIZE | Chunk Size |
| CHUNK_OVERLAP | Chunk Overlap |
| TOP_K | Retrieved Chunks |
| MAX_MEMORY_TURNS | Conversation Memory |
| API_PORT | FastAPI Port |
| FRONTEND_PORT | Streamlit Port |

---

# ▶ Running the Project

## Recommended

```bash
python run.py
```

---

## Manual

### Backend

```bash
uvicorn backend.api:app --reload
```

### Frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

# 🔍 RAG Pipeline

```text
User Query
     │
     ▼
Conversation Memory
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Similarity Search
     │
     ▼
Relevant Documents
     │
     ▼
Prompt Builder
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Safe Response + Citations
```

---

# 📈 Future Improvements

- 🔐 Authentication
- 💾 Persistent Chat History
- 🌍 Multilingual Support
- 🔍 Hybrid Search (BM25 + Dense)
- 🐳 Docker Deployment
- ☁ Cloud Deployment
- 📊 Evaluation Dashboard
- 📱 Responsive UI

---

# 📚 Documentation

Additional documentation:

- 📄 docs/Architecture.md
- 📄 docs/Logic.md

---

# ⚠ Medical Disclaimer

This project provides **general educational healthcare information only**.

It **does not diagnose diseases**.

It **does not prescribe medications**.

Always consult a licensed healthcare professional for medical concerns.

If you believe you are experiencing a medical emergency, contact your local emergency services immediately.

---

# 👨‍💻 Author

**Devansh Negi**

Backend & AI Engineer

- 💼 LinkedIn: https://linkedin.com/in/devansh-negi005
- 💻 GitHub: https://github.com/devanshnegi88

---

# ⭐ Support

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub.

Contributions, suggestions, and feedback are always welcome.
