# 🤖 Cloudwalk Chatbot (Nimbus Challenge – Level 1)

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about **CloudWalk**, its **products (e.g., InfinitePay)**, **mission**, and **brand values** using **public web sources**.

## 🎯 Goal
Build a chatbot that explains what CloudWalk is, its products, mission, and brand values.

## ⚙️ Features
- Retrieval-Augmented Generation (RAG) using **LangChain + Chroma**
- Uses **OpenAI** (or any compatible API) for embeddings & generation
- **Streamlit** chat interface
- Sources from CloudWalk’s official website, InfinitePay, and Wikipedia
- Includes **3 sample conversations** in `sample_conversations.md`

## 🧱 Architecture
1. **ingest.py** – Fetches and embeds data into Chroma
2. **app.py** – Streamlit chatbot interface
3. **rag_pipeline.py** – Shared helper functions

## 🧩 Setup Instructions
```bash
pip install -r requirements.txt
cp config.example.env .env
python ingest.py
streamlit run app.py
```

## 🧪 Sample Conversations
See `sample_conversations.md`.

## 🧑‍💻 Author
**Maneesh Polavarapu**
