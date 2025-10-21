# Developed by Maneesh Polavarapu for Nimbus Challenge
# Contact: polavarapumaneesh.info@gmail.com

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

def load_env():
    load_dotenv()
    os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-large")
    os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

def get_or_create_db(persist_dir="./vectorstore/chroma"):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    return Chroma(collection_name="cloudwalk", embedding_function=embeddings, persist_directory=persist_dir)

def retrieve_context(db, query, k=5):
    return db.similarity_search(query, k=k)

def call_llm(question, context, temperature=0.2):
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=temperature)
    prompt = ChatPromptTemplate.from_template(
        """
You are a concise company explainer for CloudWalk and its products.
Use the CONTEXT below to answer the QUESTION.

If the answer isn’t found, say “I don’t know.”

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely in 5–8 sentences.
"""
    )
    msg = prompt.format_messages(context=context, question=question)
    response = llm.invoke(msg)
    return response.content.strip()

def format_sources_block(docs):
    seen, links = set(), []
    for d in docs:
        src = d.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            links.append(f"- {src}")
    return "**Sources:**\n" + "\n".join(links)
