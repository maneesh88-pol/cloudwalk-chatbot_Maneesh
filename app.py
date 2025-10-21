# Developed by Maneesh Polavarapu for Nimbus Challenge
# Contact: polavarapumaneesh.info@gmail.com

import os
import streamlit as st
from rag_pipeline import (
    get_or_create_db,
    retrieve_context,
    call_llm,
    load_env,
    format_sources_block,
)

load_env()

st.set_page_config(page_title="Cloudwalk Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Cloudwalk Chatbot")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-k passages", 2, 10, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2)
    st.caption("Built with LangChain + Chroma + OpenAI API")

db = get_or_create_db()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

prompt = st.chat_input("Ask about CloudWalk, InfinitePay, or its mission...")

if prompt:
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    docs = retrieve_context(db, prompt, k=k)
    context_text = "\n\n".join([d.page_content for d in docs])
    answer = call_llm(prompt, context_text, temperature)

    sources = format_sources_block(docs)
    full_reply = f"{answer}\n\n---\n{sources}"

    st.session_state.history.append(("bot", full_reply))
    st.chat_message("bot").markdown(full_reply)
