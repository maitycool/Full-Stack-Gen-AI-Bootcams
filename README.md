# GenAI RAG Chatbot (Production-Ready)

## Problem
Build a scalable chatbot that answers questions from PDFs using RAG.

## Architecture
User → Streamlit UI → RAG Pipeline → Vector DB → LLM → Response

## Features
- PDF ingestion
- Semantic search (FAISS)
- LLM response generation
- Modular architecture

## Tech Stack
- Python
- LangChain
- FAISS
- Streamlit
- OpenAI / HuggingFace

## How to Run
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## Future Improvements
- Add FastAPI backend
- Deploy on AWS / Azure
- Add authentication
