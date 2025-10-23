---
title: GenAI PDF Q&A (RAG · LangChain)
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.37.1"
app_file: app.py
pinned: false
---

# GenAI PDF Q&A — RAG on LangChain (Hugging Face)

**Upload PDFs → Build an index → Ask questions.**

- **Embeddings:** FastEmbed (no API key)  
- **Vector DB:** FAISS  
- **LLM:** Groq (preferred) or OpenAI

## 🔴 Live Demo

➡️ **Hugging Face Space:** https://huggingface.co/spaces/SathishLella/genai-pdf-qa-chatbot-groq
[![Dashboard preview](GenAI_pdf_reader.png)](https://huggingface.co/spaces/SathishLella/genai-pdf-qa-chatbot-groq)



*(Click to open the hosted app.)*

---

## 🔐 Secrets (Spaces → Settings → Variables & secrets)

```bash
# Required (preferred):
GROQ_API_KEY=your_groq_key
# Optional: choose a current Groq model
GROQ_MODEL=llama-3.1-8b-instant

# Optional fallback (if you want to use OpenAI instead of Groq):
OPENAI_API_KEY=your_openai_key
