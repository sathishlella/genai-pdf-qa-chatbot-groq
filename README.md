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

### GenAI PDF Q&A — RAG on LangChain (Streamlit)

Upload PDFs → Build an index → Ask questions.

- **Embeddings:** FastEmbed (no API key)  
- **Vector DB:** FAISS  
- **LLM:** Groq (preferred) or OpenAI

#### Secrets (Spaces → Settings → Variables & secrets)

```
GROQ_API_KEY=...      # preferred
GROQ_MODEL=llama-3.1-70b-versatile
# Optional fallback:
OPENAI_API_KEY=...
```
