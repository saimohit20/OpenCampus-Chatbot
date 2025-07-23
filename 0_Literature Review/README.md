# Literature Review

This folder contains research and reference materials that inform the design and development of the OpenCampus Chatbot project. Below is a summary of key sources and their relevance to this work.

---

##  Key Sources & Insights

### 1. Agentic Retrieval‑Augmented Generation: A Survey on Agentic RAG
- **Link:** [Agentic RAG](https://arxiv.org/abs/2501.09136)
- **Objective:** Explore how agentic elements—such as tool use, planning, and step-by-step reasoning—enhance traditional RAG systems.
- **Methods:** Reviewed and classified existing RAG designs, comparing standard vs agentic pipelines.
- **Outcomes:** Found that agentic RAG systems offer superior adaptability and performance in complex tasks.
- **Relation to the Project:** Justifies the use of custom tools and multi-tool orchestration (via Gemini/Gemma) in this project, rather than relying on a single-prompt design. The chatbot leverages agentic RAG principles for more flexible and robust information retrieval and reasoning.

---

### 2. Web Scraping + RAG System (WebRAG‑style)
- **Link:** [WebRAG](https://github.com/arkeodev/scraper)
- **Objective:** Build a pipeline that scrapes web content, embeds it, stores it in a vector database, and supports question-answering via a user interface.
- **Methods:** Utilizes Playwright for scraping, llama-index for embeddings, FAISS for indexing, and Streamlit for the frontend.
- **Outcomes:** Delivers a fully functional flow from automated web scraping to RAG-based QA.
- **Relation to the Project:** Mirrors the project pipeline: scraping OpenCampus.edu with Playwright, processing JSON, indexing vectors locally with FAISS, and querying through Streamlit + Gemini/Gemma. This validates the end-to-end approach used here.

---

### 3. Why RAG Systems Fail and How to Fix Them
- **Link:** [Analytics Vidhya Blog](https://www.analyticsvidhya.com/blog/2025/03/why-rag-systems-fail-and-how-to-fix-them/)
- **Objective:** Examine common failures in RAG pipelines and propose solutions, focusing on retrieval and generation errors.
- **Methods:** Detailed evaluation of challenges like query-document mismatch, embedding flaws, chunking errors, retrieval inefficiency, and generation issues.
- **Outcomes:** Offers practical fixes—better chunking, hybrid search, embedding calibration, and query re-ranking—to improve performance.
- **Relation to the Project:** Highlights areas for improvement in this repo:
  - Optimize chunking logic in `ingest_data.py`
  - Add retrieval re-ranking after FAISS queries
  - Clean and preprocess scraped JSON data
  - Consider hybrid search to reduce LLM hallucination

---

##  Summary
These sources collectively inform the project's architecture, helping the use of agentic RAG, robust web scraping, and vector search. They also provide a roadmap for future improvements, ensuring the chatbot remains accurate, efficient, and adaptable. 