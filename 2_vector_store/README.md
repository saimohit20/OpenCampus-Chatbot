# Vector Store

This folder contains scripts and data for building and managing the vector store.

---

##  Contents

- `ingest_data.py` — Script to process scraped data and build the vector index.
- `faiss_index/` — Folder containing the generated FAISS index and related files for fast similarity search.

---

##  How to Use

1. **Install dependencies (from project root):**
   ```bash
   pip install -r ../requirements.txt
   ```
2. **Run the ingestion script:**
   ```bash
   python ingest_data.py
   ```
   This will process the data and create/update the FAISS index in the `faiss_index/` folder.

---

##  Model Used

- **Google Generative AI Embeddings**
  - Model: `text-embedding-004`
  - Used to convert course data into numerical vectors for semantic search.

---

##  How the Code Works

1. **Loads Data:** Reads all course/program JSON files from the `domain_data` folder.
2. **Processes Each Course:** Combines important details (title, description, mentors, etc.) into a single text block with metadata.
3. **Splits Text into Chunks:** Breaks up long descriptions into smaller, overlapping pieces for better AI understanding.
4. **Creates Embeddings:** Uses Google’s embedding model to turn each chunk into a set of numbers that capture meaning.
5. **Builds the Vector Index:** Stores all embeddings in a FAISS index for fast, similarity-based search.
6. **Saves the Index:** Saves the FAISS index to the `faiss_index` folder for later use.
7. **Verifies Everything Worked:** Checks that the index was created correctly and contains data.

---

##  Notes
- Ensure that the scraped data from `1_scraping/domain_data/` is available before running the ingestion script.
- The vector store is used for efficient semantic search and retrieval in the chatbot model. 
