# LLMs

This folder contains the main chatbot application for the OpenCampus project.

---

##  Contents

- `main.py` — The main Streamlit app for interacting with the OpenCampus chatbot.
- `opencampus_logo.png` — Logo used in the app interface on UI.

---

##  How to Use

1. **Install dependencies (from project root):**
   ```bash
   pip install -r ../requirements.txt
   ```
2. **Run the chatbot app:**
   ```bash
   streamlit run main.py
   ```
   This will launch the chatbot interface in your web browser.

---

##  Models Used

- **Google Generative AI Embeddings**
  - Model: `text-embedding-004`
  - Used for converting course data into numerical vectors for semantic search.
- **Google Gemini (gemini-2.0-flash)**
  - Used as a conversational language model (if selected).
- **Ollama Local Models**
  - Supports local LLMs Gemma3:12b which is downloaded on my system and uses it via the Ollama server.

---

##  How the Code Works

1. **Loads the Vector Store:** Loads the FAISS index and embeddings from the `2_vector_store/faiss_index` folder.
2. **Model Selection:** Lets you choose between Google Gemini or available Ollama models in the sidebar.
3. **Initializes the Chatbot:** Sets up a Retrieval-Augmented Generation (RAG) chain to answer questions using both the LLM and the vector store.
4. **Handles User Input & decide the tool:** Accepts questions from the user and determines the tool (e.g., list courses, get details, compare courses, or general questions).
5. **Retrieves Relevant Data:** Searches the vector store for relevant course information based on the user's question.
6. **Generates Responses:** Uses the selected LLM to generate structured, helpful answers using the retrieved data.
7. **Displays Results:** Shows the conversation and results in a user-friendly chat interface.

---

##  Notes
- Make sure the FAISS index is built and available in `2_vector_store/faiss_index` before running the app.
- You need a valid Google API key for using Google models.
- Ollama must be running locally to use local LLMs. You can check through http://localhost:11434
