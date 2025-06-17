import streamlit as st
import os
import json
import google.generativeai as genai
import chromadb

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"     # Directory where your ChromaDB instance is persisted
COLLECTION_NAME = "opencampus_courses"

# --- Custom Gemini Embedding Function Class (as in ingest_data_to_vector_db.py) ---
class GeminiEmbeddingFunction:
    """
    A custom embedding function for ChromaDB using Google Gemini's text-embedding-004 model.
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable not set. Please set it to use the Gemini API.")
            st.stop() # Stop Streamlit execution if API key is missing
        
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts using the specified Gemini embedding model.
        This method is called by ChromaDB internally.
        """
        if not input:
            return []
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=input,
                task_type="RETRIEVAL_DOCUMENT" # Specify task type for better embeddings
            )
            embeddings = [embedding.embedding for embedding in response['embeddings']]
            return embeddings
        except Exception as e:
            st.error(f"Error during Gemini embedding API call: {e}")
            return [[] for _ in input] # Return empty embeddings for failed texts
        
    def name(self) -> str:
        """
        Returns a unique name for this embedding function, required by ChromaDB.
        """
        return f"gemini_text_embedding_function_{self.model_name.replace('models/', '')}"

# --- Initialize RAG Components (Cached for Streamlit Efficiency) ---
@st.cache_resource
def initialize_rag_components():
    """
    Initializes and caches the ChromaDB client and Gemini models.
    This function runs only once when the Streamlit app starts.
    """
    print("Initializing RAG components...")
    
    # Get API key from environment variable
    api_key = "AIzaSyAdab1EdwNZtZ8yQhfwHvK3V6Ir-YDhihQ"
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not found. Please set it and restart the app.")
        st.stop()

    # Initialize Gemini Embedding Function
    gemini_ef_instance = GeminiEmbeddingFunction(api_key=api_key)

    # Initialize ChromaDB client and collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=gemini_ef_instance # Use the custom embedding function
        )
        # Check if the collection is empty, indicating ingestion might not have run
        if collection.count() == 0:
            st.warning(f"ChromaDB collection '{COLLECTION_NAME}' is empty. Have you run 'ingest_data_to_vector_db.py'?")
            st.info("The chatbot will respond based on general knowledge if the database is empty.")

    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}. Ensure '{CHROMA_DB_PATH}' exists and is valid.")
        st.stop()

    # Initialize Gemini LLM for generation
    llm_model = genai.GenerativeModel('gemini-2.0-flash')
    print("RAG components initialized successfully.")
    return collection, llm_model, gemini_ef_instance

# --- RAG Logic Function ---
def get_rag_response(user_query: str, collection, llm_model, embedding_func) -> str:
    """
    Performs the RAG process: embeds query, retrieves context, and generates response.
    """
    # 1. Embed the user query
    # The embedding_func is callable and will use the configured Gemini model
    query_embedding = embedding_func([user_query])[0] # embedding_func expects a list, returns a list of embeddings

    # 2. Retrieve relevant documents from ChromaDB
    # n_results: Number of top relevant documents to retrieve
    # You can also add `where` clause here to filter by metadata if needed (e.g., domain)
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3, # Retrieve top 3 most relevant course documents
            include=['documents', 'metadatas'] # Ask for the original text and metadata
        )
    except Exception as e:
        st.error(f"Error during ChromaDB query: {e}")
        return "I'm sorry, I couldn't retrieve information from the database."

    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    if not retrieved_docs:
        # If no relevant documents are found, tell the LLM it has no specific context
        context = "No specific course information found in the database relevant to the query."
    else:
        # Construct the context for the LLM
        context_parts = []
        for i, doc_text in enumerate(retrieved_docs):
            metadata = retrieved_metadatas[i]
            title = metadata.get('course_title', 'Unknown Course')
            url = metadata.get('course_url', '#')
            context_parts.append(f"--- Course: {title} (URL: {url}) ---\n{doc_text}")
        context = "\n\n".join(context_parts)

    # 3. Construct the prompt for the LLM
    # Important: Instruct the LLM to use ONLY the provided context
    prompt = f"""
    You are an AI assistant for OpenCampus.sh, designed to provide information and recommendations about courses.
    Answer the user's question or provide course recommendations based ONLY on the following course information.
    If the answer or recommendation cannot be found in the provided information, state that you don't have enough information about that specific topic or course, and suggest asking about available courses.
    Do not use any outside knowledge.

    Course Information (Context):
    {context}

    User Question/Interest: {user_query}

    Answer:
    """

    # 4. Generate response using LLM
    try:
        response = llm_model.generate_content(prompt)
        # Access the text from the response safely
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "I'm sorry, I couldn't generate a clear response based on the available information."
    except Exception as e:
        st.error(f"Error generating LLM response: {e}")
        return "I encountered an error trying to generate a response. Please try again."

# --- Streamlit UI ---
st.set_page_config(page_title="OpenCampus Course Chatbot", page_icon="🎓")

st.title("🎓 OpenCampus AI Course Assistant")
st.markdown("Ask me anything about the courses offered at OpenCampus.sh!")

# Initialize RAG components (cached)
collection, llm_model, gemini_ef_instance = initialize_rag_components()

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial bot message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your AI assistant for OpenCampus courses. What would you like to know today?"})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input from user
user_query = st.chat_input("Type your message...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Get RAG response
    with st.spinner("Thinking..."):
        bot_response = get_rag_response(user_query, collection, llm_model, gemini_ef_instance)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)

