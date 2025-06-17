# chatbot_app.py

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

MY_GOOGLE_API_KEY = "AIzaSyAdab1EdwNZtZ8yQhfwHvK3V6Ir-YDhihQ"



load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index" 
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-2.0-flash"


# --- Function to Load FAISS Index and Embeddings (Cached for efficiency) ---
@st.cache_resource 
def load_faiss_and_embeddings():
    try:
        google_api_key_to_use = os.getenv("GOOGLE_API_KEY", MY_GOOGLE_API_KEY)

        if not google_api_key_to_use or google_api_key_to_use == "YOUR_GOOGLE_API_KEY_HERE":
            st.error("Google API Key is not set. Please replace 'YOUR_GOOGLE_API_KEY_HERE' in the code or set the GOOGLE_API_KEY environment variable.")
            st.stop()

        # Pass the resolved API key directly to the embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=google_api_key_to_use)
        faiss_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.success(f"FAISS index loaded successfully from '{FAISS_INDEX_PATH}'.")
        return faiss_db, embeddings
    except Exception as e:
        st.error(f"Error loading FAISS index or embedding model. Make sure '{FAISS_INDEX_PATH}' exists and contains valid index files. Error: {e}")
        st.stop()


# --- Function to Initialize RAG Chain (Cached) ---
@st.cache_resource 
def initialize_rag_chain(_faiss_db_obj):
    try:
        
        google_api_key_to_use = os.getenv("GOOGLE_API_KEY", MY_GOOGLE_API_KEY)

        if not google_api_key_to_use or google_api_key_to_use == "YOUR_GOOGLE_API_KEY_HERE":
            st.error("Google API Key is not set. Please replace 'YOUR_GOOGLE_API_KEY_HERE' in the code or set the GOOGLE_API_KEY environment variable.")
            st.stop()

        # Pass the resolved API key directly to the LLM
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2, google_api_key=google_api_key_to_use)
        retriever = _faiss_db_obj.as_retriever(search_kwargs={"k": 3})
        template = """You are an educational assistant for an online learning platform.
        Answer the user's question truthfully and only based on the provided context.
        If the answer cannot be found in the context, state that you don't have enough information
        to answer from the available resources. Do not make up answers.

        Context:
        {context}

        Question: {question}

        Answer:"""
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True, 
            chain_type_kwargs={"prompt": PROMPT}
        )
        st.success("RAG chain initialized successfully.")
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing RAG chain. Error: {e}")
        st.stop()


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Educational Chatbot", page_icon="🎓")
    st.title("🎓 Educational Course Assistant")
    st.markdown("Ask me anything about the courses available on the platform!")

    # Load FAISS and embeddings (cached)
    faiss_db, embeddings = load_faiss_and_embeddings()

    # Initialize RAG chain (cached)
    qa_chain = initialize_rag_chain(faiss_db)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                source_documents = response.get('source_documents', [])

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if source_documents:
                        st.subheader("Sources:")
                        for i, doc in enumerate(source_documents):
                            st.write(f"**Document {i+1}**:")
                            st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                            st.write(f"**URL:** {doc.metadata.get('url', 'N/A')}")
                            st.write(f"**Content Snippet:** {doc.page_content[:200]}...")
                            st.markdown("---")
            except Exception as e:
                answer = f"I apologize, but an error occurred while processing your request: {e}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()