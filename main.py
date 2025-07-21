import streamlit as st
import os
import requests
import json
import re
import logging
from typing import Dict, Any, List

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field # For tool input schemas
from langchain_core.output_parsers import StrOutputParser # To parse LLM output as string


# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index" # Ensure this directory exists and contains your FAISS index
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Ensure this matches your ingest_data.py
GOOGLE_API_KEY = "" # <--- IMPORTANT: REPLACE WITH YOUR ACTUAL GOOGLE API KEY
OLLAMA_BASE_URL = "http://localhost:11434" # Adjust if your Ollama server is on a different address

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Function to get available Ollama models ---
@st.cache_data
def get_ollama_models():
    """Fetches a list of locally available Ollama models from the Ollama server."""
    logging.info(f"Attempting to connect to Ollama at: {OLLAMA_BASE_URL}")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        models_data = response.json()
        models = [model['name'] for model in models_data.get('models', [])]
        logging.info(f"Found Ollama models: {', '.join(models) if models else 'None'}")
        return models
    except requests.exceptions.ConnectionError:
        st.sidebar.error(f"Could not connect to Ollama server at {OLLAMA_BASE_URL}. Please ensure Ollama is running.")
        return []
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching Ollama models: {e}")
        return []


# --- FAISS Loading Function ---
@st.cache_resource
def load_faiss_and_embeddings():
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        st.error("GOOGLE_API_KEY is empty or placeholder. Please replace 'YOUR_GOOGLE_API_KEY' with your actual API key.")
        st.stop()

    logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL_NAME}'...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        # Test a small embedding to catch API key issues early
        test = embeddings.embed_query("test query")
        print(f"Test embedding successful: {len(test)}...")
        logging.info("Embedding model initialized and tested successfully.")
    except Exception as e:
        st.error(f"ERROR: Could not initialize GoogleGenerativeAIEmbeddings. Please ensure your GOOGLE_API_KEY is correct and valid. Details: {e}")
        st.stop()

    logging.info(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"FAISS index loaded successfully. Number of vectors: {db.index.ntotal if hasattr(db.index, 'ntotal') else 'N/A'}")
        print(f"FAISS index loaded successfully. Number of vectors: {db.index.ntotal if hasattr(db.index, 'ntotal') else 'N/A'}")
        return db, embeddings
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}. Please ensure '{FAISS_INDEX_PATH}' directory exists and is valid, and run ingest_data.py if necessary.")
        st.stop()


@st.cache_resource
def get_memory():
    """Returns a new ConversationBufferMemory instance."""
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")


# --- Document Prompt Template ---

document_prompt_template = PromptTemplate(
    template="""
    --- Retrieved Document ---
    Course Title: {title}
    Content: {page_content}
    Metadata:
      URL: {url}
      Mentor: {mentor}
      Language: {language}
      Location: {location}
      Category: {category}
      ECTS: {ects}
    --------------------------
    """,
    # The input_variables should match the metadata keys from your ingested documents
    input_variables=["page_content", "title", "url", "mentor", "language", "location", "category", "ects"]
)


COMBINE_PROMPT = PromptTemplate(
    template="""
    You are "Open Campus Bot", a helpful and knowledgeable educational assistant for OpenCampus.
    Your primary goal is to provide accurate and structured course information based ONLY on the provided context.
    The context consists of retrieved documents, each clearly separated, containing course content and specific metadata fields.

    ---
    Chat History (Examine this carefully to understand the conversation flow):
    {chat_history}
    ---
    Context about OpenCampus courses (use this to extract information):
    {context}
    ---
    Current User Question: {question}
    ---

    *Instructions for EduBot's Response (Follow these rules strictly based on the user's intent and available context):*

    1.  *Analyze User Intent:* Based on the 'Current User Question' and the 'Chat History', determine if the user is asking for:
        * *A) Initial / General Course List:* This applies if the 'Chat History' is empty (new conversation) OR if the 'Current User Question' is a broad inquiry about courses (e.g., "What courses do you offer?", "List all courses?", "Courses in Tech & Coding?"). For this intent, you MUST provide a list of *all relevant course titles from the provided context*.
        * *B) Detailed Information for a Specific Course:* This applies if the 'Current User Question' explicitly mentions a specific course name and asks for "more details," "schedule," "full information," etc., OR if the 'Chat History' shows a previous general course list was provided and the 'Current User Question' is a clear follow-up requesting details for one of those courses.

    2.  *Generate Response Based on Inferred Intent:*

        * *IF Intent is (A) Initial / General Course List:*
            * **From the 'Context about OpenCampus courses', extract the 'Course Title' (which is the 'Title' from metadata) for EVERY course mentioned in the context that is relevant to the question.**
            * Present these extracted 'Course Titles' as a clear, easy-to-read bulleted list.
            * *STRICTLY DO NOT* include descriptions, mentors, day/time, information, sessions, ECTS, location, language, or URLs in this initial response.
            * *Crucially, always end this response by asking a clear, polite question:* "Would you like more detailed information about any of these courses?"

        * *IF Intent is (B) Detailed Information for a Specific Course:*
            * First, identify the *specific course name* the user is asking about in detail from the 'Current User Question' or 'Chat History'.
            * For that identified course, extract and present *ALL available details* from the context in a structured, comprehensive format. This must include:
                -   *Course Name:* (from 'Title' metadata)
                -   *Course Description:* (from 'Description' in content or summarize from page_content)
                -   *Course Mentor(s):* (from 'Mentor' metadata)
                -   *Course Language:* (from 'Language' metadata)
                -   *Course Location:* (from 'Location' metadata)
                -   *ECTS Credits:* (from 'ECTS' metadata)
                -   *Course Day & Time:* (from 'Day/Time' in content, e.g., "THURSDAY 18:00 - 19:30")
                -   *Full Course Information:* (from 'Information' section in content)
                -   *Course Sessions:* (from 'Course Sessions' section in content - list each session's date, time, topic, and location clearly)
            * *After providing ALL these details, conclude by providing the Course URL* for the most comprehensive information. For example: "For the most comprehensive details and to register, please visit the official course page: [Course URL from metadata]"
            * *Avoid Redundancy:* Ensure you don't repeat information already given in a prior initial list for the same course; focus on the new, detailed information.

    5.  *Fallback / No Information:*
        * If no relevant course information is found in the context for the 'Current User Question', or if the question is completely out of scope for OpenCampus courses, politely state: "I apologize, but I don't have enough information about that specific topic from our current OpenCampus course materials. Can I help with anything else about OpenCampus courses?" Do not invent answers.
        * For general greetings (e.g., "Hi", "Hello"), respond appropriately as EduBot and offer to help with course information.

    ---
    EduBot's Answer:
    """,
    input_variables=["chat_history", "context", "question"]
)

# --- Tool Input Schemas ---
class CompareCoursesInput(BaseModel):
    course1_name: str = Field(description="The full name or a clear identifier of the first course to compare.")
    course2_name: str = Field(description="The full name or a clear identifier of the second course to compare.")

# --- TOOL SELECTION PROMPT ---
TOOL_SELECTION_PROMPT = PromptTemplate(
    template="""
    You are an AI assistant whose sole purpose is to determine the user's intent and select the appropriate tool.
    Based on the user's query, output one of the following JSON structures.
    Ensure the JSON is well-formed and can be directly parsed.

    1.  To list ALL general course categories or a broad overview of courses (e.g., "What do you offer?", "List all courses?", "What kind of programs are there?", 'provide courses'):
        ```json
        {{ "tool": "list_categories" }}
        ```

    2.  To list ALL courses *within a specific, clearly named category* (e.g., "Show me courses in Tech & Coding", "List all programs under Business & Startup", "Courses for Creative, Social & Sustainable"):
        ```json
        {{ "tool": "list_category_courses", "category": "Identified Category Name" }}
        ```
        Examples of Identified Category Names (use these exact names if matched):
        "Creative, Social & Sustainable", "Business & Startup", "Tech & Coding", "Degrees programs".
        This tool should ONLY be selected if the user clearly asks for *all* courses *within* one of these defined categories, without specifying a sub-topic or a quality like 'fundamental' or 'beginner'.

    3.  To get DETAILED information about a SPECIFIC COURSE (e.g., "tell me about Python Programming", "details for Web Development", "schedule of Machine Learning"):
        ```json
        {{ "tool": "get_course_details", "course_name": "Identified Course Name" }}
        ```
        Extract the exact or most likely course name the user is asking about details for. Be precise. If unsure or if no specific course is named, output "general_qa".

    4.  To COMPARE two or more SPECIFIC COURSES side-by-side (e.g., "Compare Python Programming and Web Development", "How is Machine Learning different from Data Science?", "Contrast R and Python"):
        ```json
        {{ "tool": "compare_courses", "course1_name": "Identified Course 1 Name", "course2_name": "Identified Course 2 Name" }}
        ```
        You MUST identify at least two distinct course names from the user's query for this tool. If only one or no course names are clearly identified for comparison, output "general_qa".

    5.  For any other query, including:
        * Questions about *specific topics* not explicitly a category (e.g., "electronics", "Arduino", "R programming", "web design", "AI", "robotics").
        * Questions asking for courses with a specific *quality* (e.g., "fundamental courses", "beginner-friendly", "advanced", "project-based", "introductory").
        * General queries, greetings, out-of-scope questions, or if intent is unclear for tools 2, 3, or 4.
        ```json
        {{ "tool": "general_qa" }}
        ```

    ---
    User Query: {query}
    ---
    Your Output (JSON only):
    """,
    input_variables=["query"]
)


@st.cache_resource(hash_funcs={ChatOllama: lambda _: None, ChatGoogleGenerativeAI: lambda _: None})
def initialize_rag_chain(_faiss_db, selected_llm_model_name):
    
    logging.info(f"Initializing RAG chain with model: {selected_llm_model_name}")
    llm = None
    memory = get_memory()

    if selected_llm_model_name == "gemini-2.0-flash":
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
            st.error("Google API Key is required for Google Generative AI models. Cannot initialize LLM.")
            return None
        try:
            llm = ChatGoogleGenerativeAI(model=selected_llm_model_name, temperature=0.2, google_api_key=GOOGLE_API_KEY)
            logging.info(f"Successfully initialized Google model: {selected_llm_model_name}")
        except Exception as e:
            st.error(f"Error initializing Google model '{selected_llm_model_name}': {e}. Please check your API key and model availability.")
            return None
    else: # Assume it's an Ollama model
        if not selected_llm_model_name:
            st.warning("No Ollama model selected or available. Cannot initialize LLM.")
            return None
        try:
            llm = ChatOllama(model=selected_llm_model_name, temperature=0.2, base_url=OLLAMA_BASE_URL)
            # A small invoke to ensure Ollama connection is live
            llm.invoke("Hello", config={"stream": False}) # Simple test call
            logging.info(f"Successfully initialized Ollama model: {selected_llm_model_name}")
        except Exception as e:
            st.error(f"Error initializing Ollama model '{selected_llm_model_name}': {e}. Is Ollama server running and model available? Check terminal where Ollama server runs for more details.")
            return None

    if llm:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=_faiss_db.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 documents for context
            memory=memory,
            combine_docs_chain_kwargs={"prompt": COMBINE_PROMPT, "document_prompt": document_prompt_template},
            return_source_documents=True, # Important for debugging and potential source display
            get_chat_history=lambda h: h # Function to format chat history for the prompt
        )
        logging.info("RAG chain with conversational memory successfully created.")
        return qa_chain
    logging.warning("LLM was not initialized, returning None for qa_chain.")
    return None


@st.cache_resource(hash_funcs={ChatOllama: lambda _: None, ChatGoogleGenerativeAI: lambda _: None})
def get_tool_selection_llm(selected_llm_model_name):

    logging.info(f"Initializing Tool Selection LLM with model: {selected_llm_model_name}")
    try:
        # Use a low temperature for more deterministic output for routing
        if selected_llm_model_name == "gemini-2.0-flash":
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
        else:
            # For Ollama, you might use a smaller, faster model if available (e.g., "llama3:8b")
            return ChatOllama(model=selected_llm_model_name, temperature=0.0, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        logging.error(f"Error initializing tool selection LLM: {e}")
        st.error(f"Could not initialize tool selection LLM: {e}. Please check model availability/API key.")
        return None


# --- Helper functions for compare_courses tool ---
def get_course_document_by_name_or_fuzzy(query_name: str, faiss_db: FAISS, llm_for_routing: Any) -> (Document | None, str | None): # type: ignore
    
    logging.info(f"Attempting to find document for course query: '{query_name}'")
    
    # Perform a semantic search on the FAISS DB
    retrieved_docs = faiss_db.similarity_search(query_name, k=5) # Increased k for better candidate pool

    if not retrieved_docs:
        logging.info(f"No documents retrieved for '{query_name}'.")
        return None, None

    selection_prompt = PromptTemplate(
        template="""
        The user asked for information about a course. You have retrieved the following course titles. Based on these and the user's query, identify the most likely course.

        Example:
        User query: "python basics"
        Available Titles:
        - Python Programming: Beginner to Practitioner
        - Advanced Python
        - Introduction to Data Science
        Response: Python Programming: Beginner to Practitioner

        Example:
        User query: "machine learning"
        Available Titles:
        - Machine Learning with TensorFlow
        - Deep Learning Fundamentals
        Response: Machine Learning with TensorFlow

        Example:
        User query: "history of computers"
        Available Titles:
        - Python Programming: Beginner to Practitioner
        - Advanced Python
        Response: NONE

        ---
        Current User Query: "{user_query}"
        Retrieved Course Titles:
        {documents_titles_formatted}

        Based on the current user query and the retrieved titles, which of these is the *most likely exact course name* the user is referring to?
        If none are a strong match, or if the user's query seems to be about a general topic rather than a specific course, respond with "NONE".
        Otherwise, respond ONLY with the exact course title from the "Retrieved Course Titles" list.
        Response:
        """,
        input_variables=["user_query", "documents_titles_formatted"]
    )

    documents_titles_formatted = "\n".join([f"- {d.metadata.get('title', 'Untitled')}" for d in retrieved_docs])
    
    llm_selection_query = selection_prompt.format(
        user_query=query_name,
        documents_titles_formatted=documents_titles_formatted
    )
    
    logging.info(f"LLM Course Selection Prompt for fuzzy match:\n{llm_selection_query}")
    selected_title_response = llm_for_routing.invoke(llm_selection_query).content.strip()
    logging.info(f"LLM Course Selection Response for fuzzy match: '{selected_title_response}'")

    if selected_title_response.upper() == "NONE":
        logging.info(f"LLM determined no strong course match for '{query_name}'.")
        return None, None
    
    # Find the document corresponding to the selected title
    for doc in retrieved_docs:
        if doc.metadata.get('title', '').strip() == selected_title_response:
            logging.info(f"Found specific document for '{query_name}': {doc.metadata.get('title')}")
            return doc, doc.metadata.get('title', selected_title_response) # Return the exact title
    
    # Fallback if LLM selected a title not in the top k (shouldn't happen often)
    logging.warning(f"LLM selected '{selected_title_response}' but it was not found in retrieved docs for '{query_name}'.")
    return None, None


def extract_structured_course_data(doc: Document) -> Dict[str, Any]:
    """Extracts relevant structured data from a LangChain Document for comparison."""
    if not doc:
        return {}

    data = {
        "title": doc.metadata.get("title", "N/A"),
        "url": doc.metadata.get("url", "#"),
        "mentor": doc.metadata.get("mentor", "N/A"),
        "language": doc.metadata.get("language", "N/A"),
        "location": doc.metadata.get("location", "N/A"),
        "category": doc.metadata.get("category", "N/A"),
        "ects": doc.metadata.get("ects", "N/A"),
    }

    content = doc.page_content
    desc_match = re.search(r"Description:\s*(.*?)(?=\n(?:What you will learn|Information|Course Sessions|Overview):|\n{2,}|\Z)", content, re.DOTALL | re.IGNORECASE)
    data["description"] = desc_match.group(1).strip() if desc_match else "No description available."
    if data["description"] == "": # Handle cases where description might be empty but match was found
        data["description"] = "No description available."


    # Look for "What you will learn:" followed by content, until the next major section heading
    learn_match = re.search(r"What you will learn:\s*(.*?)(?=\n(?:Information|Course Sessions|Overview|Description):|\n{2,}|\Z)", content, re.DOTALL | re.IGNORECASE)
    if learn_match:
        learnings_raw = learn_match.group(1).strip()
        # Split by lines, filter out empty ones, and clean up bullet points/hyphens
        data["learnings"] = [re.sub(r"^[-\*\d\.]+\s*", "", item).strip() for item in learnings_raw.split('\n') if item.strip()]
        if not data["learnings"]: # If list is empty after stripping, set default
            data["learnings"] = ["No specific learning outcomes listed."]
    else:
        data["learnings"] = ["No specific learning outcomes listed."]
    
    return data

# --- Compare Courses Tool Function ---

def compare_courses(course1_name: str, course2_name: str, faiss_db: FAISS, llm_for_routing: Any) -> str:
   
    logging.info(f"Starting comparison for: '{course1_name}' vs '{course2_name}'")

    # Step 1: Find the actual course documents using fuzzy matching
    doc1, identified_name1 = get_course_document_by_name_or_fuzzy(course1_name, faiss_db, llm_for_routing)
    doc2, identified_name2 = get_course_document_by_name_or_fuzzy(course2_name, faiss_db, llm_for_routing)

    # Step 2: Handle "course not found" scenarios
    not_found_messages = []
    if not doc1:
        not_found_messages.append(f"I couldn't find a course similar to '{course1_name}'.")
    if not doc2:
        not_found_messages.append(f"I couldn't find a course similar to '{course2_name}'.")

    if not_found_messages:
        return (
            "I apologize, but " + " and ".join(not_found_messages) +
            "\n\nPlease see the available courses in the sidebar or ask 'What courses do you offer?'"
            " if you'd like a general list."
        )

    # Step 3: Extract structured data from found documents
    course1_data = extract_structured_course_data(doc1)
    course2_data = extract_structured_course_data(doc2)

    # Ensure names are accurate for the table headers
    course1_display_name = course1_data.get("title", identified_name1)
    course2_display_name = course2_data.get("title", identified_name2)

    # Step 4: Category check and warning (Simplified: no multi-turn confirmation for now)
    category1 = course1_data.get("category", "N/A")
    category2 = course2_data.get("category", "N/A")

    category_mismatch_warning = ""
    if category1 != category2:
        category_mismatch_warning = (
            f"Please note: '{course1_display_name}' is in the '{category1}' category, "
            f"while '{course2_display_name}' is in the '{category2}' category.\n\n"
        )
        logging.info(f"Category mismatch detected: {category1} vs {category2}")

    # Step 5: Use LLM to format the comparison table
    
    comparison_formatter_prompt = PromptTemplate(
        template="""
        You are an expert at comparing educational courses.
        Given the following structured information for two courses, create a concise, side-by-side Markdown table comparison.
        Only include the following features: "Course Name", "Short Description", "Key Learnings", "ECTS Credits", "Language", and "More Information".
        Do NOT include Mentor or Location.
        Summarize "Short Description" and "Key Learnings" from the provided data into a few clear bullet points if they are long.
        Ensure the table headers are the course names. Provide the URL as a clickable Markdown link like this: `[Visit Course Page]({{URL}})`.

        Course 1 Details:
        - Title: {c1_title}
        - Description: {c1_description}
        - Learnings: {c1_learnings}
        - Category: {c1_category}
        - ECTS: {c1_ects}
        - Language: {c1_language}
        - URL: {c1_url}

        Course 2 Details:
        - Title: {c2_title}
        - Description: {c2_description}
        - Learnings: {c2_learnings}
        - Category: {c2_category}
        - ECTS: {c2_ects}
        - Language: {c2_language}
        - URL: {c2_url}

        ---
        Comparison Table:
        """,
        input_variables=[
            "c1_title", "c1_description", "c1_learnings", "c1_category", "c1_ects", "c1_language", "c1_url",
            "c2_title", "c2_description", "c2_learnings", "c2_category", "c2_ects", "c2_language", "c2_url"
        ]
    )

    # Convert learning list to string for prompt
    c1_learnings_str = "\n".join(course1_data.get("learnings", []))
    c2_learnings_str = "\n".join(course2_data.get("learnings", []))

    comparison_llm_chain = comparison_formatter_prompt | llm_for_routing | StrOutputParser() # Use tool_selection_llm for formatting

    formatted_table = comparison_llm_chain.invoke({
        "c1_title": course1_data.get("title"),
        "c1_description": course1_data.get("description"),
        "c1_learnings": c1_learnings_str,
        "c1_category": course1_data.get("category"),
        "c1_ects": course1_data.get("ects"),
        "c1_language": course1_data.get("language"),
        "c1_url": course1_data.get("url"),
        "c2_title": course2_data.get("title"),
        "c2_description": course2_data.get("description"),
        "c2_learnings": c2_learnings_str,
        "c2_category": course2_data.get("category"),
        "c2_ects": course2_data.get("ects"),
        "c2_language": course2_data.get("language"),
        "c2_url": course2_data.get("url")
    })

    return category_mismatch_warning + formatted_table


# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="EduBot: OpenCampus Course Assistant", layout="wide")

    # --- Initialize ALL session state variables at the very top of main() ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initial_greeting_done" not in st.session_state:
        st.session_state.initial_greeting_done = False
    if "all_categories" not in st.session_state:
        st.session_state.all_categories = [
            "Creative, Social & Sustainable",
            "Business & Startup",
            "Tech & Coding",
            "Degrees programs"
        ]
    # --- END Session State Initialisation ---

    # Sidebar for model selection
    with st.sidebar:
        # Use columns to place logo and text beside each other
        col_logo, col_text = st.columns([1, 4]) # Adjusted ratios

        with col_logo:
            st.image("opencampus_logo.png", width=40) # Smaller width

        with col_text:
            # Changed to h2, and added negative margin-left for closer placement
            st.markdown("<h2 style='margin-top: -8px; margin-bottom: 0px; margin-left: -10px;'>opencampus</h2>", unsafe_allow_html=True)

        st.header("Model Selection") # Model selection heading in sidebar

        ollama_models = get_ollama_models()
        google_models = ["gemini-2.0-flash"]
        available_models = ollama_models + google_models

        if not available_models:
            st.error("No LLM models are available. Please ensure Ollama is running with models pulled, or your Google API Key is set.")
            st.stop()

        default_model = None
        default_index = 0

        if "gemma3:12b" in ollama_models:
            default_model = "gemma3:12b"
        elif ollama_models:
            default_model = ollama_models[0]
        elif "gemini-2.0-flash" in google_models:
            default_model = "gemini-2.0-flash"

        try:
            if default_model:
                default_index = available_models.index(default_model)
        except ValueError:
            default_index = 0

        selected_llm_model = st.selectbox(
            "Choose your LLM:",
            options=available_models,
            index=default_index
        )
        logging.info(f"Selected LLM Model: {selected_llm_model}")

        if not selected_llm_model:
            st.warning("No LLM model selected. Please select a model to proceed.")
            st.stop()
    
    # Main content area
    st.markdown("""
        <style>
        h3 {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Display Initial Welcome Message ---
    if not st.session_state.initial_greeting_done:
        welcome_message = "Hello, welcome to OpenCampus! How can I help you with courses today?"
        st.chat_message("assistant").markdown(welcome_message)
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        st.session_state.initial_greeting_done = True
        # Add some empty space to push the input field down
        for _ in range(8):
            st.empty()
    else:
        
        if not any(msg["role"] == "user" for msg in st.session_state.messages):
            for _ in range(8):
                st.empty()

    # Load FAISS and embeddings (cached)
    faiss_db, embeddings = load_faiss_and_embeddings()

    # Initialize RAG chain
    qa_chain = initialize_rag_chain(faiss_db, selected_llm_model)
    memory = get_memory() # Get the memory instance for the chain


    # Initialize Tool Selection LLM (uses the same selected model, for simplicity)
    tool_selection_llm = get_tool_selection_llm(selected_llm_model)


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Avoid re-displaying the initial greeting after the first run if it's the only message
        if message["content"] == "Hello, welcome to OpenCampus! How can I help you with courses today?" and \
           message["role"] == "assistant" and \
           len(st.session_state.messages) == 1 and \
           st.session_state.initial_greeting_done: # Ensure it's the very first message
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input
    if prompt := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if qa_chain and tool_selection_llm:
                with st.spinner("Thinking..."):
                    llm_answer = ""
                    
                    try:
                        # Step 1: Use LLM to determine the initial tool/intent
                        tool_prompt_formatted = TOOL_SELECTION_PROMPT.format(query=prompt)
                        logging.info(f"Tool Selection Prompt:\n{tool_prompt_formatted}")
                        
                        tool_response_text = tool_selection_llm.invoke(tool_prompt_formatted).content
                        logging.info(f"Tool Selection LLM Raw Response: {tool_response_text}")

                        action = {"tool": "general_qa"} # Default action if parsing fails
                        try:
                            # Clean the response to ensure valid JSON (remove markdown code blocks)
                            clean_response = tool_response_text.strip()
                            if clean_response.startswith("```json"):
                                clean_response = clean_response[len("```json"):].strip()
                            if clean_response.endswith("```"):
                                clean_response = clean_response[:-len("```")].strip()
                            
                            action = json.loads(clean_response)
                        except json.JSONDecodeError:
                            logging.warning(f"Tool selection LLM did not return valid JSON. Falling back to general_qa. Raw response: '{tool_response_text}'")
                            
                        logging.info(f"Determined Initial Action: {action}")

                        # --- Step 2: Execute action based on determined tool with a unified fallback ---
                        current_action_tool = action["tool"]
                        
                        # Use a flag to track if the answer has been set by a specific tool
                        answer_set = False

                        if current_action_tool == "list_categories":
                            llm_answer = ("OpenCampus offers courses related to these categories:\n\n" +
                                        "\n".join([f"* {cat}" for cat in st.session_state.all_categories]) +
                                        "\n\nPlease tell me which category you are interested in!")
                            answer_set = True

                        elif current_action_tool == "list_category_courses":
                            category_name = action.get("category")
                            
                            if category_name and category_name in st.session_state.all_categories:
                                temp_prompt_for_llm = f"List all courses available in the '{category_name}' category."
                                logging.info(f"Invoking QA chain for category courses: {temp_prompt_for_llm}")
                                
                                response = qa_chain.invoke({"question": temp_prompt_for_llm})
                                category_list_answer = response.get("answer", "No answer generated.")
                                
                                # --- Refinement Logic: Check if specific query intent was met by category list ---
                                specific_query_keywords = [
                                    "arduino", "electronics", "microcontroller", "robotics", "raspberry pi", "circuit",
                                    "web development", "python programming", "data science", "machine learning", "maker"
                                ]
                                
                                found_specific_keyword_in_prompt = False
                                for kw in specific_query_keywords: # Iterating over specific_query_keywords, not prompt.lower()
                                    if kw in prompt.lower():
                                        found_specific_keyword_in_prompt = True
                                        break
                                
                                # Check if any specific keyword from the original prompt is *not* in the category list answer
                                if found_specific_keyword_in_prompt and \
                                   not any(kw in category_list_answer.lower() for kw in specific_query_keywords):
                                    
                                    logging.info(f"Original prompt contained specific keyword(s) not directly addressed by category list. Directly invoking general_qa. Keyword(s): {', '.join([kw for kw in specific_query_keywords if kw in prompt.lower()])}")
                                    # This path will fall through to the 'if not answer_set' block for general_qa
                                    pass 
                                else:
                                    llm_answer = category_list_answer # Use the category list answer as it seems relevant
                                    answer_set = True
                                    
                            else: # Invalid category name
                                llm_answer = (f"I couldn't identify a valid category from your request. "
                                            f"The available categories are: {', '.join(st.session_state.all_categories)}. "
                                            "Can I help with something else about OpenCampus courses?")
                                logging.warning(f"Invalid category identified by tool LLM: {category_name}. User prompt: {prompt}")
                                answer_set = True # Answer is set, no need for general_qa fallback

                        elif current_action_tool == "get_course_details":
                            course_name = action.get("course_name")
                            if course_name:
                                logging.info(f"Invoking QA chain for course details for: '{course_name}' with original prompt.")
                                response = qa_chain.invoke({"question": prompt})
                                llm_answer = response.get("answer", "No answer generated.")
                                answer_set = True
                            else:
                                llm_answer = ("I couldn't identify a specific course to get details for. "
                                              "Please specify the course name (e.g., 'tell me about Python Programming'). "
                                              "Can I help with something else about OpenCampus courses?")
                                logging.warning(f"No specific course name identified by tool LLM for details. User prompt: {prompt}")
                                answer_set = True # Answer is set, no need for general_qa fallback

                        elif current_action_tool == "compare_courses":
                            course1_name = action.get("course1_name")
                            course2_name = action.get("course2_name")

                            if course1_name and course2_name:
                                logging.info(f"Invoking compare_courses tool for: '{course1_name}' and '{course2_name}'")
                                # Pass faiss_db and tool_selection_llm (which is suitable for formatting tasks)
                                llm_answer = compare_courses(course1_name, course2_name, faiss_db, tool_selection_llm)
                                answer_set = True
                            else:
                                llm_answer = ("I need two course names to compare. Could you please specify them (e.g., 'compare course A and course B')?")
                                logging.warning(f"Compare courses tool called without two course names. Action: {action}. User prompt: {prompt}")
                                answer_set = True # Answer is set, no need for general_qa fallback

                        # --- General QA / Fallback Logic ---
                        if not answer_set:
                            # If the current_action_tool was anything other than "general_qa" initially,
                            # it means we've fallen through/rerouted here.
                            if current_action_tool != "general_qa":
                                logging.info(f"Falling back to general_qa after initial tool '{current_action_tool}' did not yield a direct answer.")

                            out_of_scope_keywords = ["translate", "summarize", "define", "what is", "how to", "convert", "calculation", "meaning of"]
                            is_out_of_scope_query = False
                            if any(keyword in prompt.lower() for keyword in out_of_scope_keywords):
                                course_attribute_keywords = ["course description", "mentor", "language", "location", "ects", "sessions", "information"]
                                if not any(attr_kw in prompt.lower() for attr_kw in course_attribute_keywords):
                                    is_out_of_scope_query = True

                            if is_out_of_scope_query:
                                llm_answer = ("Sorry, I couldn't fulfill that request because it's currently outside my scope. "
                                              "I am still in development and primarily focused on providing information about OpenCampus courses. "
                                              "Perhaps I can be more helpful in my next version for tasks like that!")
                            else:
                                logging.info("Invoking general QA chain (initial general_qa or final fallback).")
                                response = qa_chain.invoke({"question": prompt}) # Use the original prompt for general semantic search
                                llm_answer = response.get("answer", "No answer generated.")
                                
                                # --- DEBUGGING OUTPUT TO CONSOLE (Retrieved Documents) ---
                                if response and response.get("source_documents"):
                                    source_documents = response.get("source_documents", [])
                                    if source_documents:
                                        logging.info("--- Retrieved Documents (for Debug - General QA Fallback) ---")
                                        for i, doc in enumerate(source_documents):
                                            logging.info(f"Document {i+1}:")
                                            logging.info(f"  Metadata: {json.dumps(doc.metadata, indent=2)}")
                                            logging.info(f"  Page Content:\n{doc.page_content[:200]}...")
                                        logging.info("--- End Retrieved Documents ---")
                                # --- END DEBUGGING OUTPUT ---
                            answer_set = True # Ensure answer_set is True after this block
                            
                        # If for some reason answer_set is still False, this would be an unexpected case.
                        if not answer_set:
                            llm_answer = "An unexpected issue occurred and I couldn't generate a response. Please try again."
                            logging.error(f"Logic error: llm_answer was not set. Final tool state: {current_action_tool}. Prompt: {prompt}")

                    except Exception as e:
                        st.error(f"Error during tool selection or RAG chain invocation: {e}")
                        logging.error(f"Error in main processing loop: {e}", exc_info=True)
                        llm_answer = f"Sorry, an unexpected internal error occurred: {e}. Please try again or refresh the page."
                
                st.markdown(llm_answer)
                st.session_state.messages.append({"role": "assistant", "content": llm_answer})

            else:
                st.warning("Chatbot is not fully initialized. Please ensure your LLM configuration is valid and models are selected/loaded successfully in the sidebar.")
                st.session_state.messages.append({"role": "assistant", "content": "Chatbot not initialized. Please check LLM configuration."})


if __name__ == "__main__":
    main()
