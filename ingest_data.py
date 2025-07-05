# ingest_data.py

import os
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# --- Configuration ---
DOMAIN_DATA_DIR = "domain_data"
FAISS_INDEX_PATH = "faiss_index" # Directory to save the FAISS index
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Recommended for Google models

# --- IMPORTANT: Your Google API Key ---
# Replace 'YOUR_GOOGLE_API_KEY' with your actual API key
# For production, consider environment variables or secret management.
GOOGLE_API_KEY = "AIzaSyAdab1EdwNZtZ8yQhfwHvK3V6Ir-YDhihQ"  # Set your Google API key here 

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(directory_path: str) -> list[dict]:
    """
    Loads all JSON files from a specified directory.
    Injects a 'category' key into each course dictionary based on the filename.
    """
    all_data = []
    if not os.path.exists(directory_path):
        logging.error(f"Error: Directory '{directory_path}' not found.")
        logging.error(f"Please create a '{directory_path}' folder and place your JSON files inside.")
        return all_data

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            # Extract category from filename (e.g., 'programming_courses.json' -> 'programming_courses')
            category = os.path.splitext(filename)[0]
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item['category'] = category # Add category to each item
                            all_data.append(item)
                    else: # Assuming a single course object in the file
                        if isinstance(data, dict):
                            data['category'] = category # Add category to the single item
                        all_data.append(data)
                logging.info(f"Loaded data from: {filename} (Category: {category})")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading {filename}: {e}")
    return all_data

def process_course_data(course: dict) -> Document:
    """
    Combines relevant fields from a course dictionary into a single text string
    and extracts metadata. Includes the extracted 'category' in content and metadata.
    Returns a LangChain Document object.
    """
    details = course.get('details', {})
    
    # Combine what_we_will_learn and mentor lists into strings, handling missing keys gracefully
    what_we_will_learn_str = ", ".join(details.get('what_we_will_learn', []))
    mentor_str = ", ".join(details.get('mentor', []))

    # Construct the content string from relevant fields
    # Prioritize category to give it prominence in the content
    content_parts = []
    if course.get('category'):
        content_parts.append(f"Category: {course['category'].replace('_', ' ').title()}") # Make it more readable
    if course.get('title'):
        content_parts.append(f"Title: {course['title']}")
    if details.get('description'):
        content_parts.append(f"Description: {details['description']}")
    if what_we_will_learn_str:
        content_parts.append(f"What you will learn: {what_we_will_learn_str}")
    if details.get('information'):
        content_parts.append(f"Information: {details['information']}")
    if mentor_str:
        content_parts.append(f"Mentor(s): {mentor_str}")
    if details.get('day_time_summary'):
        content_parts.append(f"Day/Time: {details['day_time_summary']}")
    if details.get('location'):
        content_parts.append(f"Location: {details['location']}")
    if details.get('language'):
        content_parts.append(f"Language: {details['language']}")
    if details.get('ects'):
        content_parts.append(f"ECTS: {details['ects']}")
    
    # Join parts with double newlines for better readability for the LLM
    page_content = "\n\n".join(content_parts)

    # Prepare metadata for the document
    metadata = {
        "title": course.get('title'),
        "url": course.get('url'),
        "mentor": mentor_str,
        "language": details.get('language'),
        "location": details.get('location'),
        "ects": details.get('ects'),
        "category": course.get('category') # Include category in metadata
    }
    
    # Remove metadata entries that are None or empty strings to keep it clean
    metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

    return Document(page_content=page_content, metadata=metadata)

def main():
    # 1. Ensure the domain_data folder exists
    if not os.path.exists(DOMAIN_DATA_DIR):
        logging.error(f"The directory '{DOMAIN_DATA_DIR}' does not exist.")
        logging.error("Please create this folder in your project directory and place your JSON data files inside it.")
        logging.info("Exiting ingestion process.")
        return

    # 2. Load all JSON data from the specified directory
    logging.info(f"\nLoading data from '{DOMAIN_DATA_DIR}'...")
    raw_courses_data = load_json_data(DOMAIN_DATA_DIR)
    if not raw_courses_data:
        logging.warning("No JSON files found or data loaded. Please ensure your JSON files are in the 'domain_data' folder.")
        logging.info("Exiting.")
        return

    # 3. Process raw data into LangChain Document objects
    logging.info(f"Processing {len(raw_courses_data)} courses into initial LangChain documents...")
    all_base_documents = []
    for course_data in raw_courses_data:
        # Check if 'details' key exists and is a dictionary, otherwise skip
        if isinstance(course_data, dict) and 'details' in course_data and isinstance(course_data['details'], dict):
            all_base_documents.append(process_course_data(course_data))
        else:
            logging.warning(f"Skipping malformed course data (missing 'details' or not a dict): {course_data.get('title', 'Unknown Title')}")


    if not all_base_documents:
        logging.warning("No valid course documents were generated. Check your JSON structure.")
        return

    # 4. Split documents into smaller, overlapping chunks
    logging.info("Splitting documents into smaller, overlapping chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk (Experiment with 500-1500)
        chunk_overlap=200,    # Overlap between chunks (Experiment with 100-300)
        length_function=len,  # Use character length
        # Consider specific separators if your content has predictable structures
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    # The splitter will intelligently break `page_content` and preserve `metadata`
    chunked_documents = text_splitter.split_documents(all_base_documents)
    logging.info(f"Original documents: {len(all_base_documents)}, Created chunks: {len(chunked_documents)}")

    if not chunked_documents:
        logging.warning("No chunks were created after splitting. Check your input data or splitter settings.")
        return

    # 5. Initialize Google Gemini Embeddings
    logging.info(f"Initializing embedding model: '{EMBEDDING_MODEL_NAME}' with provided API key...")
    try:
        # Pass the API key directly to the constructor
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        # Test a small embedding to catch API key issues early
        _ = embeddings.embed_query("test query")
        logging.info("Embedding model initialized and tested successfully.")
    except Exception as e:
        logging.error(f"ERROR: Could not initialize GoogleGenerativeAIEmbeddings.")
        logging.error(f"Please ensure your GOOGLE_API_KEY variable is correctly set and valid in the code.")
        logging.error(f"Details: {e}")
        return

    # 6. Create and Save FAISS Index
    logging.info("Creating FAISS index from document chunks and embeddings...")
    try:
        db = FAISS.from_documents(chunked_documents, embeddings)
    except Exception as e:
        logging.error(f"ERROR: Failed to create FAISS index. This might be due to issues with embeddings or data.")
        logging.error(f"Details: {e}")
        return
    
    # Ensure the directory for FAISS index exists
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)
    logging.info(f"FAISS index created and saved successfully to '{FAISS_INDEX_PATH}'")
    
    # --- Verification Step ---
    logging.info("Verifying FAISS index...")
    try:
        # Load the database to verify it exists and contains vectors
        # allow_dangerous_deserialization=True is necessary when loading FAISS index with custom LangChain classes (like Documents)
        loaded_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        if hasattr(loaded_db, 'index') and loaded_db.index.ntotal > 0:
            logging.info(f"Verified: FAISS index loaded successfully with {loaded_db.index.ntotal} vectors.")
        else:
            logging.warning("Warning: FAISS index loaded but appears to be empty or corrupted.")
    except Exception as e:
        logging.error(f"Verification Error: Could not load or verify FAISS index from '{FAISS_INDEX_PATH}'. Details: {e}")

    logging.info("\nIngestion complete!")


if __name__ == "__main__":
    main()