# ingest_data.py

import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# --- Configuration ---
DOMAIN_DATA_DIR = "domain_data"
FAISS_INDEX_PATH = "faiss_index" # Directory to save the FAISS index
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Recommended for Google models

def load_json_data(directory_path: str) -> list[dict]:
    """Loads all JSON files from a specified directory."""
    all_data = []
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        print(f"Please create a '{directory_path}' folder and place your JSON files inside.")
        return all_data

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Assuming each JSON file contains a list of course objects, or a single course object
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                print(f"Loaded data from: {filename}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while loading {filename}: {e}")
    return all_data

def process_course_data(course: dict) -> Document:
    """
    Combines relevant fields from a course dictionary into a single text string
    and extracts metadata. Returns a LangChain Document object.
    """
    details = course.get('details', {})
    
    # Combine what_we_will_learn and mentor lists into strings, handling missing keys gracefully
    what_we_will_learn_str = ", ".join(details.get('what_we_will_learn', []))
    mentor_str = ", ".join(details.get('mentor', []))

    # Construct the content string from relevant fields
    content_parts = []
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
        # You can add more metadata as needed here from the 'details' or 'course' object
    }
    
    # Remove metadata entries that are None or empty strings to keep it clean
    metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

    return Document(page_content=page_content, metadata=metadata)

def main():
    # 1. Ensure the domain_data folder exists
    if not os.path.exists(DOMAIN_DATA_DIR):
        print(f"Error: The directory '{DOMAIN_DATA_DIR}' does not exist.")
        print("Please create this folder in your project directory and place your JSON data files inside it.")
        print("Exiting ingestion process.")
        return

    # 2. Load all JSON data from the specified directory
    print(f"\nLoading data from '{DOMAIN_DATA_DIR}'...")
    raw_courses_data = load_json_data(DOMAIN_DATA_DIR)
    if not raw_courses_data:
        print("No JSON files found or data loaded. Please ensure your JSON files are in the 'domain_data' folder.")
        print("Exiting.")
        return

    # 3. Process raw data into LangChain Document objects
    print(f"Processing {len(raw_courses_data)} courses into initial LangChain documents...")
    all_base_documents = []
    for course_data in raw_courses_data:
        # Check if 'details' key exists and is a dictionary, otherwise skip
        if isinstance(course_data, dict) and 'details' in course_data and isinstance(course_data['details'], dict):
            all_base_documents.append(process_course_data(course_data))
        else:
            print(f"Skipping malformed course data (missing 'details' or not a dict): {course_data.get('title', 'Unknown Title')}")


    if not all_base_documents:
        print("No valid course documents were generated. Check your JSON structure.")
        return

    # 4. Split documents into smaller, overlapping chunks
    print("Splitting documents into smaller, overlapping chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap between chunks
        length_function=len,  # Use character length
        is_separator_regex=False, # Treat separators as literal strings
    )
    # The splitter will intelligently break `page_content` and preserve `metadata`
    chunked_documents = text_splitter.split_documents(all_base_documents)
    print(f"Original documents: {len(all_base_documents)}, Created chunks: {len(chunked_documents)}")

    # 5. Initialize Google Gemini Embeddings
    print(f"Initializing embedding model: '{EMBEDDING_MODEL_NAME}'...")
    try:
        # The GOOGLE_API_KEY environment variable is automatically picked up here
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Could not initialize GoogleGenerativeAIEmbeddings.")
        print(f"Please ensure your GOOGLE_API_KEY environment variable is correctly set and valid.")
        print(f"Details: {e}")
        return

    # 6. Create and Save FAISS Index
    print("Creating FAISS index from document chunks and embeddings...")
    try:
        db = FAISS.from_documents(chunked_documents, embeddings)
    except Exception as e:
        print(f"ERROR: Failed to create FAISS index. This might be due to issues with embeddings or data.")
        print(f"Details: {e}")
        return
    
    # Ensure the directory for FAISS index exists
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index created and saved successfully to '{FAISS_INDEX_PATH}'")
    print("\nIngestion complete!")


if __name__ == "__main__":
    main()