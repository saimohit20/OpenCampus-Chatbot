import os
import json
import asyncio
import google.generativeai as genai
import chromadb
# No longer need to import embedding_functions from chromadb.utils directly,
# as we're defining our own custom class.

# --- Configuration ---
DOMAIN_DATA_DIR = "domain_data"  # Directory where your JSON files are stored
CHROMA_DB_PATH = "chroma_db"     # Directory to persist your ChromaDB instance
COLLECTION_NAME = "opencampus_courses"

# --- Custom Gemini Embedding Function Class ---
# This class implements the interface expected by ChromaDB for a custom embedding function.
class GeminiEmbeddingFunction:
    """
    A custom embedding function for ChromaDB using Google Gemini's text-embedding-004 model.
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        if not api_key:
            # In Canvas, the API key is auto-injected if left empty.
            # Locally, the user MUST set it as an env var or directly.
            print("WARNING: GOOGLE_API_KEY environment variable not set for GeminiEmbeddingFunction. This might prevent API calls.")
        
        # Configure the Gemini API globally. This is done once.
        # It's important that this `configure` happens before `genai.embed_content` is called.
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=input,
                task_type="RETRIEVAL_DOCUMENT"
            )
            # Debug: print the response if 'embeddings' is missing
            if 'embeddings' not in response:
                print("Gemini API response missing 'embeddings':", response)
                return [[] for _ in input]
            embeddings = [embedding.embedding for embedding in response['embeddings']]
            return embeddings
        except Exception as e:
            print(f"ERROR during Gemini embedding API call for texts: {input[0][:50]}... : {e}")
            return [[] for _ in input]

    def name(self) -> str:
        """
        Returns a unique name for this embedding function, required by ChromaDB.
        """
        return f"gemini_text_embedding_function_{self.model_name.replace('models/', '')}"

# --- Data Ingestion Function ---
async def ingest_data_to_vector_db():
    print("--- Starting Data Ingestion to Vector Database ---")

    # 1. Get Google API Key
    # In Canvas, this will likely be empty but auto-injected by the environment.
    # Locally, ensure GOOGLE_API_KEY env var is set.
    # api_key = os.getenv("GOOGLE_API_KEY", "") 
    api_key = "AIzaSyAdab1EdwNZtZ8yQhfwHvK3V6Ir-YDhihQ"
    
    # 2. Instantiate our custom GeminiEmbeddingFunction
    # This instance will be passed to ChromaDB.
    gemini_ef_instance = GeminiEmbeddingFunction(api_key=api_key, model_name="models/text-embedding-004")
    print(f"Gemini Embedding Function initialized with model: {gemini_ef_instance.model_name}")

    # 3. Initialize ChromaDB client
    # This will create the 'chroma_db' directory if it doesn't exist,
    # or load the existing database from it.
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"ChromaDB client initialized at: {CHROMA_DB_PATH}")

    # 4. Get or create the collection with our custom embedding function
    # ChromaDB will use `gemini_ef_instance` to embed documents when they are added.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=gemini_ef_instance # Pass the INSTANCE of our custom class
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready and using '{gemini_ef_instance.name()}' for embeddings.")

    # 5. Load all scraped course data
    all_courses_data = []
    print(f"\nLoading course data from '{DOMAIN_DATA_DIR}'...")
    if not os.path.exists(DOMAIN_DATA_DIR):
        print(f"ERROR: Domain data directory '{DOMAIN_DATA_DIR}' does not exist. Please run the scraper first.")
        return

    for filename in os.listdir(DOMAIN_DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DOMAIN_DATA_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    domain_courses = json.load(f)
                    # Assuming each item in the domain_courses list has a 'details' key
                    # and 'details' contains the actual course info
                    for course_item in domain_courses:
                        if 'details' in course_item and course_item['details']:
                            all_courses_data.append(course_item)
                        else:
                            print(f"  WARNING: Course '{course_item.get('title', 'Untitled')}' from '{filename}' has no 'details' or empty details. Skipping.")
            except Exception as e:
                print(f"  ERROR loading {filepath}: {e}")
    
    if not all_courses_data:
        print("No course data found to ingest. Exiting.")
        return

    print(f"Loaded {len(all_courses_data)} courses across all domains.")

    # 6. Prepare documents, metadata, and IDs for ChromaDB
    documents_to_add = [] # List to hold the text content for embedding
    metadatas_to_add = [] # List to hold the metadata dicts
    ids_to_add = []       # List to hold unique IDs for each document

    for course in all_courses_data:
        course_details = course.get('details', {})
        course_title = course.get('title', 'N/A Course')
        course_url = course.get('url', f"no_url_found_{len(ids_to_add)}") # Use URL as a robust unique ID
        
        # Ensure domain_name is available from the initial scrape if not already in 'course' dict
        # For this pipeline, let's explicitly add it to the course object if it's implicitly derived
        domain_name_from_path = os.path.basename(os.path.dirname(filepath)).replace('_', ' ').title() # Heuristic
        # If your initial scrape already adds 'domain_name' to each course dict in `all_courses_data`,
        # then you can use `course.get('domain_name', domain_name_from_path)`.
        # For simplicity, we'll assume the domain comes from the file name for now if not present.
        domain_name = filename.replace('.json', '').replace('_', ' ').title() # Derive from current JSON filename

        # Construct a comprehensive text document for the course.
        # This text string will be sent to the embedding model.
        doc_content_parts = [f"Title: {course_title}"]
        if course_details.get('description'):
            doc_content_parts.append(f"Description: {course_details['description']}")
        if course_details.get('information'): # Your combined field
            doc_content_parts.append(f"Additional Info: {course_details['information']}")
        if course_details.get('what_we_will_learn'):
            doc_content_parts.append(f"Learning Goals: {'; '.join(course_details['what_we_will_learn'])}")
        if course_details.get('mentor'):
            doc_content_parts.append(f"Mentors: {', '.join(course_details['mentor'])}")
        if course_details.get('day_time_summary'):
            doc_content_parts.append(f"Schedule: {course_details['day_time_summary']}")
        if course_details.get('ects'):
            doc_content_parts.append(f"ECTS: {course_details['ects']}")
        if course_details.get('location'):
            doc_content_parts.append(f"Location: {course_details['location']}")
        if course_details.get('language'):
            doc_content_parts.append(f"Language: {course_details['language']}")
        
        # Add session details if available and well-structured
        if course_details.get('course_sessions'):
            session_strings = []
            for session in course_details['course_sessions']:
                s_date = session.get('date', 'N/A')
                s_time = session.get('time', 'N/A')
                s_topic = session.get('topic', 'N/A')
                s_location = session.get('location_text', 'N/A')
                session_strings.append(f"Date: {s_date}, Time: {s_time}, Topic: {s_topic}, Location: {s_location}")
            if session_strings:
                doc_content_parts.append(f"Sessions: {' | '.join(session_strings)}")
        
        # Join all parts to form the final document string for embedding
        full_doc_content = "\n".join(doc_content_parts).strip()
        
        documents_to_add.append(full_doc_content)
        ids_to_add.append(course_url) # Use the course URL as a unique ID in ChromaDB
        metadatas_to_add.append({
            "course_title": course_title,
            "course_url": course_url,
            "domain": domain_name, # Storing the domain as metadata
            "ects": course_details.get('ects'),
            "language": course_details.get('language'),
            # Add any other metadata you might want to filter by or display later
        })

    # 7. Add documents to ChromaDB in batches
    # It's good practice to add in batches to optimize API calls and memory usage.
    batch_size = 50 # Adjust based on your API rate limits and data size
    
    print(f"\nAdding {len(documents_to_add)} documents to ChromaDB in batches of {batch_size}...")
    for i in range(0, len(documents_to_add), batch_size):
        batch_docs = documents_to_add[i:i + batch_size]
        batch_metadatas = metadatas_to_add[i:i + batch_size]
        batch_ids = ids_to_add[i:i + batch_size]
        
        try:
            # The `collection.add` method automatically calls the embedding_function
            # that was configured when the collection was created.
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"  Successfully added batch {i//batch_size + 1}/{(len(documents_to_add) + batch_size - 1) // batch_size}")
            # Add a small delay between batches if you hit API rate limits
            # time.sleep(1) 
        except Exception as e:
            print(f"  ERROR adding batch starting at index {i} (IDs: {batch_ids[0]}...): {e}")
            # Consider more robust error handling here, e.g., logging failed IDs for retry.

    print(f"\nData ingestion complete. Total documents in collection: {collection.count()}.")

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure the domain_data directory exists, otherwise the scraper must be run first
    if not os.path.exists(DOMAIN_DATA_DIR):
        print(f"Error: '{DOMAIN_DATA_DIR}' directory not found.")
        print("Please run the web scraping pipeline first to generate the necessary JSON data.")
        exit(1) # Exit with an error code

    # Ensure the ChromaDB persistence path exists
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    asyncio.run(ingest_data_to_vector_db())