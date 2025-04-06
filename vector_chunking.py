import logging
import requests
# from sentence_transformers import SentenceTransformer # Now potentially loaded by Chroma
import os
from PyPDF2 import PdfReader
import json
import datetime
import csv
# from functools import lru_cache # Caching might be less useful if DB is fast

# --- NEW IMPORTS ---
from langchain.text_splitter import RecursiveCharacterTextSplitter # Example splitter
import chromadb # Vector Database client
# If not using langchain splitter, you'd implement your own chunking logic here

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENT_FOLDER = "/home/sd22750/rag_implementation/docs" # <--- Verify this path
API_URL_GENERATE = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
OUTPUT_CSV_FILE = "rag_comparison_log.csv"
# --- Chunking Config ---
CHUNK_SIZE = 1000 # Characters per chunk (tune this)
CHUNK_OVERLAP = 150 # Characters overlap between chunks (helps context flow)
# --- ChromaDB Config ---
CHROMA_PATH = "./chroma_db" # Directory to store the persistent DB
COLLECTION_NAME = "rag_documents"
# Use the same embedding model for consistency
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RETRIEVAL_K = 3 # Retrieve top 3 chunks now

# --- Global Chroma Client and Collection ---
chroma_client = None
collection = None
rag_initialized = False

def initialize_or_load_vector_db():
    """Initializes ChromaDB client, gets/creates collection, and loads/embeds docs if needed."""
    global chroma_client, collection, rag_initialized
    if rag_initialized:
        return True

    logger.info("--- Initializing ChromaDB and Loading Documents ---")
    try:
        # 1. Initialize Chroma Client (persistent)
        # settings=chromadb.Settings(allow_reset=True) # Use allow_reset=True during dev if you need to clear often
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # 2. Get or Create Collection (specify embedding function)
        # Chroma uses SentenceTransformer internally if name is provided
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"} # Use cosine or l2 distance
            # embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME) # Let Chroma handle embedding
            # Alternatively, pre-embed and pass vectors if you prefer manual control
        )

        # 3. Check if DB needs population (simple check: count items)
        if collection.count() == 0:
            logger.info(f"Collection '{COLLECTION_NAME}' is empty. Processing and adding documents from {DOCUMENT_FOLDER}...")
            if not os.path.exists(DOCUMENT_FOLDER):
                 logger.error(f"Document folder not found: {DOCUMENT_FOLDER}")
                 return False

            # --- Document Loading and Chunking ---
            docs_to_add = []
            metadata_to_add = []
            ids_to_add = []
            doc_id_counter = 0

            # Use Langchain's splitter as an example
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )

            for file_name in os.listdir(DOCUMENT_FOLDER):
                file_path = os.path.join(DOCUMENT_FOLDER, file_name)
                full_text = None
                try:
                    if file_name.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            full_text = f.read()
                    elif file_name.endswith(".pdf"):
                        reader = PdfReader(file_path)
                        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
                        if text_parts:
                            full_text = "\n".join(text_parts)

                    if full_text:
                        # Chunk the document text
                        chunks = text_splitter.split_text(full_text)
                        logger.info(f"Split '{file_name}' into {len(chunks)} chunks.")

                        for i, chunk in enumerate(chunks):
                            # Create unique ID for each chunk
                            chunk_id = f"{file_name}_chunk_{i}"
                            ids_to_add.append(chunk_id)
                            docs_to_add.append(chunk)
                            metadata_to_add.append({
                                "source": file_name, # Store metadata!
                                "chunk_index": i
                            })
                            doc_id_counter += 1
                    else:
                        logger.warning(f"No text content extracted from {file_name}")
                except Exception as e:
                    logger.error(f"Failed processing file {file_name}: {e}")

            # --- Add Chunks to ChromaDB (Batch Add) ---
            if docs_to_add:
                logger.info(f"Adding {len(docs_to_add)} chunks to Chroma collection '{COLLECTION_NAME}'...")
                # ChromaDB handles embedding here if embedding_function is set
                collection.add(
                    documents=docs_to_add,
                    metadatas=metadata_to_add,
                    ids=ids_to_add
                )
                logger.info("Finished adding documents to ChromaDB.")
            else:
                 logger.warning("No documents processed or added to the database.")
                 # Decide if this constitutes a failure
                 # return False

        else:
            logger.info(f"Found existing collection '{COLLECTION_NAME}' with {collection.count()} documents.")

        rag_initialized = True
        logger.info("--- RAG Vector DB Initialized Successfully ---")
        return True

    except ImportError as e:
         logger.error(f"Import error during RAG init: {e}. Ensure langchain, chromadb, sentence-transformers are installed.")
         return False
    except Exception as e:
        logger.error(f"Failed to initialize/load ChromaDB: {e}")
        return False

# --- Core Querying Functions (ollama_generate remains the same) ---
def ollama_generate(prompt, model=MODEL_NAME, api_url=API_URL_GENERATE):
     # ... (same as previous implementation) ...
    headers = {"Content-Type": "application/json"}
    data = { "model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.7} }
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get('response', 'Error: Response key not found in JSON').strip()
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out querying model {model}.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed querying model {model}: {e}")
        return f"Error: API Request Failed - {e}"
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from model {model}: {response.text}")
        return "Error: JSON Decode Error"
    except Exception as e:
        logger.error(f"An unexpected error occurred querying model {model}: {e}")
        return f"Error: Unexpected error - {e}"


def query_base_model(query):
    """ Queries the base model directly without RAG. """
    logger.info(f"Querying BASE model ({MODEL_NAME})...")
    return ollama_generate(query) # Simple prompt

# --- RAG Pipeline Functions (Updated for ChromaDB) ---

def retrieve_from_vector_db(query, k=RETRIEVAL_K):
    """ Retrieves document chunks from ChromaDB. """
    if not rag_initialized or not collection:
        logger.warning("RAG Vector DB not initialized. Skipping retrieval.")
        return [], [] # Return empty lists for content and metadata

    logger.info(f"RAG: Querying vector DB for top {k} chunks...")
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas', 'distances'] # Request content, metadata, and distance
        )

        if not results or not results.get('ids') or not results['ids'][0]:
             logger.warning("Vector DB query returned no results.")
             return [], []

        # Extract the needed information (handle potential structure variations)
        retrieved_docs_content = results['documents'][0] if results.get('documents') else []
        retrieved_metadatas = results['metadatas'][0] if results.get('metadatas') else []
        retrieved_distances = results['distances'][0] if results.get('distances') else []

        logger.info(f"RAG: Retrieved {len(retrieved_docs_content)} chunks.")
        # Log metadata for debugging relevance
        for i, meta in enumerate(retrieved_metadatas):
            logger.info(f"  - Chunk {i}: Source={meta.get('source', 'N/A')}, Index={meta.get('chunk_index', 'N/A')}, Distance={retrieved_distances[i]:.4f}")

        return retrieved_docs_content, retrieved_metadatas # Return content and metadata

    except Exception as e:
        logger.error(f"Error during ChromaDB query: {e}")
        return [], []

# Optional: Context Truncation (keep if needed)
# def truncate_context(context, max_tokens=MAX_CONTEXT_TOKENS): ...

# @lru_cache(maxsize=100) # Re-evaluate caching benefit with DB
def run_rag_pipeline_with_vector_db(query):
    """ Runs the full RAG pipeline using the vector database. """
    if not rag_initialized:
        logger.error("RAG pipeline cannot run because initialization failed.")
        return "Error: RAG components not initialized.", []

    logger.info(f"Running RAG pipeline for query...")
    retrieved_docs, retrieved_metadatas = retrieve_from_vector_db(query)

    if not retrieved_docs:
        logger.warning("RAG: No relevant document chunks found. Asking model without specific context.")
        context = "No specific context was retrieved from the local document database."
        retrieved_sources_str = "N/A" # Indicate no files retrieved
    else:
        context = "\n---\n".join(retrieved_docs) # Join the chunk text
        # Create a summary of sources from metadata
        sources = set(meta.get('source', 'Unknown') for meta in retrieved_metadatas)
        retrieved_sources_str = "; ".join(sorted(list(sources)))

    # --- Context Truncation (Optional) ---
    # context = truncate_context(context)
    # ------------------------------------

    # --- RAG Prompt ---
    prompt = f"""Based ONLY on the following context, please provide a concise and accurate answer to the query. If the context does not contain the answer, state that the information is not available in the provided document excerpts.

Context:
{context}
---
Query: {query}

Answer:"""

    logger.info(f"Querying RAG model ({MODEL_NAME} with context from {retrieved_sources_str})...")
    response = ollama_generate(prompt)
    # Return response AND the summary string of sources used
    return response, retrieved_sources_str


# --- CSV Logging (write_to_csv remains mostly the same, adjust field name) ---
def write_to_csv(timestamp, query, base_response, rag_response, rag_sources_str):
    """ Appends a record to the CSV file. """
    # ... (CSV writing logic from previous example, but use rag_sources_str
    #      for the 'RAG_Retrieved_Files' column or rename the column) ...
    file_exists = os.path.isfile(OUTPUT_CSV_FILE)
    try:
        with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            # Adjust field name if desired
            fieldnames = ['Timestamp', 'Query', 'Base_Model_Response', 'RAG_Pipeline_Response', 'RAG_Context_Sources']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Timestamp': timestamp,
                'Query': query,
                'Base_Model_Response': base_response,
                'RAG_Pipeline_Response': rag_response,
                'RAG_Context_Sources': rag_sources_str # Store the sources string
            })
        logger.info(f"Results successfully appended to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Failed to write to CSV file {OUTPUT_CSV_FILE}: {e}")

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":

    # --- Initialize RAG Vector DB ONCE ---
    if not initialize_or_load_vector_db():
        logger.critical("Exiting due to critical RAG vector database initialization failure.")
        exit()

    # --- Main Interaction Loop ---
    while True:
        user_query = input("Enter your query (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            print("Exiting.")
            break # Exit the loop

        if not user_query:
            continue # Ignore empty input

        logger.info(f"Processing query: '{user_query}'")
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Run Base Model Query ---
        base_answer = query_base_model(user_query)
        print("\n--- Base Model Response ---")
        print(base_answer)
        print("-" * 27)

        # --- Run RAG Pipeline Query ---
        rag_answer, rag_sources = run_rag_pipeline_with_vector_db(user_query)
        print("\n--- RAG Pipeline Response ---")
        print(f"(Context based on chunks from: {rag_sources})")
        print(rag_answer)
        print("-" * 29)

        # --- Log Results to CSV ---
        write_to_csv(current_timestamp, user_query, base_answer, rag_answer, rag_sources)

        print(f"\nComparison results saved to {OUTPUT_CSV_FILE}\n")