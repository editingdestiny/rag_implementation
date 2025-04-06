import logging
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader
import json
import datetime
import csv # Import the CSV module
from functools import lru_cache # Keep caching if desired

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENT_FOLDER = "/home/sd22750/rag_implementation/docs" # <--- Verify this path
API_URL_GENERATE = "http://localhost:11434/api/generate" # Ollama generate endpoint
# API_URL_CHAT = "http://localhost:11434/api/chat" # Or use chat endpoint if preferred
MODEL_NAME = "deepseek-r1:1.5b" # Ensure this model is running in Ollama
RETRIEVAL_K = 2
OUTPUT_CSV_FILE = "rag_comparison_log.csv" # Name of the output table file
# MAX_CONTEXT_TOKENS = 2048 # Optional for truncation

# --- RAG Setup (Adapted from deepseek.py) ---

def load_documents_from_folder(folder_path):
    # (Keep the robust version from the previous example)
    documents = []
    filenames = []
    if not os.path.exists(folder_path):
        logger.error(f"Error: Document folder '{folder_path}' does not exist.")
        return documents, filenames
    logger.info(f"Loading documents from: {folder_path}")
    # ... (rest of the loading logic for txt/pdf) ...
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        doc_content = None
        try:
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    doc_content = file.read()
            elif file_name.endswith(".pdf"):
                reader = PdfReader(file_path)
                text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
                if text_parts:
                    doc_content = "\n".join(text_parts)
                else:
                    logger.warning(f"Could not extract text from PDF: {file_name}")
            if doc_content:
                documents.append(doc_content)
                filenames.append(file_name)
            else:
                 logger.warning(f"No content read from file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to read or process file {file_name}: {e}")
    logger.info(f"Successfully loaded {len(documents)} documents.")
    return documents, filenames

# --- Initialize Global RAG Components ---
documents, document_filenames = [], []
encoder = None
index = None
rag_initialized = False

def initialize_rag():
    global documents, document_filenames, encoder, index, rag_initialized
    if rag_initialized:
        return True

    logger.info("--- Initializing RAG Components ---")
    try:
        documents, document_filenames = load_documents_from_folder(DOCUMENT_FOLDER)
        if not documents:
            logger.warning("No documents loaded. RAG pipeline will lack context.")
            # Decide if you want RAG to be considered "initialized" even without docs
            # rag_initialized = True
            # return rag_initialized
            # OR: Consider RAG setup failed if no docs
            return False

        logger.info("Initializing sentence transformer model...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Encoding documents...")
        embeddings = encoder.encode(documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        logger.info(f"Embeddings created with dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index created with {index.ntotal} vectors.")
        rag_initialized = True
        logger.info("--- RAG Components Initialized Successfully ---")
        return True
    except ImportError as e:
         logger.error(f"Import error during RAG init: {e}. Make sure sentence-transformers and faiss-cpu are installed.")
         return False
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        # Clean up potentially partially initialized components
        documents, document_filenames, encoder, index = [], [], None, None
        return False

# --- Core Querying Functions ---

def ollama_generate(prompt, model=MODEL_NAME, api_url=API_URL_GENERATE):
    """ Sends a prompt to Ollama generate endpoint, returns response text or error message. """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False, # Keep it simple for direct comparison logging
        "options": {"temperature": 0.7}
    }
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        response_data = response.json()
        # Adjust key based on actual API response ('response' is typical for generate)
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
    # Simple prompt: just the query
    prompt = query
    return ollama_generate(prompt)

# --- RAG Pipeline Functions ---

def retrieve(query, k=RETRIEVAL_K):
    """ Retrieves documents for RAG. """
    if not rag_initialized or not index or index.ntotal == 0:
        logger.warning("RAG not initialized or index empty. Skipping retrieval.")
        return [], [] # Return empty lists for docs and names

    logger.info(f"RAG: Retrieving top {k} documents for query...")
    try:
        query_vector = encoder.encode([query]).astype('float32')
        distances, indices = index.search(query_vector, k)
        retrieved_docs_content = [documents[i] for i in indices[0]]
        retrieved_docs_names = [document_filenames[i] for i in indices[0]] # Get names
        logger.info(f"RAG: Retrieved content from files: {retrieved_docs_names}")
        return retrieved_docs_content, retrieved_docs_names # Return content and names
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")
        return [], []

# Optional: Context Truncation (keep if needed)
# def truncate_context(context, max_tokens=MAX_CONTEXT_TOKENS): ...

@lru_cache(maxsize=100)
def run_rag_pipeline(query):
    """ Runs the full RAG pipeline. """
    if not rag_initialized:
        logger.error("RAG pipeline cannot run because initialization failed.")
        return "Error: RAG components not initialized.", []

    logger.info(f"Running RAG pipeline for query...")
    retrieved_docs, retrieved_filenames = retrieve(query)

    if not retrieved_docs:
        logger.warning("RAG: No relevant documents found. Asking model without specific context.")
        context = "No specific context was retrieved from the local documents."
    else:
        context = "\n---\n".join(retrieved_docs)

    # --- Context Truncation (Optional) ---
    # context = truncate_context(context)
    # ------------------------------------

    # --- RAG Prompt ---
    prompt = f"""Based ONLY on the following context, please provide a concise and accurate answer to the query. If the context does not contain the answer, state that the information is not available in the provided documents.

Context:
{context}
---
Query: {query}

Answer:"""

    logger.info(f"Querying RAG model ({MODEL_NAME} with context)...")
    response = ollama_generate(prompt)
    # Return response AND the names of docs used (or empty list)
    return response, retrieved_filenames


# --- CSV Logging ---
def write_to_csv(timestamp, query, base_response, rag_response, retrieved_files):
    """ Appends a record to the CSV file. """
    file_exists = os.path.isfile(OUTPUT_CSV_FILE)
    try:
        with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Timestamp', 'Query', 'Base_Model_Response', 'RAG_Pipeline_Response', 'RAG_Retrieved_Files']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader() # Write header only if file is new

            writer.writerow({
                'Timestamp': timestamp,
                'Query': query,
                'Base_Model_Response': base_response,
                'RAG_Pipeline_Response': rag_response,
                'RAG_Retrieved_Files': "; ".join(retrieved_files) # Join filenames list
            })
        logger.info(f"Results successfully appended to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Failed to write to CSV file {OUTPUT_CSV_FILE}: {e}")

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":

    # --- Initialize RAG components ONCE ---
    if not initialize_rag():
        logger.warning("RAG initialization failed. RAG responses will indicate errors.")
        # You could choose to exit here if RAG is essential:
        # print("Exiting due to RAG initialization failure.")
        # exit()

    # --- Get User Query ---
    user_query = input("Enter your query (or type 'quit' to exit): ")

    if user_query.lower() == 'quit':
        print("Exiting.")
        exit()

    logger.info(f"Processing query: '{user_query}'")

    # --- Get Timestamp ---
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Run Base Model Query ---
    base_answer = query_base_model(user_query)
    print("\n--- Base Model Response ---")
    print(base_answer)
    print("-" * 27)

    # --- Run RAG Pipeline Query ---
    rag_answer, rag_files = run_rag_pipeline(user_query)
    print("\n--- RAG Pipeline Response ---")
    if rag_files: # Only print if files were retrieved
        print(f"(Context based on: {'; '.join(rag_files)})")
    print(rag_answer)
    print("-" * 29)


    # --- Log Results to CSV ---
    write_to_csv(current_timestamp, user_query, base_answer, rag_answer, rag_files)

    print(f"\nComparison results saved to {OUTPUT_CSV_FILE}")