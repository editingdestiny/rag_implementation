# run_comparison.py
import argparse
import requests
import json
import logging
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from datetime import datetime

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENT_FOLDER = "/home/sd22750/rag_implementation/docs" # Make sure this is correct!
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Or /api/chat
BASE_MODEL_NAME = "deepseek-r1:1.5b" # Or your specific base model name in Ollama
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RETRIEVAL_K = 2

# --- Global variables for RAG components (Load once if script is long-running, or load on demand) ---
# Consider loading these outside the functions if efficiency is critical,
# but loading inside might be safer for command-line execution models.
# For simplicity here, we load inside the RAG function. Re-evaluate if slow.
encoder = None
index = None
documents = None
embeddings_loaded = False

# --- Document Loading & RAG Initialization ---
# (Keep your existing load_documents_from_folder function)
def load_documents_from_folder(folder_path):
    # ... (Your existing function - ensure it logs errors well) ...
    # ... Make sure it returns an empty list if folder doesn't exist or is empty ...
    docs = []
    logger.info(f"Attempting to load documents from: {folder_path}")
    if not os.path.exists(folder_path):
        logger.error(f"Error: Document folder does not exist at '{folder_path}'")
        return docs
    # ...(rest of your loading logic)...
    logger.info(f"Finished loading. Total documents loaded: {len(docs)}")
    return docs


def initialize_rag_components(force_reload=False):
    global encoder, index, documents, embeddings_loaded
    if embeddings_loaded and not force_reload:
        logger.info("RAG components already initialized.")
        return True

    logger.info("Initializing RAG components...")
    documents = load_documents_from_folder(DOCUMENT_FOLDER)
    if not documents:
        logger.error("RAG initialization failed: No documents loaded.")
        return False

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info("Encoding documents...")
        embeddings = encoder.encode(documents, show_progress_bar=False) # Turn off progress bar for script execution

        if embeddings.size == 0:
             logger.error("RAG initialization failed: No embeddings generated (documents might be empty or unreadable).")
             return False

        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index (dimension: {dimension})...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index created with {index.ntotal} entries.")
        embeddings_loaded = True
        return True

    except Exception as e:
        logger.error(f"Error during RAG initialization: {e}", exc_info=True)
        embeddings_loaded = False
        return False

# --- Model Interaction Functions ---
def query_ollama(model_name, prompt):
    logger.info(f"Querying Ollama model '{model_name}'...")
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "prompt": prompt, "stream": False} # No streaming for script

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data, timeout=120) # Increased timeout
        response.raise_for_status()
        response_data = response.json()
        # Adjust key based on API ('response' for generate, 'message.content' for chat)
        return response_data.get('response', 'Error: Response key not found in Ollama output')
    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out for model {model_name}.")
        return "Error: Ollama request timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed for {model_name}: {e}")
        return f"Error: Ollama API Request Failed - {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred querying {model_name}: {e}", exc_info=True)
        return f"Error: Unexpected error querying {model_name}."

def run_rag_query(query, use_existing_docs=True):
    global encoder, index, documents # Use global components

    # Initialize RAG components if not already done, or if forced reload needed (e.g., based on 'use_existing_docs')
    # For this setup, we assume 'use_existing_docs' means use the components if loaded, otherwise init.
    # A real 'upload' scenario would require more complex logic here.
    if not embeddings_loaded:
        if not initialize_rag_components():
             return "Error: Failed to initialize RAG system. Cannot perform RAG query."

    if not index or encoder is None:
         return "Error: RAG components (index or encoder) not available."

    try:
        logger.info(f"Performing RAG retrieval for query: '{query}'")
        query_vector = encoder.encode([query]).astype('float32')
        _, indices = index.search(query_vector, RETRIEVAL_K)
        relevant_docs = [documents[i] for i in indices[0]]
        context = "\n---\n".join(relevant_docs)

        # Simple truncation (consider token-based if necessary)
        max_context_len = 3500 # Adjust based on model limits / typical context size
        if len(context) > max_context_len:
            context = context[:max_context_len] + "..."
            logger.warning("Truncated RAG context due to length.")

        # Construct RAG prompt (adapt as needed)
        rag_prompt = f"""Using ONLY the context provided below, answer the query accurately.

Context:
{context}
---
Query: {query}

Answer:"""
        # Use the same base model but with the RAG prompt
        return query_ollama(BASE_MODEL_NAME, rag_prompt)

    except Exception as e:
        logger.error(f"Error during RAG query execution: {e}", exc_info=True)
        return "Error: Failed during RAG query processing."

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison between base model and RAG.")
    parser.add_argument("--query", type=str, required=True, help="The user's question.")
    # Add argument for document handling if needed, but keep simple for now
    # parser.add_argument("--use_docs", type=str, default="existing", help="Specify 'existing' or potentially path to new doc")

    args = parser.parse_args()
    user_query = args.query

    results = {
        "query": user_query,
        "base_answer": None,
        "rag_answer": None,
        "timestamp": datetime.now().isoformat()
    }

    # 1. Run Base Model Query
    logger.info("--- Running Base Model Query ---")
    # Construct a simple prompt for the base model
    base_prompt = f"Question: {user_query}\nAnswer:"
    results["base_answer"] = query_ollama(BASE_MODEL_NAME, base_prompt)

    # 2. Run RAG Query
    # For now, we assume "use existing docs". A real upload needs more logic.
    logger.info("--- Running RAG Enabled Query ---")
    results["rag_answer"] = run_rag_query(user_query) # Assumes using existing docs in folder

    # 3. Output results as JSON string to stdout
    print(json.dumps(results))