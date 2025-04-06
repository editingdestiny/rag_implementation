import logging
#import sentence-transformers
logging.basicConfig(level=logging.DEBUG)
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from functools import lru_cache
import os
from PyPDF2 import PdfReader
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DOCUMENT_FOLDER = "/home/sd22750/rag_implementation/docs"  
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
RETRIEVAL_K = 3  # Number of documents to retrieve

def load_documents_from_folder(folder_path):
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return documents

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())

        elif file_name.endswith(".pdf"):
            reader = PdfReader(file_path)
            content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            documents.append(content)

    return documents

# Load documents dynamically
documents = load_documents_from_folder(DOCUMENT_FOLDER)

# Initialize embedding model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents and create FAISS index
embeddings = encoder.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))


# Inside your retrieve function in deepseek.py
def retrieve(query, k=RETRIEVAL_K):
    logger.info(f"Retrieving top {k} documents for query: '{query}'")
    query_vector = encoder.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # --- ADD THIS FOR SHOWCASE ---
    logger.info(f"Retrieved document indices: {indices[0]}")
    # Optional: Log snippets or filenames if you have them mapped
    # logger.info(f"Retrieved distances: {distances[0]}") # Can show confidence
    print(f"--- Debug: Retrieved {len(retrieved_docs)} document chunks. ---")
    # --- END ADDITION ---

    return retrieved_docs


def generate(prompt):
    """
    Send the prompt to the DeepSeek model via Ollama's API and handle streaming responses.
    """
    logger.info("Sending prompt to the DeepSeek model...")

    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True  # Enable streaming mode
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()  # Raise an error for HTTP failures

        # Read streamed response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = line.decode("utf-8")
                    parsed_data = json.loads(json_data)
                    if "response" in parsed_data:
                        full_response += parsed_data["response"]
                    if parsed_data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    logger.error("Error parsing streamed JSON: %s", e)

        return full_response.strip() if full_response else "No response received."

    except requests.exceptions.RequestException as e:
        logger.error("API request failed: %s", e)
        return f"Error: {e}"

def truncate_context(context, max_tokens=2048):
    """
    Truncate the context to ensure it doesn't exceed the model's maximum token limit.
    """
    from transformers import AutoTokenizer

    logger.info("Truncating context to fit within token limits...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your tokenizer if needed
    tokens = tokenizer.encode(context)
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)
    
    return context


@lru_cache(maxsize=100)
def rag_pipeline(query):
    relevant_docs = retrieve(query)
    context = "\n---\n".join(relevant_docs) # Use separator for clarity
    context = truncate_context(context)

    prompt = f"""Using ONLY the context provided below, answer the query accurately. Do not add information not present in the context.

    Context:
    {context}

    Query: {query}

    Answer:"""

        # Generate a response from the model
    print (context)
    return generate(prompt)


if __name__ == "__main__":
    query = [
        "What was the net revenue reported in 4Q23?",
        "how much credit was give for US small businesses?"
            ]
    
    results = {}
    for query in query:
        print(f"\nQuerying RAG'ed Model:\n{query}")
        answer = rag_pipeline(query)
        print(f"\nRAGGed Model Response:\n{answer}")
        print("-" * 30)
        results[query] = answer
        print("#" * 30)
        

    #Optionally save results
    with open("RAG_results.json", "w") as f:
        json.dump(results, f, indent=2)
