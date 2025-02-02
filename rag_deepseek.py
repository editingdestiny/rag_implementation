import logging
logging.basicConfig(level=logging.DEBUG)
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable parameters
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
RETRIEVAL_K = 2  # Number of documents to retrieve

# Knowledge base setup (can be loaded from a file if needed)
documents = [
    "The current date is Friday, January 31, 2025.",
    "DeepSeek-R1:1.5B is a language model developed by DeepSeek.",
    "RAG stands for Retrieval-Augmented Generation."
]

# Initialize the sentence transformer model
logger.info("Loading sentence transformer model...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents and create FAISS index
logger.info("Encoding documents and creating FAISS index...")
embeddings = encoder.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))


def retrieve(query, k=RETRIEVAL_K):
    """
    Retrieve the top-k most relevant documents for a given query.
    """
    logger.info("Retrieving relevant documents...")
    query_vector = encoder.encode([query]).astype('float32')
    _, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]


def generate(prompt):
    """
    Send the prompt to the DeepSeek model via Ollama's API.
    """
    logger.info("Sending prompt to the DeepSeek model...")
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "prompt": prompt
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for HTTP failures

        # Debugging: Log raw response text
        logger.debug("Raw Response Text: %s", response.text)

        # Attempt to parse JSON
        try:
            parsed_response = response.json()
            return parsed_response.get('response', None)
        except ValueError as e:
            logger.error("Error while parsing JSON: %s", e)
            logger.error("Raw Response Text: %s", response.text)
            return f"Error: Unable to parse API response. Raw response: {response.text}"
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
    """
    Full RAG pipeline: Retrieve relevant documents and generate a response.
    """
    relevant_docs = retrieve(query)
    
    # Combine retrieved documents into context
    context = "\n".join(relevant_docs)

    # Optional: Truncate context if it's too long
    context = truncate_context(context)

    # Create the prompt
    prompt = f"""Given the following context and query, provide an accurate and up-to-date answer:

Context:
{context}

Query: {query}

Answer:"""

    # Generate a response from the model
    return generate(prompt)


if __name__ == "__main__":
    query = input("Enter your query: ")
    
    logger.info("Processing query: %s", query)
    
    answer = rag_pipeline(query)
    
    if answer:
        print("\nAnswer:", answer)
    else:
        print("\nFailed to generate an answer. Check logs for details.")
