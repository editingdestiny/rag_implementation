# direct_query.py
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:11434/api/generate" # Or /api/chat if using chat endpoint
MODEL_NAME = "deepseek-r1:1.5b"

def query_base_model(prompt):
    logger.info(f"Querying base model '{MODEL_NAME}' directly...")
    headers = {"Content-Type": "application/json"}
    data = {"model": MODEL_NAME, "prompt": prompt, "stream": False} 

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        # Adjust key based on actual API response ('response' for generate, 'message.content' for chat maybe)
        return response_data.get('response', 'Error: Response key not found')
    except requests.exceptions.RequestException as e:
        logger.error("API request failed: %s", e)
        return f"Error: API Request Failed - {e}"
    except json.JSONDecodeError as e:
         logger.error("Failed to decode JSON response: %s", response.text)
         return f"Error: JSON Decode Error - {e}"
    except Exception as e:
         logger.error("An unexpected error occurred: %s", e)
         return f"Error: Unexpected error - {e}"


if __name__ == "__main__":
    # --- DEFINE YOUR TEST QUERIES HERE ---
    test_queries = [
        "Who is the CIO?", # Specific knowledge likely only in your docs
        "What percentage of managed  portfolios finished with double digit returns in 2024.", # Time-sensitive knowledge
        "According to the 'jpm_10k_q125.pdf', whathow many employees are there in Asia-Pacific?", # Document-specific detail
        "Explain the company's policy on remote work as updated last month." # Recent, specific info
    ]

    results = {}
    for query in test_queries:
        print(f"\nQuerying Base Model:\n{query}")
        answer = query_base_model(query)
        print(f"\nBase Model Response:\n{answer}")
        print("-" * 30)
        results[query] = answer

    #Optionally save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)