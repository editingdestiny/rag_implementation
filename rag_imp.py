import requests
import json
from datetime import datetime
import os
from PyPDF2 import PdfReader  # Install this library using `pip install PyPDF2`

# Hardcoded folder path for documents
DOCUMENT_FOLDER = "/home/sd22750/rag_implementation/docs"

# Function to save conversation to a log file
def log_conversation(query, response):
    with open("conversation_log.txt", "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n[{timestamp}] Query: {query}\n")
        log_file.write(f"[{timestamp}] Response: {response}\n")
        log_file.write("-" * 80 + "\n")

# Function to generate a response using deepseek-r1
def generate_response(prompt, model="deepseek-r1:1.5b"):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    full_response = ""

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=30)

        if response.status_code == 200:
            print("\nModel Response:")
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        data = json.loads(chunk.decode("utf-8"))
                        partial_response = data.get("response", "")
                        print(partial_response, end="", flush=True)
                        full_response += partial_response
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        print(f"\nError decoding chunk: {e}")
            print()
        else:
            print(f"Error generating response: {response.status_code} - {response.text}")
            full_response = f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        print("\nError: The request timed out. The model took too long to respond.")
        full_response = "Error: The request timed out."

    return full_response

# Function to load all .txt and .pdf files from a folder
def load_documents_from_folder(folder_path):
    documents = {}
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return documents

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    documents[file_name] = content
                    print(f"Loaded text file: {file_name}")
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")

        elif file_name.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
                documents[file_name] = content.strip()
                print(f"Loaded PDF file: {file_name}")
            except Exception as e:
                print(f"Error reading PDF '{file_name}': {e}")

    return documents

# Function to let the user select a document from the list
def select_document(documents):
    if not documents:
        print("No documents available.")
        return None

    print("\nAvailable Documents:")
    for idx, doc_name in enumerate(documents.keys(), start=1):
        print(f"{idx}. {doc_name}")

    while True:
        try:
            choice = int(input("\nEnter the number of the document you want to use: "))
            if 1 <= choice <= len(documents):
                selected_doc_name = list(documents.keys())[choice - 1]
                return selected_doc_name, documents[selected_doc_name]
            else:
                print("Invalid choice. Please select a valid document number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main function with document selection and querying
def main():
    print("Welcome to the Query System! Type 'exit' to terminate.")
    
    # Load documents from the hardcoded folder
    print(f"\nLoading documents from folder: {DOCUMENT_FOLDER}")
    documents = load_documents_from_folder(DOCUMENT_FOLDER)

    while True:
        sentence = input("\nEnter a sentence or query (or type 'exit' to quit): ")
        
        if sentence.lower() == 'exit':
            print("Goodbye!")
            break

        # Let user select a document
        selected_doc_name, selected_doc_content = select_document(documents)
        if not selected_doc_content:
            continue

        # Check if user wants summarization or querying
        if sentence.lower() == "summarize document":
            prompt = f"Please summarize the following document:\n\n{selected_doc_content[:2000]}\n\nSummary:"
            response = generate_response(prompt)
            print(f"\nSummary of '{selected_doc_name}':\n{response}")
            log_conversation(f"Summarize Document: {selected_doc_name}", response)
        
        else:
            prompt = f"Context: {selected_doc_content[:2000]}\n\nQuestion: {sentence}\n\nAnswer:"
            response = generate_response(prompt)
            print(f"\nResponse based on '{selected_doc_name}':\n{response}")
            log_conversation(sentence, response)

if __name__ == "__main__":
    main()
