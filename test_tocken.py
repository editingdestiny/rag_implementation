import requests
import subprocess
import json

# Function to get available models from Ollama
def get_available_models():
    try:
        # Run the `ollama list` command to fetch available models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse the output into a list of models
            lines = result.stdout.strip().split("\n")[1:]  # Skip the header line
            models = [line.split()[0] for line in lines]  # Extract model names
            return models
        else:
            print("Error fetching models. Make sure Ollama is running.")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# Filter out unsupported models (e.g., embedding-only models)
def filter_supported_models(models):
    supported_models = []
    for model in models:
        if "bge" not in model:  # Exclude embedding models like bge-m3
            supported_models.append(model)
    return supported_models

# Get the list of available models
models = get_available_models()

# Filter out unsupported models
models = filter_supported_models(models)

# Check if any supported models are available
if not models:
    print("No supported models are available. Exiting.")
    exit()

# Display the available supported models to the user
print("Available Models:")
for i, model in enumerate(models):
    print(f"{i + 1}. {model}")

# Ask the user to select a model
while True:
    try:
        choice = int(input("Select a model by entering its number: "))
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            break
        else:
            print("Invalid choice. Please select a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"You selected: {selected_model}")

# Ask the user to input a sentence
sentence = input("Enter a sentence to check its tokenization: ")

# Create the prompt to ask about tokenization
PROMPT = f"How many tokens are in this sentence: '{sentence}'? Provide details about the tokenization."

# Define the API endpoint for Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Prepare the payload for the API request
payload = {
    "model": selected_model,
    "prompt": PROMPT,
    "stream": True  # Enable streaming mode
}

try:
    # Send a POST request to Ollama's API with streaming enabled
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True,timeout=300)  # Set a timeout of 30 seconds

    # Check if the request was successful
    if response.status_code == 200:
        print("\nModel Response:")
        # Process the streaming response chunk by chunk
        for chunk in response.iter_lines():
            if chunk:
                try:
                    # Decode each chunk as JSON and extract the "response" field
                    data = json.loads(chunk.decode("utf-8"))
                    print(data.get("response", ""), end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"\nError decoding chunk: {e}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

except requests.exceptions.Timeout:
    print("\nError: The request timed out. The model took too long to respond.")
