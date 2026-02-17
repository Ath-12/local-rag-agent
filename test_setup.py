from llama_index.llms.ollama import Ollama

# Initialize the LLM
# We set request_timeout to 60.0 because running locally on CPU/Hybrid can sometimes be slow
llm = Ollama(model="llama3:latest", request_timeout=60.0)

print("1. Sending a test message to llama3:latest...")
resp = llm.complete("What is the capital of France?")
print(f"2. Response received: {resp}")