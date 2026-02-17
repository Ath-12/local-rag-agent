import sys
import os

# 1. Import the "Organs"
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("--- Starting RAG Setup ---")

# 2. Configure the Brain (Llama 3)
# We increase timeout because reading a whole PDF takes a moment
Settings.llm = Ollama(model="llama3:latest", request_timeout=300.0)

# 3. Configure the Translator (Embeddings)
# We use a specific, small, fast model from HuggingFace that runs on your CPU
print("1. Loading Embedding Model (This downloads ~80MB the first time)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Load the Data (The "Reading" Phase)
# This looks inside the 'data' folder and reads EVERY file it finds
print("2. Reading files from 'data' folder...")
try:
    documents = SimpleDirectoryReader("data").load_data()
    print(f"   -> Found {len(documents)} pages of text.")
except Exception as e:
    print(f"Error reading data: {e}")
    sys.exit()

# 5. Create the Index (The "Memorizing" Phase)
# This turns text into numbers and stores them in RAM
print("3. Building the Vector Index (Memorizing)...")
index = VectorStoreIndex.from_documents(documents)

# 6. Ask a Question (The "Test")
print("--- RAG Ready! ---")
while True:
    user_input = input("\nAsk a question about your document (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    # Create a query engine (the thing that searches the index)
    query_engine = index.as_query_engine()
    
    print("Thinking...")
    response = query_engine.query(user_input)
    
    print("\nAnswer:")
    print(response)