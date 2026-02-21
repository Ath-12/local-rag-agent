import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local RAG Agent", page_icon="🤖", layout="centered")
st.header("🤖 Local RAG Agent (Llama 3)")

# --- 1. SESSION STATE SETUP ( The "Short-Term Memory" ) ---
# Streamlit refreshes the page every time you click a button.
# We use "session_state" to keep variables alive between refreshes.

if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores the chat history

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None # Stores the RAG engine so we don't reload it every second

# --- 2. SIDEBAR SETUP ( The "Control Panel" ) ---
with st.sidebar:
    st.title("Settings")
    st.write("Current Model: **Llama 3:latest**")
    
    # Button to Load Data
    if st.button("🔄 Load/Reload Data"):
        with st.spinner("Loading and Indexing your data..."):
            # A. Setup the Brain & Translator
            Settings.llm = Ollama(model="llama3:latest", request_timeout=300.0)
            Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # B. Read the Files
            try:
                documents = SimpleDirectoryReader("data").load_data()
                st.sidebar.success(f"Loaded {len(documents)} docs.")
                
                # C. Build the Index (Memory)
                index = VectorStoreIndex.from_documents(documents)
                
                # D. Save the Engine to Session State
                st.session_state.query_engine = index.as_query_engine()
                st.sidebar.success("Index Ready!")
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# --- 3. CHAT INTERFACE ( The "UI" ) ---

# A. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# B. Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # 1. Save user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check if the engine is ready
    if st.session_state.query_engine is None:
        with st.chat_message("assistant"):
            st.error("⚠️ Please click 'Load Data' in the sidebar first!")
    else:
        # 3. Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
                
                # 4. Save AI message to history
                st.session_state.messages.append({"role": "assistant", "content": response.response})