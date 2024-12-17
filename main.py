import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import sys
from dotenv import load_dotenv
import os
# streamlit run main.py


# Load environment variables
load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLM_Directory = os.getenv("LLM_Directory")
sys.path.append(r'%s'%(LLM_Directory))

# Local imports
from ai_support import *
from htmlTemplates import *

embedding_options = ['bge-base-en', 'stella', 'stella_lite', 'MiniLM-L6', 'OpenAI']
llm_options = ['OpenAI', 'meta-llama', 'Qwen', 'MiniChat', 'sentence-transformers', 'Ollama3.2', 'Ollama3']

# Disable Hugging Face symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize session state for chat history, index, and conversation chain
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

st.title("📄 Document-Based Chatbot")

st.sidebar.header("Configuration")

# Directory input
directory = st.sidebar.text_input(
    "Enter the directory path containing your documents:",
    value='./data'
)

# Dropdown for Embedding Model Selection
emb_option = st.sidebar.selectbox(
    "Select Embedding Model:",
    options=embedding_options,
    index=embedding_options.index('MiniLM-L6')  # Default selection
)

# Dropdown for LLM Model Selection
llm_option = st.sidebar.selectbox(
    "Select LLM Model:",
    options=llm_options,
    index=llm_options.index('Ollama3.2')  # Default selection
)

load_docs = st.sidebar.button("Load Documents")

if load_docs:
    with st.spinner("Loading documents and setting up the index..."):
        try:
            documents = SimpleDirectoryReader(directory).load_data()
            # Set up embedding model
            Settings.embed_model = get_embedding(emb_option=emb_option)
            # Set up LLM
            Settings.llm = get_LLM(llm_option=llm_option, env_vars=os.environ)
            # Create the vector index
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.session_state.documents_loaded = True
            st.success("Documents loaded and index created successfully!")
        except Exception as e:
            st.error(f"Error loading documents: {e}")

# Display selected models in the sidebar
if st.session_state.documents_loaded:
    st.sidebar.subheader("Selected Models")
    st.sidebar.write(f"**Embedding Model:** {emb_option}")
    st.sidebar.write(f"**LLM Model:** {llm_option}")

if st.session_state.documents_loaded:
    st.header("💬 Chat with Your Documents")
    # Use a form to handle user input and submission
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        with st.spinner("Generating response..."):
            try:
                query_engine = st.session_state.index.as_query_engine()
                response = query_engine.query(user_input)
                
                # Ensure response is a string
                response_text = str(response)
                
                # Update chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response_text
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")

    st.markdown("---")
    st.subheader("Chat History")

    # Create a scrollable chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history in reverse order (newest first)
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please load your documents to start chatting.")
