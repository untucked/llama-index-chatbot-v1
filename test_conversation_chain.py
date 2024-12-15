# test_conversation_chain.py

from ai_support import get_embedding, get_LLM, get_conversation_chain
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from dotenv import load_dotenv

def test_conversation_chain():
    # Load environment variables
    load_dotenv()
    env_vars = os.environ

    # Load sample documents
    directory = './data'  # Ensure this directory exists and contains documents
    documents = SimpleDirectoryReader(directory).load_data()

    # Initialize embeddings and LLM
    embedding = get_embedding(emb_option='sentence_trans')
    Settings.embed_model = embedding
    Settings.llm = get_LLM(llm_option='Ollama3.2', env_vars=env_vars)

    index = VectorStoreIndex.from_documents(documents, 
                                            embed_model=embedding
                                            )
    
    # Initialize the Conversational Retrieval Chain
    conversation_chain = get_conversation_chain(Settings, index, embedding)
    
    # Test a sample query
    query = "How does this company make money?"
    response = conversation_chain({"question": query})
    
    print("Question:", query)
    print("Answer:", response['answer'])

if __name__ == "__main__":
    test_conversation_chain()
