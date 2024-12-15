from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import sys
from dotenv import load_dotenv
import os
# local
sys.path.append(r'C:\Users\eylan\Documents\llama')
from ai_support import get_embedding, get_LLM

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# path to LLMs

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

documents = SimpleDirectoryReader('./data').load_data()

# bge-base embedding model
Settings.embed_model = get_embedding(emb_option = 'sentence_trans')
# ollama
Settings.llm = get_LLM(llm_option='Ollama3.2', env_vars=os.environ)
# Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()
response = query_engine.query("How does this company make money?")
print(response)