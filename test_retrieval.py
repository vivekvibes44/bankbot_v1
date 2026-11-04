from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
)

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
)

vector_store = FAISS.load_local(
    str(VECTOR_STORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

question = "What is the personal loan interest rate?"

documents = vector_store.similarity_search(
    question,
    k=3,
    filter={"category": "loans"},
    fetch_k=10,
)

for doc in documents:
    print("\nCategory:", doc.metadata.get("category"))
    print("Content:", doc.page_content)