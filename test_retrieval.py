
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
)

# Path to saved FAISS index
BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

# Load existing FAISS index
vector_store = FAISS.load_local(
    str(VECTOR_STORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Ask a test question
question = "What is the interest rate for a Ready Bank personal loan?"

print(f"\nQuestion: {question}\n")

results = retriever.invoke(question)

for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Category:", doc.metadata.get("category"))
    print("Content:")
    print(doc.page_content)
    print()