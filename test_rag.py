
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
)

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"

# Embeddings
embeddings = OpenAIEmbeddings(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

# Load FAISS
vector_store = FAISS.load_local(
    str(VECTOR_STORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0,
)

# Ask a question
question = "What is the minimum balance for the savings account?"

# Retrieve relevant chunks
documents = retriever.invoke(question)

context = "\n\n".join(
    document.page_content for document in documents
)

# Prompt
prompt = f"""
You are the Ready Bank Knowledge Assistant.

Answer the user's question using ONLY the context below.

Rules:
- Do not invent banking policies.
- If the answer is not in the context, say you don't have enough
  information in the available Ready Bank documents.
- Clearly state that policies are fictional when relevant.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nQuestion:", question)
print("\nAnswer:")
print(response.content)

print("\nSources:")
for document in documents:
    print(
        document.metadata.get("source"),
        "Page:",
        document.metadata.get("page"),
    )