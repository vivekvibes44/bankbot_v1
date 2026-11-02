from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
)

# Test Groq LLM
print("Testing Groq LLM...")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0,
)

response = llm.invoke("Reply with exactly: Groq connection successful")
print("Groq response:", response.content)


# Test OpenRouter embeddings
print("\nTesting OpenRouter embeddings...")

embeddings = OpenAIEmbeddings(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

vector = embeddings.embed_query("Aarohan Bank savings account")

print("Embedding successful!")
print("Vector dimensions:", len(vector))