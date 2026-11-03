import os
from dotenv import load_dotenv

load_dotenv()

# Provider settings
LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "openrouter"

# API keys
OPENROUTER_API_KEY = os.getenv("OR_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq LLM
LLM_MODEL = "openai/gpt-oss-20b"

# OpenRouter embeddings
EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b:free"

# API endpoints
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
