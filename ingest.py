from pathlib import Path
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"


def load_pdfs():
    documents = []

    for pdf_path in KNOWLEDGE_DIR.rglob("*.pdf"):
        category = pdf_path.parent.name

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(
                                    pdf_path.relative_to(BASE_DIR)
                                ),
                                "page": page_number,
                                "category": category,
                            },
                        )
                    )

    return documents


def create_embeddings():
    return OpenAIEmbeddings(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )


def build_vector_store():
    print("Loading PDF documents...")
    documents = load_pdfs()

    if not documents:
        raise ValueError(
            "No PDF text found. Check the knowledge folder."
        )

    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = create_embeddings()

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(VECTOR_STORE_DIR))

    print("FAISS index saved successfully!")
    print(f"Location: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_vector_store()