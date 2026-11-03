from pathlib import Path
import csv
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


def load_documents():
    documents = []

    for file_path in KNOWLEDGE_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        file_type = file_path.suffix.lower()
        category = file_path.parent.name
        source = str(file_path.relative_to(BASE_DIR))

        # PDF documents
        if file_type == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": source,
                                    "page": page_number,
                                    "category": category,
                                    "source_type": "pdf",
                                },
                            )
                        )

        # Markdown and plain text documents
        elif file_type in [".md", ".txt"]:
            text = file_path.read_text(encoding="utf-8")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "category": category,
                            "source_type": file_type[1:],
                        },
                    )
                )

        # CSV documents: one document per row
        elif file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    content = "\n".join(
                        f"{key}: {value}" for key, value in row.items() if key and value
                    )

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "category": category,
                                "source_type": "csv",
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
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found. Check the knowledge folder and supported file types."
        )

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    embeddings = create_embeddings()

    print("Creating FAISS vector store one chunk at a time...")

    # Embed the first chunk and create the vector store
    vector_store = FAISS.from_documents(
        [chunks[0]],
        embeddings,
    )

    print(f"Embedded chunk 1 of {len(chunks)}")

    # Embed and add remaining chunks individually
    for index, chunk in enumerate(chunks[1:], start=2):
        vector_store.add_documents([chunk])

        print(f"Embedded chunk {index} of {len(chunks)}")

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(VECTOR_STORE_DIR))

    print("FAISS index saved successfully!")
    print(f"Location: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_vector_store()
