from ingest import load_documents, create_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = load_documents()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)
embeddings = create_embeddings()

batch_size = 3

for i in range(3):
    print(f"Testing chunk {i + 1} individually...")

    try:
        result = embeddings.embed_documents(
            [chunks[i].page_content]
        )
        print("Success!")

    except Exception as e:
        print(f"Failed on chunk {i + 1}:")
        print(e)