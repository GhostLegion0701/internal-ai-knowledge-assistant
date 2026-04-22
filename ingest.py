from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def main():
    print("Loading documents...")

    loader = DirectoryLoader(
        "documents",
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    print("Embeddings created and stored in ChromaDB.")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()