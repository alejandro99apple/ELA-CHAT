import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings


load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    # Cheack if the directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist. Please create it and add your PDF documents.")
    
    # Load all PDF documents from the specified directory
    loader = DirectoryLoader(
        path=docs_path, 
        glob="**/*.pdf", 
        show_progress=True, 
        loader_cls=PyPDFLoader)
    

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in {docs_path}. Please check the directory and try again.")
    
    
    """ for i, doc in enumerate(documents[:2]):
        print(f"Document {i+1}: ")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:50]}...")  # Print the first 200 characters of the content
        #print(f"  Metadata: {doc.metadata}") """


    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    """ print("-"*50)
    print(f"Split into {len(split_docs)} chunks.")

    for i, doc in enumerate(split_docs[:5]):
        print(f"Chunk {i+1}: ")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:200]}...")  # Print the first 200 characters of the content
        print("-"*50) """

    return split_docs


def create_vector_store(chunks, persistent_directory="db/chroma_db"):
    print("Creating vector store...")

    # Create embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created with {vector_store._collection.count()} vectors.")

    # Persist the vector store to disk
    return vector_store


def main():
    print("Starting ingestion pipeline...")

    #1 Load documents
    documents = load_documents()

    #2 Split documents
    split_docs = split_documents(documents, chunk_size=500, chunk_overlap=100)

    #3 Create vector store
    vector_store = create_vector_store(split_docs)


if __name__ == "__main__":
    main()