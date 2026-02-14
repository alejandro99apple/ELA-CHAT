import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI


def main():
    load_dotenv()

    persistent_directory = "db/chroma_db"
    query = "What ALS stand for?"


    # -------------------------
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    relevant_docs = retriever.invoke(query)

    print(f"Query: {query}")
    print("---Context---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i} metadata: {doc.metadata}...")
        print(f"\ntext: {doc.page_content}")
        print("----" * 20)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # -------------------------
    # NUEVO MODELO (LLAMA 70B)
    # -------------------------
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct:groq",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know based on the provided documents.'"
            },
            {
                "role": "user",
                "content": f"""
                
                Question:
                {query}

                Context:
                {context}
                """
            }
        ],
        temperature=0.2,
        max_tokens=300,
    )

    print("\n---Answer---")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
