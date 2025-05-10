import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from flipkart.data_converter import dataconverter

# Load environment variables
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Initialize Sentence Transformer Embedding
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def chunk_list(data, chunk_size=50):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def data_ingestion(status=None):
    # Create Astra DB vector store connection
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="flipkart1",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE
    )

    if status is None:
        docs = dataconverter()

        insert_ids = []
        for batch in chunk_list(docs, chunk_size=50):  # Insert in chunks of 50
            try:
                ids = vstore.add_documents(batch)
                insert_ids.extend(ids)
            except Exception as e:
                print(f"❌ Failed to insert batch: {e}")

        return vstore, insert_ids
    else:
        return vstore

def product_search(query: str, k: int = 5):
    vstore = data_ingestion(status=True)  # Don't ingest again, just reuse
    results = vstore.similarity_search(query, k=k)

    print(f"\nTop {len(results)} results for query: '{query}'")
    seen = set()
    for i, res in enumerate(results):
        if res.page_content not in seen:
            print(f"\nResult {i+1}:\n{res.page_content}\nMetadata: {res.metadata}")
            seen.add(res.page_content)

if __name__ == "__main__":
    vstore, insert_ids = data_ingestion(None)
    print(f"\n✅ Inserted {len(insert_ids)} documents into Astra DB.")

    query = "Which budget laptops have the best bass-heavy sound quality?"
    product_search(query)
