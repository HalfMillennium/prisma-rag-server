import hashlib
import json
from botocore.exceptions import NoCredentialsError
import chromadb
from chromadb import Settings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import logging
import uuid
from logic.utils import CHROMA_PATH, DATA_PATH, HASHES_FILE, embed_text

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def main():
    try:
        generate_data_store()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_embedding_to_db(chunks)

def load_documents() -> list[Document]:
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.md")
        documents = loader.load()
        return documents
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return []
    except IOError as e:
        logging.error(f"Error reading file: {e}")
        return []

def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        document = chunks[10]
        logging.info(document.page_content)
        logging.info(document.metadata)

    return chunks

def compute_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_hashes() -> set:
    if os.path.exists(HASHES_FILE):
        with open(HASHES_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_hashes(hashes: set):
    with open(HASHES_FILE, 'w') as f:
        json.dump(list(hashes), f)

def get_embeddings_from_db():
    """
    Retrieves all embeddings, along with their associated documents, metadata, and IDs, from the ChromaDB.
    """
    try:
        # Initialize the persistent client
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Load the existing collection
        collection = client.get_or_create_collection(name="document_embeddings")
        collection.similarity_search_with_relevance_scores("test", k=10)
        # Fetch all embeddings and associated data
        all_data = collection.get(include=["documents", "metadatas", "embeddings"])

        # Extract and return the data
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        embeddings = all_data.get("embeddings", [])

        logging.info(f"Retrieved {len(embeddings)} embeddings from the database.")
        return {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
            "embeddings": embeddings,
        }

    except Exception as e:
        logging.error(f"Failed to retrieve embeddings from the database: {e}")
        return None


def save_embedding_to_db(chunks: list[Document]):
    # Get and print all embeddings currently in the ChromaDB
    get_embeddings_from_db()

    # Load existing hashes
    existing_hashes = load_hashes()
    new_hashes = set()

    # Filter out duplicate chunks
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = compute_hash(chunk.page_content)
        if chunk_hash not in existing_hashes:
            unique_chunks.append(chunk)
            new_hashes.add(chunk_hash)

    if not unique_chunks:
        logging.info("No new unique chunks to add.")
        return

    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create embeddings for the unique chunks
    embeddings = [embed_text(chunk.page_content) for chunk in unique_chunks]

    # Save the embeddings and metadata to the local database
    try:
        # Initialize the persistent client
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Create or load a collection
        collection = client.get_or_create_collection(name="document_embeddings")

        # Prepare data for insertion
        documents = [chunk.page_content for chunk in unique_chunks]
        metadatas = [chunk.metadata for chunk in unique_chunks]
        ids = [str(uuid.uuid4()) for _ in unique_chunks]

        # Add data to the collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        logging.info(f"Saved {len(unique_chunks)} embeddings to the database.")
    except Exception as e:
        logging.error(f"Failed to save embeddings to the database: {e}")
        return

    # Update the hashes file
    existing_hashes.update(new_hashes)
    save_hashes(existing_hashes)


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info(f"Cleared the database at {CHROMA_PATH}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()