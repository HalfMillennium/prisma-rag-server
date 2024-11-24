import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import openai
import os
from utils import CHROMA_PATH, embed_text

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

        # Initialize the persistent client
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Load the existing collection
    collection = client.get_or_create_collection(name="document_embeddings")
    query_embedding = embed_text(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    #if len(results) == 0 or results[0][1] < 0.5:  # Lower the threshold to 0.5
    #    print(f"Unable to find matching results.")
    #    return
    print('results:', results)
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    context_text = "\n\n---\n\n".join(documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [metadata.get("source", None) for metadata in metadatas]
    formatted_response = f"Response: {response_text}\nSources: {set(sources)}"
    print(formatted_response)


if __name__ == "__main__":
    main()