# Import dependencies
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
# from fastmcp import FastMCP
from qdrant_client import QdrantClient
from openai import OpenAI


# Load variables defined in the .env file 
load_dotenv()

PORT = os.environ.get("PORT", 8000)

# Initialize the MCP server with a unique service name
mcp = FastMCP("qdrant-rag-server", host="0.0.0.0", port=PORT)

# Initialize the Qdrant client using URL and API key from environment variables
qdrant = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

# Initialize the OpenAI client (expects OPENAI_API_KEY in environment variables)
openai = OpenAI()

async def embed_query(query: str) -> list[float]:
    """
    Convert a user query string into a dense vector embedding
    using OpenAI's text-embedding-3-large model.
    """
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    
    # Return the embedding vector from the first (and only) input
    return response.data[0].embedding

async def search_qdrant(embedding: list[float], collection: str, top_k: int):
    """
    Perform a vector similarity search in the specified Qdrant collection
    using the provided embedding.
    """
    return qdrant.query_points(
        collection_name=collection,
        query=embedding,
        limit=top_k
    )

def combine_context(search_results):
    """
    Combine the retrieved document chunks into a single context string.
    Safely handles missing payload fields.
    """
    context = ""

    # Iterate over retrieved points from Qdrant
    for i in search_results.points:
        # Extract the text content from the payload, if present
        context += i.payload['content']
    return context

@mcp.tool()
async def retrieve_context(query: str, collection: str, top_k: int=10) -> dict:
    """
    MCP tool that retrieves relevant document context for a user query.

    This function:
    1. Embeds the query
    2. Searches Qdrant for similar vectors
    3. Combines retrieved text into a single context block

    The returned context is intended to be passed to an LLM
    for final answer generation.
    """

    # Generate embedding for the user query
    embedding = await embed_query(query)

    # Perform vector search in Qdrant
    results = await search_qdrant(embedding, collection, top_k)

     # Combine retrieved document chunks into a single context string
    context = combine_context(results)
    
    # Return context in a structured format for downstream use
    return context

if __name__ == "__main__":
    mcp.run(transport="streamable-http")