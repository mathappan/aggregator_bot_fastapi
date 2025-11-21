import logging
import os
from typing import List
import voyageai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv('.env.txt'))

VOYAGE_API_KEY = os.environ['VOYAGE_API_KEY']
voyageai.api_key = VOYAGE_API_KEY
vo_client = voyageai.AsyncClient()
VOYAGE_MODEL = "voyage-3-large"

logger = logging.getLogger(__name__)


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using VoyageAI.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        response = await vo_client.embed(
            texts=texts,
            model=VOYAGE_MODEL,
            input_type="document"
        )
        
        embeddings = response.embeddings
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return []


async def get_single_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text string.
    
    Args:
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    try:
        embeddings = await get_embeddings([text])
        return embeddings[0] if embeddings else []
    except Exception as e:
        logger.error(f"Error generating single embedding: {str(e)}")
        return []


async def rerank_documents(query: str, documents: List[str], top_k: int = 10):
    """
    Rerank documents based on relevance to the query using VoyageAI reranking.
    
    Args:
        query: The search query text
        documents: List of document texts to rerank
        top_k: Number of top results to return
        
    Returns:
        Reranking results with relevance scores
    """
    try:
        reranked = await vo_client.rerank(
            query=query,
            documents=documents,
            model="rerank-2",
            top_k=top_k
        )
        logger.info(f"Reranked {len(documents)} documents, returned top {len(reranked.results)}")
        return reranked
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        return None