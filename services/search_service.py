import asyncio
import logging
from typing import List, Dict, Any

from .voyageai_service import get_single_embedding
from .redis_service import search_products_by_vector, SearchResult
from .groq_service import generate_search_query

logger = logging.getLogger(__name__)


async def search_single_recommendation(recommendation: str, max_results: int = 25, budget: float = None, chat_uuid: str = None, gender: str = None) -> Dict[str, Any]:
    """
    Search for products based on a single stylist recommendation.
    
    Args:
        recommendation: The stylist recommendation text
        max_results: Maximum number of search results to return
        budget: Optional budget filter (maximum price)
        
    Returns:
        Dictionary containing the recommendation and its search results
    """
    try:
        # Get embedding for the recommendation
        embedding = await get_single_embedding(recommendation)
        
        if not embedding:
            logger.warning(f"Failed to get embedding for recommendation: {recommendation}")
            return {
                'recommendation': recommendation,
                'products': [],
                'error': 'Failed to generate embedding'
            }
        
        # Search for products using the embedding
        search_results = await search_products_by_vector(embedding, max_results, budget, chat_uuid, recommendation, True, gender)
        
        # Results are already dictionaries
        products = search_results
        
        logger.info(f"Found {len(products)} products for recommendation: {recommendation}")
        
        return {
            'recommendation': recommendation,
            'products': products,
            'total_results': len(products)
        }
        
    except Exception as e:
        logger.error(f"Error searching for recommendation '{recommendation}': {str(e)}")
        return {
            'recommendation': recommendation,
            'products': [],
            'error': str(e)
        }


async def search_parallel_recommendations(recommendations: List[str], max_results_per_item: int = 25, budget: float = None, chat_uuid: str = None, gender: str = None) -> List[Dict[str, Any]]:
    """
    Search for products based on multiple stylist recommendations in parallel.
    
    Args:
        recommendations: List of stylist recommendation texts
        max_results_per_item: Maximum number of search results per recommendation
        budget: Optional budget filter (maximum price)
        
    Returns:
        List of dictionaries, each containing a recommendation and its search results
    """
    try:
        logger.info(f"Starting parallel search for {len(recommendations)} recommendations")
        
        # Create tasks for parallel execution
        tasks = [
            search_single_recommendation(recommendation, max_results_per_item, budget, chat_uuid, gender)
            for recommendation in recommendations
        ]
        
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in parallel search for recommendation {i}: {str(result)}")
                processed_results.append({
                    'recommendation': recommendations[i] if i < len(recommendations) else f"Unknown {i}",
                    'products': [],
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        total_products = sum(len(result.get('products', [])) for result in processed_results)
        logger.info(f"Parallel search completed. Found {total_products} total products across {len(recommendations)} recommendations")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in parallel recommendations search: {str(e)}")
        return [
            {
                'recommendation': rec,
                'products': [],
                'error': 'Parallel search failed'
            }
            for rec in recommendations
        ]


async def search_products_direct(user_text: str, max_results: int = 25, budget: float = None, chat_uuid: str = None, context: str = None, gender: str = None) -> Dict[str, Any]:
    """
    Search for products directly based on user's search intent (SEARCH intent).
    
    Args:
        user_text: The user's search request
        max_results: Maximum number of search results to return
        budget: Optional budget filter (maximum price)
        chat_uuid: Chat session identifier for caching
        
    Returns:
        Dictionary containing search results
    """
    try:
        # Generate optimized search query
        query_result = await generate_search_query(user_text, context=context)
        search_query = query_result.get('search_query', user_text)
        
        logger.info(f"Direct search for: '{search_query}' (original: '{user_text}')")
        
        # Get embedding for the search query
        embedding = await get_single_embedding(search_query)
        
        if not embedding:
            logger.warning(f"Failed to get embedding for search query: {search_query}")
            return {
                'search_query': search_query,
                'products': [],
                'error': 'Failed to generate embedding'
            }
        
        # Search for products using the embedding
        search_results = await search_products_by_vector(
            embedding, 
            max_results, 
            budget, 
            chat_uuid, 
            search_query,
            True,  # enable_llm_filtering
            gender
        )
        
        logger.info(f"Found {len(search_results)} products for direct search: {search_query}")
        
        return {
            'search_query': search_query,
            'original_query': user_text,
            'products': search_results,
            'total_results': len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Error in direct product search '{user_text}': {str(e)}")
        return {
            'search_query': user_text,
            'products': [],
            'error': str(e)
        }