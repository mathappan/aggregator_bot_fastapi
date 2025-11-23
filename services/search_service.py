import asyncio
import logging
from typing import List, Dict, Any

from .voyageai_service import get_single_embedding
from .redis_service import search_products_by_vector, SearchResult
from .groq_service import generate_search_query
from config import GROQ_API_KEY, GROQ_MODEL
from groq import AsyncGroq
from prompts import IMAGE_SEARCH_PROMPT

logger = logging.getLogger(__name__)

# Initialize Groq client for image analysis
client = AsyncGroq(api_key=GROQ_API_KEY)
semaphore = asyncio.Semaphore(10)  # Limit concurrent requests


async def analyze_image_for_search(image_base64: str, user_text: str = None, context: str = None) -> str:
    """
    Analyze image to generate search query for product search.
    
    Args:
        image_base64: Base64 encoded image data
        user_text: User's additional text input
        context: Conversation context
        
    Returns:
        Search query string based on image analysis
    """
    async with semaphore:
        try:
            # Build conversation for image analysis
            base_text = "Analyze this clothing/apparel image and generate a search query to find similar items."
            
            if user_text:
                base_text += f" User context: {user_text}"
            if context:
                base_text += f" Additional context: {context}"
            
            conversation = [
                {
                    "role": "system",
                    "content": IMAGE_SEARCH_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": base_text
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=conversation,
                max_tokens=100,
                temperature=0.3  # Lower temperature for consistent search queries
            )
            
            search_query = response.choices[0].message.content.strip()
            
            # Clean up the search query
            search_query = search_query.replace('"', '').replace("'", '').strip()
            
            logger.info(f"Generated search query from image: '{search_query}'")
            return search_query
            
        except Exception as e:
            logger.error(f"Error analyzing image for search: {str(e)}")
            # Fallback to user text or generic search
            fallback = user_text if user_text else "clothing apparel"
            logger.info(f"Using fallback search query: '{fallback}'")
            return fallback


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


async def search_products_direct(user_text: str, max_results: int = 25, budget: float = None, chat_uuid: str = None, context: str = None, gender: str = None, preferences = None, image_base64: str = None) -> Dict[str, Any]:
    """
    Search for products directly based on user's search intent (SEARCH intent).
    
    Args:
        user_text: The user's search request
        max_results: Maximum number of search results to return
        budget: Optional budget filter (maximum price)
        chat_uuid: Chat session identifier for caching
        context: Conversation context
        gender: Gender filter for products
        preferences: User preferences
        image_base64: Base64 encoded image for image-based search
        
    Returns:
        Dictionary containing search results
    """
    try:
        # Handle image-based or text-based search
        if image_base64:
            # Analyze image to get search query
            search_query = await analyze_image_for_search(image_base64, user_text=user_text, context=context)
            logger.info(f"Image-based search query: '{search_query}' (user_text: '{user_text}')")
        else:
            # Generate optimized search query from text
            query_result = await generate_search_query(user_text, context=context, preferences=preferences)
            search_query = query_result.get('search_query', user_text)
            logger.info(f"Text-based search query: '{search_query}' (original: '{user_text}')")
        
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