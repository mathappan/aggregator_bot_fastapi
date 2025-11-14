import logging
import numpy as np
from typing import List, Dict, Any
from redis import asyncio as aioredis
from redis.commands.search.query import Query
from .voyageai_service import rerank_documents
from .groq_service import check_product_relevance

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv('.env.txt'))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_INDEX_NAME = "idx:product_text_description_embedding"

logger = logging.getLogger(__name__)

# Initialize async Redis client
redis_client_async = aioredis.Redis(
    host='redis-19403.c241.us-east-1-4.ec2.redns.redis-cloud.com',
    port=19403,
    decode_responses=True,
    username="default",
    password="LkHHfYttUJPNchVhbHtDNN6HrxgZC6OW"
)


class SearchResult:
    """Represents a product search result."""
    def __init__(self, doc):
        self.vector_score = float(doc.vector_score) if hasattr(doc, 'vector_score') else 0.0
        self.product_id = doc.product_id if hasattr(doc, 'product_id') else ""
        self.text_description = doc.text_description if hasattr(doc, 'text_description') else ""
        self.product_type = doc.product_type if hasattr(doc, 'product_type') else ""
        self.price1 = doc.price1 if hasattr(doc, 'price1') else ""
        self.price2 = doc.price2 if hasattr(doc, 'price2') else ""
        self.price3 = doc.price3 if hasattr(doc, 'price3') else ""
        self.price4 = doc.price4 if hasattr(doc, 'price4') else ""
        self.price5 = doc.price5 if hasattr(doc, 'price5') else ""
        self.price6 = doc.price6 if hasattr(doc, 'price6') else ""
        self.max_price = doc.max_price if hasattr(doc, 'max_price') else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            'vector_score': self.vector_score,
            'product_id': self.product_id,
            'text_description': self.text_description,
            'product_type': self.product_type,
            'prices': {
                'price1': self.price1,
                'price2': self.price2,
                'price3': self.price3,
                'price4': self.price4,
                'price5': self.price5,
                'price6': self.price6
            },
            'max_price': self.max_price
        }


async def store_combined_ids_cache(chat_uuid: str, combined_ids: List[str]) -> None:
    """
    Store combined_ids in Redis cache for a specific chat_uuid.
    
    Args:
        chat_uuid: The chat session identifier
        combined_ids: List of combined_ids to cache
    """
    try:
        cache_key = f"chat_cache:{chat_uuid}:combined_ids"
        # Add new combined_ids to the existing set
        if combined_ids:
            await redis_client_async.sadd(cache_key, *combined_ids)
            # Set expiration to 24 hours
            await redis_client_async.expire(cache_key, 86400)
            logger.info(f"Cached {len(combined_ids)} combined_ids for chat_uuid: {chat_uuid}")
    except Exception as e:
        logger.error(f"Error storing combined_ids cache for chat_uuid {chat_uuid}: {str(e)}")


async def get_cached_combined_ids(chat_uuid: str) -> List[str]:
    """
    Retrieve cached combined_ids for a specific chat_uuid.
    
    Args:
        chat_uuid: The chat session identifier
        
    Returns:
        List of cached combined_ids
    """
    try:
        cache_key = f"chat_cache:{chat_uuid}:combined_ids"
        cached_ids = await redis_client_async.smembers(cache_key)
        logger.info(f"Retrieved {len(cached_ids)} cached combined_ids for chat_uuid: {chat_uuid}")
        return list(cached_ids) if cached_ids else []
    except Exception as e:
        logger.error(f"Error retrieving combined_ids cache for chat_uuid {chat_uuid}: {str(e)}")
        return []


async def search_products_by_vector(query_embedding: List[float], limit: int = 25, budget: float = None, chat_uuid: str = None, query_text: str = None, enable_llm_filtering: bool = True, gender: str = None) -> List[Dict[str, Any]]:
    """
    Search for products using vector similarity with optional price filtering.
    
    Args:
        query_embedding: The embedding vector for the search query
        limit: Maximum number of results to return
        budget: Optional budget filter (maximum price)
        
    Returns:
        List of product metadata dictionaries
    """
    try:
        # Get cached combined_ids for exclusion if chat_uuid is provided
        excluded_combined_ids = []
        if chat_uuid:
            excluded_combined_ids = await get_cached_combined_ids(chat_uuid)
        
        # Build filters
        filters = []
        
        # Add price filter if budget is specified
        if budget is not None:
            price_filter = f"@max_price:[0 ({budget}]"
            filters.append(price_filter)
        
        # Add exclusion filter for cached combined_ids
        if excluded_combined_ids:
            exclude_ids_filter = " ".join([f"-@combined_id:{pid.replace(':', '\:')}"
                                         for pid in excluded_combined_ids])
            filters.append(exclude_ids_filter)
        
        # Add gender filter if specified
        if gender:
            # Include unisex when gender is male or female
            if gender.lower() in ["male", "female"]:
                allowed_genders = [gender.lower(), "unisex"]
            else:
                allowed_genders = [gender.lower()]
            
            if len(allowed_genders) == 1:
                # Use simple term search for single gender
                gender_filter = f"@gender:{{{allowed_genders[0]}}}"
            else:
                # For multiple genders, create OR filter
                gender_filters = [f"@gender:{{{allowed_gender}}}" for allowed_gender in allowed_genders]
                gender_filter = "(" + " | ".join(gender_filters) + ")"
            
            filters.append(gender_filter)
        
        # Combine all filters
        if filters:
            combined_filter = " ".join(filters)
            filtered_redis_query = (
                Query(f'({combined_filter})=>[KNN 50 @vector $query_vector AS vector_score]')
                .sort_by('vector_score')
                .return_fields('vector_score', 'internal_id', 'text_description', 'max_price', 'combined_id', 'gender')
                .paging(0, limit)
                .dialect(2)
            )
        else:
            filtered_redis_query = (
                Query('(*)=>[KNN 50 @vector $query_vector AS vector_score]')
                .sort_by('vector_score')
                .return_fields('vector_score', 'internal_id', 'text_description', 'max_price', 'combined_id', 'gender')
                .paging(0, limit)
                .dialect(2)
            )
        
        search_results = await redis_client_async.ft('idx:product_text_description_embedding').search(
            filtered_redis_query,
            {
                'query_vector': np.array(query_embedding, dtype=np.float32).tobytes()
            }
        )
        
        metadata_product_keys = [i['id'].replace('product_text_description', 'product_metadata') for i in search_results.docs]
        fields = ['title', 'image_url', 'product_url', 'max_price', 'store_name']

        pipe = redis_client_async.pipeline()
        for field in fields:
            pipe.json().mget(metadata_product_keys, f'.{field}')

        results = await pipe.execute()

        # Transpose the results and create dictionaries
        metadata = [
            dict(zip(fields, values)) 
            for values in zip(*results)
        ]
        
        # Apply reranking if query_text is provided and we have results
        if query_text and metadata and search_results.docs:
            try:
                # Extract text descriptions for reranking
                documents = []
                product_data = []
                
                for i, doc in enumerate(search_results.docs):
                    if hasattr(doc, 'text_description') and i < len(metadata):
                        documents.append(doc.text_description)
                        # Store both the metadata and the combined_id for later use
                        product_data.append({
                            'metadata': metadata[i],
                            'combined_id': doc['combined_id']
                        })
                
                if documents:
                    # Rerank the documents
                    reranked = await rerank_documents(query_text, documents, top_k=min(10, len(documents)))
                    
                    if reranked and reranked.results:
                        # Create reranked results with relevance scores
                        reranked_metadata = []
                        combined_ids_to_cache = []
                        
                        for r in reranked.results:
                            original_data = product_data[r.index]
                            # Add relevance score to metadata
                            reranked_item = original_data['metadata'].copy()
                            reranked_item['relevance_score'] = r.relevance_score
                            reranked_metadata.append(reranked_item)
                            
                            # Collect combined_ids for caching
                            if original_data['combined_id']:
                                combined_ids_to_cache.append(original_data['combined_id'])
                        
                        metadata = reranked_metadata
                        
                        # Apply LLM filtering if enabled
                        if enable_llm_filtering and query_text and metadata:
                            try:
                                import asyncio
                                
                                # Create filtering tasks for all products
                                filtering_tasks = []
                                for i, r in enumerate(reranked.results):
                                    # Use the original document from reranked results
                                    if r.index < len(documents):
                                        task = check_product_relevance(query_text, documents[r.index])
                                        filtering_tasks.append((i, task))
                                
                                # Execute all filtering tasks in parallel
                                if filtering_tasks:
                                    results = await asyncio.gather(*[task for _, task in filtering_tasks], return_exceptions=True)
                                    
                                    # Filter products based on LLM results
                                    filtered_metadata = []
                                    filtered_combined_ids = []
                                    
                                    for (item_idx, _), result in zip(filtering_tasks, results):
                                        if isinstance(result, Exception):
                                            logger.warning(f"LLM filtering failed for item {item_idx}: {result}")
                                            # Include item if filtering fails
                                            if item_idx < len(metadata):
                                                filtered_metadata.append(metadata[item_idx])
                                                if item_idx < len(combined_ids_to_cache):
                                                    filtered_combined_ids.append(combined_ids_to_cache[item_idx])
                                        elif result.get('match', False):
                                            # Include item if LLM says it matches
                                            if item_idx < len(metadata):
                                                item_with_filter_reason = metadata[item_idx].copy()
                                                item_with_filter_reason['llm_filter_reason'] = result.get('reason', '')
                                                filtered_metadata.append(item_with_filter_reason)
                                                if item_idx < len(combined_ids_to_cache):
                                                    filtered_combined_ids.append(combined_ids_to_cache[item_idx])
                                        else:
                                            logger.info(f"LLM filtered out product: {result.get('reason', 'No reason provided')}")
                                    
                                    metadata = filtered_metadata
                                    combined_ids_to_cache = filtered_combined_ids
                                    
                                    logger.info(f"LLM filtering: {len(filtered_metadata)}/{len(reranked_metadata)} products passed filter")
                                
                            except Exception as e:
                                logger.error(f"Error during LLM filtering: {str(e)}, using unfiltered results")
                        
                        # Cache the reranked combined_ids
                        if chat_uuid and combined_ids_to_cache:
                            await store_combined_ids_cache(chat_uuid, combined_ids_to_cache)
                        
                        logger.info(f"Final results: {len(metadata)} products after reranking and filtering")
                    else:
                        logger.warning("Reranking failed, using original results")
                        # Fall back to original caching logic
                        if chat_uuid and search_results.docs:
                            combined_ids_to_cache = [doc.get('combined_id', '') for doc in search_results.docs if doc.get('combined_id')]
                            if combined_ids_to_cache:
                                await store_combined_ids_cache(chat_uuid, combined_ids_to_cache)
            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}, using original results")
                # Fall back to original caching logic
                if chat_uuid and search_results.docs:
                    combined_ids_to_cache = [doc.get('combined_id', '') for doc in search_results.docs if doc.get('combined_id')]
                    if combined_ids_to_cache:
                        await store_combined_ids_cache(chat_uuid, combined_ids_to_cache)
        else:
            # Original caching logic when no reranking
            if chat_uuid and search_results.docs:
                combined_ids_to_cache = [doc.get('combined_id', '') for doc in search_results.docs if doc.get('combined_id')]
                if combined_ids_to_cache:
                    await store_combined_ids_cache(chat_uuid, combined_ids_to_cache)
        
        logger.info(f"Found {len(metadata)} products for vector search")
        return metadata
        
    except Exception as e:
        logger.error(f"Error searching products by vector: {str(e)}")
        return []


async def test_redis_connection() -> bool:
    """Test Redis connection."""
    try:
        await redis_client_async.ping()
        logger.info("Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        return False