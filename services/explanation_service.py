import asyncio
import logging
from typing import List, Dict, Any, Optional

from models import ProductSearchResult, WhyThisWorksExplanation
from .groq_service import generate_product_explanation
from .tag_service import generate_recommendation_tag

logger = logging.getLogger(__name__)


class ExplanationContext:
    """Context data for generating product explanations."""
    
    def __init__(self, 
                 user_text: str = None,
                 conversation_context: str = None,
                 image_analysis: str = None,
                 occasion: str = None,
                 user_preferences: Dict[str, Any] = None,
                 preferences = None):
        self.user_text = user_text or ""
        self.conversation_context = conversation_context or ""
        self.image_analysis = image_analysis or ""
        self.occasion = occasion or ""
        self.user_preferences = user_preferences or {}
        self.preferences = preferences
    
    def to_context_string(self) -> str:
        """Convert context to a string for LLM processing."""
        context_parts = []
        
        if self.user_text:
            context_parts.append(f"User request: {self.user_text}")
        
        if self.conversation_context:
            context_parts.append(f"Conversation context: {self.conversation_context}")
            
        if self.image_analysis:
            context_parts.append(f"Image analysis: {self.image_analysis}")
            
        if self.occasion:
            context_parts.append(f"Occasion: {self.occasion}")
            
        if self.user_preferences:
            pref_str = ", ".join([f"{k}: {v}" for k, v in self.user_preferences.items()])
            context_parts.append(f"User preferences: {pref_str}")
            
        # Add new style preferences
        if self.preferences and (self.preferences.moods or self.preferences.vibes or self.preferences.occasions):
            pref_parts = []
            if self.preferences.moods:
                pref_parts.append(f"Moods: {', '.join(self.preferences.moods)}")
            if self.preferences.vibes:
                pref_parts.append(f"Vibes: {', '.join(self.preferences.vibes)}")
            if self.preferences.occasions:
                pref_parts.append(f"Occasions: {', '.join(self.preferences.occasions)}")
            
            if pref_parts:
                context_parts.append(f"Style preferences - {' | '.join(pref_parts)}")
        
        return " | ".join(context_parts)


async def generate_explanation_for_product(
    recommendation: str,
    product: ProductSearchResult,
    context: ExplanationContext
) -> Optional[WhyThisWorksExplanation]:
    """
    Generate a 'Why This Works' explanation for a single product.
    
    Args:
        recommendation: The stylist recommendation this product fulfills
        product: The product details
        context: User context for personalized explanations
        
    Returns:
        WhyThisWorksExplanation object or None if generation fails
    """
    try:
        explanation_data = await generate_product_explanation(
            recommendation=recommendation,
            product_title=product.title,
            context=context.to_context_string(),
            preferences=context.preferences
        )
        
        if explanation_data:
            return WhyThisWorksExplanation(**explanation_data)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to generate explanation for product '{product.title}': {str(e)}")
        return None


async def add_explanations_to_products(
    products_with_recommendations: List[Dict[str, Any]],
    context: ExplanationContext,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Add explanations to multiple product recommendation groups in parallel.
    
    Args:
        products_with_recommendations: List of dicts containing 'recommendation' and 'products'
        context: User context for explanations
        max_concurrent: Maximum concurrent explanation generations
        
    Returns:
        Updated list with explanations added to products
    """
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_product_group(group: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single recommendation group."""
            async with semaphore:
                recommendation = group.get('recommendation', '')
                products = group.get('products', [])
                
                # Generate explanations for each product in parallel
                explanation_tasks = [
                    generate_explanation_for_product(
                        recommendation=recommendation,
                        product=ProductSearchResult(**product),
                        context=context
                    )
                    for product in products
                ]
                
                explanations = await asyncio.gather(*explanation_tasks, return_exceptions=True)
                
                # Add explanations and tags to products
                for i, product in enumerate(products):
                    if i < len(explanations) and not isinstance(explanations[i], Exception):
                        explanation = explanations[i]
                        if explanation:
                            product['why_this_works'] = explanation.dict()
                            
                            # Generate recommendation tag based on explanation and preferences
                            try:
                                tag = generate_recommendation_tag(explanation, context.preferences)
                                product['recommendation_tag'] = tag
                                logger.info(f"Generated tag '{tag}' for product: {product.get('title', 'Unknown')[:30]}...")
                            except Exception as e:
                                logger.error(f"Failed to generate tag for product: {str(e)}")
                                product['recommendation_tag'] = "Style Essential"
                
                return group
        
        # Process all groups in parallel
        tasks = [process_product_group(group) for group in products_with_recommendations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception processing group {i}: {str(result)}")
                # Return original group without explanations
                processed_results.append(products_with_recommendations[i])
            else:
                processed_results.append(result)
        
        explanation_count = sum(
            len([p for p in group.get('products', []) if 'why_this_works' in p])
            for group in processed_results
        )
        
        logger.info(f"Successfully generated {explanation_count} product explanations")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in add_explanations_to_products: {str(e)}")
        return products_with_recommendations


def create_explanation_context(
    user_text: str = None,
    conversation_context: str = None,
    image_analysis: str = None,
    preferences = None,
    **kwargs
) -> ExplanationContext:
    """
    Factory function to create ExplanationContext from API inputs.
    
    Args:
        user_text: User's text input
        conversation_context: Context from chat history
        image_analysis: Results from image analysis
        **kwargs: Additional context parameters
        
    Returns:
        ExplanationContext object
    """
    return ExplanationContext(
        user_text=user_text,
        conversation_context=conversation_context,
        image_analysis=image_analysis,
        occasion=kwargs.get('occasion'),
        user_preferences=kwargs.get('user_preferences', {}),
        preferences=preferences
    )


async def add_explanation_to_single_product(
    product: ProductSearchResult,
    recommendation: str,
    context: ExplanationContext
) -> ProductSearchResult:
    """
    Add explanation and tag to a single ProductSearchResult.
    
    Args:
        product: The product to explain
        recommendation: The recommendation it fulfills
        context: User context
        
    Returns:
        Updated ProductSearchResult with explanation and tag
    """
    explanation = await generate_explanation_for_product(recommendation, product, context)
    
    if explanation:
        product.why_this_works = explanation
        
        # Generate recommendation tag
        try:
            tag = generate_recommendation_tag(explanation, context.preferences)
            product.recommendation_tag = tag
            logger.info(f"Generated tag '{tag}' for single product: {product.title[:30]}...")
        except Exception as e:
            logger.error(f"Failed to generate tag for single product: {str(e)}")
            product.recommendation_tag = "Style Essential"
    
    return product