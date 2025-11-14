import asyncio
import base64
import json
import logging
from typing import List
from groq import AsyncGroq
from fastapi import HTTPException

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY, GROQ_MODEL, MAX_CONCURRENT_REQUESTS
from prompts import (
    APPAREL_RECOMMENDATION_PROMPT, TEXT_CLASSIFICATION_PROMPT,
    TEXT_RECOMMENDATION_PROMPT, GENERAL_FASHION_ASSISTANT_PROMPT
)

logger = logging.getLogger(__name__)

# Initialize Groq client and semaphore
client = AsyncGroq(api_key=GROQ_API_KEY)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def build_conversation(image_data: str = None, user_text: str = None, context: str = None) -> list:
    """Build conversation for Groq API with optional image analysis and user text."""
    if image_data:
        base_text = "You are a personal stylist. Analyze this clothing item and the context from the user to recommend the ideal pieces of apparel to complement it."
        if user_text:
            base_text += f" Additional context from user: {user_text}"
        if context:
            base_text += f" Conversation context: {context}"
        
        return [
            {
                "role": "system",
                "content": APPAREL_RECOMMENDATION_PROMPT
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
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    else:
        # Text-only conversation
        base_text = f"You are a personal stylist. Based on this user text, recommend ideal pieces of apparel: {user_text}"
        if context:
            base_text += f" Conversation context: {context}"
        
        return [
            {
                "role": "system",
                "content": APPAREL_RECOMMENDATION_PROMPT
            },
            {
                "role": "user",
                "content": base_text
            }
        ]


async def analyze_image_with_groq(image_data: str = None, user_text: str = None, context: str = None) -> str:
    """Send image and/or text to Groq API for analysis and get apparel recommendation."""
    async with semaphore:
        try:
            conversation = build_conversation(image_data, user_text, context)
            
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=conversation,
                max_tokens=1000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            logger.info(f"Successfully received response from Groq API for image analysis - {response.choices[0].message.content}")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to analyze image: {str(e)}"
            )


async def classify_text_intent(user_text: str) -> str:
    """Classify user text to determine if it needs recommendations or general fashion assistance."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": TEXT_CLASSIFICATION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                max_tokens=100,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            logger.info("Successfully classified text intent")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error classifying text intent: {str(e)}")
            # Default to general_agent on error
            return '{"classification": "general_agent"}'


async def get_text_recommendations(user_text: str, context: str = None) -> str:
    """Get apparel recommendations based on text description."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": TEXT_RECOMMENDATION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"{user_text}{f' Conversation context: {context}' if context else ''}"
                    }
                ],
                max_tokens=1000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            logger.info(f"Successfully generated text-based recommendations - { response.choices[0].message.content}")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text recommendations: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate recommendations: {str(e)}"
            )


async def get_general_fashion_response(user_text: str) -> str:
    """Get general fashion assistance response."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": GENERAL_FASHION_ASSISTANT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                max_tokens=1000,
                temperature=0.8
            )
            
            logger.info("Successfully generated general fashion response")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating general fashion response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate fashion response: {str(e)}"
            )


def encode_image_to_base64(image_content: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_content).decode('utf-8')


async def classify_user_intent(user_text: str, has_image: bool = False) -> dict:
    """
    Classify user intent into one of three categories: COMPLEMENT, SEARCH, or AMBIGUOUS.
    
    Args:
        user_text: The user's text input
        has_image: Whether the user uploaded an image
        
    Returns:
        Dict with 'intent' and 'reason'
    """
    system_prompt = f'''
    You are an expert fashion assistant that classifies user intent.

    User Input: "{user_text}"
    Has Image: {has_image}

    Classify the user's intent into exactly ONE of these categories:

    COMPLEMENT - User wants items that go WITH their outfit/item. Keywords: "what goes with", "complement", "match with", "pair with", "style with", "wear with"

    SEARCH - User wants to FIND similar or same items. Keywords: "find", "looking for", "where to buy", "similar to", "like this", "search for", "want this"

    AMBIGUOUS - Cannot determine intent clearly, or the request is too vague

    Rules:
    - If user has an image and NO text, classify as COMPLEMENT
    - Be decisive - choose the most likely intent based on language patterns
    - Only use AMBIGUOUS for genuinely unclear requests

    Output format (JSON):
    {{
      "intent": "COMPLEMENT|SEARCH|AMBIGUOUS",
      "reason": "Brief explanation of classification"
    }}
    '''
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Intent classification: {result.get('intent', 'UNKNOWN')}")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying user intent: {str(e)}")
            # Default to COMPLEMENT if classification fails
            return {"intent": "COMPLEMENT", "reason": f"Error in classification: {str(e)}"}


async def generate_search_query(user_text: str, context: str = None) -> dict:
    """
    Convert user message into an optimized search query for product search.
    
    Args:
        user_text: The user's search request
        
    Returns:
        Dict with 'search_query'
    """
    system_prompt = f'''
    You are an expert at converting user requests into effective product search queries.

    User Request: "{user_text}"
    {f"Conversation Context: {context}" if context else ""}

    Your task:
    Generate a clean, optimized search query that will find the best matching products.

    Guidelines:
    - Make the search query specific but not overly restrictive
    - Remove filler words and focus on searchable terms
    - If user mentions specific brands, include them
    - Consider synonyms for better matching
    - Use conversation context to better understand the request

    Output format (JSON):
    {{
      "search_query": "optimized search terms"
    }}
    '''
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=100
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Generated search query: {result.get('search_query', '')}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            # Fallback to using original user text as search query
            return {"search_query": user_text}


async def generate_context_from_chat_history(chat_history: dict, history_images: dict, current_text: str = None) -> str:
    """
    Generate contextual information from chat history to enhance current request understanding.
    
    Args:
        chat_history: Parsed chat history with user_messages and bot_messages
        history_images: Dict of history image field names to image files
        current_text: Current user input text
        
    Returns:
        Generated context string to enhance understanding
    """
    try:
        # Build conversation summary
        conversation_summary = []
        
        # Process user messages
        user_messages = chat_history.get("user_messages", [])
        bot_messages = chat_history.get("bot_messages", [])
        
        # Create chronological conversation flow
        for user_msg in user_messages[-3:]:  # Last 3 user messages
            content = user_msg.get("content", "")
            if content.startswith("chathistoryimage"):
                conversation_summary.append(f"User uploaded an image ({content})")
            else:
                conversation_summary.append(f"User: {content}")
        
        # Add bot message summaries
        for bot_msg in bot_messages[-2:]:  # Last 2 bot responses
            response_data = bot_msg.get("response_data", {})
            response_type = response_data.get("response_type", "unknown")
            
            if response_type == "COMPLEMENT":
                sections = response_data.get("data", {}).get("sections", [])
                items = [section.get("title", "") for section in sections]
                conversation_summary.append(f"Bot recommended: {', '.join(items)}")
            elif response_type == "SEARCH":
                sections = response_data.get("data", {}).get("sections", [])
                if sections:
                    search_query = sections[0].get("title", "")
                    conversation_summary.append(f"Bot searched for: {search_query}")
            elif response_type == "AMBIGUOUS":
                conversation_summary.append("Bot asked for clarification")
        
        # Create context prompt
        context_prompt = f'''
        You are analyzing a conversation to provide context for the current fashion request.
        
        Conversation History:
        {chr(10).join(conversation_summary)}
        
        Current User Input: "{current_text or 'New image uploaded'}"
        
        Based on this conversation history, provide contextual insights that will help generate better fashion recommendations. Consider:
        1. User's style preferences shown in previous requests
        2. Items already recommended or searched for
        3. Conversation flow and user's evolving needs
        4. Any patterns in their fashion interests
        
        Generate a concise context summary (2-3 sentences) that captures the most relevant information for improving the current recommendation.
        '''
        
        async with semaphore:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": context_prompt
                    }
                ],
                temperature=0.4,
                max_tokens=200
            )
            
            context = response.choices[0].message.content
            logger.info(f"Generated context from chat history: {context[:100]}...")
            return context
            
    except Exception as e:
        logger.error(f"Error generating context from chat history: {str(e)}")
        return ""


async def handle_ambiguous_intent_with_context(user_text: str, conversation_context: str) -> dict:
    """
    Handle ambiguous intent by generating a general fashion response using chat history context.
    
    Args:
        user_text: The user's current message
        conversation_context: Already generated context from chat history
        
    Returns:
        Dict with 'content' for the response
    """
    context = conversation_context or "No previous context available."
    
    system_prompt = f'''
    You are an expert fashion assistant helping a user with their fashion needs.
    
    Current user message: "{user_text}"
    
    Previous conversation context:
    {context}
    
    Based on the conversation history and current message, provide a helpful general fashion response that addresses their query. Use the context to make your response more personalized and relevant.
    
    Output format (JSON):
    {{
      "content": "A helpful general fashion response addressing their query based on context"
    }}
    '''
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user", 
                        "content": system_prompt
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("Generated general response for ambiguous intent with context")
            return result
            
        except Exception as e:
            logger.error(f"Error handling ambiguous intent with context: {str(e)}")
            return {
                "content": "I'd be happy to help with your fashion needs! Could you provide a bit more detail about what you're looking for?"
            }


async def check_product_relevance(recommendation: str, product_description: str) -> dict:
    """
    Check if a product is relevant to the recommendation using LLM.
    
    Args:
        recommendation: The stylist recommendation text
        product_description: The product's text description
        
    Returns:
        Dict with 'match' boolean and 'reason' string
    """
    system_prompt = f'''
    You are an expert fashion stylist evaluating product relevance.

    Your task is to determine if a product matches a styling recommendation.

    Recommendation: {recommendation}
    Product Description: {product_description}

    Analyze if this product is a good match for the recommendation based on:
    1. Style compatibility 
    2. Occasion appropriateness
    3. Color/pattern coordination potential
    4. Overall fashion sense

    Be strict but reasonable in your evaluation. Only approve products that genuinely complement the recommendation.

    Output format (JSON):
    {{
      "match": true/false,
      "reason": "Brief explanation of why it matches or doesn't match"
    }}
    '''
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Product relevance check: {result.get('match', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking product relevance: {str(e)}")
            # Default to allowing the product if LLM fails
            return {"match": True, "reason": f"Error in relevance check: {str(e)}"}