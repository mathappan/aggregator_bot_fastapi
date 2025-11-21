import logging
import time
import json
from typing import Union
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from models import (
    UnifiedResponse, ResponseType, ResponseStatus, SectionType, Section, 
    ResponseData, ResponseMetadata, ProductSearchResult, ClarificationOption,
    ApparelRecommendation, CompleteApparelResponse, GeneralFashionResponse, 
    ErrorResponse, RecommendationWithProducts, CombinedInput, DirectSearchResponse, AmbiguousResponse,
    UserPreferences
)
from config import MAX_FILE_SIZE_MB, ALLOWED_IMAGE_TYPES
from services.groq_service import analyze_image_with_groq, encode_image_to_base64, classify_user_intent, get_text_recommendations, get_general_fashion_response, generate_context_from_chat_history
from services.search_service import search_parallel_recommendations, search_products_direct
from services.explanation_service import add_explanations_to_products, create_explanation_context
from utils.parsers import parse_recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Apparel Recommendation API",
    description="AI-powered apparel recommendations based on image analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_user_preferences(preferences_json: str) -> UserPreferences:
    """Parse and validate user preferences JSON."""
    if not preferences_json:
        return UserPreferences()  # Empty defaults
    
    try:
        preferences_data = json.loads(preferences_json)
        return UserPreferences(**preferences_data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Invalid preferences format: {e}")
        return UserPreferences()  # Graceful fallback


@app.post("/recommend-apparel-with-text", response_model=UnifiedResponse)
async def recommend_apparel_with_text(
    request: Request,
    chat_uuid: str = Form(...),
    image: UploadFile = File(None),
    text: str = Form(None),
    budget: str = Form(None),
    gender: str = Form(None),
    chat_history: str = Form(None),
    preferences: str = Form(None)
):
    """
    Upload an image and provide text context to get AI-powered apparel recommendations with product searches.
    
    Args:
        image: Image file (JPEG, PNG, WebP supported)
        text: Additional text context from the user
        
    Returns:
        CompleteApparelResponse: Recommendations with parallel product searches
    """
    start_time = time.time()
    
    # Validate that at least one input is provided
    if not image and not text:
        raise HTTPException(
            status_code=400,
            detail="At least one input (image or text) must be provided"
        )
    
    # Validate image if provided
    content = None
    if image:
        if not image.content_type or image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image. Supported types: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
        
        # Check file size
        content = await image.read()
        max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_size_bytes:
            raise HTTPException(
                status_code=400, 
                detail=f"Image file too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
    
    try:
        # Encode image to base64 if image is provided
        image_base64 = None
        if content:
            image_base64 = encode_image_to_base64(content)
        
        print("user text - ", text)
        print("chat uuid - ", chat_uuid)
        print("budget - ", budget)
        print("gender - ", gender)
        print("has image - ", image is not None)
        
        # Parse user preferences
        preferences_obj = parse_user_preferences(preferences)
        print("preferences - ", preferences_obj.dict() if preferences_obj else "None")
        
        # Parse chat history and extract history images
        parsed_chat_history = None
        history_images = {}
        
        if chat_history:
            try:
                parsed_chat_history = json.loads(chat_history)
                print(f"Parsed chat history with {len(parsed_chat_history.get('user_messages', []))} user messages")
            except json.JSONDecodeError as e:
                print(f"Failed to parse chat_history: {e}")
                parsed_chat_history = {"user_messages": [], "bot_messages": []}
        
        # Extract history images from form data
        form_data = await request.form()
        for field_name, field_value in form_data.items():
            if field_name.startswith("chathistoryimage") and hasattr(field_value, 'file'):
                history_images[field_name] = field_value
                print(f"Found history image: {field_name}")
        
        print(f"Extracted {len(history_images)} history images")
        
        # Generate context from chat history if available
        conversation_context = ""
        if parsed_chat_history and (parsed_chat_history.get("user_messages") or parsed_chat_history.get("bot_messages")):
            conversation_context = await generate_context_from_chat_history(
                parsed_chat_history, 
                history_images, 
                current_text=text
            )
            print(f"Generated conversation context: {conversation_context[:100]}...")
        
        # Step 1: Determine intent based on input type
        if image and not text:
            # Image only → COMPLEMENT intent
            intent = "COMPLEMENT"
            print(f"Image-only input → Intent: {intent}")
        elif text:
            # Text included → Use LLM for intent detection
            intent_result = await classify_user_intent(text, has_image=bool(image))
            intent = intent_result.get("intent", "COMPLEMENT")
            print(f"Intent classified as: {intent} - {intent_result.get('reason', '')}")
        else:
            raise HTTPException(status_code=400, detail="At least one input (image or text) must be provided")
        
        # Parse budget if provided
        budget_float = None
        if budget:
            try:
                budget_float = float(budget)
                print(f"Using budget filter: {budget_float}")
            except ValueError:
                print(f"Invalid budget format: {budget}, ignoring budget filter")
        
        # Step 2: Process based on intent and return unified response
        processing_time = time.time() - start_time
        
        if intent == "AMBIGUOUS":
            # Generate general response using LLM with chat history context
            from services.groq_service import handle_ambiguous_intent_with_context
            
            llm_response = await handle_ambiguous_intent_with_context(text or "", conversation_context)
            
            return UnifiedResponse(
                response_type=ResponseType.GENERAL,
                status=ResponseStatus.SUCCESS,
                data=ResponseData(
                    sections=[],
                    clarification_options=[],
                    message=llm_response.get("content", "I'd be happy to help with your fashion needs!")
                ),
                metadata=ResponseMetadata(
                    processing_time_seconds=round(processing_time, 2),
                    original_query=text,
                    detected_items=[]
                )
            )
        
        elif intent == "SEARCH":
            # Direct product search with unified response
            if not text:
                raise HTTPException(status_code=400, detail="Text input required for product search")
            
            search_result = await search_products_direct(text, max_results=25, budget=budget_float, chat_uuid=chat_uuid, context=conversation_context, gender=gender, preferences=preferences_obj)
            
            # Convert search results to Pydantic models
            raw_products = search_result.get('products', [])
            
            # Generate explanations for search products
            if raw_products:
                explanation_context = create_explanation_context(
                    user_text=text,
                    conversation_context=conversation_context,
                    preferences=preferences_obj
                )
                
                # Format products for explanation generation (same structure as COMPLEMENT flow)
                search_group = [{
                    'recommendation': search_result.get('search_query', text),
                    'products': raw_products
                }]
                
                products_with_explanations = await add_explanations_to_products(
                    search_group,
                    explanation_context,
                    max_concurrent=3
                )
                
                # Extract products with explanations
                explained_products = products_with_explanations[0].get('products', []) if products_with_explanations else raw_products
                products = [ProductSearchResult(**product) for product in explained_products]
            else:
                products = []
            
            # Check if any products were found
            if not products:
                response = UnifiedResponse(
                    response_type=ResponseType.SEARCH,
                    status=ResponseStatus.ERROR,
                    data=ResponseData(
                        sections=[],
                        clarification_options=[]
                    ),
                    metadata=ResponseMetadata(
                        processing_time_seconds=round(processing_time, 2),
                        original_query=text,
                        search_query=search_result.get('search_query', text),
                        detected_items=[]
                    ),
                    error=search_result.get('error') or "Looks like we don't have anything matching right now. You could try a broader search."
                )
            else:
                # Create search results section
                search_section = Section(
                    section_type=SectionType.SEARCH_RESULTS,
                    title=search_result.get('search_query', text),
                    description=f"Items matching your search for '{text}'",
                    products=products,
                    total_results=search_result.get('total_results', 0),
                    show_more_available=len(products) < search_result.get('total_results', 0)
                )
                
                response = UnifiedResponse(
                    response_type=ResponseType.SEARCH,
                    status=ResponseStatus.SUCCESS,
                    data=ResponseData(
                        sections=[search_section],
                        clarification_options=[]
                    ),
                    metadata=ResponseMetadata(
                        processing_time_seconds=round(processing_time, 2),
                        original_query=text,
                        search_query=search_result.get('search_query', text),
                        detected_items=[]
                    ),
                    error=search_result.get('error')
                )
            
            logger.info(f"Successfully processed SEARCH request with {len(products)} products in {processing_time:.2f}s")
            return response
        
        elif intent == "COMPLEMENT":
            # Complementary recommendation logic with unified response
            if image:
                # Handle image-based or image+text recommendations
                recommendation_text = await analyze_image_with_groq(image_base64, user_text=text, context=conversation_context)
                recommendation = parse_recommendation(recommendation_text)
            else:
                # Text-only complementary recommendations
                recommendation_text = await get_text_recommendations(text, context=conversation_context)
                recommendation = parse_recommendation(recommendation_text)
            
            # Search for products in parallel for each recommendation
            search_results = await search_parallel_recommendations(
                recommendation.complementary_items,
                max_results_per_item=5,
                budget=budget_float,
                chat_uuid=chat_uuid,
                gender=gender
            )
            
            # Generate explanations for products
            explanation_context = create_explanation_context(
                user_text=text,
                conversation_context=conversation_context,
                image_analysis=recommendation_text if image else None,
                preferences=preferences_obj
            )
            
            search_results_with_explanations = await add_explanations_to_products(
                search_results,
                explanation_context,
                max_concurrent=3  # Limit concurrent explanations for performance
            )
            
            # Convert search results to sections
            sections = []
            for result in search_results_with_explanations:
                products = [ProductSearchResult(**product) for product in result.get('products', [])]
                
                # Only create section if products were found
                if products:
                    section = Section(
                        section_type=SectionType.RECOMMENDATION,
                        title=result['recommendation'],
                        description=f"Perfect {result['recommendation'].lower()} to complement your style",
                        products=products,
                        total_results=result.get('total_results', 0),
                        show_more_available=len(products) < result.get('total_results', 0)
                    )
                    sections.append(section)
            
            # Check if any products were found
            if not sections:
                response = UnifiedResponse(
                    response_type=ResponseType.COMPLEMENT,
                    status=ResponseStatus.ERROR,
                    data=ResponseData(
                        sections=[],
                        clarification_options=[]
                    ),
                    metadata=ResponseMetadata(
                        processing_time_seconds=round(processing_time, 2),
                        original_query=text,
                        detected_items=recommendation.complementary_items
                    ),
                    error="Looks like we don't have matching recommendations for these products yet. Want me to suggest something else?"
                )
            else:
                response = UnifiedResponse(
                    response_type=ResponseType.COMPLEMENT,
                    status=ResponseStatus.SUCCESS,
                    data=ResponseData(
                        sections=sections,
                        clarification_options=[]
                    ),
                    metadata=ResponseMetadata(
                        processing_time_seconds=round(processing_time, 2),
                        original_query=text,
                        detected_items=recommendation.complementary_items
                    )
                )
            
            total_products = sum(len(section.products) for section in sections)
            logger.info(f"Successfully processed COMPLEMENT request with {total_products} products in {processing_time:.2f}s")
            return response
        
        else:
            # Fallback error case
            return UnifiedResponse(
                response_type=ResponseType.GENERAL,
                status=ResponseStatus.ERROR,
                data=ResponseData(sections=[], clarification_options=[]),
                metadata=ResponseMetadata(
                    processing_time_seconds=round(processing_time, 2),
                    original_query=text,
                    detected_items=[]
                ),
                error=f"Unknown intent: {intent}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        processing_time = time.time() - start_time
        
        # Return error in unified format
        return UnifiedResponse(
            response_type=ResponseType.GENERAL,
            status=ResponseStatus.ERROR,
            data=ResponseData(sections=[], clarification_options=[]),
            metadata=ResponseMetadata(
                processing_time_seconds=round(processing_time, 2),
                original_query=text,
                detected_items=[]
            ),
            error=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "apparel-recommendation-api"}


if __name__ == "__main__":
    import uvicorn
    
    # Configure all multipart size limits to 10MB
    import starlette.formparsers
    starlette.formparsers.max_multipart_size = 10 * 1024 * 1024  # 10MB total
    
    # Also need to patch the multipart parser class
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_part_size = 10 * 1024 * 1024  # 10MB per part
    
    uvicorn.run(app, host="0.0.0.0", port=8000)