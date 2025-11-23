import logging
from typing import Optional
from models import WhyThisWorksExplanation, UserPreferences
from .groq_service import generate_contextual_tag

logger = logging.getLogger(__name__)

# Tag categories with specific tags for different contexts
TAG_CATEGORIES = {
    "confidence": ["Confidence Builder", "Power Piece", "Authority Essential"],
    "versatile": ["Versatile Essential", "Mix & Match Hero", "Wardrobe Staple"],
    "occasion_work": ["Meeting Ready", "Boardroom Essential", "Professional Power"],
    "occasion_networking": ["Networking Perfect", "Connection Builder", "Event Ready"],
    "occasion_date": ["Date Perfect", "Romance Ready", "Evening Essential"],
    "occasion_casual": ["Weekend Warrior", "Casual Chic", "Everyday Essential"],
    "style_professional": ["Executive Choice", "Business Essential", "Corporate Chic"],
    "style_edgy": ["Statement Maker", "Edge Essential", "Bold Choice"],
    "style_romantic": ["Romantic Essential", "Feminine Touch", "Soft Power"],
    "style_casual": ["Effortless Style", "Relaxed Essential", "Comfort Chic"],
    "fit": ["Perfect Fit", "Flattering Choice", "Silhouette Enhancer"],
    "trend": ["Trend Forward", "Modern Essential", "Style Update"],
    "classic": ["Classic Investment", "Timeless Choice", "Heritage Essential"],
    "creative": ["Creative Expression", "Artistic Essential", "Unique Find"]
}

# Fallback tags for when no specific category matches
FALLBACK_TAGS = [
    "Style Essential",
    "Perfect Pick", 
    "Recommended Choice",
    "Smart Selection"
]

# Keywords to look for in explanation content
CONTENT_KEYWORDS = {
    "confidence": ["confident", "authority", "power", "bold", "strong", "commanding"],
    "versatile": ["versatile", "multiple", "mix", "match", "wardrobe", "essential", "staple"],
    "professional": ["professional", "work", "business", "corporate", "executive", "office"],
    "casual": ["casual", "relaxed", "everyday", "comfortable", "effortless"],
    "classic": ["classic", "timeless", "investment", "traditional", "enduring"],
    "trend": ["trend", "modern", "contemporary", "current", "updated"],
    "edgy": ["edgy", "statement", "bold", "striking", "dramatic"],
    "romantic": ["romantic", "feminine", "soft", "elegant", "graceful"],
    "fit": ["fit", "flattering", "silhouette", "shape", "proportion", "balanced"]
}


async def generate_recommendation_tag(
    explanation: WhyThisWorksExplanation, 
    preferences: Optional[UserPreferences] = None,
    product_title: str = "",
    product_description: str = "",
    context: str = None
) -> str:
    """
    Generate a recommendation tag using Groq LLM based on explanation content and user preferences.
    
    Args:
        explanation: The why this works explanation
        preferences: User style preferences
        product_title: Product title/name
        product_description: Product description
        context: Additional styling context
        
    Returns:
        A concise tag describing why this product was recommended
    """
    try:
        # Convert explanation to dict for LLM processing
        explanation_dict = {}
        if explanation:
            explanation_dict = {
                'summary': explanation.summary if hasattr(explanation, 'summary') else None,
                'styling_principle': explanation.styling_principle if hasattr(explanation, 'styling_principle') else None,
                'occasion_logic': explanation.occasion_logic if hasattr(explanation, 'occasion_logic') else None,
                'colour_logic': explanation.colour_logic if hasattr(explanation, 'colour_logic') else None,
                'proportion_logic': explanation.proportion_logic if hasattr(explanation, 'proportion_logic') else None,
                'versatility_logic': explanation.versatility_logic if hasattr(explanation, 'versatility_logic') else None
            }
        
        # Convert preferences to dict for LLM processing
        preferences_dict = None
        if preferences:
            preferences_dict = {
                'moods': preferences.moods if hasattr(preferences, 'moods') and preferences.moods else [],
                'vibes': preferences.vibes if hasattr(preferences, 'vibes') and preferences.vibes else [],
                'occasions': preferences.occasions if hasattr(preferences, 'occasions') and preferences.occasions else []
            }
        
        # Use Groq LLM to generate contextual tag
        tag = await generate_contextual_tag(
            product_title=product_title or "Product",
            product_description=product_description or "",
            explanation=explanation_dict,
            user_preferences=preferences_dict,
            context=context
        )
        
        logger.info(f"Generated LLM-based tag: '{tag}' for product: {product_title[:30]}...")
        return tag
        
    except Exception as e:
        logger.error(f"Error generating recommendation tag with LLM: {str(e)}")
        # Fallback to rule-based system
        try:
            fallback_tag = _get_fallback_tag_rule_based(explanation, preferences)
            logger.info(f"Using rule-based fallback tag: {fallback_tag}")
            return fallback_tag
        except Exception as fallback_error:
            logger.error(f"Fallback tag generation also failed: {str(fallback_error)}")
            return "Style Essential"


def _get_fallback_tag_rule_based(
    explanation: WhyThisWorksExplanation, 
    preferences: Optional[UserPreferences] = None
) -> str:
    """
    Fallback rule-based tag generation when LLM fails.
    """
    try:
        # Step 1: Try preference-based tags (highest priority)
        if preferences:
            pref_tag = _get_preference_based_tag(explanation, preferences)
            if pref_tag:
                return pref_tag
        
        # Step 2: Try content-based tags
        content_tag = _get_content_based_tag(explanation)
        if content_tag:
            return content_tag
        
        # Step 3: Fallback tag
        return FALLBACK_TAGS[0]  # "Style Essential"
        
    except Exception as e:
        logger.error(f"Error in rule-based fallback: {str(e)}")
        return "Style Essential"


def _get_preference_based_tag(
    explanation: WhyThisWorksExplanation, 
    preferences: UserPreferences
) -> Optional[str]:
    """
    Generate tag based on user preferences combined with explanation content.
    """
    # Combine all explanation text for analysis
    explanation_text = _get_explanation_text(explanation).lower()
    
    # Check mood-based tags
    if preferences.moods:
        for mood in preferences.moods:
            if mood == "confident":
                if any(keyword in explanation_text for keyword in ["structure", "sharp", "authority", "power"]):
                    return "Confidence Builder"
                elif any(keyword in explanation_text for keyword in ["professional", "business"]):
                    return "Power Piece"
            
            elif mood == "creative":
                if any(keyword in explanation_text for keyword in ["unique", "artistic", "interesting"]):
                    return "Creative Expression"
            
            elif mood == "bold":
                if any(keyword in explanation_text for keyword in ["statement", "striking", "dramatic"]):
                    return "Statement Maker"
    
    # Check vibe-based tags
    if preferences.vibes:
        for vibe in preferences.vibes:
            if vibe == "professional":
                if any(keyword in explanation_text for keyword in ["meeting", "work", "business"]):
                    return "Meeting Ready"
                elif any(keyword in explanation_text for keyword in ["executive", "corporate"]):
                    return "Executive Choice"
            
            elif vibe == "edgy":
                if any(keyword in explanation_text for keyword in ["bold", "statement", "edge"]):
                    return "Edge Essential"
    
    # Check occasion-based tags
    if preferences.occasions:
        for occasion in preferences.occasions:
            if occasion == "work":
                return "Meeting Ready"
            elif occasion == "networking":
                return "Networking Perfect"
            elif occasion == "date":
                return "Date Perfect"
            elif occasion == "casual":
                return "Casual Chic"
    
    return None


def _get_content_based_tag(explanation: WhyThisWorksExplanation) -> Optional[str]:
    """
    Generate tag based on explanation content analysis.
    """
    explanation_text = _get_explanation_text(explanation).lower()
    
    # Analyze different explanation fields with priorities
    
    # High priority: styling_principle and occasion_logic
    if explanation.styling_principle:
        principle_text = explanation.styling_principle.lower()
        
        if any(keyword in principle_text for keyword in CONTENT_KEYWORDS["confidence"]):
            return TAG_CATEGORIES["confidence"][0]  # "Confidence Builder"
        elif any(keyword in principle_text for keyword in CONTENT_KEYWORDS["versatile"]):
            return TAG_CATEGORIES["versatile"][0]  # "Versatile Essential"
        elif any(keyword in principle_text for keyword in CONTENT_KEYWORDS["classic"]):
            return TAG_CATEGORIES["classic"][0]  # "Classic Investment"
    
    # Medium priority: versatility_logic
    if explanation.versatility_logic:
        return TAG_CATEGORIES["versatile"][1]  # "Mix & Match Hero"
    
    # Check occasion_logic
    if explanation.occasion_logic:
        occasion_text = explanation.occasion_logic.lower()
        
        if any(keyword in occasion_text for keyword in ["work", "meeting", "business"]):
            return TAG_CATEGORIES["occasion_work"][0]  # "Meeting Ready"
        elif any(keyword in occasion_text for keyword in ["networking", "event"]):
            return TAG_CATEGORIES["occasion_networking"][0]  # "Networking Perfect"
        elif any(keyword in occasion_text for keyword in ["date", "evening"]):
            return TAG_CATEGORIES["occasion_date"][0]  # "Date Perfect"
    
    # Check proportion_logic and silhouette_logic
    if explanation.proportion_logic or explanation.silhouette_logic:
        return TAG_CATEGORIES["fit"][0]  # "Perfect Fit"
    
    # Check trend_relevance
    if explanation.trend_relevance:
        return TAG_CATEGORIES["trend"][0]  # "Trend Forward"
    
    # General content analysis
    for category, keywords in CONTENT_KEYWORDS.items():
        if any(keyword in explanation_text for keyword in keywords):
            if category in ["confidence"]:
                return TAG_CATEGORIES["confidence"][0]
            elif category in ["versatile"]:
                return TAG_CATEGORIES["versatile"][0]
            elif category in ["professional"]:
                return TAG_CATEGORIES["style_professional"][0]
            elif category in ["classic"]:
                return TAG_CATEGORIES["classic"][0]
            elif category in ["trend"]:
                return TAG_CATEGORIES["trend"][0]
    
    return None


def _get_explanation_text(explanation: WhyThisWorksExplanation) -> str:
    """
    Combine all explanation fields into a single text for analysis.
    """
    text_parts = []
    
    if explanation.summary:
        text_parts.append(explanation.summary)
    if explanation.colour_logic:
        text_parts.append(explanation.colour_logic)
    if explanation.proportion_logic:
        text_parts.append(explanation.proportion_logic)
    if explanation.texture_logic:
        text_parts.append(explanation.texture_logic)
    if explanation.occasion_logic:
        text_parts.append(explanation.occasion_logic)
    if explanation.styling_principle:
        text_parts.append(explanation.styling_principle)
    if explanation.silhouette_logic:
        text_parts.append(explanation.silhouette_logic)
    if explanation.versatility_logic:
        text_parts.append(explanation.versatility_logic)
    if explanation.trend_relevance:
        text_parts.append(explanation.trend_relevance)
    
    return " ".join(text_parts)


def get_available_tags() -> dict:
    """
    Return all available tag categories for reference.
    Used for testing and documentation purposes.
    """
    return TAG_CATEGORIES


def validate_tag(tag: str) -> bool:
    """
    Validate that a tag is from the approved list.
    """
    all_tags = []
    for category_tags in TAG_CATEGORIES.values():
        all_tags.extend(category_tags)
    all_tags.extend(FALLBACK_TAGS)
    
    return tag in all_tags