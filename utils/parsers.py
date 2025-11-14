import json
import logging
from models import ApparelRecommendation

logger = logging.getLogger(__name__)


def parse_recommendation(response_text: str) -> ApparelRecommendation:
    """Parse the Groq JSON response into structured data."""
    try:
        # Try to parse as JSON first
        json_data = json.loads(response_text.strip())
        return ApparelRecommendation(**json_data)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from text that might have extra content
        try:
            # Look for JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                json_data = json.loads(json_str)
                return ApparelRecommendation(**json_data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        logger.warning(f"Failed to parse JSON response: {response_text[:100]}...")
        return ApparelRecommendation(
            complementary_items=["Unable to parse recommendation"]
        )
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {str(e)}")
        return ApparelRecommendation(
            complementary_items=["Error parsing response"]
        )