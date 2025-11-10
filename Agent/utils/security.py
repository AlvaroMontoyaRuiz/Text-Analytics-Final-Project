import re
from typing import Tuple, Optional

# Regex to detect common prompt injection attempts
PROMPT_INJECTION_PATTERN = re.compile(
    r"ignore.*(instructions|rules|prompt)|"
    r"disregard.*(instructions|rules|prompt)|"
    r"reveal.*(instructions|rules|prompt)|"
    r"print your (instructions|rules|prompt)|"
    r"you are now in (developer|debug) mode|"
    r"forget (your|all) (instructions|rules|prompt)",
    re.IGNORECASE | re.DOTALL
)

# Keywords indicating a user is asking for medical advice
MEDICAL_ADVICE_KEYWORDS = [
    "prescribe", "diagnose", "diagnosis", "treat", "treatment",
    "cure", "prognosis", "what should i take", "is this normal",
    "should i be worried", "medical advice"
]

# A non-negotiable disclaimer to be appended to all responses.
MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** I am an AI assistant and not a medical professional. "
    "The information provided is based on internal documents and is for "
    "informational purposes only. It is not a substitute for professional "
    "medical advice, diagnosis, or treatment. Always seek the advice of "
    "your physician or other qualified health provider."
)

def validate_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Validates user input against security guardrails.
    
    Returns: (is_valid, error_message)
    """
    
    # 1. Check for Prompt Injection (Adversarial Input)
    if PROMPT_INJECTION_PATTERN.search(user_input):
        return (False, "Your query appears to be an attempt to bypass security protocols. Request blocked.")
        
    # 2. Check for Medical Advice Seeking (Domain-Specific Risk)
    for keyword in MEDICAL_ADVICE_KEYWORDS:
        if keyword in user_input.lower():
            return (False, "I cannot provide medical advice, diagnoses, or treatment recommendations. Please consult a qualified health provider.")
            
    # 3. Check for other basic harmful intent (simple filter)
    if "harmful" in user_input.lower() or "illegal" in user_input.lower():
         return (False, "Your query contains potentially harmful content and cannot be processed.")

    return (True, None)


def apply_output_guardrails(response_text: str) -> str:
    """
    Applies security guardrails to the LLM's final output.
    """
    
    # In a clinical setting, EVERY response should have a disclaimer.
    if MEDICAL_DISCLAIMER not in response_text:
        return response_text + MEDICAL_DISCLAIMER
    
    return response_text