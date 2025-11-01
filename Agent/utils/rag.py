"""
utils/rag.py
Contains RAG-related helper functions, including custom chunking configuration
and token counting.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import tiktoken
from typing import Dict, List, Tuple

# --- Configuration based on document analysis ---
# Maps document type code (from filename) to optimal chunking parameters
# Format: (chunk_size_characters, chunk_overlap_characters)
# NOTE: Using character count here as it's simpler than managing tiktoken during splitting,
# although token count is often preferred for LLM context limits.
CHUNK_CONFIG: Dict[str, Tuple[int, int]] = {
    "POL": (600, 100),    # Policy & Protocol: Smaller chunks for specific rules (char count)
    "MED": (1000, 150),   # Medication Logs: Medium chunks for events (char count)
    "ADM": (1800, 250),   # Patient Admission: Medium-Large for context (char count)
    "DIS": (2200, 300),   # Discharge Summaries: Larger for narrative (char count)
    "CLIN": (2400, 400),  # Clinical Consults: Largest for linked sections (char count)
    "GEN": (2000, 300)    # Default/General fallback (char count)
}

# Separators are ordered from most semantic to least semantic (Recursive Strategy)
# Added common clinical section headers/patterns
CUSTOM_SEPARATORS: List[str] = [
    # Specific clinical headers (attempt to split sections)
    "\n\nHistory of Present Illness:",
    "\n\nPast Medical History:",
    "\n\nMedications:",
    "\n\nAllergies:",
    "\n\nReview of Systems:",
    "\n\nPhysical Examination:",
    "\n\nAssessment and Plan:",
    "\n\nDiagnosis:",
    "\n\nProcedure:",
    "\n\nFindings:",
    "\n\nImpression:",
    # General structural separators
    "\n\n#",          # Markdown header style (if present)
    "\n\n",          # Paragraph break (most common semantic block)
    "\n",            # Line break
    ". ",            # Sentence ending with period
    "? ",            # Sentence ending with question mark
    "! ",            # Sentence ending with exclamation mark
    ", ",            # Clause separation
    "; ",            # Clause separation
    " ",             # Word splitter
    "",              # Character splitter (last resort)
]

# --- Tiktoken Encoding Setup ---
try:
    # Use the encoding for the specific model planned for generation
    ENCODING = tiktoken.encoding_for_model("gpt-4o-mini") 
except Exception:
    # Fallback encoding if the specific model encoding isn't found
    print("Warning: gpt-4o-mini encoding not found, using cl100k_base. Token counts may be approximate.")
    ENCODING = tiktoken.get_encoding("cl100k_base")

def get_recursive_splitter(doc_type_code: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter based on the document type code.
    Uses character length for splitting configuration.
    """
    # Use .get() with a default key "GEN" to safely handle unknown doc types
    chunk_size, chunk_overlap = CHUNK_CONFIG.get(doc_type_code, CHUNK_CONFIG["GEN"]) 

    print(f"   [Splitter Setup] Doc Type: {doc_type_code}, Chunk Size (chars): {chunk_size}, Overlap (chars): {chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        separators=CUSTOM_SEPARATORS,
        chunk_size=chunk_size, # Based on character count from CHUNK_CONFIG
        chunk_overlap=chunk_overlap, # Based on character count
        length_function=len, # Use standard Python len() for character count
        is_separator_regex=False, # Treat separators as literal strings
        keep_separator=True # Keep separators to maintain context better
    )
    return splitter

def count_tokens(messages: List[Dict]) -> int:
    """
    Counts the total tokens in a list of message dicts using the loaded tiktoken encoder.
    Only considers the 'content' field of each message dictionary.
    """
    total_tokens = 0
    if not isinstance(messages, list):
         print("Warning: count_tokens expected a list of dicts, received:", type(messages))
         return 0
         
    for m in messages:
        # Check if 'm' is a dictionary and has a 'content' key
        if isinstance(m, dict) and "content" in m:
             content = m["content"]
             # Check if content is a string before encoding
             if isinstance(content, str):
                 try:
                     total_tokens += len(ENCODING.encode(content))
                 except Exception as e:
                      print(f"Warning: Error encoding content during token count: {e}. Content: '{content[:50]}...'")
             else:
                 # Handle cases where content might not be a string (e.g., None)
                  print(f"Warning: Message content is not a string, skipping token count. Content: {content}")
        # else:
            # Optionally log if a message is not in the expected format
            # print(f"Warning: Message format invalid or missing 'content', skipping token count. Message: {m}")
            
    return total_tokens

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Test splitter configuration
    pol_splitter = get_recursive_splitter("POL")
    print(f"Policy Splitter Config - Size: {pol_splitter._chunk_size}, Overlap: {pol_splitter._chunk_overlap}")
    
    unknown_splitter = get_recursive_splitter("XYZ") # Test fallback
    print(f"Unknown Splitter Config - Size: {unknown_splitter._chunk_size}, Overlap: {unknown_splitter._chunk_overlap}")

    # Test token counting
    test_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "Hi! How can I assist you today?"},
        {"role": "user", "content": None}, # Test invalid content
        "just a string" # Test invalid message format
    ]
    tokens = count_tokens(test_messages)
    print(f"\nToken count for test messages: {tokens}") 
    
    # Check encoding name
    print(f"Using Tiktoken encoding: {ENCODING.name}")
