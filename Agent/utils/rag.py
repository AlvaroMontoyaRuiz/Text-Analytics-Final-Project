"""
utils/rag.py
Contains RAG-related helper functions, including custom chunking configuration
and token counting.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import tiktoken
from typing import Dict, List, Tuple

# --- Define a single, default chunking strategy ---
DEFAULT_CHUNK_SIZE = 2000    # Characters
DEFAULT_CHUNK_OVERLAP = 300  # Characters

# Separators are ordered from most semantic to least semantic (Recursive Strategy)
# This list is still very useful for your new note structure
CUSTOM_SEPARATORS: List[str] = [
    # Specific clinical headers (attempt to split sections)
    "\n\nHistory of Present Illness:",
    "\n\nPast Medical History:",
    "\n\nSocial History:",
    "\n\nFamily History:",
    "\n\nBrief Hospital Course:",
    "\n\nMedications on Admission:",
    "\n\nDischarge Medications:",
    "\n\nDischarge Instructions:",
    "\n\nDischarge Disposition:",
    "\n\nDischarge Diagnosis:",
    "\n\nDischarge Condition:",
    "\n\nAllergies:",
    "\n\nPhysical Exam:",
    "\n\nPertinent Results:",
    # General structural separators
    "\n\n#",      # Markdown header style (if present)
    "\n\n",       # Paragraph break (most common semantic block)
    "\n",         # Line break
    ". ",         # Sentence ending with period
    "? ",         # Sentence ending with question mark
    "! ",         # Sentence ending with exclamation mark
    ", ",         # Clause separation
    "; ",         # Clause separation
    " ",          # Word splitter
    "",           # Character splitter (last resort)
]

# --- Tiktoken Encoding Setup (Unchanged) ---
try:
    ENCODING = tiktoken.encoding_for_model("gpt-4o-mini") 
except Exception:
    print("Warning: gpt-4o-mini encoding not found, using cl100k_base. Token counts may be approximate.")
    ENCODING = tiktoken.get_encoding("cl100k_base")

def get_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter with a default
    chunk size and overlap.
    """
    print(f"   [Splitter Setup] Using default chunking: Size={DEFAULT_CHUNK_SIZE}, Overlap={DEFAULT_CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        separators=CUSTOM_SEPARATORS,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len, 
        is_separator_regex=False, 
        keep_separator=True 
    )
    return splitter

def count_tokens(messages: List[Dict]) -> int:
    """
    Counts the total tokens in a list of message dicts.
    (This function is unchanged)
    """
    total_tokens = 0
    if not isinstance(messages, list):
        print("Warning: count_tokens expected a list of dicts, received:", type(messages))
        return 0
        
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            content = m["content"]
            if isinstance(content, str):
                try:
                    total_tokens += len(ENCODING.encode(content))
                except Exception as e:
                    print(f"Warning: Error encoding content during token count: {e}. Content: '{content[:50]}...'")
            else:
                print(f"Warning: Message content is not a string, skipping token count. Content: {content}")
            
    return total_tokens