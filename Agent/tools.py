"""
tools.py
Defines the specialized RAG tool(s) for the agent system.
All agents will use this single tool.
"""

from utils.pinecone_database import perform_hybrid_search, ConnectionError as PineconeConnectionError
from typing import List, Optional
import traceback
from datetime import datetime # --- NEW: Import datetime for date conversion ---

# --- NEW: Helper function to convert dates ---
def convert_date_to_iso_tool(date_str: str) -> Optional[str]:
    """Tries to convert a date string (any format) to YYYY-MM-DD."""
    if not date_str:
        return None
    try:
        # Try MM/DD/YYYY
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Try YYYY-MM-DD (if LLM is smart)
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
             # Try to parse "April 2025" - this is harder and less reliable
             # For now, we'll focus on exact date matches.
             print(f"Warning in tool: Could not parse date {date_str}. Passing as-is.")
             return date_str # Pass as-is, may fail to filter

def retrieve_patient_information(
    query: str, 
    patient_id: Optional[str] = None, 
    patient_name: Optional[str] = None,
    admission_date: Optional[str] = None,
    discharge_date: Optional[str] = None
) -> str:
    """
    Retrieves all relevant patient note chunks (history, discharge, meds)
    for a specific patient, identified by ID, Name, and/or dates.
    
    Args:
        query (str): The search query (e.g., "symptoms", "hospital course").
        patient_id (Optional[str]): The Patient_ID (Subject ID) to filter for (e.g., "10001401").
        patient_name (Optional[str]): The Patient's full name (e.g., "Casey Gray").
        admission_date (Optional[str]): The admission date, preferably YYYY-MM-DD.
        discharge_date (Optional[str]): The discharge date, preferably YYYY-MM-DD.
    
    Returns:
        str: Formatted context string or an error message.
    """
    print(f"--- [Patient_Retriever Tool] ---")
    print(f"Query: {query}")
    print(f"Patient ID: {patient_id}")
    print(f"Patient Name: {patient_name}")
    print(f"Admission Date: {admission_date}")
    print(f"Discharge Date: {discharge_date}")

    if not patient_id and not patient_name:
        return "Error from tool: No patient identifier (either patient_id or patient_name) was provided."
    
    # --- NEW: Convert dates to standard format before filtering ---
    admission_date_iso = convert_date_to_iso_tool(admission_date) if admission_date else None
    discharge_date_iso = convert_date_to_iso_tool(discharge_date) if discharge_date else None
    
    try:
        context = perform_hybrid_search(
            query_text=query,
            patient_id_filter=patient_id,
            patient_name_filter=patient_name,
            admission_date_filter=admission_date_iso,
            discharge_date_filter=discharge_date_iso,
            k=10 # Retrieve 10 chunks
        )
        return context
    except PineconeConnectionError as e:
        print(f"Error in Patient_Retriever: {e}")
        return "Error: Could not connect to the document database."
    except Exception as e:
        print(f"Unexpected error in Patient_Retriever: {e}")
        traceback.print_exc()
        return f"An unexpected error occurred during retrieval: {e}"