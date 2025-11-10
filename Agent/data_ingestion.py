"""
data_ingestion.py
Script to orchestrate document loading using Unstructured elements, 
custom chunking, and hybrid vector database population.
"""

import os
import re
# --- MODIFIED: Added Optional ---
from typing import List, Dict, Tuple, Optional 
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader 
from langchain_core.documents import Document 
# Import utilities
from utils.rag import get_recursive_splitter 
from utils.pinecone_database import store_documents_in_pinecone 
from datetime import datetime # Import datetime for date conversion

# Load environment variables
load_dotenv()

# Configuration
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/" 

# Ensure the processed directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- Helper function to convert dates ---
def convert_date_to_iso(date_str: str) -> Optional[str]:
    """Converts MM/DD/YYYY to YYYY-MM-DD for consistent filtering."""
    try:
        # Parse the date assuming MM/DD/YYYY format
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        # Convert to ISO 8601 format (YYYY-MM-DD)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        print(f"   -> Warning: Could not parse date string: {date_str}")
        return None


def extract_metadata_from_text(file_path: str, content: str) -> Dict[str, str]:
    """
    Extracts structured metadata (Patient_ID, Patient_Name, Note_ID, Dates)
    from the file name and document content.
    """
    file_name = os.path.basename(file_path)
    
    # 1. Extract Patient_Name (e.g., "Name: Casey Gray")
    name_match = re.search(r"Name:\s*([A-Za-z]+\s+[A-Za-z]+)", content, re.IGNORECASE)
    patient_name = name_match.group(1).strip() if name_match else "N/A"

    # 2. Extract Patient_ID (Subject ID) from content
    patient_id_match = re.search(r"Subject ID:\s*(\d+)", content, re.IGNORECASE)
    patient_id = patient_id_match.group(1).strip() if patient_id_match else "N/A"

    # 3. Extract Note_ID from content
    note_id_match = re.search(r"Note ID:\s*([\w-]+)", content, re.IGNORECASE)
    note_id = note_id_match.group(1).strip() if note_id_match else file_name.replace(".docx", "")
    
    # 4. Extract Admission and Discharge Dates
    admission_date_match = re.search(r"Admission Date:\s*(\d{2}/\d{2}/\d{4})", content)
    discharge_date_match = re.search(r"Discharge Date:\s*(\d{2}/\d{2}/\d{4})", content)

    admission_date_iso = None
    if admission_date_match:
        admission_date_iso = convert_date_to_iso(admission_date_match.group(1))
        
    discharge_date_iso = None
    if discharge_date_match:
        discharge_date_iso = convert_date_to_iso(discharge_date_match.group(1))

    # 5. Compile BASE metadata dictionary
    metadata = {
        "source": file_path, 
        "Note_ID": note_id,
        "Patient_ID": patient_id,
        "Patient_Name": patient_name,
    }
    
    # Add dates to metadata *only if they exist*
    if admission_date_iso:
        metadata["Admission_Date"] = admission_date_iso
    if discharge_date_iso:
        metadata["Discharge_Date"] = discharge_date_iso

    return metadata

def ingest_documents(raw_data_dir: str = RAW_DATA_PATH) -> List[Document]:
    """
    Orchestrates loading documents, extracting metadata, and splitting
    them into large, meaningful chunks.
    """
    final_documents: List[Document] = []
    
    print(f"\n--- Starting Document Ingestion from '{raw_data_dir}' (Single Mode) ---")
    
    files_to_process = [f for f in os.listdir(raw_data_dir) if f.endswith(".docx") and not f.startswith('~')]
    
    if not files_to_process:
        print("No .docx files found in the raw data directory.")
        return []
        
    print(f"Found {len(files_to_process)} .docx files to process...")

    for filename in files_to_process:
        file_path = os.path.join(raw_data_dir, filename)
        print(f"\nProcessing file: {filename}")
        
        # 1. Load document in "single" mode
        try:
            print(f"   -> Loading full document: {filename}")
            loader = UnstructuredWordDocumentLoader(file_path, mode="single", strategy="hi_res") 
            docs = loader.load()
            if not docs:
                print(f"   -> No content extracted from {filename}. Skipping.")
                continue
            
            doc = docs[0] 
            print(f"   -> Document loaded.")
                
        except Exception as e:
            print(f"Error loading document {filename}: {e}. Skipping file.")
            continue 

        # 2. Extract metadata from the single doc's content
        try:
            content = doc.page_content
            base_metadata = extract_metadata_from_text(file_path, content)
            
            if base_metadata.get("Patient_ID", "N/A") == "N/A":
                print(f"   -> CRITICAL: No Subject ID found in content for {filename}. Skipping file.")
                continue
            
            print(f"   -> Extracted Metadata: Patient={base_metadata.get('Patient_Name')}, ID={base_metadata.get('Patient_ID')}")
            print(f"   -> Admission: {base_metadata.get('Admission_Date', 'N/A')}, Discharge: {base_metadata.get('Discharge_Date', 'N/A')}")

            # Set the extracted metadata on the single document
            doc.metadata = base_metadata

        except Exception as e:
            print(f"   -> Error extracting base metadata for {filename}: {e}. Skipping file.")
            continue
            
        # 3. Split the *single* large document
        try:
            splitter = get_recursive_splitter() 
            chunk_size = splitter._chunk_size 
            chunk_overlap = splitter._chunk_overlap
        except Exception as e:
             print(f"   -> Error getting splitter: {e}. Aborting.")
             continue

        print(f"   -> Chunking full document with default strategy: Size={chunk_size}, Overlap={chunk_overlap}")

        try:
            chunked_documents = splitter.split_documents([doc]) 
        except Exception as e:
            print(f"   -> Error during document splitting for {filename}: {e}. Skipping file.")
            continue
            
        print(f"   -> Created {len(chunked_documents)} final chunks.") 

        # 4. Add chunk_id and Section_Header to each chunk's metadata
        for i, chunk in enumerate(chunked_documents):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {} 
            chunk.metadata["chunk_id"] = f"chunk-{i:03d}"
            
            content_start = chunk.page_content.lstrip()
            
            # Find the best matching header from our separators
            best_match = "General"
            for header in reversed(get_recursive_splitter()._separators):
                 if header.strip() and content_start.startswith(header.strip()):
                     best_match = header.strip().replace(":", "")
                     break
            chunk.metadata["Section_Header"] = best_match
            
        final_documents.extend(chunked_documents)

    print(f"\n--- Ingestion Finished ---")
    return final_documents

# --- Main execution block ---
if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH) or not os.path.isdir(RAW_DATA_PATH):
        print(f"Error: Directory '{RAW_DATA_PATH}' not found or is not a directory.")
        print("Please create it and place your .docx files inside.")
    else:
        # STEP 1: Load elements, add headers, and chunk documents
        documents_for_pinecone = ingest_documents()
        
        print(f"\nTotal Chunks Created Across All Files: {len(documents_for_pinecone)}")

        if documents_for_pinecone:
            print("\nSample Metadata from first few chunks:")
            for doc in documents_for_pinecone[:min(3, len(documents_for_pinecone))]:
                note_id = doc.metadata.get('Note_ID', 'N/A')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                header = doc.metadata.get('Section_Header', 'N/A')
                patient_id = doc.metadata.get('Patient_ID', 'N/A')
                patient_name = doc.metadata.get('Patient_Name', 'N/A')
                admission_date = doc.metadata.get('Admission_Date', 'N/A')
                print(f"   - Note: {note_id}, Chunk: {chunk_id}, Header: '{header}', Patient: {patient_name} ({patient_id}), Admitted: {admission_date}")

        # STEP 2: Embed and store documents in Pinecone
        if documents_for_pinecone:
            print("\n--- Starting Vector Database Upload ---")
            try:
                # Filter out any potential empty chunks
                valid_docs_to_store = []
                for doc in documents_for_pinecone:
                    if doc.page_content and not doc.page_content.strip().isspace():
                        valid_docs_to_store.append(doc)
                
                print(f"   -> Storing {len(valid_docs_to_store)} non-empty chunks in Pinecone.")
                if valid_docs_to_store:
                    store_documents_in_pinecone(valid_docs_to_store)
                    print("\n✅ Successfully indexed documents in Pinecone.")
                else:
                    print("\nNo valid chunks to store.")
                    
            except Exception as e:
                print(f"\n❌ Error during Pinecone upload: {e}")
                print("Please check your Pinecone API key, index names, and network connection in '.env' file.")
        else:
            print("\nNo documents processed, skipping Pinecone upload.")