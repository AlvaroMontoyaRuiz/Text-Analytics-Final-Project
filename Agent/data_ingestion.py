"""
data_ingestion.py
Script to orchestrate document loading using Unstructured elements, 
custom chunking, and hybrid vector database population.
Leverages element types for section header detection.
"""

import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader 
from langchain_core.documents import Document 
# Import utilities
from utils.rag import get_recursive_splitter 
from utils.pinecone_database import store_documents_in_pinecone 

# Load environment variables (must be present in a .env file)
load_dotenv()

# Configuration
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/" 
DOC_CATEGORY_MAP = {
    "POL": "Policy & Protocol",
    "MED": "Medication Administration Logs",
    "ADM": "Patient Admission Reports",
    "DIS": "Discharge Summaries",
    "CLIN": "Clinical History & Consults"
}

# Ensure the processed directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def extract_metadata_from_text(file_path: str, content: str) -> Dict[str, str]:
    """
    Extracts structured metadata (Doc_Type_Code, Patient_ID)
    from the file name and COMBINED document content based on known patterns.
    Section_Header is added later based on elements.
    """
    file_name = os.path.basename(file_path)
    
    # 1. Extract Doc_Type_Code (e.g., ADM, CLIN) - Allow 3 or 4 letters
    match = re.match(r"([A-Z]{3,4})-\d+", file_name)
    doc_type_code = match.group(1) if match else "GEN"
    
    # 2. Extract Patient_ID (MRN) from combined content
    mrn_match = re.search(r"Patient ID:\s*(MRN\s*\d+)", content, re.IGNORECASE)
    patient_id = " ".join(mrn_match.group(1).split()) if mrn_match else "N/A"
    
    # 3. Compile BASE metadata dictionary (Section_Header added later)
    metadata = {
        "source": file_path, 
        "Doc_Type_Code": doc_type_code,
        "Doc_Category": DOC_CATEGORY_MAP.get(doc_type_code, "General Document"),
        "Patient_ID": patient_id,
        # Section_Header is intentionally omitted here, added per-element later
    }
    return metadata

# --- load_and_cache_document is no longer used for the primary loading path ---
# It could be kept for other purposes or removed if only element-based loading is needed.
# def load_and_cache_document(docx_path: str) -> str: ... (Previous implementation)


def ingest_documents(raw_data_dir: str = RAW_DATA_PATH) -> List[Document]:
    """
    Orchestrates loading documents as elements, adding header metadata based on element type,
    chunking the documents, and preparing them for Pinecone.
    """
    final_documents: List[Document] = []
    
    print(f"\n--- Starting Document Ingestion from '{raw_data_dir}' (Element Mode) ---")
    
    files_to_process = [f for f in os.listdir(raw_data_dir) if f.endswith(".docx") and not f.startswith('~')]
    
    if not files_to_process:
        print("No .docx files found in the raw data directory.")
        return []
        
    print(f"Found {len(files_to_process)} .docx files to process...")

    for filename in files_to_process:
        file_path = os.path.join(raw_data_dir, filename)
        print(f"\nProcessing file: {filename}")
        
        # 1. Load document using Unstructured in "elements" mode
        try:
            print(f"   -> Loading elements from: {filename}")
            # strategy="hi_res" can sometimes yield better element categorization
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements", strategy="hi_res") 
            elements = loader.load() # This returns a list of LangChain Document objects
            if not elements:
                 print(f"   -> No elements extracted from {filename}. Skipping.")
                 continue
            print(f"   -> Extracted {len(elements)} elements.")
                 
        except ImportError as ie:
            print(f"Error loading {filename}: Missing dependency - {ie}. Please ensure all required libraries are installed.")
            continue # Skip this file
        except Exception as e:
            print(f"Error loading elements from {filename}: {e}. Skipping file.")
            continue # Skip this file

        # 2. Extract base metadata (needs combined text for MRN search)
        try:
            # Combine page_content of elements for metadata extraction
            combined_content = "\n\n".join([el.page_content for el in elements if el.page_content])
            base_metadata = extract_metadata_from_text(file_path, combined_content)
            doc_type_code = base_metadata["Doc_Type_Code"]
        except Exception as e:
            print(f"   -> Error extracting base metadata for {filename}: {e}. Skipping file.")
            continue
            
        # 3. Process elements to add Section_Header metadata
        documents_with_headers: List[Document] = []
        last_header = "Document Start" # Default header
        
        print("   -> Assigning section headers based on element types...")
        for element in elements:
            # Check the 'category' metadata provided by unstructured
            element_category = element.metadata.get('category', '').lower()
            
            # Identify potential headers (Titles are common, sometimes 'UncategorizedText' acts as one)
            # Adjust this list based on observing unstructured's output for your specific documents
            is_header = element_category in ['title', 'header', 'subheadline'] 
            # Heuristic: Short, standalone lines might also be headers
            is_standalone_short_line = len(element.page_content.split()) < 10 and '\n' not in element.page_content.strip()

            if is_header or (element_category == 'uncategorizedtext' and is_standalone_short_line):
                # Clean up header text (remove extra spaces/newlines)
                current_header_text = re.sub(r'\s+', ' ', element.page_content).strip()
                # Update last_header only if the text is meaningful
                if current_header_text:
                     last_header = current_header_text
                     # print(f"      Found Header: '{last_header}' (Category: {element_category})") # Optional: Debug print

            # Create a new metadata dict for this element's document
            # Start with base metadata, add the current header, and include element-specific metadata
            element_metadata = base_metadata.copy()
            element_metadata["Section_Header"] = last_header
            # Add potentially useful element metadata (like page number if available)
            element_metadata["element_category"] = element_category # Store the detected category
            if 'page_number' in element.metadata:
                 element_metadata['page_number'] = element.metadata['page_number']

            # Create a new Document with the element's content and the enriched metadata
            documents_with_headers.append(
                Document(page_content=element.page_content, metadata=element_metadata)
            )
        
        # 4. Get the specialized recursive splitter
        try:
            splitter = get_recursive_splitter(doc_type_code) 
            chunk_size = splitter._chunk_size 
            chunk_overlap = splitter._chunk_overlap
        except Exception as e:
             print(f"   -> Error getting splitter for doc type '{doc_type_code}' in {filename}: {e}. Using default splitter.")
             splitter = get_recursive_splitter("GEN") 
             chunk_size = splitter._chunk_size
             chunk_overlap = splitter._chunk_overlap

        print(f"   -> Chunking {len(documents_with_headers)} processed elements with strategy for '{doc_type_code}': Size={chunk_size}, Overlap={chunk_overlap}")

        # 5. Split the processed documents (with headers in metadata)
        try:
            # Use split_documents which preserves and propagates metadata
            chunked_documents = splitter.split_documents(documents_with_headers) 
        except Exception as e:
            print(f"   -> Error during document splitting for {filename}: {e}. Skipping file.")
            continue
            
        print(f"   -> Created {len(chunked_documents)} final chunks.")

        # 6. Add chunk_id to each chunk's metadata
        for i, chunk in enumerate(chunked_documents):
            # Ensure metadata exists before modifying
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                 chunk.metadata = {}
            chunk.metadata["chunk_id"] = f"chunk-{i:03d}"
            
        final_documents.extend(chunked_documents)

    print(f"\n--- Ingestion Finished ---")
    return final_documents

# --- Main execution block remains the same ---
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
                 # Construct a readable summary of the metadata
                 source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                 chunk_id = doc.metadata.get('chunk_id', 'N/A')
                 header = doc.metadata.get('Section_Header', 'N/A')
                 patient = doc.metadata.get('Patient_ID', 'N/A')
                 print(f"  - File: {source_file}, Chunk: {chunk_id}, Header: '{header}', Patient: {patient}")

        # STEP 2: Embed and store documents in Pinecone
        if documents_for_pinecone:
            print("\n--- Starting Vector Database Upload ---")
            try:
                store_documents_in_pinecone(documents_for_pinecone)
                print("\n✅ Successfully indexed documents in Pinecone.")
            except Exception as e:
                print(f"\n❌ Error during Pinecone upload: {e}")
                print("Please check your Pinecone API key, index names, and network connection in '.env' file.")
        else:
            print("\nNo documents processed, skipping Pinecone upload.")
