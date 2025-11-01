"""
Test RAG Processing: test_processing.py

This script runs the full document loading, caching, chunking, and metadata
extraction pipeline defined in data_ingestion.py. It does NOT upload anything
to Pinecone, allowing for inspection of chunk format and cache file creation.
"""
import os
import json
from data_ingestion import ingest_documents, PROCESSED_DATA_PATH # Import core ingestion logic
from utils.rag import CHUNK_CONFIG # For displaying chunk size info

def print_test_summary(documents):
    """Prints a summary of the total chunk count and the first few chunk metadata/content."""
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY | Total Chunks Created: {len(documents)}")
    print(f"{'='*60}")
    
    if not documents:
        print("No chunks were created. Check the data/raw/ folder and file naming conventions.")
        return

    # Print the first three chunks created for format verification
    for i in range(min(3, len(documents))):
        doc = documents[i]
        
        # Structure the ID as Pinecone expects it
        pinecone_id = f"{doc.metadata['Doc_Type_Code']}-{doc.metadata['chunk_id']}"
        
        # --- FIX: Define the cleaned string before using it in the f-string ---
        content_preview = doc.page_content[:200].replace('\n', ' ')
        # -------------------------------------------------------------------
        
        print(f"\n--- Chunk {i+1} Verification ---")
        print(f"Pinecone ID: {pinecone_id}")
        print(f"Document Source: {os.path.basename(doc.metadata['source'])}")
        print(f"Target Chunk Size: {CHUNK_CONFIG.get(doc.metadata['Doc_Type_Code'])}")
        print(f"Section Header: {doc.metadata['Section_Header']}")
        print(f"Patient ID: {doc.metadata['Patient_ID']}")
        print(f"Content Preview (First 200 chars):")
        print(f"--------------------------------------------------")
        print(f"{content_preview}...") # Use the pre-cleaned variable
        print(f"--------------------------------------------------")
        
        # Optional: Dump full metadata for deep inspection
        # print(f"Full Metadata Dump:\n{json.dumps(doc.metadata, indent=2)}")


if __name__ == "__main__":
    
    # Run the full ingestion pipeline defined in data_ingestion.py
    documents_for_inspection = ingest_documents()
    
    # Print the detailed summary
    print_test_summary(documents_for_inspection)

    print("\n[VERIFICATION COMPLETE]")
    print(f"Please check the '{PROCESSED_DATA_PATH}' folder now to see the cached .txt files.")
    print("If the output above looks correct, proceed to uncomment the Pinecone line in data_ingestion.py.")
