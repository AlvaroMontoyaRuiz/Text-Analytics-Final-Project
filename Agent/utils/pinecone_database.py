"""
utils/pinecone_database.py
Handles connection, index creation, document upsertion, and hybrid search 
using Pinecone's Integrated Embedding models.
"""

import os
import time
import uuid 
from dotenv import load_dotenv 
from pinecone import Pinecone # Assuming standard import, adjust if using GRPC client specifically
from langchain_core.documents import Document 
from typing import List, Dict, Optional

# Load environment variables at the module level if this script might be run standalone
# Or ensure it's loaded by the main script (data_ingestion.py, app.py) before calling functions here.
# load_dotenv() # Uncomment if running this file directly for testing

# --- Configuration (Must be set in .env file) ---
DENSE_INDEX_NAME = os.getenv("DENSE_INDEX_NAME", "semantic-clinical-rag")
SPARSE_INDEX_NAME = os.getenv("SPARSE_INDEX_NAME", "lexical-clinical-rag")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Field Mapping ---
# This MUST match the key used when preparing records for upsert
TEXT_FIELD = "page_content" 

# --- Pinecone Client Initialization (Singleton Pattern) ---
_pc_client: Optional[Pinecone] = None

class ConnectionError(Exception):
    """Custom exception for Pinecone connection/configuration errors."""
    pass

def get_pinecone_client() -> Pinecone:
    """
    Initializes and returns the Pinecone client using a singleton pattern.
    Reads API key from environment variables.
    Raises ConnectionError if initialization fails.
    """
    global _pc_client 
    if _pc_client:
        return _pc_client

    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is missing or empty.")
        
        print("Initializing Pinecone client...")
        _pc_client = Pinecone(api_key=api_key)
        # Optionally, add a quick check like listing indexes to confirm connection
        _pc_client.list_indexes() 
        print("Pinecone client initialized successfully.")
        return _pc_client
        
    except ValueError as ve:
        raise ConnectionError(f"Configuration Error: {ve}. Ensure .env file exists, is loaded, and PINECONE_API_KEY is defined.")
    # Removed specific ApiException handling, catch general Exception
    except Exception as e:
        # Catch other potential errors during init (network issues, API errors, etc.)
        raise ConnectionError(f"Pinecone Client Initialization Failed: {e}. Check API key validity, network connection, or Pinecone service status.")


def _wait_for_index_ready(client: Pinecone, index_name: str, timeout: int = 300):
    """Waits for a Pinecone index to become ready."""
    print(f"Waiting for index '{index_name}' to become active (up to {timeout}s)...")
    start_time = time.time()
    while True:
        try:
            status = client.describe_index(index_name).status
            if status['ready']:
                print(f"Index '{index_name}' is active.")
                return True
            if status['state'] == 'Failed':
                 raise ConnectionError(f"Index '{index_name}' entered Failed state during creation.")
                 
        # Catch general exceptions including potential API errors like 404
        except Exception as e:
             # Check if it's likely a temporary "not found yet" error vs. a persistent one
             if "not found" in str(e).lower() and (time.time() - start_time < timeout * 0.5): # Heuristic
                  print(f"   Index '{index_name}' not found yet, waiting...")
             else:
                  raise ConnectionError(f"Error checking status for index '{index_name}': {e}")

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Index '{index_name}' did not become ready within {timeout} seconds.")
            
        time.sleep(10) # Wait longer between checks

def create_hybrid_index_if_not_exists(force_recreate: bool = False):
    """
    Creates both the dense (semantic) and sparse (keyword) indexes if they do not exist.
    Uses Pinecone's integrated embedding models. Can optionally delete existing indexes.
    """
    client = get_pinecone_client() 
    
    indexes_to_create = [
        {"name": DENSE_INDEX_NAME, "model": "llama-text-embed-v2", "type": "Dense"},
        {"name": SPARSE_INDEX_NAME, "model": "pinecone-sparse-english-v0", "type": "Sparse"}
    ]

    try:
        active_indexes_response = client.list_indexes()
        # Handle potential differences in response structure (list of dicts vs object)
        if isinstance(active_indexes_response, list):
             active_indexes = [idx.get('name') for idx in active_indexes_response if isinstance(idx, dict) and 'name' in idx]
        elif hasattr(active_indexes_response, 'indexes'): # Handle SDK v3+ structure
             active_indexes = [idx.name for idx in active_indexes_response.indexes]
        else:
             print("Warning: Unexpected format for list_indexes response. Cannot reliably check existing indexes.")
             active_indexes = [] # Assume no indexes exist if format is wrong

    except Exception as e: # Catch general exception
        print(f"Error listing indexes: {e}. Cannot proceed with index creation/check.")
        raise ConnectionError("Failed to list Pinecone indexes.") from e


    for index_info in indexes_to_create:
        index_name = index_info["name"]
        model = index_info["model"]
        index_type = index_info["type"]

        if index_name in active_indexes:
            if force_recreate:
                 print(f"Deleting existing {index_type} index: {index_name}...")
                 try:
                      client.delete_index(index_name)
                      # Wait for deletion, checking status periodically might be more robust
                      print("Waiting for index deletion...")
                      time.sleep(30) # Simple wait, adjust as needed
                      active_indexes.remove(index_name) # Update local list
                 except Exception as e: # Catch general exception
                      print(f"Warning: Failed to delete index {index_name}: {e}. Skipping recreation if it still exists.")

            else:
                 print(f"{index_type} index '{index_name}' already exists. Skipping creation.")
                 continue # Skip to the next index if not forcing recreate

        # Re-check existence after potential deletion attempt
        # Fetching list again might be safer if deletion status is uncertain
        try:
             # Re-fetch and parse active indexes
             current_indexes_response = client.list_indexes()
             if isinstance(current_indexes_response, list):
                  current_indexes = [idx.get('name') for idx in current_indexes_response if isinstance(idx, dict) and 'name' in idx]
             elif hasattr(current_indexes_response, 'indexes'):
                  current_indexes = [idx.name for idx in current_indexes_response.indexes]
             else:
                  current_indexes = active_indexes # Use potentially stale list if re-fetch fails
        except Exception:
             print("Warning: Could not re-verify index list after deletion attempt.")
             current_indexes = active_indexes # Use potentially stale list

        if index_name not in current_indexes:
            print(f"Creating {index_type} index: {index_name} with model {model}...")
            try:
                # Use create_index_for_model for integrated embedding setup
                client.create_index_for_model( 
                    name=index_name,
                    cloud=CLOUD,
                    region=REGION,
                    embed={
                        "model": model, 
                        # Maps Pinecone's internal 'text' field to our TEXT_FIELD ('page_content')
                        "field_map": {"text": TEXT_FIELD} 
                    }
                    # Metric and dimension are inferred by create_index_for_model
                )
                print(f"{index_type} index '{index_name}' creation initiated.")
                # Wait for the index to be ready after initiating creation
                _wait_for_index_ready(client, index_name) 

            except Exception as e: # Catch general exception
                 print(f"Error creating {index_type} index '{index_name}': {e}. Please check configuration and Pinecone status.")
                 raise ConnectionError(f"Failed to create index {index_name}") from e


def store_documents_in_pinecone(documents: List[Document], batch_size: int = 96):
    """
    Upserts chunked LangChain Documents to both dense and sparse Pinecone indexes.
    Uses integrated embedding - requires records with the TEXT_FIELD key.
    Ensures indexes exist and are ready before upserting.
    """
    if not documents:
        print("No documents provided to store in Pinecone.")
        return

    client = get_pinecone_client()
        
    # Ensure indexes exist and wait for them
    try:
        create_hybrid_index_if_not_exists(force_recreate=False) 
        # The create function now includes waits, so explicit waits might be redundant
    except (ConnectionError, TimeoutError, Exception) as e: # Catch general Exception
         print(f"Error ensuring indexes are ready: {e}. Aborting upsert.")
         return # Stop if indexes aren't ready

    # 1. Prepare records for upserting
    records_to_upsert = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document) or not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
             print(f"Warning: Skipping invalid document object at index {i}.")
             continue
             
        # Generate a robust, unique ID combining source filename and chunk ID
        try:
            doc_metadata = doc.metadata if doc.metadata else {}
            source_filename = os.path.basename(doc_metadata.get('source', f'doc_{i}'))
            chunk_id = doc_metadata.get('chunk_id', f'chunk_{i}')
            # Ensure ID is within Pinecone's length limits (e.g., 512 chars)
            record_id = f"{source_filename}-{chunk_id}"[:512] 
        except Exception as e:
             print(f"Warning: Error creating record ID for doc {i}: {e}. Using fallback ID.")
             record_id = f"fallback-{uuid.uuid4()}" # Ensure unique ID even if metadata is bad


        # Prepare the record structure for integrated embedding
        record = {
            # Use '_id' for integrated embedding records
            "_id": record_id, 
            # This key (e.g., 'page_content') MUST match TEXT_FIELD and field_map
            TEXT_FIELD: doc.page_content or "", # Ensure content is a string
            # Spread the rest of the metadata fields
            **(doc.metadata if doc.metadata else {}) 
        }
        records_to_upsert.append(record)

    if not records_to_upsert:
         print("No valid records prepared for upserting.")
         return

    # 2. Upsert to BOTH Indexes in Batches
    total_records = len(records_to_upsert)
    print(f"\nPrepared {total_records} records for upsert.")

    for index_name, index_type in [(DENSE_INDEX_NAME, "Dense"), (SPARSE_INDEX_NAME, "Sparse")]:
        try:
            index = client.Index(index_name)
            print(f"Upserting to {index_type} index '{index_name}' (Batch size: {batch_size})...")
            
            success_count = 0
            for i in range(0, total_records, batch_size):
                batch = records_to_upsert[i : i + batch_size]
                try:
                    # Use upsert_records for integrated embedding
                    # Pinecone SDK v4+ returns None on success, raises error on failure.
                    index.upsert_records(
                        namespace="__default__", # Using default namespace
                        records=batch
                    )
                    processed_count = len(batch) # Assume success if no exception
                    success_count += processed_count
                    print(f"  Batch {(i//batch_size) + 1}/{ (total_records + batch_size - 1)//batch_size }: Processed {processed_count} records for upsert.")
                    
                # Catch more general exception now
                except Exception as e:
                     print(f"  Batch {(i//batch_size) + 1}: Error during upsert to {index_name}: {e}. Skipping batch.")
                     # Consider retry logic here for transient errors

            print(f"Finished upserting to {index_type} index '{index_name}'. Attempted to process {success_count}/{total_records} records.")

        except Exception as e: # Catch general exception
            print(f"Error connecting to or upserting into {index_type} index '{index_name}': {e}")
             
    print("\nPinecone upsert process completed.")


# --- CORRECTED RESULT PARSING V4 & UPDATED k DEFAULT: perform_hybrid_search ---
def perform_hybrid_search(query_text: str, patient_id_filter: Optional[str] = None, k: int = 10, alpha: float = 0.7) -> str: # Changed k default to 10
    """
    Performs a hybrid search (dense + sparse) across Pinecone indexes using text search
    for integrated embedding. Applies optional metadata filtering, combines results 
    using alpha weighting, and returns a formatted context string.

    Args:
        query_text: The user's search query.
        patient_id_filter: The specific Patient_ID (e.g., "MRN 123456") to filter by.
        k: The number of results to retrieve from EACH index before combining. Now defaults to 10.
        alpha: Weighting factor for dense results (0.0 to 1.0).

    Returns:
        Formatted context string or an error/no results message.
    """
    if not query_text:
        return "Please provide a query."
        
    if not (0.0 <= alpha <= 1.0):
        print("Warning: Alpha must be between 0.0 and 1.0. Using default 0.7.")
        alpha = 0.7

    try:
        client = get_pinecone_client()
        dense_index = client.Index(DENSE_INDEX_NAME)
        sparse_index = client.Index(SPARSE_INDEX_NAME)
    except ConnectionError as e: 
        print(f"Error connecting to Pinecone indexes: {e}")
        return "Error: Could not connect to the document database."
    except Exception as e: 
         print(f"Unexpected error getting Pinecone indexes: {e}")
         return "Error: Could not connect to the document database."

    # 1. Create metadata filter
    metadata_filter = {}
    if patient_id_filter and patient_id_filter.strip().upper() not in ["N/A", ""]:
        metadata_filter = {"Patient_ID": {"$eq": patient_id_filter}}
        print(f"Applying metadata filter: {metadata_filter}")
    else:
        print("No Patient_ID filter applied.")

    # 2. Perform Dense and Sparse Searches using the correct payload for text search
    dense_hits = []
    sparse_hits = []
    dense_response = None
    sparse_response = None
    try:
        print(f"Performing dense search for '{query_text[:50]}...' (k={k})")
        # Use the documented payload structure for text search with integrated embedding
        dense_response = dense_index.search(
            namespace="__default__",
            query={
                "inputs": {"text": query_text},
                "top_k": k, # Uses the k parameter (now default 10)
                "filter": metadata_filter 
            },
            # Specify fields explicitly to ensure they are returned
            fields=[TEXT_FIELD, "source", "Patient_ID", "Section_Header", "chunk_id"] 
        )
        
        # --- ROBUST RESULT ACCESS using direct key access in try/except ---
        try:
            dense_hits = dense_response['result']['hits']
            if not isinstance(dense_hits, list):
                 print(f"Warning: Expected list for dense 'hits', got {type(dense_hits)}. Response: {dense_response}")
                 dense_hits = [] # Reset if not a list
            else:
                 print(f"  -> Dense search returned {len(dense_hits)} results.")
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Error accessing dense search results via keys ('result'/'hits'): {e}. Response: {dense_response}")
            dense_hits = []
        # --- End Correction ---

        print(f"Performing sparse search for '{query_text[:50]}...' (k={k})")
        sparse_response = sparse_index.search(
            namespace="__default__",
            query={
                 "inputs": {"text": query_text},
                 "top_k": k, # Uses the k parameter (now default 10)
                 "filter": metadata_filter
            },
            fields=[TEXT_FIELD, "source", "Patient_ID", "Section_Header", "chunk_id"]
        )

        # --- ROBUST RESULT ACCESS using direct key access in try/except ---
        try:
            sparse_hits = sparse_response['result']['hits']
            if not isinstance(sparse_hits, list):
                print(f"Warning: Expected list for sparse 'hits', got {type(sparse_hits)}. Response: {sparse_response}")
                sparse_hits = [] # Reset if not a list
            else:
                 print(f"  -> Sparse search returned {len(sparse_hits)} results.")
        except (KeyError, TypeError, AttributeError) as e:
              print(f"Warning: Error accessing sparse search results via keys ('result'/'hits'): {e}. Response: {sparse_response}")
              sparse_hits = []
        # --- End Correction ---

    # Catch general exception 
    except Exception as e:
        print(f"Error during Pinecone search API call: {e}")
        # Optionally print responses for debugging
        # print("Dense Response during error:", dense_response)
        # print("Sparse Response during error:", sparse_response)
        return "Error: Failed to retrieve documents from the database due to API error."

    # 3. Combine and Re-score Results using Weighted Fusion (alpha)    
    combined_scores: Dict[str, float] = {}
    metadata_cache: Dict[str, Dict] = {} # Cache metadata by ID

    # --- HIT PROCESSING (Should be correct now with dict access) ---
    # Process dense results 
    for hit in dense_hits:
        record_id = hit.get('_id') 
        score = hit.get('_score', 0.0) 
        metadata = hit.get('fields', {}) 

        if record_id and score is not None: 
            combined_scores[record_id] = combined_scores.get(record_id, 0.0) + (alpha * score)
            if record_id not in metadata_cache:
                metadata_cache[record_id] = metadata if metadata else {}
        else:
             print(f"Warning: Skipping dense hit with missing _id or _score: {hit}")

    # Process sparse results 
    for hit in sparse_hits:
        record_id = hit.get('_id') 
        score = hit.get('_score', 0.0)
        metadata = hit.get('fields', {})

        if record_id and score is not None:
            combined_scores[record_id] = combined_scores.get(record_id, 0.0) + ((1 - alpha) * score)
            if record_id not in metadata_cache:
                 metadata_cache[record_id] = metadata if metadata else {}
        else:
             print(f"Warning: Skipping sparse hit with missing _id or _score: {hit}")
    # --- End Hit Processing ---

    if not combined_scores:
        print("No relevant documents found after combining search results.")
        return "No relevant context found in the documents for your query."

    # 4. Sort combined results by score (descending) and take top K
    sorted_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    # --- Ensure final result also respects k ---
    num_results_to_take = min(k, len(sorted_results)) # Use k here too, not just for retrieval
    top_results = sorted_results[:num_results_to_take] 

    print(f"Combined and ranked {len(top_results)} results (out of {len(combined_scores)} unique).")

    # 5. Format the final context string
    context_str = ""
    if not top_results:
        print("No results after ranking.")
        return "No relevant context found after ranking."
        
    for i, (record_id, score) in enumerate(top_results):
        # --- METADATA ACCESS (Should be correct) ---
        metadata = metadata_cache.get(record_id, {}) 
        
        source_file = os.path.basename(metadata.get('source', 'Unknown Source'))
        patient_id = metadata.get('Patient_ID', 'N/A')
        section = metadata.get('Section_Header', 'No Section Header')
        chunk_id = metadata.get('chunk_id', 'N/A')
        text_content = metadata.get(TEXT_FIELD, "Error: Content missing in metadata") 
        # --- End Metadata Access ---

        context_str += f"--- Context Chunk [{i+1}] (Score: {score:.4f}) ---\n"
        context_str += f"Source: {source_file} (Chunk: {chunk_id})\n"
        context_str += f"Patient_ID: {patient_id}\n"
        context_str += f"Section: {section}\n"
        context_str += f"Content: {text_content}\n\n"
    
    return context_str.strip()

