

import os
import time
import uuid 
from dotenv import load_dotenv 
from pinecone import Pinecone 
from langchain_core.documents import Document 
from typing import List, Dict, Optional
import streamlit as st # --- NEW: Import Streamlit ---

# --- MODIFIED: Load .env only if it exists (for local ingestion) ---
load_dotenv() 

# --- MODIFIED: Configuration ---
# Try st.secrets first, then fall back to os.getenv()
# This makes the file work BOTH locally and on the cloud.
DENSE_INDEX_NAME = st.secrets.get("DENSE_INDEX_NAME", os.getenv("DENSE_INDEX_NAME", "semantic-clinical-rag"))
SPARSE_INDEX_NAME = st.secrets.get("SPARSE_INDEX_NAME", os.getenv("SPARSE_INDEX_NAME", "lexical-clinical-rag"))
CLOUD = st.secrets.get("PINECONE_CLOUD", os.getenv("PINECONE_CLOUD", "aws"))
REGION = st.secrets.get("PINECONE_REGION", os.getenv("PINECONE_REGION", "us-east-1"))

TEXT_FIELD = "page_content" 

# --- Pinecone Client Initialization (Singleton Pattern) ---
_pc_client: Optional[Pinecone] = None

class ConnectionError(Exception):
    """Custom exception for Pinecone connection/configuration errors."""
    pass

def get_pinecone_client() -> Pinecone:
    """
    Initializes and returns the Pinecone client using a singleton pattern.
    Reads API key from Streamlit secrets (cloud) or .env (local).
    """
    global _pc_client 
    if _pc_client:
        return _pc_client

    try:
        # --- MODIFIED: Try st.secrets first, then os.getenv ---
        api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is missing or empty.")
        
        print("Initializing Pinecone client...")
        _pc_client = Pinecone(api_key=api_key)
        _pc_client.list_indexes() 
        print("Pinecone client initialized successfully.")
        return _pc_client
        
    except ValueError as ve:
        raise ConnectionError(f"Configuration Error: {ve}. Ensure .env or st.secrets is defined.")
    except Exception as e:
        raise ConnectionError(f"Pinecone Client Initialization Failed: {e}. Check API key, network, or service status.")


def _wait_for_index_ready(client: Pinecone, index_name: str, timeout: int = 300):
    # (This function is unchanged)
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
        except Exception as e:
            if "not found" in str(e).lower() and (time.time() - start_time < timeout * 0.5): 
                print(f"   Index '{index_name}' not found yet, waiting...")
            else:
                raise ConnectionError(f"Error checking status for index '{index_name}': {e}")
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Index '{index_name}' did not become ready within {timeout} seconds.")
        time.sleep(10)

def create_hybrid_index_if_not_exists(force_recreate: bool = False):
    # (This function is unchanged)
    client = get_pinecone_client() 
    indexes_to_create = [
        {"name": DENSE_INDEX_NAME, "model": "llama-text-embed-v2", "type": "Dense"},
        {"name": SPARSE_INDEX_NAME, "model": "pinecone-sparse-english-v0", "type": "Sparse"}
    ]
    try:
        active_indexes_response = client.list_indexes()
        if isinstance(active_indexes_response, list):
            active_indexes = [idx.get('name') for idx in active_indexes_response if isinstance(idx, dict) and 'name' in idx]
        elif hasattr(active_indexes_response, 'indexes'): 
            active_indexes = [idx.name for idx in active_indexes_response.indexes]
        else:
            print("Warning: Unexpected format for list_indexes response.")
            active_indexes = []
    except Exception as e: 
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
                    print("Waiting for index deletion...")
                    time.sleep(30)
                    active_indexes.remove(index_name)
                except Exception as e: 
                    print(f"Warning: Failed to delete index {index_name}: {e}. Skipping recreation if it still exists.")
            else:
                print(f"{index_type} index '{index_name}' already exists. Skipping creation.")
                continue 
        try:
            current_indexes_response = client.list_indexes()
            if isinstance(current_indexes_response, list):
                current_indexes = [idx.get('name') for idx in current_indexes_response if isinstance(idx, dict) and 'name' in idx]
            elif hasattr(current_indexes_response, 'indexes'):
                current_indexes = [idx.name for idx in current_indexes_response.indexes]
            else:
                current_indexes = active_indexes
        except Exception:
            print("Warning: Could not re-verify index list after deletion attempt.")
            current_indexes = active_indexes 
        if index_name not in current_indexes:
            print(f"Creating {index_type} index: {index_name} with model {model}...")
            try:
                client.create_index_for_model( 
                    name=index_name,
                    cloud=CLOUD,
                    region=REGION,
                    embed={
                        "model": model, 
                        "field_map": {"text": TEXT_FIELD} 
                    }
                )
                print(f"{index_type} index '{index_name}' creation initiated.")
                _wait_for_index_ready(client, index_name) 
            except Exception as e: 
                print(f"Error creating {index_type} index '{index_name}': {e}.")
                raise ConnectionError(f"Failed to create index {index_name}") from e

def store_documents_in_pinecone(documents: List[Document], batch_size: int = 96):
    # (This function is unchanged)
    if not documents:
        print("No documents provided to store in Pinecone.")
        return
    client = get_pinecone_client()
    try:
        create_hybrid_index_if_not_exists(force_recreate=False) 
    except (ConnectionError, TimeoutError, Exception) as e:
        print(f"Error ensuring indexes are ready: {e}. Aborting upsert.")
        return
    records_to_upsert = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document) or not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
            print(f"Warning: Skipping invalid document object at index {i}.")
            continue
        try:
            doc_metadata = doc.metadata if doc.metadata else {}
            note_id = doc_metadata.get('Note_ID', f'note_{i}')
            chunk_id = doc_metadata.get('chunk_id', f'chunk_{i}')
            record_id = f"{note_id}-{chunk_id}"[:512] 
        except Exception as e:
            print(f"Warning: Error creating record ID for doc {i}: {e}. Using fallback ID.")
            record_id = f"fallback-{uuid.uuid4()}" 
        record = {
            "_id": record_id, 
            TEXT_FIELD: doc.page_content or "",
            **(doc.metadata if doc.metadata else {}) 
        }
        records_to_upsert.append(record)
    if not records_to_upsert:
        print("No valid records prepared for upserting.")
        return
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
                    index.upsert_records(
                        namespace="__default__", 
                        records=batch
                    )
                    processed_count = len(batch)
                    success_count += processed_count
                    print(f"   Batch {(i//batch_size) + 1}/{ (total_records + batch_size - 1)//batch_size }: Processed {processed_count} records for upsert.")
                except Exception as e:
                    print(f"   Batch {(i//batch_size) + 1}: Error during upsert to {index_name}: {e}. Skipping batch.")
            print(f"Finished upserting to {index_type} index '{index_name}'. Attempted to process {success_count}/{total_records} records.")
        except Exception as e:
            print(f"Error connecting to or upserting into {index_type} index '{index_name}': {e}")
    print("\nPinecone upsert process completed.")


# --- MODIFIED: perform_hybrid_search ---
# This function is unchanged from our previous version, but included for completeness.
def perform_hybrid_search(
    query_text: str, 
    patient_id_filter: Optional[str] = None,
    patient_name_filter: Optional[str] = None, 
    admission_date_filter: Optional[str] = None,
    discharge_date_filter: Optional[str] = None,
    k: int = 10, 
    alpha: float = 0.7
) -> str:
    """
    Performs a hybrid search (dense + sparse) across Pinecone indexes.
    Applies metadata filtering for Patient_ID/Name and optionally dates.
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
    filter_parts = []
    
    identifier_parts = []
    if patient_id_filter:
        identifier_parts.append({"Patient_ID": {"$eq": patient_id_filter}})
        print(f"Applying Patient_ID filter: {patient_id_filter}")
        
    if patient_name_filter:
        identifier_parts.append({"Patient_Name": {"$eq": patient_name_filter}})
        print(f"Applying Patient_Name filter: {patient_name_filter}")
    
    if not identifier_parts:
         print("Warning: No Patient_ID or Patient_Name filter applied.")
    elif len(identifier_parts) == 1:
        filter_parts.append(identifier_parts[0])
    else:
        filter_parts.append({"$or": identifier_parts})
        
    if admission_date_filter:
        filter_parts.append({"Admission_Date": {"$eq": admission_date_filter}})
        print(f"Applying Admission_Date filter: {admission_date_filter}")
        
    if discharge_date_filter:
        filter_parts.append({"Discharge_Date": {"$eq": discharge_date_filter}})
        print(f"Applying Discharge_Date filter: {discharge_date_filter}")

    if len(filter_parts) == 1:
        metadata_filter = filter_parts[0]
    elif len(filter_parts) > 1:
        metadata_filter = {"$and": filter_parts}
        
    print(f"Combined metadata filter: {metadata_filter}")

    # 2. Perform Dense and Sparse Searches
    dense_hits = []
    sparse_hits = []
    dense_response = None
    sparse_response = None
    try:
        print(f"Performing dense search for '{query_text[:50]}...' (k={k})")
        dense_response = dense_index.search(
            namespace="__default__",
            query={
                "inputs": {"text": query_text},
                "top_k": k,
                "filter": metadata_filter 
            },
            fields=["page_content", "source", "Patient_ID", "Patient_Name", "Note_ID", "Section_Header", "chunk_id", "Admission_Date", "Discharge_Date"] 
        )
        try:
            dense_hits = dense_response['result']['hits']
            if not isinstance(dense_hits, list):
                dense_hits = []
            else:
                print(f"   -> Dense search returned {len(dense_hits)} results.")
        except (KeyError, TypeError, AttributeError):
            dense_hits = []

        print(f"Performing sparse search for '{query_text[:50]}...' (k={k})")
        sparse_response = sparse_index.search(
            namespace="__default__",
            query={
                "inputs": {"text": query_text},
                "top_k": k,
                "filter": metadata_filter
            },
            fields=["page_content", "source", "Patient_ID", "Patient_Name", "Note_ID", "Section_Header", "chunk_id", "Admission_Date", "Discharge_Date"]
        )
        try:
            sparse_hits = sparse_response['result']['hits']
            if not isinstance(sparse_hits, list):
                sparse_hits = []
            else:
                print(f"   -> Sparse search returned {len(sparse_hits)} results.")
        except (KeyError, TypeError, AttributeError):
            sparse_hits = []
    except Exception as e:
        print(f"Error during Pinecone search API call: {e}")
        return "Error: Failed to retrieve documents from the database due to API error."

    # 3. Combine and Re-score Results (Unchanged)
    combined_scores: Dict[str, float] = {}
    metadata_cache: Dict[str, Dict] = {} 
    for hit in dense_hits:
        record_id = hit.get('_id') 
        score = hit.get('_score', 0.0) 
        metadata = hit.get('fields', {}) 
        if record_id and score is not None: 
            combined_scores[record_id] = combined_scores.get(record_id, 0.0) + (alpha * score)
            if record_id not in metadata_cache:
                metadata_cache[record_id] = metadata if metadata else {}
    for hit in sparse_hits:
        record_id = hit.get('_id') 
        score = hit.get('_score', 0.0)
        metadata = hit.get('fields', {})
        if record_id and score is not None:
            combined_scores[record_id] = combined_scores.get(record_id, 0.0) + ((1 - alpha) * score)
            if record_id not in metadata_cache:
                metadata_cache[record_id] = metadata if metadata else {}
    if not combined_scores:
        print("No relevant documents found after combining search results.")
        return "No relevant context found in the documents for your query."

    # 4. Sort combined results (Unchanged)
    sorted_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    num_results_to_take = min(k, len(sorted_results)) 
    top_results = sorted_results[:num_results_to_take] 
    print(f"Combined and ranked {len(top_results)} results (out of {len(combined_scores)} unique).")

    # 5. Format the final context string
    context_str = ""
    if not top_results:
        print("No results after ranking.")
        return "No relevant context found after ranking."
        
    for i, (record_id, score) in enumerate(top_results):
        metadata = metadata_cache.get(record_id, {}) 
        
        note_id = metadata.get('Note_ID', 'Unknown Note')
        patient_id = metadata.get('Patient_ID', 'N/A')
        patient_name = metadata.get('Patient_Name', 'N/A')
        section = metadata.get('Section_Header', 'N/A')
        chunk_id = metadata.get('chunk_id', 'N/A')
        admission_date = metadata.get('Admission_Date', 'N/A')
        text_content = metadata.get(TEXT_FIELD, "Error: Content missing in metadata") 

        context_str += f"--- Context Chunk [{i+1}] (Score: {score:.4f}) ---\n"
        context_str += f"Patient: {patient_name} (ID: {patient_id})\n"
        context_str += f"Note ID: {note_id} (Admission: {admission_date})\n"
        context_str += f"Section: {section}\n"
        context_str += f"Content: {text_content}\n\n"
    
    return context_str.strip()