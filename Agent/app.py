import streamlit as st
from openai import OpenAI, APIError # Import specific APIError
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json
import re
from typing import List, Dict, Optional, Tuple # Added Tuple

# Import SQLite/Auth files
from database import ChatDatabase
from auth import AuthManager

# Import RAG utilities
from utils.pinecone_database import perform_hybrid_search, ConnectionError as PineconeConnectionError
from utils.rag import count_tokens

# Import Security Guardrails
from utils.security import validate_input, apply_output_guardrails

# --- Configuration Constants ---
APP_TITLE = "Smart Chat Pro - Clinical Assistant"
APP_ICON = "🤖"
DEFAULT_SYSTEM_PROMPT = "You are a helpful clinical assistant. Answer questions based ONLY on the provided context. Cite sources using [Source File (Chunk: chunk-id), Patient_ID: ID, Section: Section Name]. If the context is insufficient, state that you cannot answer from the provided documents."
OPENAI_MODEL = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 120000 # Rough limit for gpt-4o-mini (adjust based on actual limits)
MAX_MESSAGES_TO_SEND = 15 # Limit history sent to LLM to prevent exceeding limits
DB_FILENAME = "chat_app.db"
SESSION_DIR = "session"
DB_PATH = os.path.join(SESSION_DIR, DB_FILENAME) # Use os.path.join for cross-platform compatibility

# Load environment variables
load_dotenv()

# --- Initialize Clients ---
# Add error handling for client initialization
try:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)

    # Ensure session directory exists (moved here from database.py for clarity)
    os.makedirs(SESSION_DIR, exist_ok=True)

    db = ChatDatabase(db_path=DB_PATH)
    auth = AuthManager(db_path=DB_PATH)

    # Perform a quick check during init (optional but recommended)
    # db._get_connection().close() # Check DB connection
    # Consider adding a check for Pinecone connection here if desired

except Exception as e:
    st.error(f"Error during initialization: {e}")
    st.stop()


# --- Helper Functions (Mostly Unchanged) ---
def export_session_json(session_id: str) -> str:
    """Exports session data to a JSON string."""
    try:
        messages = db.get_session_messages(session_id)
        # Ensure user_id exists before fetching sessions
        if "user_id" not in st.session_state or not st.session_state.user_id:
            return json.dumps({"error": "User not logged in"}, indent=2)

        sessions = db.get_user_sessions(st.session_state.user_id)
        session = next((s for s in sessions if s["session_id"] == session_id), None)

        # Convert datetime objects to strings for JSON serialization if necessary
        # (SQLite usually stores them compatibly, but good practice)
        if session:
             session["created_at"] = str(session.get("created_at", ""))
             session["updated_at"] = str(session.get("updated_at", ""))
        for msg in messages:
             msg["timestamp"] = str(msg.get("timestamp", ""))

        export_data = {
            "session_name": session["session_name"] if session else "Unknown",
            "created_at": session["created_at"] if session else "",
            "messages": messages,
            "total_tokens": session["total_tokens"] if session else 0
        }
        return json.dumps(export_data, indent=2, default=str) # Add default=str for safety
    except Exception as e:
        print(f"Error exporting session {session_id}: {e}")
        return json.dumps({"error": f"Failed to export session: {e}"}, indent=2)

def search_sessions(query: str) -> list:
    """Searches session names and message content for a query."""
    try:
        if "user_id" not in st.session_state or not st.session_state.user_id:
            return [] # Return empty if user not logged in

        if not query:
            return db.get_user_sessions(st.session_state.user_id)

        all_sessions = db.get_user_sessions(st.session_state.user_id)
        query_lower = query.lower()
        matching_session_ids = set()
        matching_sessions = []

        for session in all_sessions:
            session_id = session["session_id"]
            # Check session name first
            if query_lower in session.get("session_name", "").lower():
                if session_id not in matching_session_ids:
                    matching_sessions.append(session)
                    matching_session_ids.add(session_id)
                continue # Skip message search if name matched

            # If name didn't match, search messages
            messages = db.get_session_messages(session_id)
            for msg in messages:
                if msg.get("role") != "system" and query_lower in msg.get("content", "").lower():
                    if session_id not in matching_session_ids:
                         matching_sessions.append(session)
                         matching_session_ids.add(session_id)
                    break # Stop searching messages for this session once a match is found
        return matching_sessions
    except Exception as e:
        print(f"Error searching sessions for query '{query}': {e}")
        st.error(f"Error during session search: {e}")
        return []


# --- REFACTORED CORE RAG FUNCTION with Error Handling & Context Trimming ---
def generate_rag_response(system_prompt: str, messages_list: List[Dict], user_input: str) -> Tuple[Optional[str], Optional[str], int, Optional[str]]:
    """
    Performs RAG query, augments prompt, streams LLM response with error handling.
    Returns: (final_response_text, context_str, assistant_tokens, error_message)
             On success, error_message is None. On failure, final_response_text is None.
    """
    response_text = ""
    context_str = "Context retrieval skipped due to previous error."
    assistant_tokens = 0
    error_message = None

    # 1. Extract Patient ID for filtering (No changes needed)
    mrn_match = re.search(r"MRN\s*(\d+)", user_input, re.IGNORECASE)
    patient_id_filter = f"MRN {mrn_match.group(1).strip()}" if mrn_match else "N/A"

    # 2. Perform Hybrid Search (STEP 2: RETRIEVAL) with Error Handling
    try:
        with st.spinner(f"Retrieving context (Patient Filter: {patient_id_filter})..."):
            context_str = perform_hybrid_search(user_input, patient_id_filter, k=10) # k=5 documents
    except PineconeConnectionError as e:
        error_message = f"Error connecting to document database: {e}"
        st.error(error_message)
        return None, context_str, 0, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during context retrieval: {e}"
        st.error(error_message)
        # Optionally log traceback here: import traceback; traceback.print_exc()
        return None, context_str, 0, error_message

    # 3. Create RAG-Augmented System Prompt (No changes needed)
    rag_system_prompt = (
        system_prompt +
        "\n\n--- CONTEXT FOR GROUNDING ---\n"
        "Use the provided context chunks below to answer the user's question. "
        "Cite the source file, Patient_ID, and Section when relevant using the format [Source File (Chunk: chunk-id), Patient_ID: ID, Section: Section Name]. "
        "If the context is insufficient or irrelevant, state that you cannot answer based on the provided documents.\n\n"
        f"CONTEXT:\n{context_str}"
    )

    # 4. Prepare message list for API call - WITH CONTEXT TRIMMING
    # Keep system prompt, add user input, keep most recent non-system messages
    messages_to_send = [{"role": "system", "content": rag_system_prompt}]

    # Get recent history (excluding system prompt)
    history = [msg for msg in messages_list if msg["role"] != "system"][-MAX_MESSAGES_TO_SEND:]

    messages_to_send.extend(history)

    # Ensure the *current* user input is the last message if not already included
    if not history or history[-1]['content'] != user_input or history[-1]['role'] != 'user':
         messages_to_send.append({"role": "user", "content": user_input})

    # Optional: Add token check here and potentially trim more if still too long
    # current_prompt_tokens = count_tokens(messages_to_send)
    # print(f"Debug: Tokens being sent to LLM: {current_prompt_tokens}")
    # if current_prompt_tokens > SOME_INPUT_LIMIT: handle trimming further

    # 5. Get and stream assistant response (STEP 3: GENERATION) with Error Handling
    try:
        with st.chat_message("assistant"):
            st.caption(f"🤖 RAG Mode (Filter: {patient_id_filter})")

            with st.expander("View Retrieved Context"):
                st.text(context_str if context_str else "No context was retrieved.")

            response_placeholder = st.empty()
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages_to_send,
                temperature=0.1, # Low temperature for factual grounding
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    response_placeholder.write(response_text + "▌") # Add cursor during streaming
            response_placeholder.write(response_text) # Write final complete text

    except APIError as e:
        error_message = f"OpenAI API Error: {e}"
        st.error(error_message)
        return None, context_str, 0, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during response generation: {e}"
        st.error(error_message)
        # Optionally log traceback here
        return None, context_str, 0, error_message

    # 6. Apply Output Guardrail
    final_response_text = apply_output_guardrails(response_text)

    # Update UI *if* disclaimer was added (to ensure it's visible after streaming stops)
    if final_response_text != response_text and 'response_placeholder' in locals():
           response_placeholder.write(final_response_text)

    # 7. Count tokens for the final response (including disclaimer)
    try:
        assistant_tokens = count_tokens([{"content": final_response_text}])
    except Exception as e:
        print(f"Warning: Could not count tokens for final response: {e}")
        assistant_tokens = 0 # Default to 0 if counting fails

    return final_response_text, context_str, assistant_tokens, None # Return None for error on success


# --- Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Add subtle hover effects */
    .stButton>button:hover {
        border-color: #f63366;
        color: #f63366;
    }
    .stDownloadButton>button:hover {
        border-color: #f63366;
        color: #f63366;
    }
    /* Customize chat messages slightly */
     .stChatMessage {
         padding: 1rem;
         border-radius: 0.5rem;
         margin-bottom: 1rem;
         border: 1px solid transparent; /* Base border */
     }
     /* Optional: Style user message differently */
     /* .stChatMessage[data-testid="chat-message-container-user"] {
          background-color: #f0f2f6;
     } */
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Ensures all keys exist
default_states = {
    "authenticated": False,
    "user_id": None,
    "username": None,
    "current_session_id": None,
    "messages": [],
    "search_query": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "regenerate_flag": False,
    "rag_enabled": True # Keep RAG always enabled as per previous logic
}
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Login/Register Page ---
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1.5, 1]) # Adjust column width
    with col2:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown("### Your AI-Powered Clinical Document Assistant")
        st.markdown("---")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                st.subheader("Welcome Back!")
                login_username = st.text_input("Username", key="login_user")
                login_password = st.text_input("Password", type="password", key="login_pass")
                submit = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submit:
                    try:
                        success, user_id = auth.authenticate_user(login_username, login_password)
                        if success and user_id:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_id
                            st.session_state.username = login_username
                            st.success("Login successful!")
                            st.rerun() # Use rerun instead of experimental_rerun
                        else:
                            st.error("Invalid username or password")
                    except Exception as e:
                        st.error(f"Login failed: {e}")

        with tab2:
            with st.form("register_form"):
                st.subheader("Create Account")
                reg_username = st.text_input("Choose Username", key="reg_user")
                reg_email = st.text_input("Email (optional)", key="reg_email")
                reg_password = st.text_input("Choose Password", type="password", key="reg_pass")
                reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
                submit = st.form_submit_button("Create Account", use_container_width=True)
                if submit:
                    if not reg_username or not reg_password:
                        st.error("Username and password are required")
                    elif reg_password != reg_password2:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        try:
                            success, message_or_id = auth.create_user(reg_username, reg_password, reg_email)
                            if success:
                                st.success("Account created successfully! Please login.")
                            else:
                                st.error(f"Registration failed: {message_or_id}")
                        except Exception as e:
                            st.error(f"Registration error: {e}")

# --- Main Application UI ---
else:
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"#### Logged in as: {st.session_state.username}")

        # Logout Button
        if st.button("🚪 Logout", use_container_width=True):
            # Clear sensitive session state keys
            keys_to_clear = ["authenticated", "user_id", "username", "current_session_id", "messages"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Optionally keep non-sensitive states like theme settings
            st.success("Logged out successfully.")
            st.rerun()

        st.info("⚡ Hybrid RAG Enabled")
        st.divider()

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                session_id = str(uuid.uuid4())
                # Use a more descriptive default name
                session_name = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                # --- Use DEFAULT_SYSTEM_PROMPT always when creating new chat ---
                system_prompt = DEFAULT_SYSTEM_PROMPT

                db.create_session(session_id, st.session_state.user_id, session_name)
                # Only add system message if it's not empty/None
                if system_prompt:
                     db.add_message(session_id, "system", system_prompt, tokens=0)
                     # --- Initialize session state with the DEFAULT prompt ---
                     st.session_state.messages = [{"role": "system", "content": system_prompt}]
                else:
                     st.session_state.messages = [] # Start with empty if no system prompt

                # --- Set the session state system prompt to default too ---
                st.session_state.system_prompt = system_prompt
                st.session_state.current_session_id = session_id
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create new chat: {e}")

        st.divider()

        # Search Sessions
        st.markdown("#### Chat History")
        search_query = st.text_input("🔍 Search sessions", key="search_input", placeholder="Search by name or content...")

        # Fetch and display sessions
        sessions = search_sessions(search_query) # Already includes error handling

        if sessions:
             st.markdown(f"Found {len(sessions)} session(s)")
             # Use a scrollable container for long session lists
             with st.container(height=400): # Adjust height as needed
                for session in sessions:
                    session_id = session['session_id']
                    with st.container(border=True): # Add border for visual separation
                        col1, col2, col3 = st.columns([0.6, 0.2, 0.2]) # Adjust widths
                        with col1:
                            is_current = session_id == st.session_state.current_session_id
                            button_type = "primary" if is_current else "secondary"
                            # Use session name in button, add tooltip with date?
                            if st.button(session["session_name"],
                                         key=f"session_{session_id}",
                                         use_container_width=True,
                                         type=button_type,
                                         help=f"Updated: {session['updated_at']}"):
                                try:
                                    st.session_state.current_session_id = session_id
                                    db_messages = db.get_session_messages(session_id)
                                    # Convert DB format to session state format (role, content)
                                    loaded_messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]

                                    # Find or set system prompt
                                    system_prompt_content = DEFAULT_SYSTEM_PROMPT # Always load default
                                    if loaded_messages and loaded_messages[0]["role"] == "system":
                                        # Use the prompt stored in DB if available
                                        system_prompt_content = loaded_messages[0]["content"]
                                    # Ensure system prompt exists in messages list for logic
                                    # Add default system prompt if missing from loaded messages
                                    if not loaded_messages or loaded_messages[0]["role"] != "system":
                                         loaded_messages.insert(0, {"role": "system", "content": system_prompt_content})


                                    st.session_state.messages = loaded_messages
                                    # --- Update session state prompt when loading ---
                                    st.session_state.system_prompt = system_prompt_content
                                    st.rerun()
                                except Exception as e:
                                     st.error(f"Error loading session {session_id}: {e}")

                        with col2:
                            try:
                                export_json = export_session_json(session_id)
                                st.download_button("📥",
                                                 data=export_json,
                                                 file_name=f"{session.get('session_name', 'chat_export')}_{session_id[:8]}.json",
                                                 mime="application/json",
                                                 key=f"export_{session_id}",
                                                 help="Export chat as JSON")
                            except Exception as e:
                                 st.error(f"Export failed: {e}")

                        with col3:
                            if st.button("🗑️", key=f"delete_{session_id}", help="Delete chat"):
                                try:
                                    db.delete_session(session_id)
                                    # If deleting the current session, reset chat area
                                    if st.session_state.current_session_id == session_id:
                                        st.session_state.current_session_id = None
                                        st.session_state.messages = []
                                    st.success(f"Deleted '{session['session_name']}'")
                                    st.rerun()
                                except Exception as e:
                                     st.error(f"Failed to delete session: {e}")
        else:
            st.caption("No sessions found." if search_query else "No chat history yet.")

    # --- Main Chat Area ---
    if st.session_state.current_session_id:
        # Get current session details (needed for title, options)
        current_session_name = "Chat" # Default
        current_total_tokens = 0
        try:
             sessions = db.get_user_sessions(st.session_state.user_id) # Fetch might be needed if name changes
             current_session = next((s for s in sessions if s["session_id"] == st.session_state.current_session_id), None)
             if current_session:
                  current_session_name = current_session['session_name']
                  current_total_tokens = current_session['total_tokens']
        except Exception as e:
             st.error(f"Could not load session details: {e}")


        # --- Header and Options ---
        col1, col_spacer, col2, col3 = st.columns([3, 1, 1, 1]) # Add spacer
        with col1:
             # Use Markdown for potentially larger titles
             st.markdown(f"### {current_session_name}")
        with col2:
             # Use message count excluding system prompt
             msg_count = len([m for m in st.session_state.messages if m["role"] != "system"])
             st.metric("Messages", msg_count)
        with col3:
             # --- MODIFIED POPOVER: Removed System Prompt Editing ---
             with st.popover("⚙️ Options", use_container_width=True):
                  st.markdown("##### Session Management") # Changed header
                  new_name = st.text_input(
                       "Rename Session:",
                       value=current_session_name,
                       key="popover_rename_input"
                  )

                  if st.button("Save Changes", use_container_width=True, type="primary"):
                        RERUN_NEEDED = False
                        try:
                                # Update name if changed
                                if new_name and new_name != current_session_name:
                                    db.rename_session(st.session_state.current_session_id, new_name)
                                    current_session_name = new_name # Update local variable for immediate display
                                    RERUN_NEEDED = True # Rerun to update sidebar
                                    st.toast("Session renamed.")
                                
                                # --- Removed System Prompt Update Logic ---

                                if RERUN_NEEDED:
                                    st.rerun() # Rerun only if name changed
                                else: # If only prompt was changed (now removed), no action needed
                                    st.toast("No changes detected.") # Or remove toast if nothing happened

                        except Exception as e:
                                st.error(f"Failed to save changes: {e}")

                  st.divider()
                  try:
                       export_json_data = export_session_json(st.session_state.current_session_id)
                       st.download_button(
                            "📥 Export Chat",
                            data=export_json_data,
                            file_name=f"{current_session_name}_{st.session_state.current_session_id[:8]}.json",
                            mime="application/json",
                            use_container_width=True
                       )
                  except Exception as e:
                       st.error("Export failed.")
        st.divider()

        # --- Display Chat Messages ---
        # Calculate current tokens for warning
        current_display_tokens = count_tokens(st.session_state.messages)
        if current_display_tokens > MAX_CONTEXT_TOKENS * 0.9: # Warn at 90%
            st.warning(f"Chat history is long ({current_display_tokens:,} tokens). Older messages might be excluded from context sent to the AI (limit: ~{MAX_MESSAGES_TO_SEND} turns).")

        # Display conversation history (skip system message in UI)
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.write(msg["content"]) # Use write for markdown rendering

        # --- Handle Regeneration Flag ---
        # This block runs if the regenerate button was clicked on the *previous* run
        if st.session_state.regenerate_flag:
            st.session_state.regenerate_flag = False # Reset flag
            user_input_to_regenerate = None

            # Ensure there are at least user and assistant messages to remove
            if len(st.session_state.messages) >= 3:
                try:
                    # Remove last ASSISTANT message (DB and state)
                    # Corrected: remove_last_message returns details now
                    db.remove_last_message(st.session_state.current_session_id) 
                    st.session_state.messages.pop()

                    # Remove last USER message (DB and state), keep its content
                    last_user_msg_details = db.remove_last_message(st.session_state.current_session_id)
                    st.session_state.messages.pop()

                    if last_user_msg_details and last_user_msg_details.get("role") == "user":
                        user_input_to_regenerate = last_user_msg_details.get("content")
                    else:
                        st.error("Failed to retrieve the last user message for regeneration.")

                except Exception as e:
                    st.error(f"Error during regeneration setup: {e}")
            else:
                st.error("Not enough messages in history to regenerate.")

            # If we successfully got the last user input, trigger generation automatically
            if user_input_to_regenerate:
                print("Triggering regeneration for:", user_input_to_regenerate) # Debug
                # Set user_input variable so the main processing block runs
                user_input = user_input_to_regenerate
                # Set a temporary flag to prevent re-adding user message during regen processing
                st.session_state._regenerating_now = True
            else:
                 user_input = None # Ensure main block doesn't run if setup failed

        # --- Normal User Input ---
        else:
            # Input bar and Regenerate button
            col_input, col_regenerate = st.columns([5, 1]) # Adjust ratio
            with col_input:
                user_input = st.chat_input("Ask about clinical documents...")
            with col_regenerate:
                # Disable regenerate if history is too short
                can_regenerate = len(st.session_state.messages) >= 3
                if st.button("🔄",
                             help="Regenerate last response",
                             use_container_width=True,
                             disabled=not can_regenerate):
                    st.session_state.regenerate_flag = True
                    st.rerun() # Rerun immediately to trigger the flag logic above

        # --- Main Processing Logic (Handles both new input and regeneration) ---
        if user_input:
            # 1. Validate Input (Security)
            is_valid, error_message = validate_input(user_input)

            if not is_valid:
                # Show guardrail message inline
                with st.chat_message("assistant", avatar="🚨"):
                    st.error(f"Input Blocked: {error_message}")
                # DO NOT save blocked input or proceed further

            else:
                # Input is valid, proceed with RAG

                # 2. Add USER message to UI state *first* for immediate display
                # Use a flag to prevent adding during regeneration
                is_regeneration = st.session_state.pop('_regenerating_now', False) # Check and remove flag
                if not is_regeneration:
                     st.session_state.messages.append({"role": "user", "content": user_input})
                     with st.chat_message("user"):
                          st.write(user_input)

                     # 3. Add USER message to DB
                     try:
                          user_tokens = count_tokens([{"content": user_input}])
                          db.add_message(st.session_state.current_session_id, "user", user_input, tokens=user_tokens)
                     except Exception as e:
                          st.error(f"Failed to save your message: {e}")
                          # Decide whether to stop or continue generation


                # 4. Generate RAG response (includes API call and streaming)
                # Pass the *current* messages list (including the latest user input)
                final_response, context, assistant_tokens, generation_error = generate_rag_response(
                    # --- Always use the prompt from session state ---
                    st.session_state.system_prompt,
                    st.session_state.messages,
                    user_input # Pass the specific input being processed
                )

                # 5. Handle response or error
                if generation_error:
                    # Error already shown by generate_rag_response
                    pass # Keep UI clean, error shown via st.error
                elif final_response is not None:
                    # 6. Add successful ASSISTANT message to UI state
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

                    # 7. Add successful ASSISTANT message to DB
                    try:
                        db.add_message(st.session_state.current_session_id, "assistant", final_response, tokens=assistant_tokens)
                    except Exception as e:
                        st.error(f"Failed to save assistant's message: {e}")

                # 8. No automatic rerun needed here.


    # --- Dashboard View (No Current Session Selected) ---
    else:
        st.title(f"{APP_ICON} Welcome to {APP_TITLE}")
        st.markdown("Select a chat from the sidebar or start a new one.")
        st.divider()

        # Display Dashboard Metrics
        st.subheader("📊 Session Dashboard")
        try:
             sessions = db.get_user_sessions(st.session_state.user_id)
             total_sessions = len(sessions)
             total_messages = 0
             total_tokens_all = 0
             if sessions:
                 total_messages = sum(len([m for m in db.get_session_messages(s["session_id"]) if m["role"] != "system"]) for s in sessions)
                 total_tokens_all = sum(s.get('total_tokens', 0) for s in sessions)

             # Rough cost estimate for gpt-4o-mini ($0.15 / 1M input, $0.60 / 1M output - average cost $0.375/1M)
             estimated_cost = (total_tokens_all / 1_000_000) * 0.375

             col1, col2, col3, col4 = st.columns(4)
             col1.metric("📁 Total Chats", total_sessions)
             col2.metric("💬 Total Messages", total_messages)
             col3.metric("🎯 Total Tokens Used", f"{total_tokens_all:,}")
             col4.metric("💰 Est. Cost", f"${estimated_cost:.3f}") # Show more precision for low costs
        except Exception as e:
            st.error(f"Could not load dashboard metrics: {e}")

        st.divider()
        # Optionally show recent sessions here too if desired
        st.caption("👈 Use the sidebar to manage your chats.")

