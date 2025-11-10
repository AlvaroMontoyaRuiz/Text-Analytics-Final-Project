import streamlit as st
from openai import OpenAI, APIError 
import os
# from dotenv import load_dotenv # --- REMOVED ---
import uuid
from datetime import datetime
import json
import re
import traceback 
from typing import List, Dict, Optional, Tuple 

# --- ADD THIS LINE ---
import google.generativeai as genai 

# --- NEW: Import OpenAI-based agent classes ---
from agents import ManagerAgent, PatientHistoryAgent, DischargeAgent

# Import SQLite/Auth files
from database import ChatDatabase
from auth import AuthManager

# Import RAG utilities
from utils.pinecone_database import perform_hybrid_search, ConnectionError as PineconeConnectionError
from utils.rag import count_tokens

# Import Security Guardrails
from utils.security import validate_input, apply_output_guardrails

# Configuration Constants
APP_TITLE = "Smart Chat Pro - Clinical Assistant"
APP_ICON = "🤖"
OPENAI_MODEL = "gpt-4o-mini" 
WORKER_MODEL = "gpt-4o-mini" # The OpenAI model for workers
MANAGER_MODEL = "gemini-1.5-pro-latest" 

MAX_CONTEXT_TOKENS = 120000 
MAX_MESSAGES_TO_SEND = 15 
DB_FILENAME = "chat_app.db"
SESSION_DIR = "session"
DB_PATH = os.path.join(SESSION_DIR, DB_FILENAME) 

# --- Initialize Clients (Both OpenAI and Gemini using st.secrets) ---

# --- MODIFIED: Initialize Google Gemini Client ---
try:
    # Use st.secrets.get()
    google_api_key = st.secrets.get('GEMINI_API_KEY')
    if not google_api_key:
        st.error("GEMINI_API_KEY not found. Please add it to your Streamlit Cloud secrets.")
        st.stop()
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Error initializing Google Gemini client: {e}")
    st.stop()

# --- MODIFIED: Initialize OpenAI Client ---
try:
    # Use st.secrets.get()
    openai_api_key = st.secrets.get('OPENAI_API_KEY')
    if not openai_api_key:
        st.error("OpenAI API key not found. Please add it to your Streamlit Cloud secrets.")
        st.stop()
    client = OpenAI(api_key=openai_api_key) 
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

# --- Initialize DB/Auth (Unchanged) ---
try:
    os.makedirs(SESSION_DIR, exist_ok=True)
    db = ChatDatabase(db_path=DB_PATH)
    auth = AuthManager(db_path=DB_PATH)
except Exception as e:
    st.error(f"Error during DB/Auth initialization: {e}")
    st.stop()

# --- MODIFIED: Initialize Hybrid Agents ---
try:
    manager_agent = ManagerAgent(
        model_name=MANAGER_MODEL
    )
    
    patient_history_agent = PatientHistoryAgent(
        client=client, 
        model_name=WORKER_MODEL
    )
    
    discharge_agent = DischargeAgent(
        client=client, 
        model_name=WORKER_MODEL
    )
    
    LAST_RETRIEVED_CONTEXT = {"content": ""}

except Exception as e:
    st.error(f"Fatal Error: Failed to initialize agents: {e}. Check 'agents.py' and API keys.")
    st.stop()


# --- Helper Functions (export_session_json, search_sessions) ---
def export_session_json(session_id: str) -> str:
    try:
        messages = db.get_session_messages(session_id)
        if "user_id" not in st.session_state or not st.session_state.user_id:
            return json.dumps({"error": "User not logged in"}, indent=2)
        sessions = db.get_user_sessions(st.session_state.user_id)
        session = next((s for s in sessions if s["session_id"] == session_id), None)
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
        return json.dumps(export_data, indent=2, default=str)
    except Exception as e:
        print(f"Error exporting session {session_id}: {e}")
        return json.dumps({"error": f"Failed to export session: {e}"}, indent=2)

def search_sessions(query: str) -> list:
    try:
        if "user_id" not in st.session_state or not st.session_state.user_id:
            return []
        if not query:
            return db.get_user_sessions(st.session_state.user_id)
        all_sessions = db.get_user_sessions(st.session_state.user_id)
        query_lower = query.lower()
        matching_session_ids = set()
        matching_sessions = []
        for session in all_sessions:
            session_id = session["session_id"]
            if query_lower in session.get("session_name", "").lower():
                if session_id not in matching_session_ids:
                    matching_sessions.append(session)
                    matching_session_ids.add(session_id)
                continue
            messages = db.get_session_messages(session_id)
            for msg in messages:
                if msg.get("role") != "system" and query_lower in msg.get("content", "").lower():
                    if session_id not in matching_session_ids:
                        matching_sessions.append(session)
                        matching_session_ids.add(session_id)
                    break
        return matching_sessions
    except Exception as e:
        print(f"Error searching sessions for query '{query}': {e}")
        st.error(f"Error during session search: {e}")
        return []

# --- Page Configuration (Unchanged) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Unchanged) ---
st.markdown("""
<style>
    /* ... (your CSS styles) ... */
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization (Unchanged) ---
default_states = {
    "authenticated": False,
    "user_id": None,
    "username": None,
    "current_session_id": None,
    "messages": [],
    "search_query": "",
    "regenerate_flag": False,
    "rag_enabled": True 
}
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Login/Register Page (Unchanged) ---
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1.5, 1]) 
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
                            st.rerun() 
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
    # --- Sidebar (Unchanged) ---
    with st.sidebar:
        st.markdown(f"#### Logged in as: {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            keys_to_clear = ["authenticated", "user_id", "username", "current_session_id", "messages"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Logged out successfully.")
            st.rerun()
        st.info("⚡ Multi-Agent RAG Enabled") 
        st.divider()
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                session_id = str(uuid.uuid4())
                session_name = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                db.create_session(session_id, st.session_state.user_id, session_name)
                st.session_state.messages = [] 
                st.session_state.current_session_id = session_id
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create new chat: {e}")
        st.divider()
        st.markdown("#### Chat History")
        search_query = st.text_input("🔍 Search sessions", key="search_input", placeholder="Search by name or content...")
        sessions = search_sessions(search_query) 
        if sessions:
            st.markdown(f"Found {len(sessions)} session(s)")
            with st.container(height=400): 
                for session in sessions:
                    session_id = session['session_id']
                    with st.container(border=True): 
                        col1, col2, col3 = st.columns([0.6, 0.2, 0.2]) 
                        with col1:
                            is_current = session_id == st.session_state.current_session_id
                            button_type = "primary" if is_current else "secondary"
                            if st.button(session["session_name"],
                                        key=f"session_{session_id}",
                                        use_container_width=True,
                                        type=button_type,
                                        help=f"Updated: {session['updated_at']}"):
                                try:
                                    st.session_state.current_session_id = session_id
                                    db_messages = db.get_session_messages(session_id)
                                    loaded_messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
                                    st.session_state.messages = [m for m in loaded_messages if m["role"] != "system"]
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
                                    if st.session_state.current_session_id == session_id:
                                        st.session_state.current_session_id = None
                                        st.session_state.messages = []
                                    st.success(f"Deleted '{session['session_name']}'")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete session: {e}")
        else:
            st.caption("No sessions found." if search_query else "No chat history yet.")

    # --- Main Chat Area (Unchanged) ---
    if st.session_state.current_session_id:
        current_session_name = "Chat" 
        current_total_tokens = 0
        try:
            sessions = db.get_user_sessions(st.session_state.user_id) 
            current_session = next((s for s in sessions if s["session_id"] == st.session_state.current_session_id), None)
            if current_session:
                current_session_name = current_session['session_name']
                current_total_tokens = current_session['total_tokens']
        except Exception as e:
            st.error(f"Could not load session details: {e}")
        col1, col_spacer, col2, col3 = st.columns([3, 1, 1, 1]) 
        with col1:
            st.markdown(f"### {current_session_name}")
        with col2:
            msg_count = len(st.session_state.messages) 
            st.metric("Messages", msg_count)
        with col3:
            with st.popover("⚙️ Options", use_container_width=True):
                st.markdown("##### Session Management") 
                new_name = st.text_input("Rename Session:", value=current_session_name, key="popover_rename_input")
                if st.button("Save Changes", use_container_width=True, type="primary"):
                    RERUN_NEEDED = False
                    try:
                        if new_name and new_name != current_session_name:
                            db.rename_session(st.session_state.current_session_id, new_name)
                            current_session_name = new_name 
                            RERUN_NEEDED = True
                            st.toast("Session renamed.")
                        if RERUN_NEEDED:
                            st.rerun()
                        else:
                            st.toast("No changes detected.")
                    except Exception as e:
                        st.error(f"Failed to save changes: {e}")
                st.divider()
                try:
                    export_json_data = export_session_json(st.session_state.current_session_id)
                    st.download_button("📥 Export Chat", data=export_json_data, file_name=f"{current_session_name}_{st.session_state.current_session_id[:8]}.json", mime="application/json", use_container_width=True)
                except Exception as e:
                    st.error("Export failed.")
        st.divider()
        current_display_tokens = count_tokens(st.session_state.messages)
        if current_display_tokens > MAX_CONTEXT_TOKENS * 0.9: 
            st.warning(f"Chat history is long ({current_display_tokens:,} tokens). Older messages might be excluded from context sent to the AI (limit: ~{MAX_MESSAGES_TO_SEND} turns).")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"]) 

        if st.session_state.regenerate_flag:
            st.session_state.regenerate_flag = False 
            user_input_to_regenerate = None
            if len(st.session_state.messages) >= 2:
                try:
                    db.remove_last_message(st.session_state.current_session_id) 
                    st.session_state.messages.pop()
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
            if user_input_to_regenerate:
                print("Triggering regeneration for:", user_input_to_regenerate)
                user_input = user_input_to_regenerate
                st.session_state._regenerating_now = True
            else:
                user_input = None 
        else:
            col_input, col_regenerate = st.columns([5, 1]) 
            with col_input:
                user_input = st.chat_input("Ask about patient notes (e.g., 'HPI for Casey Gray')...") 
            with col_regenerate:
                can_regenerate = len(st.session_state.messages) >= 2
                if st.button("🔄", help="Regenerate last response", use_container_width=True, disabled=not can_regenerate):
                    st.session_state.regenerate_flag = True
                    st.rerun() 

        # --- MODIFIED Main Processing Logic (Multi-Agent) ---
        if user_input:
            is_valid, error_message = validate_input(user_input)
            if not is_valid:
                with st.chat_message("assistant", avatar="🚨"):
                    st.error(f"Input Blocked: {error_message}")
            else:
                is_regeneration = st.session_state.pop('_regenerating_now', False) 
                if not is_regeneration:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.write(user_input)
                    try:
                        user_tokens = count_tokens([{"content": user_input}])
                        db.add_message(st.session_state.current_session_id, "user", user_input, tokens=user_tokens)
                    except Exception as e:
                        st.error(f"Failed to save your message: {e}")

                final_response = ""
                agent_to_run = None
                LAST_RETRIEVED_CONTEXT["content"] = "" 

                try:
                    patient_id_filter = None
                    patient_name_filter = None
                    
                    id_match = re.search(r"MRN\s*(\d+)|Subject ID:\s*(\d+)", user_input, re.IGNORECASE)
                    if id_match:
                        if id_match.group(1): 
                             patient_id_filter = f"MRN {id_match.group(1).strip()}"
                        else: 
                             patient_id_filter = id_match.group(2).strip()
                    else:
                        name_match = re.search(r"for\s+([A-Za-z]+\s+[A-Za-z]+)|patient\s+([A-Za-z]+\s+[A-Za-z]+)|(Casey Gray)", user_input, re.IGNORECASE)
                        if name_match:
                            patient_name_filter = name_match.group(1) or name_match.group(2) or name_match.group(3)
                            patient_name_filter = patient_name_filter.strip()

                    
                    if not patient_id_filter and not patient_name_filter:
                        final_response = "I cannot proceed without a Patient ID or Name. Please include the patient's ID (e.g., 'Subject ID: 10001401') or full name (e.g., 'for patient Casey Gray') in your request."
                        with st.chat_message("assistant", avatar="🚨"):
                            st.error(final_response)
                    
                    else:
                        with st.spinner(f"Manager Agent ({MANAGER_MODEL}) is classifying intent..."):
                            intent = manager_agent.execute(user_input)

                        identifier_str = patient_id_filter if patient_id_filter else patient_name_filter
                        st.caption(f"🤖 Manager routed to: **{intent}** (Patient: {identifier_str})")
                        
                        if intent == "Patient History":
                            agent_to_run = patient_history_agent
                        elif intent == "Discharge":
                            agent_to_run = discharge_agent
                        
                        if agent_to_run:
                            with st.spinner(f"{agent_to_run.name} ({WORKER_MODEL}) is processing..."):
                                history_for_agent = st.session_state.messages[-MAX_MESSAGES_TO_SEND:]
                                
                                response_text = agent_to_run.run( 
                                    query=user_input,
                                    patient_id=patient_id_filter,
                                    patient_name=patient_name_filter,
                                    chat_history=history_for_agent
                                )
                                
                                LAST_RETRIEVED_CONTEXT["content"] = agent_to_run.last_context
                            
                            final_response = apply_output_guardrails(response_text)
                            
                            with st.chat_message("assistant"):
                                st.write(final_response)
                                if LAST_RETRIEVED_CONTEXT["content"]:
                                    with st.expander("View Retrieved Context"):
                                        st.text(LAST_RETRIEVED_CONTEXT["content"])

                        elif intent == "Ambiguous":
                            final_response = "Your request is unclear. Please specify if you are looking for **patient history** (symptoms, past procedures) or **discharge information** (hospital course, medications)."
                            with st.chat_message("assistant"):
                                st.write(final_response)
                        
                        else: 
                            final_response = "I am sorry, but I can only assist with tasks related to 'Patient History' or 'Discharge Summaries'."
                            with st.chat_message("assistant"):
                                st.write(final_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                
                    try:
                        assistant_tokens = count_tokens([{"content": final_response}])
                        db.add_message(st.session_state.current_session_id, "assistant", final_response, tokens=assistant_tokens)
                    except Exception as e:
                        st.error(f"Failed to save assistant's message: {e}")

                except Exception as e:
                    st.error(f"An error occurred during the agent workflow: {e}")
                    traceback.print_exc() 

    # --- Dashboard View (Unchanged) ---
    else:
        st.title(f"{APP_ICON} Welcome to {APP_TITLE}")
        st.markdown("Select a chat from the sidebar or start a new one.")
        st.divider()
        st.subheader("📊 Session Dashboard")
        try:
            sessions = db.get_user_sessions(st.session_state.user_id)
            total_sessions = len(sessions)
            total_messages = 0
            total_tokens_all = 0
            if sessions:
                total_messages = sum(len([m for m in db.get_session_messages(s["session_id"]) if m["role"] != "system"]) for s in sessions)
                total_tokens_all = sum(s.get('total_tokens', 0) for s in sessions)
            
            estimated_cost = (total_tokens_all / 1_000_000) * 0.375 
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📁 Total Chats", total_sessions)
            col2.metric("💬 Total Messages", total_messages)
            col3.metric("🎯 Total Tokens Used", f"{total_tokens_all:,}")
            col4.metric("💰 Est. Cost", f"${estimated_cost:.3f}") 
        except Exception as e:
            st.error(f"Could not load dashboard metrics: {e}")
        st.divider()
        st.caption("👈 Use the sidebar to manage your chats.")