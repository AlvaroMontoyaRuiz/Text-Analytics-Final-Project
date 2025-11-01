"""
Database operations for chat application using SQLite.
Manages users, chat sessions, and individual messages.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json # Not used directly here, but often useful with DB data
import os # Import os to handle directory creation

class ChatDatabase:
    """Manages SQLite database for chat sessions and messages"""

    def __init__(self, db_path: str = "session/chat_app.db"): # Updated default path
        """Initializes the database connection and ensures tables exist."""
        self.db_path = db_path
        # Ensure the directory exists before connecting
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes and returns a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;") # Ensure foreign key constraints are enforced
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise # Re-raise the exception to signal connection failure

    def init_database(self):
        """Creates the necessary tables if they don't already exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Users table (for authentication)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT, -- Optional
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_tokens INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')), -- Enforce roles
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens INTEGER DEFAULT 0, -- Tokens for this specific message
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Indexing for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);")

            conn.commit()
            print("Database initialized successfully.")
        except sqlite3.Error as e:
            print(f"Error initializing database tables: {e}")
        finally:
            conn.close()

    def create_session(self, session_id: str, user_id: str, session_name: str) -> bool:
        """Creates a new chat session."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, session_name, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, user_id, session_name))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
             print(f"Error: Session ID {session_id} might already exist or User ID {user_id} not found.")
             return False
        except sqlite3.Error as e:
            print(f"Error creating session {session_id}: {e}")
            return False
        finally:
            conn.close()

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Gets all sessions for a specific user, ordered by most recently updated."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row # Allows access by column name
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, session_name, created_at, updated_at, total_tokens
                FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
            sessions = [dict(row) for row in cursor.fetchall()]
            return sessions
        except sqlite3.Error as e:
            print(f"Error fetching sessions for user {user_id}: {e}")
            return []
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str, tokens: int = 0) -> bool:
        """Adds a message to a session and updates the session's token count and timestamp."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Use a transaction for atomicity
            conn.execute("BEGIN TRANSACTION;")

            # Insert the new message
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, tokens)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, tokens))

            # Update the corresponding session's total tokens and updated timestamp
            cursor.execute("""
                UPDATE sessions
                SET total_tokens = total_tokens + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (tokens, session_id))

            # Check if the update affected any row (session exists)
            if cursor.rowcount == 0:
                 raise sqlite3.Error(f"Session ID {session_id} not found during message add.")

            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding message to session {session_id}: {e}")
            conn.rollback() # Rollback transaction on error
            return False
        finally:
            conn.close()

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Gets all messages for a specific session, ordered chronologically."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, role, content, timestamp, tokens
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            messages = [dict(row) for row in cursor.fetchall()]
            return messages
        except sqlite3.Error as e:
            print(f"Error fetching messages for session {session_id}: {e}")
            return []
        finally:
            conn.close()

    def remove_last_message(self, session_id: str) -> Optional[str]:
        """
        Deletes the single most recent message (user or assistant) for a session,
        deducts its tokens from the session total, and returns the deleted message's content.
        Returns None if no message exists or an error occurs.
        """
        conn = self._get_connection()
        deleted_content: Optional[str] = None
        try:
            cursor = conn.cursor()

            # Use a transaction
            conn.execute("BEGIN TRANSACTION;")

            # 1. Find the last message's ID, tokens, and content
            cursor.execute("""
                SELECT message_id, tokens, content
                FROM messages
                WHERE session_id = ?
                ORDER BY message_id DESC
                LIMIT 1
            """, (session_id,))

            result = cursor.fetchone()

            if result:
                message_id, tokens_to_deduct, content = result
                deleted_content = content # Store content to return

                # 2. Delete the message
                cursor.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))

                # 3. Deduct tokens from the session
                cursor.execute("""
                    UPDATE sessions
                    SET total_tokens = total_tokens - ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (tokens_to_deduct, session_id))

                # Check if the update was successful (session exists)
                if cursor.rowcount == 0:
                     raise sqlite3.Error(f"Session ID {session_id} not found during token deduction.")

                conn.commit()
                print(f"Removed last message (ID: {message_id}) from session {session_id}, deducted {tokens_to_deduct} tokens.")
            else:
                 print(f"No messages found in session {session_id} to remove.")
                 conn.rollback() # No changes needed

        except sqlite3.Error as e:
            print(f"Error removing last message from session {session_id}: {e}")
            conn.rollback()
            deleted_content = None # Ensure None is returned on error
        finally:
            conn.close()

        return deleted_content # Return the content or None

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and all its associated messages."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Deleting the session will cascade delete messages due to FOREIGN KEY ON DELETE CASCADE
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            print(f"Deleted session {session_id} and associated messages.")
            return True
        except sqlite3.Error as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
        finally:
            conn.close()

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Renames an existing chat session."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET session_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (new_name, session_id))
            conn.commit()
            if cursor.rowcount > 0:
                 print(f"Renamed session {session_id} to '{new_name}'.")
                 return True
            else:
                 print(f"Session {session_id} not found for renaming.")
                 return False
        except sqlite3.Error as e:
            print(f"Error renaming session {session_id}: {e}")
            return False
        finally:
            conn.close()

    # --- User Management Methods ---

    def add_user(self, user_id: str, username: str, password_hash: str, email: Optional[str] = None) -> bool:
        """Adds a new user to the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, username, password_hash, email)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, password_hash, email))
            conn.commit()
            return True
        except sqlite3.IntegrityError: # Handles UNIQUE constraint violation for username
            print(f"Error: Username '{username}' already exists.")
            return False
        except sqlite3.Error as e:
            print(f"Error adding user {username}: {e}")
            return False
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Retrieves user details by username."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, password_hash, email, created_at
                FROM users
                WHERE username = ?
            """, (username,))
            user_data = cursor.fetchone()
            return dict(user_data) if user_data else None
        except sqlite3.Error as e:
            print(f"Error fetching user {username}: {e}")
            return None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Retrieves user details by user ID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, password_hash, email, created_at
                FROM users
                WHERE user_id = ?
            """, (user_id,))
            user_data = cursor.fetchone()
            return dict(user_data) if user_data else None
        except sqlite3.Error as e:
            print(f"Error fetching user ID {user_id}: {e}")
            return None
        finally:
            conn.close()

