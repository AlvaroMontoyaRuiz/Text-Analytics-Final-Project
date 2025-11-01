"""
Authentication module using bcrypt for password hashing and SQLite for user storage.
Works in conjunction with database.py.
"""

import bcrypt
import sqlite3 # Only needed for type hinting if database interaction was here
import uuid
from typing import Optional, Tuple
import os # Import os to handle directory existence check indirectly via ChatDatabase

# Import the database class to interact with the user table
from database import ChatDatabase 

class AuthManager:
    """Handles user authentication, registration, and password verification."""

    def __init__(self, db_path: str = "session/chat_app.db"): # Updated default path
        """Initializes the AuthManager with the path to the SQLite database."""
        # Use the ChatDatabase class which handles directory creation and connection
        self.db = ChatDatabase(db_path=db_path) 
        print("AuthManager initialized.")


    def hash_password(self, password: str) -> str:
        """Hashes a password using bcrypt."""
        try:
            salt = bcrypt.gensalt()
            hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed_bytes.decode('utf-8')
        except Exception as e:
             print(f"Error hashing password: {e}")
             raise # Re-raise error to indicate failure


    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain password against a stored bcrypt hash."""
        try:
            # Ensure hashed_password is the correct type/encoding
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except ValueError as ve:
             # Specifically catch potential "invalid salt" errors from bad hash format
             print(f"Error verifying password: Likely invalid hash format. {ve}")
             return False
        except Exception as e:
             # Catch other unexpected errors
             print(f"Error verifying password (unexpected): {e}")
             return False


    def create_user(self, username: str, password: str, email: Optional[str] = None) -> Tuple[bool, str]:
        """
        Creates a new user in the database.

        Args:
            username: The desired username (must be unique).
            password: The plain text password.
            email: Optional email address.

        Returns:
            A tuple: (success_boolean, message_or_user_id_string)
            - (True, user_id) on successful creation.
            - (False, error_message) on failure (e.g., username exists, DB error).
        """
        if not username or not password:
            return False, "Username and password cannot be empty."
            
        # Check if username already exists using the db object
        if self.db.get_user_by_username(username):
            return False, "Username already exists."

        # Generate user ID and hash password
        user_id = str(uuid.uuid4())
        try:
            password_hash = self.hash_password(password)
        except Exception as e:
             # Handle hashing error specifically
             return False, f"Failed to hash password: {e}"

        # Add user to database using the db object
        success = self.db.add_user(user_id, username, password_hash, email)
        
        if success:
            print(f"User '{username}' created successfully with ID: {user_id}")
            return True, user_id
        else:
            # The add_user method in ChatDatabase prints specific errors (like IntegrityError)
            return False, "Failed to add user to the database (check logs for details)."


    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticates a user based on username and password.

        Args:
            username: The username to authenticate.
            password: The plain text password to verify.

        Returns:
            A tuple: (success_boolean, user_id_or_None)
            - (True, user_id) if authentication succeeds.
            - (False, None) if authentication fails (user not found or wrong password).
        """
        if not username or not password:
             print("Authentication failed: Username or password empty.")
             return False, None # Cannot authenticate with empty credentials

        # Get user data from database using the db object
        user_data = self.db.get_user_by_username(username)

        if not user_data:
            print(f"Authentication failed: User '{username}' not found.")
            return False, None # User does not exist

        stored_hash = user_data.get("password_hash")
        user_id = user_data.get("user_id")

        if not stored_hash or not user_id:
             # This indicates a potential data integrity issue in the DB
             print(f"Authentication error: Missing hash or ID for user '{username}'.")
             return False, None 

        # Verify the provided password against the stored hash
        if self.verify_password(password, stored_hash):
            print(f"Authentication successful for user '{username}'.")
            return True, user_id
        else:
            print(f"Authentication failed: Incorrect password for user '{username}'.")
            return False, None


    def get_username(self, user_id: str) -> Optional[str]:
        """Gets the username associated with a given user ID."""
        if not user_id:
            return None
        
        user_data = self.db.get_user_by_id(user_id)
        
        if user_data:
            return user_data.get("username")
        else:
            print(f"Could not find username for user ID: {user_id}")
            return None

# --- Testing block removed ---

