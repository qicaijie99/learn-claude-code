#!/usr/bin/env python3
"""
login.py — Login/Authentication Module

Implements a simple but secure user login feature with:
- Password hashing (SHA-256 + salt)
- JWT-like token generation
- User registration & authentication
- Session management

Usage:
    from login import AuthManager

    auth = AuthManager()
    auth.register_user("alice", "secure_password123")
    token = auth.login("alice", "secure_password123")
    if token:
        print("Login successful!")
    else:
        print("Login failed")
"""

import hashlib
import json
import os
import time
import secrets
from pathlib import Path
from datetime import datetime

# ── Configuration ──

USERS_FILE = Path(__file__).parent / ".users.json"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour


# ── Data Persistence ──

def _load_users() -> dict:
    """Load users from the JSON file."""
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict):
    """Save users to the JSON file."""
    USERS_FILE.write_text(json.dumps(users, indent=2))


# ── Password Utilities ──

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """
    Hash a password with a salt using SHA-256.
    Returns (hashed_password, salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)  # 16 bytes = 32 hex chars
    # Combine password and salt, then hash
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash with its salt."""
    computed_hash, _ = _hash_password(password, salt)
    return computed_hash == stored_hash


# ── Token Utilities ──

def _generate_token(username: str) -> str:
    """
    Generate a simple token containing:
    - username
    - timestamp (seconds since epoch)
    - random nonce
    - HMAC-like signature

    Format: base64_json.signature
    """
    timestamp = int(time.time())
    nonce = secrets.token_hex(8)

    payload = {
        "username": username,
        "ts": timestamp,
        "nonce": nonce,
    }

    # Encode payload as base64-like hex string
    payload_str = json.dumps(payload, separators=(",", ":"))
    payload_encoded = payload_str.encode().hex()

    # Create signature using a server secret
    server_secret = _get_server_secret()
    signature = hashlib.sha256(
        (payload_str + server_secret).encode()
    ).hexdigest()[:16]

    return f"{payload_encoded}.{signature}"


def _decode_token(token: str) -> dict | None:
    """
    Decode and verify a token. Returns payload dict if valid, None otherwise.
    """
    try:
        payload_encoded, signature = token.split(".", 1)

        payload_str = bytes.fromhex(payload_encoded).decode()
        payload = json.loads(payload_str)

        # Verify signature
        server_secret = _get_server_secret()
        expected_sig = hashlib.sha256(
            (payload_str + server_secret).encode()
        ).hexdigest()[:16]

        if signature != expected_sig:
            return None

        # Check expiry
        if time.time() - payload["ts"] > TOKEN_EXPIRY_SECONDS:
            return None

        return payload
    except (ValueError, json.JSONDecodeError, KeyError, OSError):
        return None


def _get_server_secret() -> str:
    """Get or generate a server secret for token signing."""
    secret_file = Path(__file__).parent / ".server_secret"
    if secret_file.exists():
        return secret_file.read_text().strip()

    secret = secrets.token_hex(32)
    secret_file.write_text(secret)
    return secret


# ── Auth Manager ──

class AuthManager:
    """Manages user registration, login, and session validation."""

    def __init__(self):
        self.users = _load_users()

    def register_user(self, username: str, password: str) -> dict:
        """
        Register a new user.
        Returns a dict with success status and message.
        """
        if not username or not password:
            return {"success": False, "message": "Username and password required."}

        if len(password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters."}

        if username in self.users:
            return {"success": False, "message": f"User '{username}' already exists."}

        hashed, salt = _hash_password(password)
        self.users[username] = {
            "hash": hashed,
            "salt": salt,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "role": "user",
        }
        _save_users(self.users)
        return {"success": True, "message": f"User '{username}' registered successfully."}

    def login(self, username: str, password: str) -> dict:
        """
        Authenticate a user and return a token.
        Returns dict with success status, token, and message.
        """
        if username not in self.users:
            return {"success": False, "message": "Invalid username or password."}

        user = self.users[username]
        if not _verify_password(password, user["hash"], user["salt"]):
            return {"success": False, "message": "Invalid username or password."}

        # Update last login
        user["last_login"] = datetime.utcnow().isoformat()
        _save_users(self.users)

        token = _generate_token(username)
        return {"success": True, "token": token, "message": "Login successful."}

    def validate_token(self, token: str) -> dict:
        """
        Validate a token and return user info if valid.
        Returns dict with valid status and user info.
        """
        payload = _decode_token(token)
        if payload is None:
            return {"valid": False, "message": "Invalid or expired token."}

        username = payload["username"]
        if username not in self.users:
            return {"valid": False, "message": "User no longer exists."}

        return {
            "valid": True,
            "username": username,
            "role": self.users[username].get("role", "user"),
            "token_issued_at": datetime.utcfromtimestamp(payload["ts"]).isoformat(),
        }

    def list_users(self) -> list[str]:
        """List registered usernames (safe, no passwords)."""
        return list(self.users.keys())

    def delete_user(self, username: str) -> dict:
        """Delete a user account."""
        if username not in self.users:
            return {"success": False, "message": f"User '{username}' not found."}
        del self.users[username]
        _save_users(self.users)
        return {"success": True, "message": f"User '{username}' deleted."}


# ── CLI / Demo ──

def main():
    """Simple command-line login demo."""
    auth = AuthManager()
    print("=== Login System Demo ===")
    print("Commands: register <user> <pass>, login <user> <pass>, validate <token>, users, quit")

    while True:
        try:
            cmd = input("\n> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd[0] == "quit":
            break
        elif cmd[0] == "register" and len(cmd) == 3:
            result = auth.register_user(cmd[1], cmd[2])
            print(result["message"])
        elif cmd[0] == "login" and len(cmd) == 3:
            result = auth.login(cmd[1], cmd[2])
            print(result["message"])
            if result.get("token"):
                print(f"  Token: {result['token'][:50]}...")
        elif cmd[0] == "validate" and len(cmd) == 2:
            result = auth.validate_token(cmd[1])
            if result["valid"]:
                print(f"  Valid! User: {result['username']}, Role: {result['role']}")
            else:
                print(f"  Invalid: {result['message']}")
        elif cmd[0] == "users":
            users = auth.list_users()
            print(f"  Registered users: {users}")
        else:
            print("  Unknown command. Try: register, login, validate, users, quit")


if __name__ == "__main__":
    main()
