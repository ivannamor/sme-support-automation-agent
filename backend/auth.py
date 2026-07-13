"""
Authentication helpers for the role-based access system.

Clients need no account — they select the Client role directly.
Installers and managers authenticate with username + bcrypt password.
"""

import bcrypt
from typing import Optional
from backend.database import get_user, create_user


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def authenticate(username: str, password: str) -> Optional[str]:
    """
    Returns the user's role string ('installer' or 'manager') on success,
    or None if credentials are invalid.
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user["role"]


def register_user(username: str, password: str, role: str) -> bool:
    """
    Creates a new user account. Returns False if username already exists.
    Only 'installer' and 'manager' roles are allowed.
    """
    if role not in ("installer", "manager"):
        raise ValueError(f"Invalid role: {role}. Must be 'installer' or 'manager'.")
    return create_user(username, hash_password(password), role)
