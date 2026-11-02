"""
Authentication helpers for the role-based access system.

Clients need no account — they select the Client role directly.
Installers and managers authenticate with username + bcrypt password.

Failed logins are counted in the database, per username (reloading the page
must not reset them, and a client IP isn't reliable behind a proxy). Unknown
usernames are counted and hashed the same way, so neither the reply nor its
timing reveals which accounts exist. The trade-off: anyone can lock an account
for LOGIN_LOCKOUT_SECONDS by failing on purpose.
"""

import logging
import time
from dataclasses import dataclass
from functools import cache

import bcrypt

from backend.database import (
    clear_login_attempts,
    create_user,
    get_user,
    register_login_attempt,
    update_user_password,
)

logger = logging.getLogger("Auth")

MAX_FAILED_LOGINS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60
MIN_PASSWORD_LENGTH = 12
BCRYPT_MAX_BYTES = 72  # bcrypt rejects longer passwords
# Accounts created by older versions of scripts/create_users.py.
KNOWN_DEFAULT_PASSWORDS = {"change-me"}


@dataclass
class LoginResult:
    # "ok" | "invalid" | "locked" | "default_password"
    status: str
    role: str | None = None  # only set when status == "ok"


def _now() -> float:
    return time.time()


def _attempt_key(username: str) -> str:
    return username.strip().casefold()[:255]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


@cache
def _dummy_hash() -> str:
    return hash_password("no account has this password")


def authenticate(username: str, password: str) -> LoginResult:
    """
    Checks the credentials. The attempt is counted before the password check,
    so parallel requests can't slip past the limit.
    """
    key = _attempt_key(username)
    failures = register_login_attempt(key, _now(), LOGIN_LOCKOUT_SECONDS, MAX_FAILED_LOGINS)
    if failures > MAX_FAILED_LOGINS:
        logger.warning("LOGIN_LOCKED")
        return LoginResult("locked")

    user = get_user(username)
    matches = verify_password(password, user["password_hash"] if user else _dummy_hash())
    if not user or not matches:
        logger.info("LOGIN_FAILED")
        return LoginResult("invalid")

    clear_login_attempts(key)
    if password in KNOWN_DEFAULT_PASSWORDS:
        logger.warning("LOGIN_REFUSED_DEFAULT_PASSWORD")
        return LoginResult("default_password")
    return LoginResult("ok", role=user["role"])


def _validate_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"The password must have at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode()) > BCRYPT_MAX_BYTES:
        raise ValueError(f"The password must fit in {BCRYPT_MAX_BYTES} bytes.")


def register_user(username: str, password: str, role: str) -> bool:
    """
    Creates a new user account. Returns False if username already exists.
    Only 'installer' and 'manager' roles are allowed.
    """
    if role not in ("installer", "manager"):
        raise ValueError(f"Invalid role: {role}. Must be 'installer' or 'manager'.")
    _validate_new_password(password)
    return create_user(username, hash_password(password), role)


def set_password(username: str, password: str) -> bool:
    """Changes an account's password and lifts any lockout. False if it doesn't exist."""
    _validate_new_password(password)
    if not update_user_password(username, hash_password(password)):
        return False
    clear_login_attempts(_attempt_key(username))
    return True
