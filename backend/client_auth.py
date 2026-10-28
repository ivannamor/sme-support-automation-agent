"""
Email verification for client sessions (one-time code).

Clients have no account — anyone can browse products or request a site
survey anonymously. Only actions on EXISTING tickets (seeing them, accepting
or rejecting an estimate, reading installation reports) require proving the
client owns the ticket's email: request_code() emails a 6-digit code,
verify_code() checks it, and the UI then stores the verified email for that
session (see verified_client_email_var in backend/log_context.py).

Verification is deliberately UI-only — there is no chat tool for it, so the
LLM never takes part in authentication and prompt injection can't fake it.

Demo mode: when SMTP isn't configured AND DEMO_MODE=true, the code is
returned to the caller so the UI can show it on screen. Off by default.
"""

import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from backend.company_config import COMPANY_CONTEXT
from backend.database import get_connection
from backend.email_utils import is_email_configured, send_email
from backend.secrets_provider import get_secret

logger = logging.getLogger("ClientAuth")

CODE_TTL_SECONDS = 10 * 60
RESEND_COOLDOWN_SECONDS = 60
MAX_ATTEMPTS = 5

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now() -> float:
    return time.time()


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(email)))


def is_demo_mode() -> bool:
    return get_secret("DEMO_MODE", "false").strip().lower() == "true"


def _hash_code(email: str, code: str) -> str:
    # Bound to the email so a hash can't be replayed for another address.
    return hashlib.sha256(f"{email}:{code}".encode("utf-8")).hexdigest()


@dataclass
class CodeRequestResult:
    # "sent" | "demo" | "cooldown" | "invalid_email" | "email_unavailable"
    status: str
    demo_code: Optional[str] = None  # only set when status == "demo"


def request_code(email: str) -> CodeRequestResult:
    """Creates a fresh one-time code for `email` and delivers it."""
    email = normalize_email(email)
    if not is_valid_email(email):
        return CodeRequestResult("invalid_email")

    deliver_by_email = is_email_configured()
    if not deliver_by_email and not is_demo_mode():
        logger.warning("CLIENT_CODE_UNAVAILABLE: SMTP not configured and DEMO_MODE off")
        return CodeRequestResult("email_unavailable")

    now = _now()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT created_at FROM client_verification_codes WHERE email = ?", (email,)
        ).fetchone()
        if row and now - row["created_at"] < RESEND_COOLDOWN_SECONDS:
            return CodeRequestResult("cooldown")

        code = f"{secrets.randbelow(10**6):06d}"
        conn.execute(
            """
            INSERT INTO client_verification_codes (email, code_hash, expires_at, attempts, created_at)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(email) DO UPDATE SET
                code_hash = excluded.code_hash, expires_at = excluded.expires_at,
                attempts = 0, created_at = excluded.created_at
            """,
            (email, _hash_code(email, code), now + CODE_TTL_SECONDS, now),
        )
        conn.commit()
    finally:
        conn.close()

    if not deliver_by_email:
        logger.info("CLIENT_CODE_ISSUED_DEMO")
        return CodeRequestResult("demo", demo_code=code)

    sent = send_email(
        to=email,
        subject=f"Your {COMPANY_CONTEXT['company_name']} verification code",
        body=(
            f"Your verification code is: {code}\n\n"
            f"It expires in {CODE_TTL_SECONDS // 60} minutes. "
            "If you didn't request it, you can ignore this email."
        ),
    )
    if not sent:
        _delete_code(email)
        return CodeRequestResult("email_unavailable")
    logger.info("CLIENT_CODE_SENT")
    return CodeRequestResult("sent")


def verify_code(email: str, code: str) -> bool:
    """
    True if `code` is the current, unexpired code for `email`. A code works
    once; after MAX_ATTEMPTS wrong guesses it's dead and a new one is needed.
    """
    email = normalize_email(email)
    code = (code or "").strip()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT code_hash, expires_at, attempts FROM client_verification_codes WHERE email = ?",
            (email,),
        ).fetchone()
        if not row or _now() > row["expires_at"] or row["attempts"] >= MAX_ATTEMPTS:
            return False

        if hmac.compare_digest(row["code_hash"], _hash_code(email, code)):
            conn.execute("DELETE FROM client_verification_codes WHERE email = ?", (email,))
            conn.commit()
            logger.info("CLIENT_EMAIL_VERIFIED")
            return True

        conn.execute(
            "UPDATE client_verification_codes SET attempts = attempts + 1 WHERE email = ?", (email,)
        )
        conn.commit()
        logger.warning("CLIENT_CODE_MISMATCH")
        return False
    finally:
        conn.close()


def _delete_code(email: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM client_verification_codes WHERE email = ?", (email,))
        conn.commit()
    finally:
        conn.close()
