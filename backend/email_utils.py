"""
Minimal SMTP email utility.

Configured via environment variables (see backend/secrets_provider.py —
locally via .env, in Docker/hosted environments via your deployment
platform or secret manager):
    SMTP_HOST     e.g. smtp.gmail.com
    SMTP_PORT     e.g. 587
    SMTP_USERNAME sender address
    SMTP_PASSWORD app password / SMTP credential
    MANAGER_EMAIL where manager alerts are delivered
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from backend.secrets_provider import get_secret

logger = logging.getLogger("EmailUtils")

SMTP_HOST     = get_secret("SMTP_HOST", "")
SMTP_PORT     = int(get_secret("SMTP_PORT", "587"))
SMTP_USERNAME = get_secret("SMTP_USERNAME", "")
SMTP_PASSWORD = get_secret("SMTP_PASSWORD", "")
MANAGER_EMAIL = get_secret("MANAGER_EMAIL", "")


def is_email_configured() -> bool:
    return all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD])


def _mask_email(address: str) -> str:
    """Masks a recipient address for logging — e.g. 'j***@example.com'."""
    local, _, domain = address.partition("@")
    if not domain or not local:
        return "***"
    return f"{local[0]}***@{domain}"


def send_email(to: str, subject: str, body: str) -> bool:
    """
    Sends a plain-text email via SMTP.
    Returns True on success, False if not configured or on error.
    """
    if not is_email_configured():
        logger.warning("EMAIL_NOT_CONFIGURED: SMTP credentials missing in .env")
        return False
    if not to:
        logger.warning("EMAIL_NO_RECIPIENT: 'to' address is empty")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USERNAME
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"EMAIL_SENT: to={_mask_email(to)} subject='{subject}'")
        return True

    except Exception as exc:
        logger.error(f"EMAIL_FAILED: {exc}", exc_info=True)
        return False


def send_manager_alert(subject: str, body: str) -> bool:
    """Convenience wrapper — sends to the configured MANAGER_EMAIL."""
    return send_email(MANAGER_EMAIL, subject, body)
