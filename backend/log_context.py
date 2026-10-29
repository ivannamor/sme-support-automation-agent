from contextvars import ContextVar

# Set once per request in DoorAssistantAgent._stream(); read by the log filter.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Set from ui_shared.handle_user_turn before each agent invocation so tools
# can stamp the correct username on pending_bookings / pending_reservations.
username_var: ContextVar[str] = ContextVar("username", default="unknown")

# Email the current client session has proven it owns (one-time code, see
# backend/client_auth.py), or "" if unverified. Set by the UI alongside
# username_var; client tools use it — never an LLM-supplied argument — to
# decide which tickets the caller may see or act on.
verified_client_email_var: ContextVar[str] = ContextVar("verified_client_email", default="")

# The session's role, set by the UI alongside username_var. LLM calls made
# outside a chat turn (dashboard buttons) are logged to usage_log under it;
# None (offline scripts, tests) means they are not logged.
role_var: ContextVar[str | None] = ContextVar("role", default=None)
