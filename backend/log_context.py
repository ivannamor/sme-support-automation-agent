from contextvars import ContextVar

# Set once per request in assistant.ask() / stream_ask(); read by the log filter.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Set from ui_shared.handle_user_turn before each agent invocation so tools
# can stamp the correct username on pending_bookings / pending_reservations.
username_var: ContextVar[str] = ContextVar("username", default="unknown")
