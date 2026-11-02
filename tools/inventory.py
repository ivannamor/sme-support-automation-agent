"""Inventory API access and installer stock reservations."""

import time
from urllib.parse import quote

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import ConnectionError, HTTPError, Timeout

from backend.database import (
    create_pending_reservation,
    get_pending_reservations,
    reopen_reservation,
    resolve_reservation,
)
from backend.log_context import username_var
from backend.secrets_provider import get_secret
from tools.common import API_URL, logger


class InventoryInput(BaseModel):
    action: str = Field(
        description="Action to perform. Supported values: 'get', 'take', or 'backorder'."
    )
    model: str = Field(description="Hardware model code, for example 'DC420' or 'DC700G-E'.")
    quantity: int = Field(default=1, description="Quantity to reserve or backorder.")


@tool(args_schema=InventoryInput)
def inventory_api_tool(action: str, model: str, quantity: int = 1) -> str:
    """
    Manages hardware inventory via the company API.

    Use:
    - 'get' to check stock and price
    - 'take' to reserve/remove items from stock
    - 'backorder' to register a backorder request
    """

    _, message = _inventory_request(action, model, quantity)
    return message


def _format_api_error(detail) -> str:
    # FastAPI validation errors (422) carry a list of {"loc", "msg", ...} dicts.
    if isinstance(detail, list):
        return "; ".join(
            str(err.get("msg", err)) if isinstance(err, dict) else str(err) for err in detail
        )
    return str(detail)


def _inventory_request(action: str, model: str, quantity: int = 1) -> tuple[bool, str]:
    """
    Calls the inventory API and returns (ok, message), so callers such as
    approve_reservation_tool branch on success instead of string-matching
    the message text.
    """
    action = action.lower().strip()
    model = model.upper().strip()
    start_time = time.time()

    allowed_actions = {"get", "take", "backorder"}

    if action not in allowed_actions:
        return False, (
            f"Inventory Error: unsupported action '{action}'. Use 'get', 'take', or 'backorder'."
        )

    if quantity < 1:
        return False, "Inventory Error: quantity must be at least 1."

    try:
        # Read per call: run.py may inject a per-session key after import.
        headers = {"X-API-Key": get_secret("INVENTORY_API_KEY", "")}

        if action == "get":
            response = requests.get(
                f"{API_URL}/inventory/{quote(model, safe='')}",
                headers=headers,
                timeout=5,
            )
        else:
            response = requests.post(
                f"{API_URL}/inventory/{action}",
                json={"model": model, "quantity": quantity},
                headers=headers,
                timeout=5,
            )

        response.raise_for_status()

        duration = round(time.time() - start_time, 2)
        logger.info(f"INVENTORY_SUCCESS: {action} on {model} in {duration}s")

        data = response.json()

        if action == "get":
            return True, str(data)

        return True, data.get("message", "Operation successful.")

    except ConnectionError:
        logger.error("INVENTORY_CONN_ERROR: API server is down.")
        return (
            False,
            "Error: Connection to inventory system failed. Ensure api_server.py is running.",
        )

    except Timeout:
        logger.error(f"INVENTORY_TIMEOUT: Request for {model} timed out.")
        return False, "Error: Inventory system timed out."

    except HTTPError as exc:
        status_code = exc.response.status_code
        try:
            error_detail = _format_api_error(exc.response.json().get("detail", "Unknown error"))
        except Exception:
            error_detail = "API response error"

        logger.warning(f"INVENTORY_HTTP_{status_code}: {error_detail}")
        return False, f"Inventory Error: {error_detail}"

    except Exception as exc:
        logger.error(f"INVENTORY_UNEXPECTED_ERROR: {str(exc)}", exc_info=True)
        return False, "Error: An unexpected error occurred in the inventory tool."


class ReservationRequestInput(BaseModel):
    model: str = Field(description="Hardware model code to reserve, e.g. 'DC420'.")
    quantity: int = Field(default=1, description="Number of units needed.")
    reason: str = Field(
        default="",
        description="Optional reason for the reservation (job reference, site name, etc.).",
    )


@tool(args_schema=ReservationRequestInput)
def request_inventory_reservation_tool(model: str, quantity: int = 1, reason: str = "") -> str:
    """
    Submits an inventory reservation request for manager approval.

    Use this when an installer needs to reserve stock for a job.
    The request will be reviewed by a manager before inventory is actually deducted.
    Do NOT use inventory_api_tool with action='take' directly.
    """
    model = model.upper().strip()

    if quantity < 1:
        return "Reservation Request Error: quantity must be at least 1."

    try:
        reservation_id = create_pending_reservation(
            model=model,
            quantity=quantity,
            requested_by=username_var.get(),
            reason=reason,
        )
        logger.info(
            f"RESERVATION_REQUEST_CREATED: id={reservation_id} model={model} qty={quantity}"
        )
        return (
            f"Reservation request for **{quantity}× {model}** submitted (reference #{reservation_id}). "
            "A manager will review and approve it before stock is deducted."
        )
    except Exception as exc:
        logger.error(f"RESERVATION_REQUEST_FAILED: {exc}", exc_info=True)
        return "Reservation Request Error: could not save your request. Please try again."


@tool
def list_pending_reservations_tool() -> str:
    """
    Lists all pending inventory reservation requests waiting for manager approval.
    Use this before approve_reservation_tool or reject_reservation_tool to see the IDs.
    Manager-only.
    """
    pending = get_pending_reservations(status="pending")
    if not pending:
        return "No pending reservation requests."
    lines = []
    for r in pending:
        entry = f"#{r['id']} — {r['quantity']}× {r['model']} requested by {r['requested_by']}"
        if r.get("reason"):
            entry += f"\n  Reason: {r['reason']}"
        lines.append(entry)
    return "\n\n".join(lines)


class ApproveReservationInput(BaseModel):
    reservation_id: int = Field(description="ID of the pending reservation request to approve.")


@tool(args_schema=ApproveReservationInput)
def approve_reservation_tool(reservation_id: int) -> str:
    """
    Approves a pending inventory reservation request and deducts the stock
    (via the same inventory API path as a manual 'take' action).
    Always call list_pending_reservations_tool first to confirm the ID.
    Manager-only.
    """
    pending = get_pending_reservations(status="pending")
    reservation = next((r for r in pending if r["id"] == reservation_id), None)
    if not reservation:
        return f"Reservation #{reservation_id} not found among pending requests."

    # Claim first: resolve_reservation only succeeds while the row is still
    # 'pending', so of two concurrent approvals exactly one deducts stock.
    if not resolve_reservation(reservation_id, "approved", username_var.get()):
        return f"Reservation #{reservation_id} not found among pending requests."

    ok, take_result = _inventory_request("take", reservation["model"], reservation["quantity"])
    if not ok:
        reopen_reservation(reservation_id)
        return f"Could not deduct stock for reservation #{reservation_id}: {take_result}"

    logger.info(f"RESERVATION_APPROVED: id={reservation_id} by={username_var.get()}")
    return f"Reservation #{reservation_id} ({reservation['quantity']}× {reservation['model']}) approved. {take_result}"


class RejectReservationInput(BaseModel):
    reservation_id: int = Field(description="ID of the pending reservation request to reject.")


@tool(args_schema=RejectReservationInput)
def reject_reservation_tool(reservation_id: int) -> str:
    """
    Rejects a pending inventory reservation request — no stock is deducted.
    Manager-only.
    """
    updated = resolve_reservation(reservation_id, "rejected", username_var.get())
    if not updated:
        return f"Reservation #{reservation_id} not found among pending requests."
    logger.info(f"RESERVATION_REJECTED: id={reservation_id} by={username_var.get()}")
    return f"Reservation #{reservation_id} has been rejected."
