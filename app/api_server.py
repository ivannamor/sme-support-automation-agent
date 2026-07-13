import sys
from pathlib import Path
import logging
import sqlite3
import time
from typing import Any, Tuple

from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "company.db"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Run standalone (python app/api_server.py) as well as via run.py — make sure
# the project root is importable either way.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FastAPI app
# ============================================================

from backend.secrets_provider import get_secret

INVENTORY_API_KEY = get_secret("INVENTORY_API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in get_secret("ALLOWED_ORIGINS", "http://localhost:8501").split(",")]

app = FastAPI(title="Door Company Inventory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)):
    if not INVENTORY_API_KEY:
        # Key not configured — allow access so local dev works out of the box.
        return
    if api_key != INVENTORY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ============================================================
# Rate limiting
# ============================================================

user_requests = {}
# All traffic to this API comes from the assistant itself (server-to-server,
# same container), so every call shares client_ip="127.0.0.1" — a 1-request/s
# limit meant to stop external abuse ends up blocking the agent's own
# back-to-back calls (e.g. a legitimate retry). Kept short only as a guard
# against a genuine runaway loop, not to throttle normal sequential tool use.
RATE_LIMIT_INTERVAL = 0.15  # seconds between requests from same IP


# ============================================================
# Logging
# ============================================================

_audit_handler = RotatingFileHandler(
    LOG_DIR / "orders_audit.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_audit_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_audit_handler, logging.StreamHandler()],
)

logger = logging.getLogger("DoorInventoryAPI")


# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Simple in-memory rate limiter for local/demo usage.
    Good enough for this project; not intended for production scaling.
    """

    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()

    last_request = user_requests.get(client_ip, 0)

    if current_time - last_request < RATE_LIMIT_INTERVAL:
        logger.warning("RATE LIMIT TRIGGERED: %s attempted rapid access.", client_ip)
        # Raising HTTPException here (inside middleware, not a route handler)
        # isn't reliably caught by FastAPI's exception handling — it surfaced
        # as an unhandled 500 instead of a clean 429. Return the response directly.
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests - Please slow down."},
        )

    user_requests[client_ip] = current_time
    return await call_next(request)


# ============================================================
# Database helpers
# ============================================================

def normalize_model(model: str) -> str:
    """
    Normalizes model codes before querying SQLite.
    """

    cleaned = model.strip().upper()

    if not cleaned:
        raise HTTPException(status_code=400, detail="Model cannot be empty.")

    return cleaned


def validate_quantity(quantity: int) -> None:
    """
    Validates inventory quantities.
    """

    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than zero.")


def query_db(query: str, params: Tuple[Any, ...] = (), commit: bool = False):
    """
    Executes a SQLite query safely against db/company.db.
    """

    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Inventory database not found at {DB_PATH}. "
                "Run backend/database.py first to initialize it."
            ),
        )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Same WAL + busy_timeout rationale as backend/database.py's
            # get_connection() — this service opens its own independent
            # connections to the same file.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            cursor = conn.cursor()
            cursor.execute(query, params)

            if commit:
                conn.commit()

            return cursor.fetchall()

    except sqlite3.Error as exc:
        logger.error("SQLITE_ERROR: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database operation failed.") from exc


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health_check():
    """
    Basic health check for app/tools.
    """

    return {
        "status": "ok",
        "database_exists": DB_PATH.exists(),
        "database_path": str(DB_PATH),
    }


@app.get("/inventory/{model}", dependencies=[Depends(require_api_key)])
def get_item(model: str):
    """
    Returns stock, price, and backorder count for a model.
    """

    model_code = normalize_model(model)

    result = query_db(
        """
        SELECT stock, price, backorder_count
        FROM inventory
        WHERE UPPER(model) = ?
        """,
        (model_code,),
    )

    if not result:
        raise HTTPException(status_code=404, detail=f"Item not found: {model_code}")

    stock, price, backorder_count = result[0]

    return {
        "model": model_code,
        "stock": stock,
        "price": price,
        "backorder": backorder_count,
    }


@app.post("/inventory/take", dependencies=[Depends(require_api_key)])
def take_item(model: str, quantity: int = 1):
    """
    Reserves/removes units from inventory.
    """

    model_code = normalize_model(model)
    validate_quantity(quantity)

    item = get_item(model_code)

    if item["stock"] < quantity:
        logger.warning(
            "FAILED TAKE: %sx %s requested. Available stock: %s.",
            quantity,
            model_code,
            item["stock"],
        )
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient stock for {model_code}. Available: {item['stock']}",
        )

    query_db(
        """
        UPDATE inventory
        SET stock = stock - ?
        WHERE UPPER(model) = ?
        """,
        (quantity, model_code),
        commit=True,
    )

    new_stock = item["stock"] - quantity

    logger.info("ORDER CONFIRMED: %sx %s removed from stock. New stock: %s.", quantity, model_code, new_stock)

    return {
        "message": f"Successfully reserved {quantity} unit(s) of {model_code}.",
        "model": model_code,
        "reserved_quantity": quantity,
        "new_stock": new_stock,
    }


@app.post("/inventory/backorder", dependencies=[Depends(require_api_key)])
def backorder_item(model: str, quantity: int = 1):
    """
    Adds requested units to backorder count.
    """

    model_code = normalize_model(model)
    validate_quantity(quantity)

    # Ensure the item exists first.
    get_item(model_code)

    query_db(
        """
        UPDATE inventory
        SET backorder_count = backorder_count + ?
        WHERE UPPER(model) = ?
        """,
        (quantity, model_code),
        commit=True,
    )

    logger.info("BACKORDER PLACED: %sx %s requested for production.", quantity, model_code)

    return {
        "message": f"Backorder for {quantity} unit(s) of {model_code} recorded successfully.",
        "model": model_code,
        "backorder_quantity": quantity,
    }


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
