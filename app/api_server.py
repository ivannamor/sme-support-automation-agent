import sys
from pathlib import Path
import hmac
import logging
import sqlite3

from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Path as PathParam, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Run standalone (python app/api_server.py) as well as via run.py — make sure
# the project root is importable either way.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import database
from backend.secrets_provider import get_secret


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

# Attached to this service's own logger rather than the root logger, so
# importing the module (e.g. from tests) doesn't reconfigure everyone else's logging.
logger = logging.getLogger("DoorInventoryAPI")
logger.setLevel(logging.INFO)
logger.addHandler(_audit_handler)


# ============================================================
# FastAPI app
# ============================================================

# Only the assistant's own tools call this API (server-to-server), so there is
# no CORS middleware: no browser origin is ever meant to reach it.
app = FastAPI(title="Door Company Inventory API")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_api_key() -> str:
    # Read per request, not at import, so the key can't be silently baked in
    # as "" by an import that happened before the environment was set.
    return get_secret("INVENTORY_API_KEY", "")


async def require_api_key(api_key: str = Security(_api_key_header)):
    """
    Fails closed: with no key configured, every protected endpoint refuses
    (503) instead of quietly running unauthenticated. run.py generates a
    per-run key when none is set, so local dev still works with zero setup.
    """
    expected = _configured_api_key()
    if not expected:
        logger.error("AUTH_MISCONFIGURED: INVENTORY_API_KEY is not set; refusing request.")
        raise HTTPException(status_code=503, detail="Inventory API is not configured.")
    if not api_key or not hmac.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ============================================================
# Schemas
# ============================================================

# Catalogue model codes are short names such as "IRIS" or "MAGIS 40".
MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$"
MAX_QUANTITY = 1000


class StockChange(BaseModel):
    model: str = Field(pattern=MODEL_PATTERN, description="Catalogue model code, e.g. 'IRIS'.")
    quantity: int = Field(default=1, ge=1, le=MAX_QUANTITY)


class InventoryItem(BaseModel):
    model: str
    stock: int
    price: float
    backorder: int


class TakeResult(BaseModel):
    message: str
    model: str
    reserved_quantity: int
    new_stock: int


class BackorderResult(BaseModel):
    message: str
    model: str
    backorder_quantity: int
    backorder_total: int


# ============================================================
# Database helpers
# ============================================================

def normalize_model(model: str) -> str:
    return model.strip().upper()


def run_query(query: str, params: tuple = (), commit: bool = False) -> list[sqlite3.Row]:
    """
    Runs one statement on the shared operational DB (backend.database owns the
    path and connection settings). Rows are fetched before committing so
    UPDATE ... RETURNING results are available.
    """
    conn = None
    try:
        conn = database.get_connection()
        rows = conn.execute(query, params).fetchall()
        if commit:
            conn.commit()
        return rows
    except sqlite3.Error as exc:
        logger.error("SQLITE_ERROR: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database operation failed.") from exc
    finally:
        if conn is not None:
            conn.close()


def _require_item(model_code: str) -> sqlite3.Row:
    rows = run_query(
        "SELECT model, stock, price, backorder_count FROM inventory WHERE UPPER(model) = ?",
        (model_code,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Item not found: {model_code}")
    return rows[0]


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health_check():
    """
    Liveness + DB check. Deliberately says nothing about paths or config.
    """
    try:
        run_query("SELECT 1 FROM inventory LIMIT 1")
    except HTTPException:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}


@app.get("/inventory/{model}", response_model=InventoryItem, dependencies=[Depends(require_api_key)])
def get_item(model: str = PathParam(pattern=MODEL_PATTERN)):
    """
    Returns stock, price, and backorder count for a model.
    """
    model_code = normalize_model(model)
    row = _require_item(model_code)
    return InventoryItem(
        model=model_code,
        stock=row["stock"],
        price=row["price"],
        backorder=row["backorder_count"],
    )


@app.post("/inventory/take", response_model=TakeResult, dependencies=[Depends(require_api_key)])
def take_item(body: StockChange):
    """
    Reserves/removes units from inventory.

    The stock check and the decrement are one atomic UPDATE: a separate
    SELECT-then-UPDATE would let two concurrent takes both pass the check
    and oversell the remaining stock.
    """
    model_code = normalize_model(body.model)
    rows = run_query(
        """
        UPDATE inventory
        SET stock = stock - ?
        WHERE UPPER(model) = ? AND stock >= ?
        RETURNING stock
        """,
        (body.quantity, model_code, body.quantity),
        commit=True,
    )

    if not rows:
        item = _require_item(model_code)  # 404 if the model doesn't exist
        logger.warning(
            "FAILED TAKE: %sx %s requested. Available stock: %s.",
            body.quantity, model_code, item["stock"],
        )
        raise HTTPException(
            status_code=409,
            detail=f"Insufficient stock for {model_code}. Available: {item['stock']}",
        )

    new_stock = rows[0]["stock"]
    logger.info("ORDER CONFIRMED: %sx %s removed from stock. New stock: %s.", body.quantity, model_code, new_stock)

    return TakeResult(
        message=f"Successfully reserved {body.quantity} unit(s) of {model_code}.",
        model=model_code,
        reserved_quantity=body.quantity,
        new_stock=new_stock,
    )


@app.post("/inventory/backorder", response_model=BackorderResult, dependencies=[Depends(require_api_key)])
def backorder_item(body: StockChange):
    """
    Adds requested units to backorder count.
    """
    model_code = normalize_model(body.model)
    rows = run_query(
        """
        UPDATE inventory
        SET backorder_count = backorder_count + ?
        WHERE UPPER(model) = ?
        RETURNING backorder_count
        """,
        (body.quantity, model_code),
        commit=True,
    )

    if not rows:
        raise HTTPException(status_code=404, detail=f"Item not found: {model_code}")

    logger.info("BACKORDER PLACED: %sx %s requested for production.", body.quantity, model_code)

    return BackorderResult(
        message=f"Backorder for {body.quantity} unit(s) of {model_code} recorded successfully.",
        model=model_code,
        backorder_quantity=body.quantity,
        backorder_total=rows[0]["backorder_count"],
    )


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    import uvicorn

    if not _configured_api_key():
        sys.exit(
            "INVENTORY_API_KEY is not set. Set it in .env, or start everything "
            "with run.py, which generates a key for the session."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # Loopback by default: only the assistant on the same host calls this API.
    # Set API_HOST=0.0.0.0 only when it runs in its own container.
    uvicorn.run(app, host=get_secret("API_HOST", "127.0.0.1"), port=8000)
