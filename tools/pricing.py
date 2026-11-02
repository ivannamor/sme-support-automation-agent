"""Ad hoc estimates, cost settings and the survey price calculation."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.database import GLASS_TYPES, VALID_COST_KEYS, get_costs, get_product, update_cost
from backend.log_context import username_var
from tools.common import logger
from tools.travel import _compute_travel


class PriceEstimateInput(BaseModel):
    model: str = Field(description="Hardware model code, e.g. 'DC420'.")
    quantity: int = Field(default=1, description="Number of units to install.")
    travel_duration_hours: float = Field(
        description="One-way travel duration in hours, from calculate_travel_time_tool."
    )
    travel_distance_km: float = Field(
        description="One-way travel distance in km, from calculate_travel_time_tool."
    )


@tool(args_schema=PriceEstimateInput)
def price_estimate_tool(
    model: str,
    quantity: int,
    travel_duration_hours: float,
    travel_distance_km: float,
) -> str:
    """
    Calculates a full price estimate for an installation job including components,
    labour, travel, fixed overhead, and VAT.

    IMPORTANT: always call calculate_travel_time_tool first and pass its
    duration (hours) and distance (km) here. Do NOT call this tool without
    travel data.
    """
    model = model.upper().strip()

    product = get_product(model)
    if not product:
        return f"Price Estimate Error: model '{model}' not found in inventory."

    costs = get_costs()
    hourly_rate = costs.get("hourly_rate", 45.0)
    cost_per_km = costs.get("travel_cost_per_km", 0.30)
    fixed_cost = costs.get("generic_install_cost", 50.0)
    vat_rate = costs.get("vat_rate", 0.22)
    install_hours = product.get("installation_hours", 1.5)
    unit_price = product.get("price", 0.0)

    component_cost = unit_price * quantity
    labour_cost = install_hours * quantity * hourly_rate
    travel_labour_cost = travel_duration_hours * hourly_rate
    travel_vehicle_cost = travel_distance_km * cost_per_km
    subtotal = component_cost + labour_cost + travel_labour_cost + travel_vehicle_cost + fixed_cost
    vat_amount = subtotal * vat_rate
    total = subtotal + vat_amount

    breakdown = (
        f"PRICE ESTIMATE — {quantity}× {model}\n\n"
        f"  Components:        {quantity} × €{unit_price:.2f}                      = €{component_cost:.2f}\n"
        f"  Installation:      {quantity} × {install_hours}h × €{hourly_rate:.2f}/h  = €{labour_cost:.2f}\n"
        f"  Travel (labour):   {travel_duration_hours:.1f}h × €{hourly_rate:.2f}/h                = €{travel_labour_cost:.2f}\n"
        f"  Travel (vehicle):  {travel_distance_km:.0f}km × €{cost_per_km:.2f}/km                = €{travel_vehicle_cost:.2f}\n"
        f"  Fixed overhead:                                          = €{fixed_cost:.2f}\n"
        f"  ─────────────────────────────────────────────────────────\n"
        f"  Subtotal:                                                = €{subtotal:.2f}\n"
        f"  VAT ({vat_rate * 100:.0f}%):                                             = €{vat_amount:.2f}\n"
        f"  ═════════════════════════════════════════════════════════\n"
        f"  TOTAL ESTIMATE:                                          €{total:.2f}\n\n"
        f"  Travel figures are estimates (no live traffic)."
    )

    logger.info(f"PRICE_ESTIMATE: model={model} qty={quantity} total={total:.2f}")
    return breakdown


class UpdateCostsInput(BaseModel):
    key: str = Field(
        description=(
            "Cost variable to update. One of: "
            "'hourly_rate' (€/hr), "
            "'travel_cost_per_km' (€/km), "
            "'generic_install_cost' (€/job), "
            "'vat_rate' (decimal, e.g. 0.22 for 22%)."
        )
    )
    value: float = Field(description="New numeric value for the cost variable.")


@tool(args_schema=UpdateCostsInput)
def update_installation_costs_tool(key: str, value: float) -> str:
    """
    Updates a cost variable used in price estimates.
    Manager-only. Changes take effect immediately on the next estimate.
    """
    if key not in VALID_COST_KEYS:
        return (
            f"Error: '{key}' is not a valid cost key. "
            f"Valid keys: {', '.join(sorted(VALID_COST_KEYS))}."
        )
    if value < 0:
        return "Error: cost value must be non-negative."

    success = update_cost(key, value)
    if not success:
        return f"Error: failed to update '{key}'."

    logger.info(f"COST_UPDATED: key={key} value={value} by={username_var.get()}")
    return f"Cost variable '{key}' updated to {value}."


# ── site-survey report review (manager) ──────────────────────────────────────

# Reference window used to scale the frame price from inventory (a flat
# per-unit price) to the real measured size — the same 1000×1500mm reference
# already used throughout the scraped catalogue's Uw/Ug spec sheets.
REFERENCE_WIDTH_M = 1.0
REFERENCE_HEIGHT_M = 1.5
REFERENCE_PERIMETER_M = 2 * (REFERENCE_WIDTH_M + REFERENCE_HEIGHT_M)


def _compute_survey_estimate(booking: dict) -> dict:
    """
    Computes the priced estimate for a surveyed ticket from its stored
    measurements/glass_type, travel to the client's address, and the current
    cost variables. Raises ValueError if the model/glass type is unknown or
    measurements are missing; propagates travel-computation exceptions.
    """
    model = booking["model"]
    product = get_product(model)
    if not product:
        raise ValueError(f"model '{model}' not found in inventory.")

    height_cm, width_cm = booking.get("height_cm"), booking.get("width_cm")
    glass_type = booking.get("glass_type")
    if not height_cm or not width_cm or not glass_type:
        raise ValueError(
            "survey measurements are incomplete — submit_survey_measurements_tool must run first."
        )

    quantity = booking.get("quantity", 1)
    costs = get_costs()
    hourly_rate = costs.get("hourly_rate", 45.0)
    cost_per_km = costs.get("travel_cost_per_km", 0.30)
    fixed_cost = costs.get("generic_install_cost", 50.0)
    vat_rate = costs.get("vat_rate", 0.22)
    glass_price = costs.get(GLASS_TYPES[glass_type], 0.0)

    travel = _compute_travel(booking.get("client_address", ""))

    perimeter_m = 2 * (height_cm / 100 + width_cm / 100)
    area_m2 = (height_cm / 100) * (width_cm / 100)

    frame_cost = (perimeter_m / REFERENCE_PERIMETER_M) * product.get("price", 0.0) * quantity
    glass_cost = area_m2 * glass_price * quantity
    labour_cost = quantity * product.get("installation_hours", 1.5) * hourly_rate
    travel_labour_cost = 2 * travel["duration_hours"] * hourly_rate
    travel_vehicle_cost = 2 * travel["distance_km"] * cost_per_km

    subtotal = (
        frame_cost
        + glass_cost
        + labour_cost
        + travel_labour_cost
        + travel_vehicle_cost
        + fixed_cost
    )
    vat_amount = subtotal * vat_rate
    total = subtotal + vat_amount

    breakdown = (
        f"PRICE ESTIMATE — {quantity}× {model} ({height_cm:.0f}×{width_cm:.0f}cm, {glass_type} glass)\n\n"
        f"  Frame ({perimeter_m:.2f}m perimeter):    = €{frame_cost:.2f}\n"
        f"  Glass ({area_m2:.2f}m² × €{glass_price:.2f}/m²): = €{glass_cost:.2f}\n"
        f"  Installation labour:                     = €{labour_cost:.2f}\n"
        f"  Travel (labour, round trip):             = €{travel_labour_cost:.2f}\n"
        f"  Travel (vehicle, round trip):             = €{travel_vehicle_cost:.2f}\n"
        f"  Fixed overhead:                           = €{fixed_cost:.2f}\n"
        f"  ─────────────────────────────────────────\n"
        f"  Subtotal:                                 = €{subtotal:.2f}\n"
        f"  VAT ({vat_rate * 100:.0f}%):                             = €{vat_amount:.2f}\n"
        f"  ═════════════════════════════════════════\n"
        f"  TOTAL ESTIMATE:                           €{total:.2f}"
    )
    return {"total": total, "breakdown": breakdown}
