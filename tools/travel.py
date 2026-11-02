"""Geocoding and driving time to an installation address."""

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import ConnectionError, HTTPError, Timeout

from tools.common import (
    GEOCODING_COUNTRY_CODES,
    INSTALLER_BASE_LABEL,
    INSTALLER_BASE_LAT,
    INSTALLER_BASE_LOCATION,
    INSTALLER_BASE_LON,
    logger,
)


class TravelTimeInput(BaseModel):
    destination: str = Field(
        description="Installation destination address or place name provided by the user."
    )
    origin: str | None = Field(
        default=None,
        description=(
            "Optional starting location. If omitted, the configured installer base "
            "location is used."
        ),
    )


def _coordinates_from_env() -> tuple[float, float, str] | None:
    if not INSTALLER_BASE_LAT or not INSTALLER_BASE_LON:
        return None

    try:
        return (
            float(INSTALLER_BASE_LAT),
            float(INSTALLER_BASE_LON),
            INSTALLER_BASE_LABEL,
        )
    except ValueError:
        logger.warning("TRAVEL_TIME_CONFIG_ERROR: invalid INSTALLER_BASE_LAT/LON.")
        return None


def _geocode_location(location: str) -> tuple[float, float, str]:
    """
    Geocodes a location using OpenStreetMap Nominatim.
    """

    params = {
        "q": location,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }

    if GEOCODING_COUNTRY_CODES:
        params["countrycodes"] = GEOCODING_COUNTRY_CODES

    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=params,
        headers={"User-Agent": "advanced-rag-turing-door-assistant/1.0"},
        timeout=10,
    )
    response.raise_for_status()

    results = response.json()
    if not results:
        country_hint = (
            f" with country filter '{GEOCODING_COUNTRY_CODES}'" if GEOCODING_COUNTRY_CODES else ""
        )
        raise ValueError(f"Could not geocode location: {location}{country_hint}")

    result = results[0]
    return (
        float(result["lat"]),
        float(result["lon"]),
        result.get("display_name", location),
    )


def _standardize_address(raw_address: str) -> str:
    """
    Standardizes a client-typed address into "City, Street Number, Postcode
    (Province)" (e.g. "Cividale del Friuli, Via Parini 8, 33043, (UD)") via
    Nominatim's structured address breakdown — otherwise the client
    directory accumulates as many inconsistent spellings of the same
    address as there are ways to type it. Falls back to the original text
    if geocoding fails or doesn't return enough structured detail.
    """
    try:
        params = {"q": raw_address, "format": "json", "limit": 1, "addressdetails": 1}
        if GEOCODING_COUNTRY_CODES:
            params["countrycodes"] = GEOCODING_COUNTRY_CODES

        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "advanced-rag-turing-door-assistant/1.0"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return raw_address

        result = results[0]
        addr = result.get("address", {})
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or ""
        )
        road = addr.get("road") or ""
        house_number = addr.get("house_number") or ""
        postcode = addr.get("postcode") or ""

        iso_province = addr.get("ISO3166-2-lvl6", "")
        province = iso_province.split("-")[-1] if "-" in iso_province else addr.get("county", "")

        street_line = " ".join(part for part in [road, house_number] if part)
        parts = [part for part in [city, street_line, postcode] if part]
        if not parts:
            return raw_address

        standardized = ", ".join(parts)
        if province:
            standardized += f" ({province})"
        return standardized

    except Exception as exc:
        logger.warning(
            f"ADDRESS_STANDARDIZATION_FAILED: {exc} — using the client's original wording"
        )
        return raw_address


def _format_duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} h {remaining_minutes} min"

    return f"{remaining_minutes} min"


def _compute_travel(destination: str, origin: str | None = None) -> dict:
    """
    Deterministic core of the travel-time calculation — geocodes origin/
    destination and queries OSRM, returning plain numbers. Used both by
    calculate_travel_time_tool (which formats this into prose for the LLM)
    and by server-side tools (survey/installation pricing and duration) that
    need real floats, not a sentence to parse. Raises ValueError/ConnectionError/
    Timeout/HTTPError exactly as the geocoding/routing calls below do — callers
    are expected to catch and translate these the same way calculate_travel_time_tool does.
    """
    destination = destination.strip()
    explicit_origin = bool(origin and origin.strip())
    origin = (origin or INSTALLER_BASE_LOCATION).strip()

    if not destination:
        raise ValueError("destination is required.")
    if not origin:
        raise ValueError(
            "origin is not configured. Set INSTALLER_BASE_LOCATION in .env or provide origin explicitly."
        )

    configured_coordinates = _coordinates_from_env() if not explicit_origin else None
    if configured_coordinates:
        origin_lat, origin_lon, origin_label = configured_coordinates
    else:
        origin_lat, origin_lon, origin_label = _geocode_location(origin)

    dest_lat, dest_lon, dest_label = _geocode_location(destination)

    route_response = requests.get(
        (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        ),
        params={"overview": "false", "alternatives": "false", "steps": "false"},
        timeout=15,
    )
    route_response.raise_for_status()

    routes = route_response.json().get("routes", [])
    if not routes:
        raise ValueError("no driving route found for those locations.")

    route = routes[0]
    return {
        "duration_hours": route["duration"] / 3600,
        "distance_km": route["distance"] / 1000,
        "origin_label": origin_label,
        "dest_label": dest_label,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
    }


@tool(args_schema=TravelTimeInput)
def calculate_travel_time_tool(destination: str, origin: str | None = None) -> str:
    """
    Estimates driving travel time from the configured installer base or a
    provided origin to the installation destination provided by the user.

    Uses INSTALLER_BASE_LAT/LON from .env when available; otherwise uses
    INSTALLER_BASE_LOCATION. Geocoding can be country-biased with
    GEOCODING_COUNTRY_CODES, for example "it" or "it,fr".
    """
    try:
        travel = _compute_travel(destination, origin)
        duration = _format_duration(travel["duration_hours"] * 3600)

        logger.info(
            f"TRAVEL_TIME_SUCCESS: {travel['origin_label']} -> {travel['dest_label']} "
            f"{travel['distance_km']:.1f} km, {duration}"
        )

        return (
            f"Estimated driving travel time: {duration}. "
            f"Distance: {travel['distance_km']:.1f} km. "
            f"Origin: {travel['origin_label']}. Destination: {travel['dest_label']}. "
            f"Coordinates used: origin ({travel['origin_lat']:.5f}, {travel['origin_lon']:.5f}), "
            f"destination ({travel['dest_lat']:.5f}, {travel['dest_lon']:.5f}). "
            "Estimate provided by OpenStreetMap/Nominatim and OSRM; actual traffic is not included."
        )

    except ValueError as exc:
        logger.warning(f"TRAVEL_TIME_GEOCODE_FAILURE: {str(exc)}")
        return f"Travel Time Error: {str(exc)}"

    except (ConnectionError, Timeout):
        logger.error("TRAVEL_TIME_CONN_ERROR: routing service unavailable.")
        return "Travel Time Error: routing service is currently unavailable."

    except HTTPError as exc:
        logger.error(f"TRAVEL_TIME_HTTP_ERROR: {str(exc)}", exc_info=True)
        return "Travel Time Error: routing service returned an error."

    except Exception as exc:
        logger.error(f"TRAVEL_TIME_UNEXPECTED_ERROR: {str(exc)}", exc_info=True)
        return "Travel Time Error: an unexpected error occurred."
