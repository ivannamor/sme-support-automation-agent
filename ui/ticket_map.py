"""The map of open tickets."""

import logging
import re
import time

import pydeck as pdk
import streamlit as st

from backend.database import ticket_code
from ui.i18n import t
from ui.tickets import _TICKET_STATUS_COLOR, _TICKET_STATUS_LABEL


def _hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def render_ticket_map(tickets: list[dict]) -> None:
    """
    Plots open/in-progress tickets as colored dots on a map using pydeck (a
    ScatterplotLayer with pixel-clamped radius, so dots stay small and legible
    at any zoom level — st.map's native size channel scales in meters and
    balloons to neighborhood-sized circles when zoomed in). Color = ticket
    status (same palette as the status badge); radius = phase (installation
    jobs render slightly larger than site surveys) as a second, independent
    channel. Hovering a dot shows a native tooltip with the ticket's details.
    Closed/cancelled tickets are excluded — they're no longer operationally
    relevant to plot.

    Tickets missing geocoded coordinates (client_lat/client_lon — set once at
    request time in request_site_survey_tool; tickets that predate this
    feature are simply null) are backfilled lazily here, once, via the same
    Nominatim lookup and persisted back to the DB (update_ticket_coordinates)
    so later renders don't repeat the lookup. If the full address doesn't
    resolve, a second attempt strips the house number and retries with just
    the street/city — Nominatim's free-text search is picky about exact
    tokenization, and a terse or slightly-off house number is a common
    reason the full address alone fails.

    A ticket that still has no coordinates after that (bad/incomplete
    address, or no address on file at all) is surfaced in a warning listing
    it by ticket code instead of just silently not appearing as a dot —
    the previous behavior gave the manager no way to know a ticket was
    missing from the map at all, let alone why.
    """
    active_tickets = [t for t in tickets if t.get("ticket_status") not in ("closed", "cancelled")]

    missing = [
        t
        for t in active_tickets
        if not (t.get("client_lat") and t.get("client_lon")) and t.get("client_address")
    ]
    if missing:
        from backend.database import update_ticket_coordinates
        from tools.travel import _geocode_location

        for i, ticket in enumerate(missing):
            if i > 0:
                time.sleep(1)  # Nominatim usage policy: max ~1 request/second
            address = ticket["client_address"]
            candidates = [address]
            # Strip a trailing house number from just the first comma-separated
            # segment (the street line) — e.g. "Montenero 11, Cividale del
            # Friuli" -> "Montenero, Cividale del Friuli". Only that segment,
            # not a blind first-digit-anywhere match, so a numeric postcode or
            # province elsewhere in the string is left untouched.
            segments = [seg.strip() for seg in address.split(",")]
            if segments and re.search(r"\d", segments[0]):
                segments[0] = re.sub(r"\s*\d+[a-zA-Z]?\s*$", "", segments[0]).strip()
            stripped = ", ".join(seg for seg in segments if seg)
            if stripped and stripped != address:
                candidates.append(stripped)
            for attempt, candidate in enumerate(candidates):
                try:
                    if attempt > 0:
                        time.sleep(1)
                    lat, lon, _ = _geocode_location(candidate)
                    ticket["client_lat"], ticket["client_lon"] = lat, lon
                    update_ticket_coordinates(ticket["id"], lat, lon)
                    break
                except Exception as exc:
                    logging.getLogger("ui_shared").warning(f"MAP_BACKFILL_GEOCODE_FAILED: {exc}")

    unplaced = [tk for tk in active_tickets if not (tk.get("client_lat") and tk.get("client_lon"))]
    if unplaced:
        no_address_label = t("map_no_address")
        lines = [
            f"- {ticket_code(tk['id'])} — {tk.get('client_address') or no_address_label}"
            for tk in unplaced
        ]
        st.warning(t("map_unplaced_tickets_warning") + "\n" + "\n".join(lines))

    def _map_label(ticket: dict) -> str:
        type_label = t(f"ticket_type_{ticket.get('ticket_type', 'window')}")
        return (
            f"{ticket_code(ticket['id'])} — {ticket.get('model', '')} ({type_label})<br>"
            f"{ticket.get('client_name') or ticket.get('requested_by', '')}<br>"
            f"{_TICKET_STATUS_LABEL.get(ticket.get('ticket_status', ''), ticket.get('ticket_status', ''))}"
        )

    rows = [
        {
            "lat": ticket["client_lat"],
            "lon": ticket["client_lon"],
            "color": _hex_to_rgb(
                _TICKET_STATUS_COLOR.get(ticket.get("ticket_status", ""), "#6B7280")
            ),
            "radius": 35 if ticket.get("ticket_phase") == "installation" else 22,
            "label": _map_label(ticket),
        }
        for ticket in active_tickets
        if ticket.get("client_lat") and ticket.get("client_lon")
    ]
    if not rows:
        st.info(t("map_no_locations"))
        return

    avg_lat = sum(r["lat"] for r in rows) / len(rows)
    avg_lon = sum(r["lon"] for r in rows) / len(rows)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=rows,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=14,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=10),
        map_style="light",
        tooltip={
            "html": "<b>{label}</b>",
            "style": {
                "backgroundColor": "#161613",
                "color": "#F3F1E9",
                "fontSize": "0.8rem",
                "fontFamily": "Montserrat, sans-serif",
            },
        },
    )
    st.pydeck_chart(deck)
