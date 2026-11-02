"""Ticket cards, status badges and summary tiles."""

import streamlit as st

from backend.database import get_product, get_ticket_log, ticket_code
from ui.i18n import t
from utils.escaping import html_md_escape, md_escape

_TICKET_STATUS_LABEL = {
    "new": "NEW",
    "open": "OPEN",
    "in_progress": "IN PROGRESS",
    "awaiting_client": "AWAITING CLIENT",
    "blocked": "BLOCKED",
    "in_review": "IN REVIEW",
    "closed": "CLOSED",
    "cancelled": "CANCELLED",
}
_TICKET_STATUS_COLOR = {
    "new": "#2563EB",
    "open": "#6B7280",
    "in_progress": "#D97706",
    "awaiting_client": "#0891B2",
    "blocked": "#DC2626",
    "in_review": "#7C3AED",
    "closed": "#16A34A",
    "cancelled": "#57534E",
}


def _status_badge(ticket_status: str) -> str:
    # Border + colored text, transparent fill — status colors stay distinct
    # per-status (functional signal, unchanged), only the *treatment* moves
    # from a solid loud fill to the flat/outline language used everywhere else.
    label = (
        t(f"status_{ticket_status}")
        if ticket_status in _TICKET_STATUS_LABEL
        else html_md_escape(ticket_status.upper())
    )
    color = _TICKET_STATUS_COLOR.get(ticket_status, "#6B7280")
    return (
        f'<span style="border:1px solid {color};color:{color};padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _phase_badge(ticket_phase: str) -> str:
    label = (
        t(f"phase_{ticket_phase}")
        if ticket_phase in ("survey", "installation")
        else html_md_escape(ticket_phase.upper())
    )
    return (
        f'<span style="border:1px solid var(--green,#1E4638);color:var(--green,#1E4638);padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _type_badge(ticket_type: str) -> str:
    label = (
        t(f"ticket_type_{ticket_type}")
        if ticket_type in ("door", "window")
        else html_md_escape(ticket_type.upper())
    )
    return (
        f'<span style="border:1px solid var(--amber,#A8712E);color:var(--amber,#A8712E);padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _stock_dot(stock: int) -> str:
    if stock > 5:
        color, label = "#1E4638", str(stock)
    elif stock > 0:
        color, label = "#A8712E", str(stock)
    else:
        color, label = "#B3261E", "0"
    return (
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{color};margin-right:4px;vertical-align:middle;"></span>'
        f'<span style="font-size:0.8rem;">{label} in stock</span>'
    )


def render_ticket_card(ticket: dict, *, show_stock: bool = True) -> None:
    """
    Renders the common ticket card body — code, model, status badge, client
    info, address, notes, and assigned installer. Used by both the manager
    board and the installer job panels so the two views stay in sync.
    Manager view additionally shows stock availability (show_stock=True);
    installers don't see inventory, so their view omits it.
    """
    ts = ticket.get("ticket_status", "new")
    phase = ticket.get("ticket_phase", "installation")
    qty = ticket.get("quantity", 1)
    name = ticket.get("client_name") or ticket.get("requested_by", "—")

    # Top accent stripe in the status color — quick visual scan without reading the badge.
    # (No negative margins: stays safely inside the card regardless of the
    # exact padding Streamlit's bordered container applies.)
    status_color = _TICKET_STATUS_COLOR.get(ts, "#6B7280")
    st.markdown(
        f'<div style="height:3px;background:{status_color};border-radius:2px;margin-bottom:0.6rem;"></div>',
        unsafe_allow_html=True,
    )

    ticket_type = ticket.get("ticket_type", "window")
    st.markdown(
        f'<span class="mono-accent">{ticket_code(ticket["id"])}</span> — {qty}× **{html_md_escape(ticket["model"])}** '
        f"{html_md_escape(ticket['date_str'])} {html_md_escape(ticket['time_str'])} &nbsp; "
        f"{_type_badge(ticket_type)} {_phase_badge(phase)} {_status_badge(ts)}",
        unsafe_allow_html=True,
    )

    client_line = f"**Client:** {md_escape(name)}"
    email = ticket.get("client_email", "")
    phone = ticket.get("client_phone", "")
    if email:
        client_line += f"  |  {md_escape(email)}"
    if phone:
        client_line += f"  |  {md_escape(phone)}"
    st.markdown(client_line)

    address = ticket.get("client_address", "")
    if address:
        st.caption(t("address_inline").format(address=md_escape(address)))

    if show_stock:
        product = get_product(ticket["model"])
        stock = product["stock"] if product else None
        if stock is not None:
            st.markdown(_stock_dot(stock), unsafe_allow_html=True)
        else:
            st.caption(t("stock_not_found"))

    if ticket.get("height_cm") and ticket.get("width_cm"):
        measurements = (
            f'<span class="mono-accent" style="color:#3F3F3F;font-size:0.8rem;">'
            f"{ticket['height_cm']:.0f}×{ticket['width_cm']:.0f}×{ticket.get('depth_cm') or 0:.0f}cm</span>"
        )
        if ticket.get("glass_type"):
            measurements += f' <span style="font-size:0.8rem;color:#3F3F3F;">· {t("glass_type_caption")} {html_md_escape(ticket["glass_type"])}</span>'
        if ticket.get("estimated_price"):
            measurements += (
                f' <span class="mono-accent" style="font-size:0.8rem;color:#3F3F3F;">'
                f"· €{ticket['estimated_price']:.2f}</span>"
            )
        st.markdown(measurements, unsafe_allow_html=True)
    if ticket.get("notes"):
        st.caption(t("notes_inline").format(notes=md_escape(ticket["notes"])))
    if ticket.get("assigned_to"):
        st.caption(
            t("installer_inline").format(
                installer=md_escape(ticket["assigned_to"]),
                date=(ticket.get("started_at") or "")[:16],
            )
        )

    log_entries = get_ticket_log(ticket["id"])
    if log_entries:
        with st.expander(t("ticket_log_label")):
            for entry in log_entries:
                line = (
                    f"`{entry['created_at'][:16]}` — **{entry['ticket_status'].replace('_', ' ').upper()}** "
                    f"by {md_escape(entry['changed_by'])}"
                )
                st.markdown(line)
                if entry.get("note"):
                    st.caption(md_escape(entry["note"]))


def render_stat_tiles(tiles: list[dict]) -> None:
    """
    Renders a row of KPI stat tiles: value in Montserrat 700, label in
    uppercase tracked Montserrat 600 — flat, hairline-bordered, no shadow,
    matching the rest of the app's card language (assets/theme.css).
    Each tile is {"label": str, "value": str}.
    """
    cards = "".join(
        f'<div style="flex:1;min-width:120px;background:#EBEAE3;border:1px solid #C7C6BE;'
        f'border-radius:4px;padding:14px 16px;">'
        f"<div style=\"font-family:'Montserrat',sans-serif;font-size:0.7rem;color:#4E4D47;"
        f'text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">{tile["label"]}</div>'
        f"<div style=\"font-family:'Montserrat',sans-serif;font-weight:700;font-size:1.5rem;color:#161613;\">{tile['value']}</div>"
        f"</div>"
        for tile in tiles
    )
    st.markdown(
        f'<div style="display:flex;gap:12px;margin:0.25rem 0 1rem 0;flex-wrap:wrap;">{cards}</div>',
        unsafe_allow_html=True,
    )


def render_ticket_summary(tickets: list[dict]) -> None:
    """
    Compact stat-tile row of ticket counts by status — same visual language as
    render_stat_tiles. Used identically on the manager dashboard (all tickets)
    and the installer dashboard (only tickets assigned to that installer).
    """
    counts: dict = {}
    for tk in tickets:
        key = tk.get("ticket_status", "new")
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        st.caption(t("no_tickets_yet"))
        return
    tiles = [
        {
            "label": (
                t(f"status_{status}") if status in _TICKET_STATUS_LABEL else status.upper()
            ).upper(),
            "value": str(count),
        }
        for status, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    render_stat_tiles(tiles)
