"""The installer's briefing for their active job."""

import streamlit as st

from backend.database import get_costs, get_product
from ui.i18n import t
from ui.session import _k
from utils.escaping import md_escape

_NO_INFO_MARKER = (
    "The provided documentation does not contain enough information to answer this reliably."
)


def render_installer_job_briefing(active_job: dict) -> None:
    """
    Renders inline INSIDE an active job's ticket card (called from the
    installer page's per-job loop, itself already inside a bordered
    container) — everything useful for that specific visit: an estimated
    time budget (on-site labor + best-effort travel time from the
    configured base location), whatever the client said in the original
    request, and — for installation-phase jobs only, not site surveys —
    installation guidance for that model pulled from the same
    installer-only RAG collections as search_installation_manuals_tool.
    Retrieval + LLM calls are cached per booking_id/language so they only
    run once, not on every rerun.
    """
    booking_id = active_job["id"]
    model = active_job.get("model", "")
    phase = active_job.get("ticket_phase", "installation")

    # ── time estimate — on-site hours source depends on phase: a survey
    # visit uses survey_hours_per_unit, an installation visit uses the
    # product's installation_hours. Using the wrong one would understate/
    # overstate the visit length. ─────────────────────────────────────────
    quantity = active_job.get("quantity", 1)
    if phase == "survey":
        hours_per_unit = get_costs().get("survey_hours_per_unit", 0.5)
    else:
        product = get_product(model)
        hours_per_unit = (product or {}).get("installation_hours", 1.5)
    on_site_hours = quantity * hours_per_unit
    time_parts = [f"{on_site_hours:.1f}h {t('briefing_on_site')}"]
    try:
        from tools.travel import _compute_travel

        address = active_job.get("client_address", "")
        if address:
            travel = _compute_travel(address)
            time_parts.append(
                f"{travel['duration_hours'] * 2:.1f}h {t('briefing_round_trip_travel')}"
            )
    except Exception:
        pass  # best-effort — no origin/geocoding available, just skip this part
    st.caption(f"{t('briefing_estimated_time')}: {' + '.join(time_parts)}")

    # ── info from the client's original request, and — separately — the
    # surveyor's own note from the site visit (a different installer's
    # observation, not something the client said, so it gets its own
    # label rather than being folded into "from the client's request"). ──
    if active_job.get("notes"):
        st.caption(t("briefing_client_request"))
        st.markdown(f"> {md_escape(active_job['notes'])}")
    if active_job.get("survey_notes"):
        st.caption(t("briefing_surveyor_note"))
        st.markdown(f"> {md_escape(active_job['survey_notes'])}")

    # ── installation guidance — installation-phase jobs only. A site
    # survey visit doesn't involve installing anything, so this section
    # (and its RAG/LLM cost) is skipped entirely for those. ───────────────
    if phase == "installation":
        lang = st.session_state.get("ux_lang_select", "English")
        cache_key = _k(f"briefing_{booking_id}_{model}_{lang}")
        if cache_key not in st.session_state:
            try:
                from rag.rag_engine import get_assistant_response
                from tools.knowledge import _INSTALLATION_COLLECTIONS

                # Phrased around "general procedures... like model X" rather than
                # "how do I install model X" — the installation manuals are generic
                # manufacturer procedure guides, not per-SKU documents, so a query
                # naming only the exact model tends to retrieve nothing useful.
                query = t("briefing_install_query").format(model=model)
                answer, _ = get_assistant_response(
                    query, allowed_collections=_INSTALLATION_COLLECTIONS
                )
                # If the model has no dedicated instructions but the answer still
                # goes on to give general guidance that does apply, swap the blunt
                # "not enough information" opener for an explicit, friendlier one
                # instead of leading with what sounds like a dead end.
                if answer.startswith(_NO_INFO_MARKER) and len(answer) > len(_NO_INFO_MARKER) + 20:
                    answer = (
                        t("briefing_no_specific_guidance").format(model=model)
                        + answer[len(_NO_INFO_MARKER) :]
                    )
                st.session_state[cache_key] = answer
            except Exception as exc:
                st.session_state[cache_key] = t("briefing_guidance_unavailable").format(error=exc)

        with st.expander(t("briefing_installation_guidance").format(model=md_escape(model))):
            st.markdown(st.session_state[cache_key])
