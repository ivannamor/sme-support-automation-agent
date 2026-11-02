"""The shared sidebar and its catalogue PDF viewer."""

import shutil
from pathlib import Path
from urllib.parse import quote as _url_quote

import streamlit as st

from backend.database import get_config, set_config
from backend.llm_provider import DEFAULT_MODEL, OPENROUTER_MODELS
from ui.i18n import _LANGS, t
from ui.session import ROLE_LABEL, _get, _k, _set, reset_chat_session

# TODO PR 11: drop with the app/ restructure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


CATALOGUE_DIR = PROJECT_ROOT / "data" / "catalogue"
STATIC_CATALOGUE_DIR = PROJECT_ROOT / "app" / "static" / "catalogue"


def _ensure_static_pdf(pdf_path: Path) -> Path:
    STATIC_CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    static_path = STATIC_CATALOGUE_DIR / pdf_path.name
    if (
        not static_path.exists()
        or static_path.stat().st_mtime < pdf_path.stat().st_mtime
        or static_path.stat().st_size != pdf_path.stat().st_size
    ):
        shutil.copy2(pdf_path, static_path)
    return static_path


@st.dialog("Catalogue PDF", width="large")
def _catalogue_pdf_dialog(pdf_url: str, pdf_name: str) -> None:
    st.markdown(
        f'<iframe src="{pdf_url}" width="100%" height="780" style="border:none;border-radius:4px;"></iframe>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<a href="{pdf_url}" target="_blank" style="font-size:0.85em;color:#14532D;">Open in full browser tab</a>',
        unsafe_allow_html=True,
    )


def render_sidebar(role: str) -> str:
    """Renders the shared sidebar. Returns the active model ID."""
    model_ids = list(OPENROUTER_MODELS.keys())
    default_idx = model_ids.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_ids else 0

    with st.sidebar:
        st.markdown(
            f"**{ROLE_LABEL.get(role, role).upper()}**"
            f"  —  {st.session_state.get('username', 'Guest')}"
        )
        if st.button(t("logout"), use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("pages/login.py")

        # ── UX language — global key so all roles share the same choice ───────
        st.divider()
        st.selectbox(
            t("language_label"),
            _LANGS,
            index=_LANGS.index(st.session_state.get("ux_lang_select", "English")),
            key="ux_lang_select",  # intentionally NOT role-namespaced
        )

        # ── model selector (manager only) ─────────────────────────────────────
        if role == "manager":
            st.divider()
            st.caption(t("model_label").upper())
            model_options = [
                f"{info['name']} ({info['provider']})" for info in OPENROUTER_MODELS.values()
            ]
            current_model_id = get_config("current_model", DEFAULT_MODEL)
            current_model_idx = (
                model_ids.index(current_model_id) if current_model_id in model_ids else default_idx
            )
            selected_display = st.selectbox(
                t("active_llm_label"),
                model_options,
                index=current_model_idx,
                label_visibility="collapsed",
            )
            new_model_id = model_ids[model_options.index(selected_display)]

            if new_model_id != current_model_id:
                set_config("current_model", new_model_id)
                reset_chat_session()
                st.rerun()

            model_info = OPENROUTER_MODELS[new_model_id]
            st.caption(f"{model_info['provider']} · {model_info['context_window']:,} tokens")
            selected_model = new_model_id
        else:
            selected_model = get_config("current_model", DEFAULT_MODEL)

        # ── voice mode ────────────────────────────────────────────────────────
        st.divider()
        voice_on = st.toggle(
            t("voice_label"), value=_get("voice_mode", False), key=_k("voice_toggle")
        )
        _set("voice_mode", voice_on)

        # ── conversation controls ─────────────────────────────────────────────
        st.divider()
        if st.button(t("clear"), use_container_width=True):
            reset_chat_session()
            st.rerun()

        # ── catalogue PDF ─────────────────────────────────────────────────────
        st.divider()
        catalogue_pdfs = sorted(CATALOGUE_DIR.glob("*.pdf"))
        if catalogue_pdfs:
            selected_pdf = st.selectbox(
                t("catalogue_select_label"),
                catalogue_pdfs,
                format_func=lambda p: p.name,
                key=_k("catalogue_pdf_select"),
                label_visibility="collapsed",
            )
            static_pdf = _ensure_static_pdf(selected_pdf)
            pdf_url = f"/app/static/catalogue/{_url_quote(static_pdf.name)}"
            if st.button(t("view_catalogue"), use_container_width=True):
                _catalogue_pdf_dialog(pdf_url, selected_pdf.name)

    return selected_model
