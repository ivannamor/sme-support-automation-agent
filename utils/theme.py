"""Visual theme injection — reads assets/theme.css and applies it via st.markdown."""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THEME_CSS_PATH = PROJECT_ROOT / "assets" / "theme.css"


def inject_custom_css() -> None:
    with open(THEME_CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
