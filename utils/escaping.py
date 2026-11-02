"""
Escaping for data shown in the UI. Streamlit renders Markdown in st.markdown,
st.caption and st.expander labels, so a client's name could carry a link or an
image that loads in the manager's browser; with unsafe_allow_html=True it could
carry raw HTML too.
"""

import html
import re

_MD_SPECIAL = re.compile(r"([\\`*_\[\]()!~#|$:])")


def md_escape(text: object) -> str:
    """For user data in Markdown: it shows literally, nothing is interpreted."""
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("<", r"\<").replace(">", r"\>")


def html_md_escape(text: object) -> str:
    """For user data in a st.markdown(..., unsafe_allow_html=True) string."""
    # "<" stays out of the backslash pass: "\&lt;" would show as a literal "&lt;".
    return html.escape(_MD_SPECIAL.sub(r"\\\1", str(text)))


def md_no_images(text: object) -> str:
    """For generated reports that keep their formatting: images never load."""
    return str(text).replace("![", r"\![")
