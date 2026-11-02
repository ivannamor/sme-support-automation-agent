"""Global CSS, the page header and the draggable column handle."""

import streamlit as st
import streamlit.components.v1 as components

from backend.company_config import COMPANY_CONTEXT

# All visual styling lives in assets/theme.css (loaded via utils/theme.py) —
# a pure-presentation stylesheet kept separate from this module's own logic.


def inject_global_css() -> None:
    from utils.theme import inject_custom_css

    inject_custom_css()


def render_page_header(title: str) -> None:
    """
    Renders a two-tier page title, sticky (stays visible regardless of scroll
    position): the company name as the bold primary line, and the page title
    (e.g. "Manager Portal") as a small tracked-uppercase subtitle beneath —
    the same lockup structure as the two-line wordmark on the real Marinig site.
    """
    # Deliberately a <div>, not an <h1> — Streamlit's own markdown renderer
    # post-processes h1-h6 tags (even raw ones from unsafe_allow_html) to add
    # anchor-link icons, and that pass drops the original `style` attribute,
    # which is what caused the low-contrast dark-on-dark-green text even with
    # an inline color. A div is untouched by that pass, so .header-title's
    # color rule in assets/theme.css applies reliably.
    st.markdown(
        f'<div class="page-header-sticky">'
        f'<div class="header-title">{COMPANY_CONTEXT["company_name"]}</div>'
        f'<div class="header-subtitle">{title}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_column_resize_handle() -> None:
    """
    Injects a thin draggable handle between the chat column and dashboard
    column (manager/installer pages) so their widths can be adjusted by hand.

    st.columns() has no native resize support, so this reaches into the
    parent document from a zero-height st.components.v1.html iframe (the
    standard trick for DOM tweaks Streamlit itself doesn't expose — same
    origin, so window.parent.document is reachable) and manipulates the two
    [data-testid="stColumn"] elements' flex-basis directly via plain JS.
    Set via style.setProperty(..., 'important') rather than plain style.flex
    assignment — assets/theme.css pins these same columns' flex to fill
    their row (needed for the independent-scroll layout) with !important,
    which would otherwise silently win over a non-important inline style.

    The columns are recreated by Streamlit on every rerun, so:
    - the mousemove/mouseup listeners are attached to the parent document
      ONCE ONLY (guarded by a flag stored on that document), and always look
      up the *current* column elements fresh rather than holding onto
      references from a previous rerun that no longer exist;
    - the chosen ratio is persisted to localStorage and re-applied by a
      short-lived poll loop every time this component remounts, so the
      layout doesn't snap back to the default 3:2 split after every rerun
      (e.g. after sending a chat message).
    """
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;

            if (!doc.__chatDashboardResizeInit) {
                doc.__chatDashboardResizeInit = true;
                doc.__chatDashboardDragging = false;

                doc.addEventListener('mousemove', function(e) {
                    if (!doc.__chatDashboardDragging) return;
                    const zone = doc.querySelector('.st-key-chat_zone');
                    if (!zone) return;
                    const row = zone.closest('[data-testid="stHorizontalBlock"]');
                    if (!row) return;
                    const cols = row.querySelectorAll(':scope > [data-testid="stColumn"]');
                    if (cols.length < 2) return;
                    const rect = row.getBoundingClientRect();
                    let ratio = (e.clientX - rect.left) / rect.width;
                    ratio = Math.min(0.8, Math.max(0.2, ratio));
                    cols[0].style.setProperty('flex', ratio + ' 1 0', 'important');
                    cols[1].style.setProperty('flex', (1 - ratio) + ' 1 0', 'important');
                    localStorage.setItem('chat_dashboard_ratio', ratio.toString());
                });
                doc.addEventListener('mouseup', function() {
                    doc.__chatDashboardDragging = false;
                });
            }

            function setup() {
                const zone = doc.querySelector('.st-key-chat_zone');
                if (!zone) return false;
                const row = zone.closest('[data-testid="stHorizontalBlock"]');
                if (!row) return false;
                const cols = row.querySelectorAll(':scope > [data-testid="stColumn"]');
                if (cols.length < 2) return false;

                const stored = localStorage.getItem('chat_dashboard_ratio');
                if (stored) {
                    const ratio = parseFloat(stored);
                    cols[0].style.setProperty('flex', ratio + ' 1 0', 'important');
                    cols[1].style.setProperty('flex', (1 - ratio) + ' 1 0', 'important');
                }

                if (row.querySelector(':scope > .resize-handle')) return true;

                const handle = doc.createElement('div');
                handle.className = 'resize-handle';
                handle.style.cssText =
                    'width:6px;flex:0 0 auto;cursor:col-resize;' +
                    'background:transparent;z-index:50;position:relative;';
                handle.addEventListener('mouseenter', function() {
                    handle.style.background = 'rgba(42,87,72,0.35)';
                });
                handle.addEventListener('mouseleave', function() {
                    handle.style.background = 'transparent';
                });
                handle.addEventListener('mousedown', function(e) {
                    doc.__chatDashboardDragging = true;
                    e.preventDefault();
                });
                cols[0].after(handle);
                return true;
            }

            let attempts = 0;
            const interval = setInterval(function() {
                attempts++;
                if (setup() || attempts > 40) clearInterval(interval);
            }, 250);
        })();
        </script>
        """,
        height=0,
    )
