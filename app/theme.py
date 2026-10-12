"""Load the project's shared, accessible visual theme."""

from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    """Inject the base component styling and project-specific palette."""
    app_dir = Path(__file__).resolve().parent
    css = "\n".join((app_dir / name).read_text(encoding="utf-8") for name in ("theme_base.css", "theme.css"))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
