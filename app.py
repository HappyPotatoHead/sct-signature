import streamlit as st

demonstration = st.Page(
    "pages/demonstration.py",
    title="Offline Signature Verification",
    icon="✒️"
)

explanation = st.Page(
    "pages/explanation.py",
    title="How it Works",
    icon="📖"
)

pg = st.navigation([demonstration, explanation])

pg.run()