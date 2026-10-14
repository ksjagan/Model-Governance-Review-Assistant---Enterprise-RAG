from __future__ import annotations

from urllib.parse import urljoin

import requests
import streamlit as st

from docgov.config import get_settings


settings = get_settings()

st.set_page_config(page_title="Model Governance Review Assistant", layout="wide")
st.title("Model Governance Review Assistant")

api_url = st.sidebar.text_input("FastAPI URL", value=settings.fastapi_url)
visual_min_score = st.sidebar.slider(
    "Visual match confidence",
    min_value=0.0,
    max_value=1.0,
    value=float(settings.visual_match_min_score),
    step=0.05,
)

uploaded_files = st.file_uploader(
    "Upload governance PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Index Documents"):
    files = [
        ("files", (uploaded.name, uploaded.getvalue(), "application/pdf"))
        for uploaded in uploaded_files
    ]
    with st.spinner("Sending PDFs to FastAPI and building the retrieval index..."):
        response = requests.post(urljoin(api_url, "/ingest"), files=files, timeout=600)
    if response.ok:
        payload = response.json()
        st.session_state["session_id"] = payload["session_id"]
        st.success("Documents indexed.")
        cols = st.columns(4)
        stats = payload["stats"]
        cols[0].metric("Documents", stats["documents"])
        cols[1].metric("Pages", stats["pages"])
        cols[2].metric("Chunks", stats["chunks"])
        cols[3].metric("Visuals", stats["visuals"])
    else:
        st.error(response.text)

session_id = st.session_state.get("session_id")

question = st.text_input("Ask a governance review question")
if question:
    if not session_id:
        st.info("Index documents first.")
    else:
        with st.spinner("Retrieving evidence and generating answer..."):
            response = requests.post(
                urljoin(api_url, "/ask"),
                json={
                    "session_id": session_id,
                    "question": question,
                    "visual_min_score": visual_min_score,
                },
                timeout=600,
            )
        if response.ok:
            payload = response.json()
            st.subheader("Answer")
            st.write(payload["answer"])

            if payload["visuals"]:
                st.subheader("Attached Images")
                for visual in payload["visuals"]:
                    image_url = urljoin(api_url, visual["image_url"])
                    st.image(image_url, caption=f"{visual['kind'].title()} - page {visual['page_number']}")
        else:
            st.error(response.text)

if session_id and st.button("Generate Governance Review"):
    with st.spinner("Generating section-wise review..."):
        response = requests.post(urljoin(api_url, "/review"), json={"session_id": session_id}, timeout=600)
    if response.ok:
        payload = response.json()
        st.subheader("Section-wise Review")
        for section in payload["sections"]:
            st.markdown(f"### {section['section'].title()}")
            st.write(section["summary"])
            for visual in section["visuals"]:
                st.image(
                    urljoin(api_url, visual["image_url"]),
                    caption=f"{visual['kind']} - page {visual['page_number']}",
                )
    else:
        st.error(response.text)
