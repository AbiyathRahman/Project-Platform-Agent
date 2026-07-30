"""Streamlit interface for selecting GitHub repository files to add to the RAG index."""

import os

import requests
import streamlit as st


st.set_page_config(page_title="Portfolio Agent", page_icon="🗂️", layout="wide")

DEFAULT_BACKEND_URL = os.getenv("PORTFOLIO_AGENT_API_URL", "http://127.0.0.1:8000")


def backend_url(path: str) -> str:
    return f"{st.session_state.backend_url.rstrip('/')}{path}"


def show_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except requests.JSONDecodeError:
        detail = response.text
    st.error(f"Backend request failed ({response.status_code}): {detail}")


if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Project Portfolio Agent")
st.caption("Choose source files from a GitHub repository, then add them to your RAG knowledge base.")

with st.sidebar:
    st.subheader("Connection")
    st.text_input("Backend URL", key="backend_url", help="The FastAPI server URL.")
    try:
        response = requests.get(backend_url("/health"), timeout=3)
        response.raise_for_status()
        st.success("Backend connected")
    except requests.RequestException:
        st.warning("Backend unavailable. Start FastAPI before loading files.")

with st.form("repository_form"):
    repo_url = st.text_input(
        "GitHub repository link",
        placeholder="https://github.com/owner/repository",
        help="Public repositories work without a token; private repositories require GITHUB_TOKEN on the backend.",
    )
    branch = st.text_input("Branch", value="main")
    load_files = st.form_submit_button("Load repository files", type="primary")

if load_files:
    try:
        response = requests.post(
            backend_url("/repository/files"),
            json={"repo_url": repo_url, "branch": branch},
            timeout=60,
        )
        if response.ok:
            st.session_state.repository = response.json()
            st.session_state.selected_files = []
        else:
            show_error(response)
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")

repository = st.session_state.get("repository")
if repository:
    files = repository["files"]
    st.subheader(f"Select files from {repository['owner']}/{repository['repo_name']}")
    st.caption(f"{len(files)} supported text and source files found on `{repository['branch']}`.")

    filter_text = st.text_input("Filter files", placeholder="e.g. src/ or README")
    visible_files = [path for path in files if filter_text.lower() in path.lower()]
    select_all = st.checkbox("Select all filtered files")
    defaults = visible_files if select_all else st.session_state.get("selected_files", [])
    selected_files = st.multiselect(
        "Files to add to RAG",
        options=visible_files,
        default=[path for path in defaults if path in visible_files],
        help="Select up to 200 files per ingestion request.",
    )
    st.session_state.selected_files = selected_files
    st.caption(f"{len(selected_files)} file(s) selected")

    if st.button("Add selected files to RAG", type="primary", disabled=not selected_files):
        if len(selected_files) > 200:
            st.error("Please select 200 files or fewer.")
        else:
            payload = {
                "owner": repository["owner"],
                "repo_name": repository["repo_name"],
                "branch": repository["branch"],
                "file_paths": selected_files,
            }
            try:
                with st.spinner("Fetching, chunking, embedding, and indexing selected files..."):
                    response = requests.post(backend_url("/ingest"), json=payload, timeout=300)
                if response.ok:
                    chunks_ingested = response.json().get("chunks_ingested", 0)
                    st.session_state.rag_ready = True
                    st.success(f"Ingestion complete: {chunks_ingested} chunks added to the RAG index.")
                else:
                    show_error(response)
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")

if st.session_state.rag_ready:
    st.divider()
    st.subheader("Ask your repository")
    st.caption("Answers are generated only from the files you added to the RAG index.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        score = source.get("score")
                        score_text = f" — relevance: {score:.3f}" if isinstance(score, float) else ""
                        st.write(f"- `{source.get('repo')}/{source.get('file_path')}`{score_text}")

    question = st.chat_input("Ask a question about the selected repository files")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        repo_filter = repository["repo_name"] if repository else None
        with st.chat_message("assistant"):
            with st.spinner("Searching the repository and writing an answer..."):
                try:
                    response = requests.post(
                        backend_url("/query"),
                        json={"question": question, "top_k": 5, "repo_filter": repo_filter},
                        timeout=120,
                    )
                    if response.ok:
                        result = response.json()
                        answer = result.get("answer") or "I couldn't generate an answer."
                        sources = result.get("sources", [])
                        st.markdown(answer)
                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    score = source.get("score")
                                    score_text = f" — relevance: {score:.3f}" if isinstance(score, float) else ""
                                    st.write(f"- `{source.get('repo')}/{source.get('file_path')}`{score_text}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    else:
                        show_error(response)
                except requests.RequestException as exc:
                    st.error(f"Could not reach the backend: {exc}")
