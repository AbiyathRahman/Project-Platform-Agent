"""Streamlit UI for Project Portfolio Agent - GitHub Repository RAG Workspace."""

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Project Portfolio Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_BACKEND_URL = os.getenv("PORTFOLIO_AGENT_API_URL", "http://127.0.0.1:8000")
DOC_EXTENSIONS = {".md", ".mdx", ".txt", ".json", ".yaml", ".yml", ".toml", ".xml"}
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs", ".rb",
    ".sql", ".css", ".html", ".cpp", ".c", ".h", ".hpp", ".cs", ".php",
}

# Session State Initialization
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repository" not in st.session_state:
    st.session_state.repository = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "ingestion_stats" not in st.session_state:
    st.session_state.ingestion_stats = None


def get_backend_url(path: str) -> str:
    return f"{st.session_state.backend_url.rstrip('/')}{path}"


def show_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except requests.JSONDecodeError:
        detail = response.text
    st.error(f"Backend Error ({response.status_code}): {detail}")


def check_backend_health() -> bool:
    try:
        res = requests.get(get_backend_url("/health"), timeout=3)
        return res.status_code == 200
    except requests.RequestException:
        return False


# Centered Title and Header
st.markdown("<h1 style='text-align: center;'>GitHub Repository RAG Workspace</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #6b7280;'>Index GitHub repository source files and ask questions grounded in your code.</p>",
    unsafe_allow_html=True,
)
st.write("")

# Backend status notice if offline
backend_online = check_backend_health()
if not backend_online:
    st.warning(f"Backend server is unreachable at {st.session_state.backend_url}. Please ensure the FastAPI backend is running.")

# Main Flow Tabs
tab1, tab2, tab3 = st.tabs([
    "1. Load Repository",
    "2. Select & Ingest Files",
    "3. Repository Q&A Chat",
])

# ==========================================
# TAB 1: LOAD REPOSITORY
# ==========================================
with tab1:
    st.subheader("Connect to GitHub Repository")
    st.caption("Enter a public GitHub repository link or owner/repository to inspect available files.")

    col1, col2 = st.columns([3, 1])

    with col1:
        repo_input = st.text_input(
            "GitHub Repository Link or Owner/Repo",
            placeholder="https://github.com/owner/repository or owner/repository",
            help="Public repositories work directly. Private repositories require GITHUB_TOKEN on backend.",
        )

    with col2:
        branch_input = st.text_input(
            "Branch",
            value="main",
            help="Target branch (e.g. main, master, dev)",
        )

    st.write("")
    fetch_button = st.button("Fetch Repository Files", type="primary", use_container_width=True)

    if fetch_button:
        if not repo_input.strip():
            st.warning("Please enter a valid GitHub repository URL.")
        elif not backend_online:
            st.error("Backend server is offline. Please start the FastAPI backend first.")
        else:
            with st.spinner("Fetching repository directory tree..."):
                try:
                    response = requests.post(
                        get_backend_url("/repository/files"),
                        json={"repo_url": repo_input.strip(), "branch": branch_input.strip()},
                        timeout=60,
                    )
                    if response.ok:
                        data = response.json()
                        st.session_state.repository = data
                        st.session_state.selected_files = []
                        st.session_state.rag_ready = False
                        st.session_state.ingestion_stats = None
                        st.success(
                            f"Successfully loaded {len(data['files'])} files from {data['owner']}/{data['repo_name']} (branch: {data['branch']})."
                        )
                        st.info("Proceed to Tab 2 to select files and start ingestion.")
                    else:
                        show_error(response)
                except requests.RequestException as exc:
                    st.error(f"Could not connect to backend: {exc}")

    # Display loaded repository metrics if available
    if st.session_state.repository:
        repo = st.session_state.repository
        st.divider()
        st.subheader("Repository Information")
        m1, m2, m3 = st.columns(3)
        m1.metric("Repository", f"{repo['owner']}/{repo['repo_name']}")
        m2.metric("Branch", repo["branch"])
        m3.metric("Supported Files Found", len(repo["files"]))

# ==========================================
# TAB 2: SELECT & INGEST FILES
# ==========================================
with tab2:
    st.subheader("Select and Ingest Files")

    repo = st.session_state.repository
    if not repo:
        st.info("No repository loaded yet. Please complete Step 1 first.")
    else:
        files = repo["files"]

        # Calculate extension breakdown
        ext_counts: dict[str, int] = {}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # File metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Files", len(files))
        m2.metric("Selected Files", len(st.session_state.selected_files))
        m3.metric("Unique Extensions", len(ext_counts))

        st.write("")
        st.write("**Quick Selection**")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        if btn_col1.button("Select All", use_container_width=True):
            st.session_state.selected_files = files[:200]
            st.rerun()

        if btn_col2.button("Select Code Only", use_container_width=True):
            st.session_state.selected_files = [f for f in files if os.path.splitext(f)[1].lower() in CODE_EXTENSIONS][:200]
            st.rerun()

        if btn_col3.button("Select Docs Only", use_container_width=True):
            st.session_state.selected_files = [f for f in files if os.path.splitext(f)[1].lower() in DOC_EXTENSIONS][:200]
            st.rerun()

        if btn_col4.button("Clear Selection", use_container_width=True):
            st.session_state.selected_files = []
            st.rerun()

        st.divider()

        # Filtering Options
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            search_query = st.text_input("Filter files by path or name", placeholder="e.g. src/ or README")
        with f_col2:
            ext_options = sorted(list(ext_counts.keys()))
            selected_exts = st.multiselect("Filter by extension", options=ext_options, default=[])

        # Apply filters
        filtered_files = files
        if search_query:
            filtered_files = [f for f in filtered_files if search_query.lower() in f.lower()]
        if selected_exts:
            filtered_files = [f for f in filtered_files if os.path.splitext(f)[1].lower() in selected_exts]

        st.caption(f"Showing {len(filtered_files)} of {len(files)} files.")

        # File Selection Multiselect
        selected_in_widget = st.multiselect(
            "Files to index into RAG",
            options=filtered_files,
            default=[f for f in st.session_state.selected_files if f in filtered_files],
            help="Maximum 200 files per ingestion request.",
        )
        st.session_state.selected_files = selected_in_widget

        st.divider()

        ingest_ready = len(st.session_state.selected_files) > 0
        st.write(f"**Ready for Ingestion:** {len(st.session_state.selected_files)} file(s) selected.")

        if st.button("Ingest Selected Files", type="primary", disabled=not ingest_ready, use_container_width=True):
            if len(st.session_state.selected_files) > 200:
                st.error("Please select 200 or fewer files per batch.")
            else:
                payload = {
                    "owner": repo["owner"],
                    "repo_name": repo["repo_name"],
                    "branch": repo["branch"],
                    "file_paths": st.session_state.selected_files,
                }
                try:
                    with st.spinner("Processing, chunking, and embedding files..."):
                        response = requests.post(get_backend_url("/ingest"), json=payload, timeout=300)
                    if response.ok:
                        res_data = response.json()
                        chunks = res_data.get("chunks_ingested", 0)
                        st.session_state.rag_ready = True
                        st.session_state.ingestion_stats = {
                            "chunks_ingested": chunks,
                            "file_count": len(st.session_state.selected_files),
                        }
                        st.success(f"Ingestion complete: {chunks} chunks added to the RAG index from {len(st.session_state.selected_files)} files.")
                        st.info("Switch to Tab 3 to start asking questions.")
                    else:
                        show_error(response)
                except requests.RequestException as exc:
                    st.error(f"Ingestion failed: {exc}")

# ==========================================
# TAB 3: REPOSITORY Q&A CHAT
# ==========================================
with tab3:
    st.subheader("Repository Questions & Answers")

    if not st.session_state.rag_ready:
        st.info("RAG index is not ready yet. Please select and ingest files in Step 2 to enable Q&A.")

    # Scope Selection & Clear Controls
    c_col1, c_col2 = st.columns([3, 1])

    repo_filter_name = None
    with c_col1:
        if st.session_state.repository:
            current_repo_name = st.session_state.repository["repo_name"]
            filter_mode = st.radio(
                "Search Scope",
                options=[f"Current Repository ({current_repo_name})", "All Indexed Repositories"],
                horizontal=True,
            )
            if "Current Repository" in filter_mode:
                repo_filter_name = current_repo_name

    with c_col2:
        st.write("")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("View Source Passages"):
                    for src in msg["sources"]:
                        score = src.get("score")
                        score_text = f" — relevance: {score:.3f}" if isinstance(score, float) else ""
                        st.write(f"- `{src.get('repo')}/{src.get('file_path')}`{score_text}")

    # Chat Input Handling
    user_query = st.chat_input("Ask a question about the repository...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching repository and generating answer..."):
                try:
                    payload = {
                        "question": user_query,
                        "top_k": 5,
                        "repo_filter": repo_filter_name,
                    }
                    response = requests.post(get_backend_url("/query"), json=payload, timeout=120)
                    if response.ok:
                        data = response.json()
                        answer = data.get("answer", "No answer generated.")
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("View Source Passages"):
                                for src in sources:
                                    score = src.get("score")
                                    score_text = f" — relevance: {score:.3f}" if isinstance(score, float) else ""
                                    st.write(f"- `{src.get('repo')}/{src.get('file_path')}`{score_text}")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    else:
                        show_error(response)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
