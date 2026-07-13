"""
US Tax & Legal RAG — Production UI
Harvey AI / Perplexity inspired Streamlit front end.

Run with:  streamlit run app.py
Backend URL is read from the BACKEND_URL environment variable
(falls back to http://127.0.0.1:8000 for local dev).
"""

import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# ==========================================================
# 1. CONFIG
# ==========================================================

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_ENDPOINT = f"{BACKEND_URL}/ask"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
STATS_ENDPOINT = f"{BACKEND_URL}/stats"

APP_TITLE = "US Tax & Legal RAG"
APP_TAGLINE = "Hybrid Search • BM25 • Vector Search • CrossEncoder • Groq"
ASSISTANT_NAME = "Legal Research Assistant"

SUGGESTED_QUESTIONS = [
    "What is IRC Section 162?",
    "What expenses qualify as ordinary and necessary business deductions?",
    "How does the statute of limitations apply to tax audits?",
    "What is the difference between a tax credit and a tax deduction?",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# 2. SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ==========================================================
# 3. THEME / CSS
# ==========================================================

st.markdown(
    """
<style>

:root {
    --accent: #8a6d3b;
    --accent-light: #b8975f;
    --bg-glass: rgba(255, 255, 255, 0.55);
    --border-glass: rgba(138, 109, 59, 0.18);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-glass: rgba(30, 30, 30, 0.55);
        --border-glass: rgba(184, 151, 95, 0.25);
    }
}

html, body, [class*="css"] {
    font-family: "Georgia", "Iowan Old Style", "Source Serif Pro", serif;
}

#MainMenu, footer, header {visibility: hidden;}

/* Hero header */
.hero {
    padding: 1.4rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(138,109,59,0.10), rgba(138,109,59,0.02));
    border: 1px solid var(--border-glass);
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.hero p {
    margin: 0.2rem 0 0 0;
    opacity: 0.75;
    font-size: 0.95rem;
}

/* Glass card */
.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(6px);
    margin-bottom: 0.8rem;
}

/* NOTE: chat bubbles come from Streamlit's native st.chat_message() —
   no custom .user-box / .ai-box classes needed, avoids double-styling. */

.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: rgba(138, 109, 59, 0.15);
    border: 1px solid var(--border-glass);
    font-size: 0.72rem;
    margin-right: 0.35rem;
}

.status-dot {
    height: 9px;
    width: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-online { background-color: #2ecc71; }
.status-offline { background-color: #e74c3c; }

.header-status {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    font-size: 0.85rem;
    float: right;
    margin-top: 0.3rem;
}

.ref-card {
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    background: rgba(138, 109, 59, 0.05);
}

.stChatInputContainer, .stButton>button {
    border-radius: 12px !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 4. HELPERS
# ==========================================================


@st.cache_data(ttl=30, show_spinner=False)
def check_backend_health():
    """Ping the backend. Returns (is_online, detail_dict)."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=4)
        if resp.status_code == 200:
            try:
                return True, resp.json()
            except ValueError:
                return True, {}
        return False, {}
    except requests.exceptions.RequestException:
        return False, {}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_corpus_stats():
    """Optional /stats endpoint on the backend. Returns dict or None."""
    try:
        resp = requests.get(STATS_ENDPOINT, timeout=4)
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.exceptions.RequestException:
        return None


def call_ask_api(question: str):
    """Calls the backend /ask endpoint. Returns a result dict."""
    try:
        resp = requests.post(ASK_ENDPOINT, json={"question": question}, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "answer": data.get("answer", ""),
                "latency": data.get("latency"),
                "model": data.get("model"),
                "references": data.get("references", []),
                "error": None,
            }
        return {
            "answer": None,
            "latency": None,
            "model": None,
            "references": [],
            "error": f"Backend error ({resp.status_code}): {resp.text[:300]}",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "answer": None,
            "latency": None,
            "model": None,
            "references": [],
            "error": f"Could not reach backend: {exc}",
        }


def copy_to_clipboard_button(text: str, key: str):
    """Renders a 'Copy Answer' button using an embedded HTML component.
    Lawyers tend to copy answers into their own documents far more often
    than they download files, so this sits alongside the download buttons."""
    safe_text = text.replace("`", "\\`").replace("</", "<\\/")
    components.html(
        f"""
        <button id="copy-btn-{key}"
            style="
                background: rgba(138,109,59,0.12);
                border: 1px solid rgba(138,109,59,0.3);
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 13px;
                width: 100%;
                cursor: pointer;
                font-family: sans-serif;
            "
            onclick="
                navigator.clipboard.writeText(`{safe_text}`);
                const b = document.getElementById('copy-btn-{key}');
                b.innerText = 'Copied ✓';
                setTimeout(() => {{ b.innerText = '📋 Copy Answer'; }}, 1500);
            "
        >📋 Copy Answer</button>
        """,
        height=42,
    )


def render_references(references):
    if not references:
        st.warning("No references were returned for this answer.")
        return
    with st.expander(f"📄 References ({len(references)})", expanded=False):
        for ref in references:
            doc_name = ref.get("document_name", "Unknown document")
            doc_type = ref.get("document_type", "N/A")
            page_start = ref.get("page_start", "?")
            page_end = ref.get("page_end", "?")
            st.markdown(
                f"""
<div class="ref-card">
<b>{doc_name}</b><br>
<span class="badge">{doc_type}</span>
<span class="badge">pp. {page_start}–{page_end}</span>
</div>
""",
                unsafe_allow_html=True,
            )


def render_assistant_message(message, index):
    with st.chat_message("assistant", avatar="⚖️"):
        st.caption(f"⚖️ {ASSISTANT_NAME}")
        st.markdown(message["content"])

        if message.get("latency") is not None or message.get("model"):
            m1, m2 = st.columns(2)
            with m1:
                if message.get("latency") is not None:
                    st.metric("⏱ Response Time", f"{message['latency']:.2f} sec")
            with m2:
                if message.get("model"):
                    st.metric("🤖 Model", message["model"])

        render_references(message.get("references", []))

        # Actions: copying tends to be used far more than downloading a file,
        # so it gets equal billing right next to the download buttons.
        act1, act2, act3 = st.columns(3)
        with act1:
            copy_to_clipboard_button(message["content"], key=f"copy-{index}")
        with act2:
            st.download_button(
                label="📥 Download (.md)",
                data=message["content"],
                file_name=f"legal_answer_{index}.md",
                mime="text/markdown",
                key=f"dl-md-{index}",
                use_container_width=True,
            )
        with act3:
            st.download_button(
                label="📥 Download (.txt)",
                data=message["content"],
                file_name=f"legal_answer_{index}.txt",
                mime="text/plain",
                key=f"dl-txt-{index}",
                use_container_width=True,
            )

        st.divider()


def render_user_message(message):
    with st.chat_message("user", avatar="👤"):
        st.markdown(message["content"])


# ==========================================================
# 5. SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("### ⚖️ US Tax & Legal RAG")
    st.caption(APP_TAGLINE)
    st.caption(f"`{BACKEND_URL}`")
    st.divider()

    # Backend online/offline status now lives top-right in the header,
    # where it's far more visible than tucked away in the sidebar.
    stats = fetch_corpus_stats()
    if stats:
        st.markdown("#### 📊 Corpus")
        s1, s2 = st.columns(2)
        with s1:
            if "document_count" in stats:
                st.metric("Documents", stats["document_count"])
        with s2:
            if "chunk_count" in stats:
                st.metric("Chunks", stats["chunk_count"])

    st.divider()
    st.markdown("#### 🏗️ Architecture")
    st.markdown(
        """
- **Retrieval:** Hybrid (BM25 + Vector)
- **Reranking:** CrossEncoder
- **Generation:** Groq
- **Backend:** FastAPI
- **Frontend:** Streamlit
"""
    )

    st.divider()
    st.markdown("#### 💡 Try asking")
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, key=f"suggest-{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_question = None
        st.rerun()

# ==========================================================
# 6. HEADER
# ==========================================================

is_online, health_detail = check_backend_health()
status_class = "status-online" if is_online else "status-offline"
status_text = "Online" if is_online else "Offline"

st.markdown(
    f"""
<div class="hero">
<span class="header-status">
    <span class="status-dot {status_class}"></span>Backend {status_text}
</span>
<h1>⚖️ {APP_TITLE}</h1>
<p>{APP_TAGLINE} · Built with FastAPI + Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 7. EMPTY STATE
# ==========================================================

if len(st.session_state.messages) == 0:
    st.info(
        """
👋 **Welcome!** Ask any question regarding:

- Acts
- Court Judgments
- Tax Documents
- Legal Commentaries

Use the suggestions in the sidebar, or type your own question below.
"""
    )

# ==========================================================
# 8. CONVERSATION HISTORY
# ==========================================================

for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        render_user_message(message)
    else:
        render_assistant_message(message, idx)

# ==========================================================
# 9. CHAT INPUT
# ==========================================================

question = st.chat_input("Ask a question about the legal documents...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()

# ==========================================================
# 10. GENERATE AI RESPONSE (only for the newest, un-answered question)
# ==========================================================

if st.session_state.messages:
    last_message = st.session_state.messages[-1]

    if (
        last_message["role"] == "user"
        and last_message["content"] != st.session_state.last_question
    ):
        st.session_state.last_question = last_message["content"]

        start = time.time()
        with st.status("Searching legal documents...", expanded=True) as status:
            st.write("🔎 Running hybrid search (BM25 + vector)...")
            st.write("📚 Reranking with CrossEncoder...")
            st.write("🤖 Generating grounded answer...")
            result = call_ask_api(last_message["content"])
            elapsed = time.time() - start

            if result["error"]:
                status.update(label="Search failed", state="error", expanded=False)
            else:
                status.update(label="Answer ready", state="complete", expanded=False)

        if result["error"]:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ {result['error']}",
                    "latency": elapsed,
                    "model": None,
                    "references": [],
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "latency": result["latency"] if result["latency"] is not None else elapsed,
                    "model": result["model"],
                    "references": result["references"],
                }
            )

        st.rerun()

# ==========================================================
# 11. FOOTER
# ==========================================================

st.divider()
st.caption(
    f"""
⚖️ {APP_TITLE} · {APP_TAGLINE}

Built with FastAPI + Streamlit
"""
)