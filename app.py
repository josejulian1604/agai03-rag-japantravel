"""
app.py
------
Streamlit UI for the Japan Travel RAG Chatbot.
Run with: streamlit run app.py
"""

import os
import csv
import streamlit as st
from datetime import datetime
from src.chatbot import JapanTravelChatbot
import html as html_lib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Japan Travel AI",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #C0392B 0%, #922B21 60%, #1a1a2e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: "🌸";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.15;
    }
    .app-header h1 {
        font-family: 'Noto Serif JP', serif;
        color: white;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.75);
        margin: 0.3rem 0 0;
        font-size: 0.9rem;
    }

    /* Chat messages */
    .msg-user {
        background: #f0f2f6;
        border-radius: 18px 18px 4px 18px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0 0.5rem 3rem;
        font-size: 0.95rem;
        color: #1a1a2e;
    }
    .msg-bot {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 18px 18px 18px 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 3rem 0.5rem 0;
        font-size: 0.95rem;
        color: #1a1a2e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Source badge */
    .source-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }
    .badge-qa {
        background: #d4edda;
        color: #155724;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .badge-vector {
        background: #cce5ff;
        color: #004085;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .source-link {
        font-size: 0.78rem;
        color: #C0392B;
        text-decoration: none;
    }
    .score-bar-wrap {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        color: #888;
    }
    .score-bar {
        height: 4px;
        border-radius: 2px;
        background: #e0e0e0;
        width: 80px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #C0392B, #e74c3c);
    }

    /* Sidebar */
    .stat-card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .stat-num {
        font-size: 1.6rem;
        font-weight: 600;
        color: #C0392B;
        font-family: 'Noto Serif JP', serif;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.1rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_resource
def load_chatbot():
    """Load chatbot once and cache it across reruns."""
    return JapanTravelChatbot()


def count_pages():
    try:
        return len([f for f in os.listdir("data/raw") if f.endswith(".json")])
    except:
        return 0


def count_qa_pairs():
    try:
        with open("data/qa_dataset.csv", encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f)) - 1
    except:
        return 0


def render_message(role: str, content: str, meta: dict = None):
    """Render a single chat message with optional source metadata."""
    if role == "user":
        st.markdown(f'<div class="msg-user">👤 {content}</div>',
                    unsafe_allow_html=True)
    else:
        # ── Render bot response as markdown (not inside HTML div) ──
        with st.container():
            st.markdown(f"🤖 {content}")

            # ── Render citation separately ──
            if meta:
                mode             = meta.get("mode", "")
                sources          = meta.get("sources", [])
                score            = meta.get("score", 0)
                matched_question = meta.get("matched_question")

                badge_class = "badge-qa" if mode == "qa_direct" else "badge-vector"
                badge_label = "Q/A Direct" if mode == "qa_direct" else "Vector Search"
                score_pct   = int(score * 100)
                score_width = int(score * 80)

                citation_html = f'''
                <div style="border-left: 3px solid #e8e8e8; padding-left: 0.8rem; margin-top: 0.5rem;">
                    <div class="source-row">
                        <span class="{badge_class}">{badge_label}</span>
                        <div class="score-bar-wrap">
                            <div class="score-bar">
                                <div class="score-fill" style="width:{score_width}px"></div>
                            </div>
                            {score_pct}% confidence
                        </div>
                    </div>'''

                if mode == "qa_direct" and matched_question:
                    import html as html_lib
                    safe_q = html_lib.escape(matched_question[:90])
                    citation_html += f'''
                    <div style="font-size:0.75rem;color:#666;margin-top:0.3rem;">
                        📌 <strong>Matched Q/A pair:</strong>
                    </div>
                    <div style="font-size:0.75rem;color:#555;margin-top:0.2rem;padding-left:0.8rem;border-left:2px solid #ddd;">
                        <em><span style="color:white;font-weight:bold">Q:</span> "{safe_q}"</em>
                    </div>'''

                    matched_answer = meta.get("matched_answer")
                    if matched_answer:
                        safe_a = html_lib.escape(matched_answer[:150])
                        citation_html += f'''
                            <div style="font-size:0.75rem;color:#555;margin-top:0.2rem;padding-left:0.8rem;border-left:2px solid #ddd;">
                                <span style="color:white;font-weight:bold">A:</span>: {safe_a}{"..." if len(matched_answer) > 150 else ""}
                            </div>'''
                if sources:
                    src   = sources[0]
                    url   = src.get("url", "#")
                    title = html_lib.escape(src.get("title", "Source")[:50]) if 'html_lib' in dir() else src.get("title", "Source")[:50]
                    citation_html += f'''
                    <div style="font-size:0.75rem;margin-top:0.2rem;">
                        📎 Source: <a class="source-link" href="{url}" target="_blank">{title}</a>
                    </div>'''

                    if mode == "vector_search" and len(sources) > 1:
                        for s in sources[1:3]:
                            citation_html += f'''
                            <div style="font-size:0.72rem;color:#aaa;">
                                📎 <a class="source-link" href="{s.get("url","#")}" target="_blank">
                                {s.get("title","")[:50]}</a> ({s.get("score",0):.0%})
                            </div>'''

                citation_html += '</div>'
                st.markdown(citation_html, unsafe_allow_html=True)
            
            st.divider()


def export_chat(messages: list) -> str:
    """Format chat history as plain text for download."""
    lines = [
        "Japan Travel AI — Chat Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        ""
    ]
    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"You: {msg['content']}")
        else:
            lines.append(f"Assistant: {msg['content']}")
            if msg.get("meta"):
                mode  = msg["meta"].get("mode", "")
                score = msg["meta"].get("score", 0)
                lines.append(f"[{mode} | confidence: {score:.0%}]")
        lines.append("")
    return "\n".join(lines)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None


# ── Load chatbot ──────────────────────────────────────────────────────────────
bot = load_chatbot()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗾 Japan Travel AI")
    st.caption("Powered by JNTO · Claude · RAG")
    st.divider()

    # Stats
    n_pages = count_pages()
    n_qa    = count_qa_pairs()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num">{n_pages}</div>
            <div class="stat-label">Pages scraped</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num">{n_qa}</div>
            <div class="stat-label">Q/A pairs</div>
        </div>""", unsafe_allow_html=True)

    # Last query info
    if st.session_state.last_meta:
        st.divider()
        meta  = st.session_state.last_meta
        mode  = meta.get("mode", "")
        score = meta.get("score", 0)

        st.markdown("**Last retrieval**")
        if mode == "qa_direct":
            st.success(f"Q/A Direct — {score:.0%} confidence")
        else:
            st.info(f"Vector Search — {score:.0%} confidence")

    st.divider()

    # Sample questions
    st.markdown("**Try asking:**")
    samples = [
        "What airports serve Tokyo?",
        "Best time for cherry blossoms?",
        "How to travel by Shinkansen?",
        "What food should I try in Osaka?",
        "Do I need a visa for Japan?",
    ]
    for sample in samples:
        if st.button(sample, use_container_width=True, key=f"sample_{sample}"):
            st.session_state["prefill"] = sample

    st.divider()

    # Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_meta = None
            bot.clear_history()
            st.rerun()
    with col_b:
        if st.session_state.messages:
            chat_text = export_chat(st.session_state.messages)
            st.download_button(
                label="💾 Export",
                data=chat_text,
                file_name=f"japan_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()
    st.markdown("""
        <a href="https://www.japan.travel/en/us/">Japan Travel</a>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("AGAI-03 CodeCademy · José Julián Gutiérrez Badilla · Software Engineer")
    


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🗾 Japan Travel AI Assistant</h1>
    <p>Ask me anything about traveling to Japan — destinations, food, transport, culture and more.</p>
</div>
""", unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    st.markdown("""<div class="msg-bot">
        🤖 Konnichiwa! 👋 I'm your Japan travel assistant, powered by the official
        JNTO travel guide. Ask me about destinations, food, transportation, culture,
        visa requirements, or the best seasons to visit.<br><br>
        <em>Where in Japan are you thinking of visiting?</em>
    </div>""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("meta"))

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")

with st.container():
    user_input = st.chat_input(
        placeholder="Ask about Japan... (e.g. What's the best time to see Mt. Fuji?)",
    )

    # Use prefill from sidebar sample buttons
    if prefill:
        user_input = prefill

# ── Process input ─────────────────────────────────────────────────────────────
if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
    })

    # Get response from chatbot
    with st.spinner("Searching Japan travel knowledge..."):
        response = bot.chat(user_input)

    meta = {
        "mode":    response["mode"],
        "sources": response["sources"],
        "score":   response["score"],
        "matched_question": response.get("matched_question"),
        "matched_answer":   response.get("matched_answer"),
    }

    # Add bot response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response["answer"],
        "meta":    meta,
    })

    st.session_state.last_meta = meta
    st.rerun()