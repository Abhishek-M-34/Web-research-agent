from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import uuid
import requests as _requests

# Load API keys from the .env file (or HF Spaces secrets)
load_dotenv()

app = Flask(__name__)
# Secret key required for Flask session (memory per browser tab)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# ── Lazy globals (initialized on first request, not at import time) ──────────
_llm        = None
_agent      = None
_checkpointer = None   # LangGraph in-memory checkpointer for conversation memory

# ── Self-awareness / identity prompt ────────────────────────────────────────
SYSTEM_PROMPT = """
You are **LangGraph Research Agent**, an AI assistant built specifically for web-based research.

## 🧠 Identity & Architecture
- **Name**: LangGraph Research Agent
- **Framework**: LangGraph (by LangChain) — a graph-based agent orchestration framework
- **LLM backbone**: Llama 3.1 8B Instant, served via the Groq inference API (ultra-low latency)
- **Search tool**: A custom `brave_search` tool that first queries the Tavily API for real-time
  web results, then falls back to DuckDuckGo if Tavily returns nothing.
- **Frontend**: A responsive single-page Flask web application styled with glassmorphism and
  dark-mode aesthetics.
- **Memory**: You maintain full conversation memory within a session using LangGraph's
  MemorySaver checkpointer. Every message in the current session is part of your context.

## ⚠️ Limitations
- **Knowledge cutoff**: Your base training data has a knowledge cutoff of early 2024. For
  anything more recent, you ALWAYS call `brave_search` first.
- **Session-scoped memory**: Your memory is limited to the current browser session. If the user
  refreshes or opens a new tab, memory resets.
- **Search reliability**: Search results depend on third-party APIs (Tavily / DuckDuckGo).
  Occasionally they may return no results or outdated pages.
- **Context window**: Very long conversations may eventually exceed the model's context window
  (~8 k tokens for this model). If that happens, earlier messages may be truncated by the LLM.
- **No file uploads**: You cannot read PDFs, images, or other files — only text messages.
- **No real-time execution**: You cannot run code or access private/authenticated websites.

## 🔧 Behavior Rules
- ALWAYS call `brave_search` first before answering any factual or current-events question.
- If search results are non-empty: summarize the findings, cite the top 2 sources (title + URL),
  and give a clear, concise answer.
- If search results are empty: state that live search returned no results, then provide a
  thorough best-effort answer from your training knowledge, noting the knowledge cutoff.
- Leverage past messages in the conversation to give contextually relevant follow-up answers.
- Never refuse to answer. Always give your best response.
- When asked about yourself (how you work, your limitations, your architecture), answer
  accurately using the information above.
"""


def get_agent():
    """Initialize the LangGraph agent + MemorySaver once, on first use."""
    global _llm, _agent, _checkpointer
    if _agent is not None:
        return _agent, _checkpointer

    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_groq import ChatGroq
    from langchain_core.tools import tool

    # Search backend helpers ─────────────────────────────────────────────────
    def _search_tavily(query: str, max_results: int = 5) -> list:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return []
        try:
            payload = {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
            resp = _requests.post(
                "https://api.tavily.com/search", json=payload, timeout=12
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "href":  r.get("url", ""),
                    "body":  r.get("content", ""),
                }
                for r in results
            ]
        except Exception as e:
            print(f"[Tavily error] {e}")
            return []

    def _search_duckduckgo(query: str, max_results: int = 5) -> list:
        try:
            from ddgs import DDGS
            results = DDGS().text(
                query, max_results=max_results, region="wt-wt", backend="api"
            )
            return results if results else []
        except Exception as e:
            print(f"[DuckDuckGo error] {e}")
            return []

    @tool
    def brave_search(query: str) -> list:
        """Search the web for up-to-date information.
        Tries Tavily first (if TAVILY_API_KEY is set), then falls back to DuckDuckGo.
        Returns a list of result dicts with keys: title, href, body.
        """
        results = _search_tavily(query)
        if results:
            return results
        results = _search_duckduckgo(query)
        return results if results else []

    # ── Build the agent with MemorySaver (enables persistent conversation memory) ──
    _checkpointer = MemorySaver()
    _llm   = ChatGroq(model="llama-3.1-8b-instant")
    _agent = create_react_agent(
        _llm,
        [brave_search],
        prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    print("✅ LangGraph agent (with memory) initialized successfully.")
    return _agent, _checkpointer


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Assign a unique session ID to each browser session (persists across messages)
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health-check endpoint so HF Spaces knows the container is up."""
    return {"status": "ok"}, 200


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clear the current session memory, start fresh."""
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "cleared", "session_id": session["session_id"]})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Ensure the session has an ID (safety net)
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    thread_id = session["session_id"]

    try:
        agent, _ = get_agent()

        # The config dict tells LangGraph which memory thread to use.
        # Using the same thread_id means the agent sees all previous messages.
        config = {"configurable": {"thread_id": thread_id}}

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        response_content = result["messages"][-1].content

        if (
            isinstance(response_content, list)
            and response_content
            and isinstance(response_content[0], dict)
        ):
            response_text = response_content[0].get("text", str(response_content))
        else:
            response_text = str(response_content)

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"[Chat error] {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
