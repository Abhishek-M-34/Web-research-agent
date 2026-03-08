from flask import Flask, render_template, request, jsonify
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests as _requests

# Load API keys from the .env file
load_dotenv()

app = Flask(__name__)

# ----- Search backend helpers -----
def _search_tavily(query: str, max_results: int = 5) -> list:
    """Use Tavily API if TAVILY_API_KEY is set."""
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
        resp = _requests.post("https://api.tavily.com/search", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [{"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")} for r in results]
    except Exception as e:
        print(f"[Tavily error] {e}")
        return []

def _search_duckduckgo(query: str, max_results: int = 5) -> list:
    """Fallback: DuckDuckGo via ddgs."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="wt-wt", backend="lite")
        return results if results else []
    except Exception as e:
        print(f"[DuckDuckGo error] {e}")
        return []

# Initialize the AI Model globally
llm = ChatGroq(model="llama-3.1-8b-instant")

# Add the web search tool (Tavily primary → DuckDuckGo fallback)
@tool
def brave_search(query: str) -> list:
    """Search the web for up-to-date information.
    Tries Tavily first (if TAVILY_API_KEY is set), then falls back to DuckDuckGo.
    Returns a list of result dictionaries with keys: title, href, body.
    If no results found, returns an empty list so the LLM uses its own knowledge.
    """
    results = _search_tavily(query)
    if results:
        return results
    results = _search_duckduckgo(query)
    if results:
        return results
    return []

tools = [brave_search]

# System prompt with clear guidance on using search
system_prompt = """
You are a helpful research assistant with access to a web search tool called `brave_search`.
The tool returns a list of dicts, each with keys: `title`, `href`, `body`.

Behavior rules:
- ALWAYS call `brave_search` first before answering any factual or current-events question.
- If the list is non-empty: summarize the findings, cite the top 2 sources (title + URL), and answer the question.
- If the list is empty: clearly say that live search returned no results, then provide a thorough
  best-effort answer from your training knowledge, noting the knowledge cutoff.
- Never refuse to answer. Always give your best response.
"""

# Create the LangGraph agent
agent = create_react_agent(llm, tools, prompt=system_prompt)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
        
    try:
        # Run the agent with the user's input
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        
        # Get the final response from the agent
        response_content = result["messages"][-1].content
        
        # Format the output just in case the LLM returns a list instead of plain string
        if isinstance(response_content, list) and len(response_content) > 0 and isinstance(response_content[0], dict):
            response_text = response_content[0].get("text", str(response_content))
        else:
            response_text = str(response_content)
            
        return jsonify({"response": response_text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
