---
title: Web Research Agent
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 LangGraph Web Research Agent

A powerful AI research assistant built with **LangGraph** and **Groq (Llama 3.1)** that searches the web to answer your questions in real time.

## Features

- 🌐 **Live Web Search** — Uses Tavily (primary) and DuckDuckGo (fallback) to find up-to-date information
- 🤖 **LangGraph ReAct Agent** — Intelligently decides when to search and how to synthesize results
- ⚡ **Powered by Groq** — Ultra-fast inference using `llama-3.1-8b-instant`
- 💬 **Clean Chat UI** — Simple web interface to interact with the agent

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq — Llama 3.1 8B Instant |
| Search (Primary) | Tavily API |
| Search (Fallback) | DuckDuckGo (ddgs) |
| Web Server | Flask + Gunicorn |
| Deployment | Docker on Hugging Face Spaces |
