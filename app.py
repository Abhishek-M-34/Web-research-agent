from flask import Flask, render_template, request, jsonify
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

# Load API keys from the .env file
load_dotenv()

app = Flask(__name__)

# Initialize the AI Model globally
llm = ChatGroq(model="llama-3.1-8b-instant")
# Add the web search tool
tools = [DuckDuckGoSearchRun()]
# Create the LangGraph agent
agent = create_react_agent(llm, tools)

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
