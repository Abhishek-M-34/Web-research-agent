# langgraph_agent.py - Research agent
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

# 1. Initialize the AI Model
llm = ChatGroq(model="llama-3.1-8b-instant")

# 2. Add the web search tool
tools = [DuckDuckGoSearchRun()]

# 3. Create the LangGraph agent
agent = create_react_agent(llm, tools)

print("=========================================================")
print("🌐 Welcome to the LangGraph Web Research Agent!")
print("   Ask me any question and I will search the web for it.")
print("   Type 'quit', 'exit', or 'q' to stop.")
print("=========================================================\n")

# 4. Interactive loop for the user
while True:
    user_input = input("🗣️ You: ")
    
    # Check if the user wants to quit
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\n👋 Goodbye! Have a great day!")
        break
        
    # Skip empty questions
    if not user_input.strip():
        continue
        
    print("\n⏳ Agent is thinking and searching the web...\n")
    
    try:
        # Run the agent with the user's input
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        
        # Get the final response from the agent
        response_content = result["messages"][-1].content
        
        # Format the output just in case the LLM returns a list instead of plain string
        if isinstance(response_content, list) and len(response_content) > 0 and isinstance(response_content[0], dict):
            response_text = response_content[0].get("text", str(response_content))
        else:
            response_text = response_content
            
        print(f"🤖 Agent:\n{response_text}\n")
        
    except Exception as e:
        print(f"⚠️ Oops! Something went wrong: {e}\n")

    print("-" * 57)
