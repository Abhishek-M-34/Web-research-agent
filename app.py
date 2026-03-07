# streamlit_app.py - Deployment file for Research agent
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="LangGraph Web Research Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🌐 LangGraph Web Research Agent")
st.markdown("Ask me any question and I will search the web for it using AI and DuckDuckGo!")

# Groq API key is loaded from .env automatically by ChatGroq
# Ensure GROQ_API_KEY is in your .env file

# --- Initialize Agent (Cached to avoid re-initializing on every interaction) ---
@st.cache_resource
def get_agent():
    # 1. Initialize the AI Model
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    # 2. Add the web search tool
    tools = [DuckDuckGoSearchRun()]
    # 3. Create the LangGraph agent
    return create_react_agent(llm, tools)

agent = get_agent()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Logic ---
if prompt := st.chat_input("What is the latest news about diffusion models?"):
    # Display user response in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("⏳ Researching the web..."):
                # Run the agent with the user's input
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                
                # Get the final response from the agent
                response_content = result["messages"][-1].content
                
                # Format the output
                if isinstance(response_content, list) and len(response_content) > 0 and isinstance(response_content[0], dict):
                    full_response = response_content[0].get("text", str(response_content))
                else:
                    full_response = str(response_content)

            # Display the result
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"⚠️ Oops! Something went wrong: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
