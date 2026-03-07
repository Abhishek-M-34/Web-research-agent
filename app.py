# streamlit_app.py - Deployment file for Research agent
import streamlit as st
from langgraph.prebuilt import create_react_agent

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

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    model_provider = st.selectbox(
        "Select AI Provider:",
        ("Google Gemini", "OpenAI", "Anthropic (Claude)", "Groq", "DeepSeek")
    )
    
    if model_provider == "Google Gemini":
        api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
        if not api_key:
            with st.expander("ℹ️ How to get a Gemini API Key"):
                st.markdown('''
                1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
                2. Sign in with your Google account.
                3. Click **Create API key**.
                4. Copy the generated key and paste it above.
                ''')
    elif model_provider == "OpenAI":
        st.caption("⚠️ **Note:** OpenAI API is a paid service. You must add credits to your OpenAI account for the key to work.")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if not api_key:
            with st.expander("ℹ️ How to get an OpenAI API Key"):
                st.markdown('''
                1. Go to the [OpenAI API Keys page](https://platform.openai.com/api-keys).
                2. Sign in or sign up.
                3. Set up a payment method under Billing.
                4. Click **Create new secret key**.
                5. Copy the key and paste it above.
                ''')
    elif model_provider == "Anthropic (Claude)":
        st.caption("⚠️ **Note:** Anthropic API is a paid service. You must add credits to your Anthropic console for the key to work.")
        api_key = st.text_input("Enter your Anthropic API Key:", type="password")
        if not api_key:
            with st.expander("ℹ️ How to get an Anthropic API Key"):
                st.markdown('''
                1. Go to the [Anthropic Console](https://console.anthropic.com/settings/keys).
                2. Sign in or sign up.
                3. Add funds in the Billing section.
                4. Click **Create Key**.
                5. Copy the key and paste it above.
                ''')
    elif model_provider == "Groq":
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if not api_key:
            with st.expander("ℹ️ How to get a Groq API Key"):
                st.markdown('''
                1. Go to the [Groq Console](https://console.groq.com/keys).
                2. Sign in or sign up.
                3. Click **Create API Key**.
                4. Copy the key and paste it above.
                ''')
    elif model_provider == "DeepSeek":
        api_key = st.text_input("Enter your DeepSeek API Key:", type="password")
        if not api_key:
            with st.expander("ℹ️ How to get a DeepSeek API Key"):
                st.markdown('''
                1. Go to the [DeepSeek Platform](https://platform.deepseek.com/).
                2. Sign in or register.
                3. Navigate to the API Keys section.
                4. Click **Create new API key**.
                5. Copy the key and paste it above.
                ''')

if not api_key:
    st.warning(f"⚠️ Please enter your {model_provider} API Key in the sidebar to continue.")
    st.stop()

# --- Initialize Agent (Cached to avoid re-initializing on every interaction) ---
@st.cache_resource
def get_agent(provider, api_key_str):
    # 1. Initialize the AI Model
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key_str)
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key_str)
    elif provider == "Anthropic (Claude)":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=api_key_str)
    elif provider == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama3-70b-8192", api_key=api_key_str)
    elif provider == "DeepSeek":
        from langchain_openai import ChatOpenAI
        # DeepSeek uses an OpenAI-compatible API
        llm = ChatOpenAI(model="deepseek-chat", api_key=api_key_str, base_url="https://api.deepseek.com/v1")
    # 2. Add the web search tool
    tools = [DuckDuckGoSearchRun()]
    # 3. Create the LangGraph agent
    return create_react_agent(llm, tools)

agent = get_agent(model_provider, api_key)

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
