import streamlit as st
import json
import requests
import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import (
    messages_from_dict,
    messages_to_dict,
    SystemMessage,
    HumanMessage
)
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: white; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    h1 { color: #38bdf8; }
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border-radius: 10px;
        border: none;
        width: 100%;
        padding: 12px;
        font-size: 14px;
    }
    .stButton > button:hover { background-color: #2563eb; color: white; }
</style>
""", unsafe_allow_html=True)

#API KEYS

API_KEY = st.secrets.get("GROQ_API_KEY", "")
MEMORY_FILE = "travel_memory.json"

# MEMORY

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return messages_from_dict(json.load(f))
    except:
        return []

def save_memory(chat_history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(messages_to_dict(chat_history), f)
    except:
        pass

def trim_history(history, max_messages=4):
    system_msgs = [m for m in history if isinstance(m, SystemMessage)]
    other_msgs = [m for m in history if not isinstance(m, SystemMessage)]
    return system_msgs + other_msgs[-max_messages:]
    
# ALL TOOLS

@tool
def weather(city: str):
    """Check current weather of a city"""
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
        if r.status_code == 200:
            c = r.json()["current_condition"][0]
            return f"Weather in {city}: {c['temp_C']}C, {c['weatherDesc'][0]['value']}, Humidity: {c['humidity']}%, Wind: {c['windspeedKmph']} kmph"
        return "City not found"
    except Exception as e:
        return f"Weather not available: {str(e)}"

@tool
def search_travel(query: str):
    """Search for any travel related information including flights, trains, hotels and prices"""
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result

@tool
def itinerary(tocity: str, days: str, budget: str):
    """Generate a day by day travel plan for a city."""
    search = DuckDuckGoSearchRun()
    result = search.run(f"best {days} day itinerary for {tocity} tourist places to visit")
    return result

# PROMPT

SYSTEM_MESSAGE = """You are a helpful travel agent assistant.

RULES:
1. For weather questions use the weather tool directly.
2. For flights, trains, hotels and prices ALWAYS use search_travel tool with a specific search query.
3. For itinerary use the itinerary tool.
4. NEVER show raw JSON or function calls to user.
5. Always answer in clean simple English.
6. For train prices search: "train ticket price from X to Y"
7. For flight prices search: "flight ticket price from X to Y"
8. For hotels search: "best hotels in X with price"
"""

# STREAMLIT
@st.cache_resource
def get_agent():
    llm = ChatGroq(api_key=API_KEY, model="llama-3.3-70b-versatile")
    tools = [weather, search_travel, itinerary]
    return create_react_agent(llm, tools)

if "chat_history" not in st.session_state:
    history = load_memory()
    if not any(isinstance(m, SystemMessage) for m in history):
        history.insert(0, SystemMessage(content=SYSTEM_MESSAGE))
    st.session_state.chat_history = history

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

with st.sidebar:
    st.markdown("## ✈️ Travel Agent")
    st.markdown("---")
    st.markdown("**What I can help with:**")
    st.markdown("🌤️ Live weather")
    st.markdown("🚆 Train prices")
    st.markdown("✈️ Flight prices")
    st.markdown("🏨 Hotel search")
    st.markdown("🗺️ Trip itinerary")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [SystemMessage(content=SYSTEM_MESSAGE)]
        st.session_state.display_messages = []
        st.session_state.prefill = ""
        st.rerun()
    st.markdown("---")
    st.caption("Built with LangGraph + Groq + Streamlit")

st.markdown("# ✈️ Travel Agent")
st.markdown("Your smart travel planning assistant — flights, trains, hotels and more.")
st.markdown("---")

st.markdown("### 💡 Try these:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🌤️ Weather in Chennai"):
        st.session_state.prefill = "What is the current weather in Chennai?"
with col2:
    if st.button("🗺️ Plan trip to Norway for 10 days"):
        st.session_state.prefill = "Plan a 10 day trip to Norway with full itinerary"
with col3:
    if st.button("🚆 Train price Chennai to Delhi"):
        st.session_state.prefill = "What is the current train ticket price from Chennai to Delhi?"
with col4:
    if st.button("✈️ Flight price Mumbai to Bangkok"):
        st.session_state.prefill = "What is the current flight ticket price from Mumbai to Bangkok?"

st.markdown("---")

for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

user_input = st.chat_input("Ask me about your trip...")

if st.session_state.prefill != "":
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

if user_input:
    agent_executor = get_agent()
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    trimmed = trim_history(st.session_state.chat_history, max_messages=4)
    with st.spinner("Searching for you..."):
        try:
            response = agent_executor.invoke({"messages": trimmed})
            st.session_state.chat_history = response["messages"]
            answer = st.session_state.chat_history[-1].content
            save_memory(st.session_state.chat_history)
            st.session_state.display_messages.append({"role": "agent", "content": answer})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.pop()
    st.rerun()
