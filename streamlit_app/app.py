# streamlit_app/app.py
import sys
import os

# Add project root to sys.path BEFORE importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from utils.logger import logger
from agents.azure_agent import initialize_azure_agent
from agents.tools import TOOLS

st.set_page_config(page_title="Shipment Chatbot", layout="wide")
st.title("🚢 Azure‑Powered Shipping Chatbot")

# ----------------------------------------------------------------------
# Initialise the agent once (cached between Streamlit reruns)
# ----------------------------------------------------------------------
@st.cache_resource
def get_agent():
    agent, _ = initialize_azure_agent(TOOLS)
    return agent

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []


def submit():
    user_msg = st.session_state.user_input
    if not user_msg:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.user_input = ""

    try:
        answer = agent.run(user_msg)
    except Exception as exc:
        logger.exception("Agent error")
        answer = f"⚠️  Something went wrong: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

st.text_input(
    label="Ask a question about shipments …",
    key="user_input",
    on_change=submit,
    placeholder="e.g. 'What is the ETA for container ABCD1234567?'",
)
