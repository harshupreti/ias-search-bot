import streamlit as st
st.set_page_config(page_title="IAS Search Bot", layout="wide")
import os

from query_engine import handle_user_query, refine_query, rollback_last_step
from session_manager import SessionManager
from debug_logger import get_logs

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --- Custom CSS ---
st.markdown("""
    <style>
        .bot-message {
            background-color: #1e1e1e;
            color: #e0e0e0;
            padding: 12px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #2c2f33;
            color: #ffffff;
            padding: 12px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 IAS Officer Search Bot")
st.write("This is v3")

# --- Session State Setup ---
for key, default in {
    "messages": [],
    "session": SessionManager(),
    "last_result": None,
    "last_confidence": None,
    "last_source": None,
    "user_feedback": None,
    "awaiting_feedback": False,
    "last_query": None,
    "feedback_action": None,
    "processing_stage": None,
    "debug_logs": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# --- Feedback-triggered fallback rerun ---
if st.session_state.feedback_action and st.session_state.last_query:
    user_input = st.session_state.last_query
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_confidence = None

    st.chat_message("user").markdown(user_input)

    stage = "🔎 Searching in Supremo dataset (RAW)..." if st.session_state.feedback_action == "force_raw" else "🔎 Searching the Web..."
    with st.spinner(stage):
        out = handle_user_query(
            user_input,
            st.session_state.session,
            user_feedback=st.session_state.feedback_action
        )
        response_markdown, result, source, debug_logs = out

    st.chat_message("user").markdown(user_input)
    st.session_state.last_result = result
    st.session_state.last_confidence = "manual_feedback"
    st.session_state.last_source = source
    st.session_state.awaiting_feedback = source in {"gold", "raw"}
    st.session_state.processing_stage = None
    st.session_state.debug_logs = debug_logs if source == "gold" else None
    st.session_state.messages.append({"role": "bot", "content": response_markdown})
    st.session_state.feedback_action = None
    st.rerun()

# --- Handle normal queries / refinement / rollback ---
elif st.session_state.last_query and st.session_state.last_confidence is None:
    user_input = st.session_state.last_query
    is_refine = st.session_state.last_result is not None

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    if is_refine and user_input.lower().strip() in {"undo", "rollback"}:
        st.session_state.processing_stage = "rollback"
        with st.spinner("🔁 Rolling back to previous result..."):
            response_markdown, restored_result = rollback_last_step(st.session_state.session)
        st.chat_message("user").markdown(user_input)
        st.session_state.last_result = restored_result
        st.session_state.last_confidence = "restored"
        st.session_state.last_source = "restored"
        st.session_state.debug_logs = None

    elif is_refine:
        st.session_state.processing_stage = "refine"
        with st.spinner("🧬 Refining your query..."):
            response_markdown, result, source, debug_logs = refine_query(
                user_input,
                st.session_state.last_result,
                st.session_state.session
            )
        st.chat_message("user").markdown(user_input)
        st.session_state.last_result = result
        st.session_state.last_confidence = "refined"
        st.session_state.last_source = source
        st.session_state.awaiting_feedback = source in {"gold", "raw"}
        st.session_state.debug_logs = debug_logs if source == "gold" else None

    else:
        st.session_state.processing_stage = "gold"
        with st.spinner("🔹 Searching in Supremo dataset..."):
            response_markdown, result, source, debug_logs = handle_user_query(
                user_input,
                st.session_state.session
            )
        st.chat_message("user").markdown(user_input)
        st.session_state.last_result = result
        st.session_state.last_confidence = "initial"
        st.session_state.last_source = source
        st.session_state.awaiting_feedback = source in {"gold", "raw"}
        st.session_state.debug_logs = debug_logs if source == "gold" else None

    st.session_state.processing_stage = None
    st.session_state.messages.append({"role": "bot", "content": response_markdown})
    st.session_state.user_feedback = None
    st.rerun()

# --- Feedback Buttons ---
if st.session_state.awaiting_feedback and st.session_state.last_source != "web":
    st.markdown("---")
    st.markdown("**Was this response helpful?**")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("👍 Satisfied"):
            st.session_state.user_feedback = None
            st.session_state.awaiting_feedback = False
            st.session_state.feedback_action = None
            st.rerun()

    with col2:
        if st.button("👎 Not satisfied, try again"):
            if st.session_state.last_source == "gold":
                st.session_state.feedback_action = "force_raw"
            elif st.session_state.last_source == "raw":
                st.session_state.feedback_action = "force_web"
            st.session_state.awaiting_feedback = False
            st.rerun()

# --- Internal Debug Logs (GOLD only) ---
if st.session_state.debug_logs:
    with st.expander("🧠 View Internal Logic (Debug Logs)", expanded=False):
        for line in st.session_state.debug_logs.splitlines():
            st.markdown(f"- {line}")

# --- Input Box ---
user_input = st.chat_input("Enter your officer search query or type 'undo' to rollback")
if user_input:
    st.session_state.last_query = user_input
    st.session_state.last_confidence = None
    if st.session_state.last_result and user_input.lower().strip() in {"undo", "rollback"}:
        st.session_state.processing_stage = "rollback"
    elif st.session_state.last_result:
        st.session_state.processing_stage = "refine"
    else:
        st.session_state.processing_stage = "gold"
    st.rerun()
