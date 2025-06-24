import streamlit as st
st.set_page_config(page_title="IAS Search Bot", layout="wide")
import os
import json

from query_engine import handle_user_query, refine_query, rollback_last_step
from session_manager import SessionManager

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
        .fallback-banner {
            background-color: #ffa726;
            color: black;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            margin: 12px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🎯 IAS Officer Search Bot")

# --- Session Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session" not in st.session_state:
    st.session_state.session = SessionManager()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = None
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "feedback_action" not in st.session_state:
    st.session_state.feedback_action = None
if "last_fallback_notice" not in st.session_state:
    st.session_state.last_fallback_notice = None

# --- Handle feedback-triggered rerun ---
if st.session_state.feedback_action and st.session_state.last_query:
    with st.spinner("Reprocessing with fallback..."):
        response_markdown, result, source = handle_user_query(
            st.session_state.last_query,
            st.session_state.session,
            user_feedback=st.session_state.feedback_action
        )
        # Determine fallback banner
        if source == "raw":
            st.session_state.last_fallback_notice = "⚠️ GOLD results insufficient, falling back to RAW..."
        elif source == "web":
            st.session_state.last_fallback_notice = "🌐 RAW results insufficient, falling back to WEB..."
        else:
            st.session_state.last_fallback_notice = None

        st.session_state.messages = []  # Auto-collapse history
        st.session_state.messages.append({"role": "user", "content": st.session_state.last_query})
        st.session_state.messages.append({"role": "bot", "content": response_markdown})
        st.session_state.last_result = result
        st.session_state.last_confidence = "manual_feedback"
        st.session_state.last_source = source
        st.session_state.feedback_action = None
        st.session_state.awaiting_feedback = (source in {"gold", "raw"})
        st.rerun()

# --- Display Messages ---
if st.session_state.last_fallback_notice:
    st.markdown(f"<div class='fallback-banner'>{st.session_state.last_fallback_notice}</div>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# --- Show feedback buttons if applicable ---
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
            else:
                st.session_state.feedback_action = None
            st.session_state.awaiting_feedback = False
            st.rerun()

# --- Chat Input ---
user_input = st.chat_input("Enter your officer search query or type 'undo' to rollback")

# --- Handle New Input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    is_refine = st.session_state.last_result is not None
    st.session_state.last_query = user_input

    with st.spinner("Processing..."):
        if is_refine and user_input.lower().strip() in {"undo", "rollback"}:
            response_markdown, restored_result = rollback_last_step(st.session_state.session)
            st.session_state.last_result = restored_result
            st.session_state.last_confidence = "restored"
            st.session_state.last_source = "restored"
            st.session_state.last_fallback_notice = None
        elif is_refine:
            response_markdown, result, source = refine_query(
                user_input, st.session_state.last_result, st.session_state.session
            )
            st.session_state.last_result = result
            st.session_state.last_confidence = "refined"
            st.session_state.last_source = source
            st.session_state.last_fallback_notice = None
        else:
            response_markdown, result, source = handle_user_query(
                user_input,
                st.session_state.session,
                user_feedback=st.session_state.user_feedback
            )
            # Set fallback banners only if fallback happened
            if source == "raw":
                st.session_state.last_fallback_notice = "⚠️ GOLD results insufficient, falling back to RAW..."
            elif source == "web":
                st.session_state.last_fallback_notice = "🌐 RAW results insufficient, falling back to WEB..."
            else:
                st.session_state.last_fallback_notice = None

            st.session_state.last_result = result
            st.session_state.last_confidence = "initial"
            st.session_state.last_source = source

    st.session_state.messages.append({"role": "bot", "content": response_markdown})
    st.session_state.user_feedback = None
    st.session_state.awaiting_feedback = (st.session_state.last_source in {"gold", "raw"})
    st.rerun()
