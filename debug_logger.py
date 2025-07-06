import streamlit as st

def log(message: str):
    print(f"[DEBUG] {message}")
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(message)

def reset_logs():
    st.session_state.debug_logs = []

def get_logs():
    return '\n'.join(st.session_state.get("debug_logs", []))
