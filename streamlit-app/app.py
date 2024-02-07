import json
import uuid
import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:5000"
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.uploaded_file_name = ""

# Region UI

st.write("RAG Chat bot")
with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload a file to query using AI")
        submitted = st.form_submit_button("Submit")

# Render the chat history onto the user's screen
for chat_message in st.session_state.chat_history:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

if prompt := st.chat_input("Enter a message..."):
    with st.chat_message(USER):
        st.markdown(prompt)
    st.session_state.chat_history.append({
        "role": USER,
        "content": prompt
    })
    
    # Send a request to the flask server to generate a response
    request_body = {
        "user_input": prompt,
        "rag": True
    }
    response = requests.post(
        url=f"{BASE_URL}/chat", 
        data=json.dumps(request_body),
        headers={'Content-Type': 'application/json'}
    )
    response_json = response.json()

    # Check that the response did not generate an error
    if not response_json["error"]:
        # Render GPT's response onto the user's screen
        assistant_message = response_json["message"]
        with st.chat_message(ASSISTANT):
            st.markdown(assistant_message)
        st.session_state.chat_history.append({
            "role": ASSISTANT,
            "content": assistant_message
        })
    else:
        st.exception(response_json["message"])

# End Region UI

if uploaded_file and (uploaded_file.name != st.session_state.uploaded_file_name):
    files = {'file': uploaded_file}
    response = requests.post(
        url=f"{BASE_URL}/files", 
        files=files,
    )
    response_json = response.json()

    if not response_json["error"]:
        response_message = response_json["message"]
        st.success(response_message)
        st.session_state.uploaded_file_name = uploaded_file.name
    else:
        st.exception(response_json["message"])

