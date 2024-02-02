import streamlit as st

st.write("RAG Chat bot")
uploaded_file = st.file_uploader("Upload a file to query using AI")

if uploaded_file is not None:
    # TODO: Add function to chunk, and insert into vector DB
    print(uploaded_file.name)