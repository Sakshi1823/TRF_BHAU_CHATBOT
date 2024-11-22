import streamlit as st
import requests

# st.title("Chat with B.H.A.U 🤖")
st.markdown("<h1 style='text-align: center;'>Chat with B.H.A.U 🤖 <small>(Beta Version)</small></h1>", unsafe_allow_html=True)

#initialise the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#display previous messages
for message in  st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("hello, whats up!")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    response = requests.post("http://127.0.0.1:8000/bhau_api", json={"query": prompt})
    response = response.json()
    response = response["result"]
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant", "content": response})


st.markdown("""
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .chat-message {
            display: flex;
            flex-direction: column;
            margin: 10px 0;
        }

        .chat-message .user {
            background-color: #D1F7C4;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            align-self: flex-start;
        }

        .chat-message .assistant {
            background-color: #D1E1F7;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            align-self: flex-end;
        }

        .stTextInput input {
            border-radius: 20px;
            padding: 15px;
            font-size: 16px;
            border: 1px solid #e1e1e1;
        }

        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 20px;
            border: none;
        }

        .stButton button:hover {
            background-color: #45a049;
        }

        .stMarkdown {
            font-family: "Arial", sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
