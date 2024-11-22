import streamlit as st
import requests
import pyttsx3
import speech_recognition as sr
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties for the TTS engine (you can adjust these as you like)
engine.setProperty('rate', 150)  # Speed of speech (higher means faster)
engine.setProperty('volume', 1)  # Volume (from 0.0 to 1.0)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to listen to the user's speech and convert it to text
def listen_to_user():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening for your command...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.write(f"You said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            st.write("Sorry, the speech recognition service is unavailable.")
            return ""

# Streamlit app title
st.title("Chat with B.H.A.U 🤖 (Voice Input/Output)")

# Initialize chat history if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Button to trigger voice input
if st.button("Click to Speak"):
    user_input = listen_to_user()
    
    if user_input:
        # Display the user input in the chat
        with st.chat_message("user"):
            st.markdown(user_input)

        # Append the user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Send the user query to the API
        response = requests.post("http://127.0.0.1:8000/bhau_api", json={"query": user_input})
        response = response.json()
        assistant_reply = response["result"]

        # Display the assistant's reply in the chat
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        # Append the assistant's reply to session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Trigger text-to-speech for the assistant's response
        engine.say(assistant_reply)
        engine.runAndWait()

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
