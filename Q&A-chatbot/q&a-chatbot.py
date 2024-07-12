# to initialize environment vairables
from dotenv import load_dotenv
load_dotenv()

# essential libraries
import streamlit as st
import os 
import google.generativeai as genai

# loading API key
genai.configure(api_key=os.getenv("gemini_api_key"))

# function to load gemini pro model and store response
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


# initializing the streamlit app
st.set_page_config(page_title='Q&A Chatbot')
st.header('Q&A Chatbot powered by Gemini')

# initializing session state to save chat history 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# recieving input
input = st.text_input('input: ', key='input')
submit = st.button('ask question')

# generating a response from the given input
if submit and input:
    response = get_gemini_response(input)
    
    # saving chat history
    st.session_state['chat_history'].append(('YOU', input))
    st.subheader('response: ')
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('BOT', chunk.text))

# displaying chat history
st.write("**********************************************")
st.subheader('chat history')
for role, text in st.session_state['chat_history']:
    st.write(f'{role} : {text}')
    st.write('***************************************')

