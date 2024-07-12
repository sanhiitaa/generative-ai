from dotenv import load_dotenv
load_dotenv() # loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai


genai.configure(api_key=os.getenv('gemini_api_key'))

# function to load geminip pro model and get responses
model=genai.GenerativeModel('gemini-pro')
def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text


# initializing streamlit app

st.set_page_config(page_title='Question & Answer demo')
st.header('Gemini Q&A Web Application')
input = st.text_input("Ask your question here: ", key="input")
submit=st.button("Ask")
# when submit is clicked
if submit:
    response = get_gemini_response(input)
    st.subheader("Response: ")
    st.write(response)