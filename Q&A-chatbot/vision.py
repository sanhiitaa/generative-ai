from dotenv import load_dotenv
load_dotenv() # loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image


genai.configure(api_key=os.getenv('gemini_api_key'))
# function to load geminip pro model and get responses
model=genai.GenerativeModel('gemini-pro-vision')
def get_gemini_response(input, image):
    if input!="":
        response = model.generate_content([input,image])
    else:
        response = model.generate_content(image)
    return response.text

# initializing streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Image Summary Generator Web App")
st.write("You can tailor the response wrt the image.")
input=st.text_input("Input Prompt: ", key="input")

# image upload option
uploaded_file = st.file_uploader("Upload an image: ", type=['jpeg', 'jpg', 'png'])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded image ", use_column_width=True)



submit=st.button("Generate a response")

# if submit clicked

if submit:
    response=get_gemini_response(input, image)
    st.subheader("Response:")
    st.write(response)