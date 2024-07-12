# loading environment variables

from dotenv import load_dotenv
load_dotenv()

# import libraries
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

genai.configure(api_key= os.getenv('gemini_api_key'))

# function to load gemini model
model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_response(image, input, prompt): # the order here will have to be the same as streamlit app initialization order
    response = model.generate_content([input, image[0], prompt])
    return response.text

#  function to
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # read the file into bytes
        bytes_data= uploaded_file.getvalue()

        image_parts = [
            {
                'mime_type': uploaded_file.type,
                'data': bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError('no file uploaded.')


# initializing streamlit app
st.set_page_config(page_title='Multi Language Invoice Extractor')

st.header('Multi Language Invoice Extractor')

# uploading the file and displaying it
uploaded_file=st.file_uploader('Upload an invoice image:', type=['jpg', 'jpeg', 
'png'])
if uploaded_file is not None:
   image = Image.open(uploaded_file)
   st.image(image, caption='Uploaded file.', use_column_width=True) 

# entering prompt
input= st.text_input("Ask your question wrt the invoice: ", key='input')

submit = st.button('Ask question')

input_prompt = '''
you are an expert in understanding invoices. the user will upload an image of the invoice
and you will have to carefully scan through the image and provide answers to any question the user asks
wrt to the invoice provided.'''

# if submit button clicked
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(image_data, input_prompt, input)
    st.subheader('the response is:')
    st.write(response)