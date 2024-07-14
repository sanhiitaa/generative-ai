import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


from langchain_google_genai import GoogleGenerativeAIEmbeddings # provides embeddings from google generative AI for text data
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # for vector embeddings for similarity search
from langchain_google_genai import ChatGoogleGenerativeAI #facilitates chat interactions with google generative AI
from langchain.chains.question_answering import load_qa_chain # loads a chain for question asnwering tasks
from langchain.prompts import PromptTemplate # constructs and manages custom prompt templates for various tasks
from dotenv import load_dotenv # loads environment variables from a .env file

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# function to read and extract text from a list of pdfs
def get_pdf_text(pdf_doc):
    text=""
    for pdf in pdf_doc:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# function to divide the text into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# creating vector store from text chunks
def get_vector_store(text_chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store= FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

# creating conversational chain by defining prompt_template
def get_conversational_chain():
    # prompt template
    prompt_template="""
                        Based on the following context from the provided PDF document, answer the question:

    Context:
    \n {context} \n

    Question:
    \n {question} \n

    Answer:

    Guidelines:
    - Answer Length: Provide mid-range answers by default. If a user explicitly requests a longer answer, then provide a detailed response.
    - Contextual Accuracy: Ensure that all answers are accurate and relevant to the content in the PDF. If the information needed to answer a question is not available in the PDF, clearly state that the answer is not available.
    - Context Adherence: Stick strictly to the context of the PDF. Do not fabricate or infer information beyond what is provided in the document.

    Example:

    User Question: 'What are the main objectives of the project outlined in the document?'

    Expected Response: 'The main objectives of the project are [summarize objectives based on the PDF].'

    User Question: 'Can you provide a detailed explanation of the methodology used?'

    Expected Response: 'The methodology used involves [detailed explanation from the PDF].'

    If the answer to a question is not in the document:

    Response: 'The information required to answer this question is not available in the PDF.'
    Be precise and ensure that all responses are based on the information contained in the PDF.
    """
                    
    # initialize the model to generate responses
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)

    # create prompt template object with the defined prompt
    prompt= PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    # load the question-answering chain using the model and the prompt template
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# function to generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs =  new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents" : docs, "question" : user_question},
        return_only_outputs=True
    )
    st.write('Reply: ', response["output_text"])

# defining main
def main():
        st.set_page_config(page_title='Chat with multiple PDFs')
        st.header('Chat with PDFs : Powered By Gemini')

        user_question = st.text_input("Ask a question from the uploaded PDF file(s)")
        submit = st.button("Submit")
        if user_question and submit:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu: ")
            pdf_docs = st.file_uploader("upload your PDF files", accept_multiple_files=True, type=['pdf'])
            if st.button ("submit and process"):
                if pdf_docs:
                    with st.spinner('preprocessing...'):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks= get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success('Done')
                else:
                    st.warning("Please upload PDF files before processing")

if __name__=="__main__":
    main()
