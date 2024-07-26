import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain # helps to setup context
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS # to perfrom similarity search - vectorstore database
from langchain_community.document_loaders import PyPDFLoader # to load pdfs from directories
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embedding techniques
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader
from langchain.schema import Document

from dotenv import load_dotenv
import os

load_dotenv()

# loading groq and google api keys
groq_api_key= os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')

# instantiating gemma model via groq API
llm = ChatGroq(api_key=groq_api_key, model="Gemma-7b-it")

# setting up prompt template
template ="""
You are a helpful and knowledgeable chatbot that answers questions based on the provided document. 
Your responses should stick to the context of the document and provide mid-size answers unless asked otherwise. 
If the answer is not available in the document, respond with "Answer not available in the document." 
Provide the most accurate response based on the question.

Context:
{context}

Question:
{input}

Answer:
"""
prompt =ChatPromptTemplate.from_template(template = template)

# function to create vector embeddings
def vector_embeddings(documents):

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # load a directory (with required documents)

        # all_text = []
        # for document in documents:
        #     loader = PyPDFLoader(document)
        #     docs = loader.load()
        #     all_text.extend(docs)

        st.session_state.all_text=""
        for pdf in documents:
            pdf_reader=PdfReader(pdf)
            for page in pdf_reader.pages:
                st.session_state.all_text+=page.extract_text()      
        # # creating a session state for the text extracted
        # st.session_state.all_texts = all_text

        # creating an instance of text splitter
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # splitting text
        split_text=st.session_state.text_splitter.split_text(st.session_state.all_text)
        st.session_state.final_documents = [Document(page_content=text) for text in split_text]
        # generating vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector store DB is ready!")

#  function to handle input prompt
def input_prompt():
    # input prompt
    if "vectors" in st.session_state:
        prompt1= st.text_input("Ask a question from the document(s)")

        import time
        if prompt1:   
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(response['answer'])

            # providing an expander
            with st.expander('document similarity search'):
                # find relevant chunks
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("---"*50)


    else:
        st.warning("Please create a vector store first")


# defining main function
def main():
    # title of the app page
    st.title('Gemma Model Document Q&A')

    # upload files
    st.write("### Upload your PDFs")

    docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=['pdf'])
    
    if docs:
        st.write("PDF(s) uploaded successfully!")
    
        # submit files and create a vector store
        if st.button("Create a vector store"):
            vector_embeddings(docs)

    # take and process input prompt
    if "vectors" in st.session_state:
        input_prompt()

if __name__ == "__main__":
    main()


    



