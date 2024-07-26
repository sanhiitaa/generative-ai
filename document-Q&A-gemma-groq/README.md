# Chat with Your PDFs!

This Streamlit app leverages the power of Langchain, Groq's Gemma LLM, and Google Generative AI Embeddings to transform your PDFs into interactive Q&A sessions.

[**Check out the app here**](https://chat-with-multiple-pdfs-09.streamlit.app/)

## Key Features:

**Handle multiple PDFs**: Easily upload multiple PDFs for analysis.
**Rapid Vector Store Creation**: Efficiently generates vector embeddings using FAISS for lightning-fast similarity search.
**Intelligent Question Answering**: Ask precise questions and receive accurate, context-aware answers from the provided documents.
**Transparent Contextual Understanding**: Explore relevant document excerpts for enhanced comprehension.
**Cutting-Edge Technology Stack**: Built with Langchain, Groq's Gemma LLM, Google Generative AI Embeddings, Streamlit, and FAISS for optimal performance.

## How it Works:

1. Upload your PDFs.
2. Create a vector store to index document content efficiently.
3. Ask your questions and get informative answers based on the provided PDFs.
4. Dive deeper into the context by exploring relevant document snippets.

## Technologies Used:

* **Streamlit**: For building the interactive web interface.
* **Langchain**: For handling document processing and chaining.
* **PyPDF2**: For extracting text from PDF files.
* **FAISS**: For creating and managing the vector store for similarity searches.
* **Google Generative AI Embeddings**: For generating vector embeddings.
* **ChatGroq (Gemma-7b-it)**: As the language model for generating responses.
* **dotenv**: For managing API keys and environment variables.

**Note**: Requires Groq and Google Cloud API keys.
