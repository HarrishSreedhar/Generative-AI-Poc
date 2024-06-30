# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from datetime import datetime
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

st.header("Chatbot")
st.title("File Uploader")
os.environ["OPENAI_API_KEY"] = "test-key"

pdf_file = st.file_uploader("Upload any PDF", type="pdf")
if pdf_file is not None:
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap  = 0,
    length_function = len,)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Enter your question")
    if user_question:
        
        matched_data = vector_store.similarity_search(user_question)
        st.write(matched_data)
        LLM = ChatOpenAI(
            openai_api_key = OPEN_AI_KEY,
            temperature = 0,
            max_tokens = 100,
            model_name = "gpt-3.5-turbo")

        chain = load_qa_chain(LLM, chain_type="stuff")
        response = chain.run(input = matched_data, question = user_question)
        st.write(response)
    else:
        st.write("Kindly enter a valid prompt get a response)
else:
    st.write("Kindly upload a valid PDF to get a response)
