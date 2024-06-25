from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader

st.header("Chatbot")
st.title("File Uploader")
pdf_file = st.file_uploader("Upload any PDF", type="pdf")


if pdf_file is not None:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        st.write(text)
