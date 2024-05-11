import streamlit as st
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
from PyPDF2 import PdfReader, PdfWriter

# Load the model & tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function to split PDF into individual pages and return paths
def split_pdfs(input_file_path):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    out_paths = []
    with open(input_file_path, "rb") as file_handle:
        inputpdf = PdfReader(file_handle)
        for i, page in enumerate(inputpdf.pages):
            output = PdfWriter()
            output.add_page(page)
            base_name = os.path.splitext(os.path.basename(input_file_path))[0]
            out_file_path = os.path.join("outputs", f"{base_name}_{i}.pdf")
            with open(out_file_path, "wb") as output_stream:
                output.write(output_stream)
            out_paths.append(out_file_path)
    return out_paths

# Text extraction and preprocessing for summarization
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_text = " ".join(text.page_content for text in texts)
    return final_text, len(final_text)

# Summarization pipeline for each PDF page
def llm_pipeline(filepath):
    input_text, input_length = file_preprocessing(filepath)
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=input_length // 8, 
        min_length=25)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# Streamlit app setup
st.set_page_config(page_title='PDF Insight', layout="wide", page_icon="📃", initial_sidebar_state="expanded")
st.title("PDF Insight")

uploaded_file = st.file_uploader("Upload the PDF", type=['pdf'], key="unique_file_uploader")
if uploaded_file is not None:
    filepath = "uploaded_pdfs/" + uploaded_file.name
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())
    
    if st.button("Summarize Each Page"):
        out_paths = split_pdfs(filepath)
        summaries = {}
        for page_path in out_paths:
            page_summary = llm_pipeline(page_path)
            page_number = os.path.basename(page_path).split('_')[-1].replace('.pdf', '')
            summaries[page_number] = page_summary

        for page, summary in summaries.items():
            st.subheader(f"Summary for Page {page}:")
            st.write(summary)
