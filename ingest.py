from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document as LC_Document
from utils import read_pdf, read_docx
import os
import re

def load_and_split_document(file_path):
    if file_path.endswith(".pdf"):
        raw_text = read_pdf(file_path)
    elif file_path.endswith(".docx"):
        raw_text = read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")


def ingest_file(file_path, persist_directory="data/chroma"):
    documents = load_and_split_document(file_path)
    if not documents:
        raise ValueError("No content extracted from document.")
    embedding_model = OllamaEmbeddings(model="llama3")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    vectordb.add_documents(documents)
    return len(documents)
