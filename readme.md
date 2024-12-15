# Llama Index Chatbot

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-blue.svg)

> You can find the tutorial for this project on [YouTube] in the future if there's enough interest (TBD)

## Introduction
------------
The **Llama Index Chatbot App (v1)** is a Python-based application that enables interactive conversations with multiple PDF documents. Users can pose questions in natural language, and the chatbot provides relevant answers derived from the content of the loaded PDFs. This tool leverages advanced language models to ensure accurate and contextually appropriate responses. Note that the chatbot's knowledge is confined to the uploaded documents.

## How It Works
------------

The application operates through the following steps:

1. **Directory Loading:** The app scans a specified directory to read multiple PDF documents and extracts their textual content.
2. **Text Chunking:** The extracted text is segmented into manageable chunks to facilitate efficient processing.
3. **Language Model Integration:** Utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. **Similarity Matching:** When a user asks a question, the app compares it against the text chunks to identify the most semantically similar sections.
5. **Response Generation:** The selected text chunks are fed into the language model to generate a coherent and relevant response based on the PDF content.

## Features
------------
- **Multiple PDF Support:** Load and interact with numerous PDF documents simultaneously.
- **Natural Language Understanding:** Ask questions in plain English and receive precise answers.
- **Chat History:** Maintain a record of all interactions for easy reference.
- **Efficient Indexing:** Rapid setup and indexing for quick query responses.
- **Scalable Embeddings:** Utilizes robust embedding models to understand and process text effectively.

## Dependencies and Installation
----------------------------
Follow these steps to set up the Llama Index Chatbot App on your local machine:

### **1. Clone the Repository**
git clone https://github.com/untucked/llama-index-chatbot-v1.git
cd llama-index-chatbot-v1
### **2. Run the Application
bash
streamlit run main.py

### **3. Interact with the App

The application will open in your default web browser.
Load Documents: Enter the directory path containing your PDF documents and click "Load Documents."
Chat: Once documents are loaded, use the chat interface to ask questions related to the content of the PDFs.