import time
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vectorstore(chunks, pdf_name):
    db_folder = 'db'
    pkl_path = os.path.join(db_folder, f"{pdf_name}.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            VectorStore = pickle.load(f)
        st.caption('Embeddings loaded locally.')
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        with open(pkl_path, "wb") as f:
            pickle.dump(VectorStore, f)
        st.caption("New embeddings computed.")

    return VectorStore

def get_conversation_chain(select_llm):
    if select_llm == 'oasst-sft-4':
        select_llm = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
    elif select_llm == 'flan-t5-xxl':
        select_llm = 'google/flan-t5-xxl'
    elif select_llm == 'falcon-7b':
        select_llm = 'tiiuae/falcon-7b-instruct'

    llm = HuggingFaceHub(repo_id=select_llm, model_kwargs={"temperature": 0.3, "max_length": 200})
    return load_qa_chain(llm=llm, chain_type="stuff")

def handle_user_input(query, VectorStore, select_llm):
    query = f"<|prompter|>{query}<|endoftext|><|assistant|>" 
    
    docs = VectorStore.similarity_search(query=query, k=4)
    
    chain = get_conversation_chain(select_llm)
    response = chain.run(input_documents=docs, question=query)
    
    print(query)
    print(response)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = response
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    load_dotenv()

    with st.sidebar:
        pdf = st.file_uploader("Upload your PDF", type='pdf')

        if pdf is not None:
            text = get_pdf_text(pdf)
            chunks = get_text_chunks(text)
            pdf_name = pdf.name[:-4]
            VectorStore = get_vectorstore(chunks, pdf_name)

        select_llm = st.selectbox(
            'Select a large language model:',
            ('oasst-sft-4', 'flan-t5-xxl', 'falcon-7b')
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        handle_user_input(query, VectorStore, select_llm)        

if __name__ == '__main__':
    main()
