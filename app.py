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
import concurrent.futures
import os

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__}: {execution_time} seconds")
        return result
    return wrapper

@timing_decorator
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@timing_decorator
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

@timing_decorator
def get_vectorstore(chunks, pdf_name):
    db_folder = 'db'
    pkl_path = os.path.join(db_folder, f"{pdf_name}.pkl")

    with st.status("Processing document...") as status:
        st.caption("Searching for existing embeddings...")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                VectorStore = pickle.load(f)
            status.update(label="Embeddings loaded!", state="complete", expanded=False)
        else:
            st.caption("Computing new data...")
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)

            with open(pkl_path, "wb") as f:
                pickle.dump(VectorStore, f)
            status.update(label="New embeddings computed!", state="complete", expanded=False)

    return VectorStore

@timing_decorator
def get_conversation_chain(select_llm):
    model_mapping = {
        'oasst-sft-4': 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
        'flan-t5-xxl': 'google/flan-t5-xxl',
        'mistralai/Mistral-7B-v0.1': 'mistralai/Mistral-7B-v0.1'
    }

    model_id = model_mapping.get(select_llm, select_llm)
    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.3, "max_token": 512})
    return load_qa_chain(llm=llm, chain_type="stuff")

@timing_decorator
def process_response(query, VectorStore, select_llm):
    chain = get_conversation_chain(select_llm)
    
    with concurrent.futures.ThreadPoolExecutor() as exe:
        future_vector_search = exe.submit(VectorStore.similarity_search, query=query, k=3)
        docs = future_vector_search.result()
        future_response = exe.submit(chain.run, input_documents=docs, question=query)
    
    return future_response.result()

@timing_decorator
def display_response(response):
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

@timing_decorator
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
            ('oasst-sft-4', 'flan-t5-xxl', 'mistralai/Mistral-7B-v0.1')
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
        
        response = process_response(query, VectorStore, select_llm)
        display_response(response)       

if __name__ == '__main__':
    main()
