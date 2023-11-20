import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

with st.sidebar:
    st.title('LLM Chat PoC')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot using:
    - Streamlit
    - LangChain
    - OpenAI
    
    ''')
    
    add_vertical_space(5)
    st.write('Made agonizingly by me....')    
    

def main():
    st.write("Hello")
    
    load_dotenv()
    
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        
        chunks = text_splitter.split_text(text = text)
        
        db_folder = 'db'
        store_name = pdf.name[:-4]
        pkl_path = os.path.join(db_folder, f"{store_name}.pkl")
        
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings loaded locally.')
        else:
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # Construct the path to the .pkl file in the 'db' folder
            pkl_path = os.path.join(db_folder, f"{store_name}.pkl")

            with open(pkl_path, "wb") as f:
                pickle.dump(VectorStore, f)
            st.write("New embeddings computed.")
        
        query = st.text_input("Ask questions about your document:")
        
        if query:
            query = f"<|prompter|>{query}<|endoftext|><|assistant|>"
            
            docs = VectorStore.similarity_search(query = query, k = 3)
            
            llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"temperature":0.8, "max_length":512})
            
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            
            st.write(response)
            print(response)     
            st.write(docs)

if __name__ == '__main__':
    main()