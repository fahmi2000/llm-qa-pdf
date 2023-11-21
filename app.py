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
import os

with st.sidebar:
    st.title('LLM Chat')
                    
def main():
    load_dotenv()
    
    with st.sidebar:
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

                pkl_path = os.path.join(db_folder, f"{store_name}.pkl")

                with open(pkl_path, "wb") as f:
                    pickle.dump(VectorStore, f)
                st.write("New embeddings computed.")
    
        select_llm = st.selectbox(
            'Select a large language model:',
            ('oasst-sft-4', 'flan-t5-xxl', 'falcon-7b')
        )
        
        if select_llm == 'oasst-sft-4':
            select_llm = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
        elif select_llm == 'flan-t5-xxl':
            select_llm = 'google/flan-t5-xxl'
        elif select_llm == 'falcon-7b':
            select_llm = 'tiiuae/falcon-7b-instruct'
        
    # st.write(select_llm)
    
    query = st.chat_input("Ask questions about your document...")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        query = f"<|prompter|>{query}<|endoftext|><|assistant|>"          
        docs = VectorStore.similarity_search(query = query, k = 4)           
        llm = HuggingFaceHub(repo_id=select_llm, model_kwargs={"temperature":0.3, "max_length":200})
        chain = load_qa_chain(llm = llm, chain_type = "stuff")
        response = chain.run(input_documents = docs, question = query)
        
        with st.chat_message("ai"):
            st.write(response)
        print(query)
        print(response)

if __name__ == '__main__':
    main()