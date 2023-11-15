import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from huggingface_hub import InferenceClient
from langchain.chains.conversation.memory import ConversationBufferMemory


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader =  PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len 
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore
    
# def get_converse_chain(vectorstore):
#     # llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
#     memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
#     converse_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm,
#         retriever = vectorstore.as_retriever(),
#         memory = memory
#     )
#     return converse_chain

def get_converse_chain(vectorstore):
    try:
        llm = InferenceClient(model="google/flan-t5-xxl")
        print("Model loaded successfully!")
    except Exception as e:
        print("Failed to load model!")
        print("Error: ", e)
        return None

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converse_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    
    return converse_chain

def handle_userinput(user_question):
    response = st.session_state.converse({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html = True)
    if "converse" not in st.session_state:
        st.session_state.converse = None
    
    st.header("PDF Chat :books:")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hello bot"), unsafe_allow_html = True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html = True)
    
    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here & click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                
                # text_chunks = get_text_chunks(raw_text)
                
                print("Raw Text:", raw_text)

                # vectorstore = get_vectorstore(text_chunks)
                # print("Vectorstore:", vectorstore)

                # st.session_state.converse = get_converse_chain(vectorstore)
                # print("Converse Chain:", st.session_state.converse)


if __name__ == '__main__':
    main()