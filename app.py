import fitz as pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HugggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Load the pdf file and assigned the texts into a variable
def get_pdf_text():
    doc = pymupdf.open("klia.pdf")
    text = ""
    for page in doc.pages:
        text += page.extract_text()
    return text

# Get the text variable and split it into sizes
def get_text_chunks():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        length_function = len,
        separators = "\n"
    )
    chunks = text_splitter.split_text(text = text)
    return chunks
    
def get_converse_chain():
    llm = HugggingFaceHub(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        model_kwargs = {
            "temperature": 0.7,
            "max_new_token": 1024,
            "top_k": 50,
            "top_p": 0.95
        }
    )
    return load_qa_chain(
        llm = llm,
        chain_type = "stuff"
    )
    
# Look for existing embeddings before computing a new one
def process_vectorstore(chunks, pdf_name):
    db_folder = 'db'  
    pkl_path = os.path.join(
        db_folder,
        f"{pdf_name}.pkl"
    )
    
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-base"
        )
        
        vectorstore = FAISS.from_texts(
            chunks, embeddings = embeddings
        )
        
        with open(pkl_path, "wb") as f:
            pickle.dump(vectorstore, f)
    
def process_query(query, vectorstore):
    chain = get_converse_chain()
    query = f"<|user|>\n{query}</s>"
    
    docs = vectorstore.similarity_search(
        query = query, k = 3
    )
    
    response = chain.run(
        input_documents = docs,
        question = query
    )
    
    return response
    
# def display_reponse():


def main():
    get_pdf_text()
    process_vectorstore()
    query = input()
    response = process_query
    
    
if __name__ == '__main__':
    main()