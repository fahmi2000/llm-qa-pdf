from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Get the text variable and split it into sizes
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        length_function = len,
        separators = "\n"
    )
    chunks = text_splitter.split_text(text = text)
    return chunks
    
def load_llm_chain():
    llm = HuggingFaceHub(
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
def process_vectorstore(chunks, pdf_path):
    db_folder = 'db'  
    pkl_path = os.path.join(
        db_folder,
        f"{pdf_path}.pkl"
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
            
    return vectorstore
    
# Build the QA chain and pass it to the LLM
def process_query(query, vectorstore):
    chain = load_llm_chain()
    query = f"<|user|>\n{query}</s>"
    
    docs = vectorstore.similarity_search(query = query, k = 3)
    
    response = chain.run(
        input_documents = docs,
        question = query
    )
    
    return response

def main():
    
    pdf = PdfFileReader("klia.pdf")
    if pdf is not None:
        try:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
                
            chunks = get_text_chunks(text)
            pdf_name = pdf.name[:-4]
            vectorstore = process_vectorstore(chunks, pdf_name)
            
            while True:
                
                query = input("Enter your question:")
                if query.lower() == 'exit':
                    print("Exiting...")
                    break
                
                response = process_query(query, vectorstore)
                print(response)
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
if __name__ == '__main__':
    main()