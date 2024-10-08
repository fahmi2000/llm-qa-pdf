from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os
import time

def get_text_file():
    print("--------------------\n", "Ingesting data...", "\n--------------------")
    file_path = "Kuala_Lumpur_International_Airport.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
def get_text_chunks(text):
    print("--------------------\n", "Splitting data...", "\n--------------------")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 200,
        length_function = len,
        separators = "\n"
    )
    chunks = text_splitter.split_text(text = text)
    return chunks

def get_vector_store(chunks):
    db_folder = 'db'
    pkl_path = os.path.join(db_folder, f"Kuala_Lumpur_International_Airport.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            vector_store = pickle.load(f)

    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-base")
        print("--------------------\n", "Computing embedding...", "\n--------------------")
        vector_store = FAISS.from_texts(chunks, embedding = embeddings)

        with open(pkl_path, "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store

def get_conversation_chain():
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={
            "temperature": 0.3,
            "max_new_token": 512,
            "top_k": 50,
            "top_p": 0.95
        }
    )
    return load_qa_chain(llm = llm, chain_type = "stuff")

def process_response(query, vector_store):
    chain = get_conversation_chain()
    template_query = f"<|system|>\nYou are a friendly customer service agent of Kuala Lumpur Internation Airport (KLIA).\n<|user|>\n{query}\n<|assistant|>"

    docs = vector_store.similarity_search_with_relevance_scores(
        query = query, 
        k = 2,
        score_threshold = 0.80
    )

    if not docs:
        response = "Sorry! I can only answer questions related to the Kuala Lumpur International Airport"

    else:
        document_strings = [doc[0] for doc in docs]
        response = chain.run(input_documents = document_strings, question = template_query)

    return response
    

def main():
    load_dotenv()
    text = get_text_file()
    chunk = get_text_chunks(text)
    vector_store = get_vector_store(chunk)
    
    while True:
        query = input("User input: ")
        
        if query.lower() == 'exit':
            break
        
        start_time = time.time()
        response = process_response(query, vector_store)
        print("--------------------\n", response, "\n--------------------")
        end_time = time.time()
        print("API Response Time: ", end_time - start_time)

if __name__ == '__main__':
    main()
