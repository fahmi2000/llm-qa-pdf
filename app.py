from dotenv import load_dotenv
from gtts import gTTS
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import concurrent.futures
import os

def get_pdf_text():
    pdf_reader = PdfReader("klia.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
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
    pkl_path = os.path.join(db_folder, f"klia.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            vector_store = pickle.load(f)

    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-base")
        vector_store = FAISS.from_texts(chunks, embedding = embeddings)

        with open(pkl_path, "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store

def get_conversation_chain():
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={
            "temperature": 0.7,
            "max_new_token": 512,
            "top_k": 50,
            "top_p": 0.95
        }
    )
    return load_qa_chain(llm=llm, chain_type="stuff")

def text_to_speech(text, language='en'):
    tts = gTTS(
        text = text, 
        lang = language, 
        slow = False
    )
    tts.save("response.mp3")
    os.system("start response.mp3")

def process_response(query, vector_store):
    chain = get_conversation_chain()
    query = f"<|system|>\nYou are a friendly customer service agent of Kuala Lumpur Internation Airport (KLIA).\n<|user|>\n{query}\n<|assistant|>"
    with concurrent.futures.ThreadPoolExecutor() as exe:
        future_vector_search = exe.submit(vector_store.similarity_search, query = query, k = 3)
        docs = future_vector_search.result()
        future_response = exe.submit(chain.run, input_documents=docs, question=query)
    
    return future_response.result()

def main():
    load_dotenv()
    text = get_pdf_text()
    chunk = get_text_chunks(text)
    vector_store = get_vector_store(chunk)
    
    while True:
        query = input("User input: ")
        
        if query.lower() == 'exit':
            break
        
        response = process_response(query, vector_store)
        print(response)
        text_to_speech(response)

if __name__ == '__main__':
    main()
