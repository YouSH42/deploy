from flask import request, redirect, url_for, current_app
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
import threading
from . import upload_bp

UPLOAD_FOLDER = 'uploads'

def process_embedding(filepath):
    # PDF 파일을 로드하고 텍스트를 자르는 부분
    print(filepath)
    loader = PyPDFLoader(filepath)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=200,
        chunk_overlap=0
    )
    docs = loader.load_and_split(text_splitter)
    
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용하는 디바이스 이름: ")
    print(device)
    
    # 모델 및 임베딩 설정
    model_name = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # 임베딩한 정보를 벡터DB에 넣는 부분
    db = Chroma.from_documents(docs, hf, persist_directory="/home/sanghyun42/practice/web/db")
    print("Documents have been embedded and stored in the VectorDB." + filepath)

@upload_bp.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # 새로운 스레드를 생성하여 임베딩 작업을 수행
        embedding_thread = threading.Thread(target=process_embedding, args=(filepath,))
        embedding_thread.start()
        
        return redirect(url_for('main.main'))


