import os
import streamlit as st
import torch
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import numpy as np

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
# =============================
# .bashrc 파일에 
# export HF_HOME="./.cache/" << 추가
# =============================

RAG_PROMPT_TEMPLATE = """당신은 주어진 답변에 자세하게 대답하는 챗봇입니다. 모르는 내용이 있다면 모른다고 답변을 해주세요. 복잡한 작업을 더 간단한 하위 작업으로 나누고, 각 단계에서 "생각"할 시간을 가지세요. 그리고 답변 끝에 참고한 문서를 표기하십시오.
아래는 질문과 그에 대한 예제 답변입니다.

Question: 장학생 선발 기준에 대해서 알려줘
Context: 장학생 선발 기준은 성적, 리더십, 사회봉사, 재정 상태 등을 종합적으로 평가하여 결정됩니다. 일반적으로 학업 성취도와 리더십이 주요 평가 요소입니다.
Answer: 장학생 선발 기준은 주로 성적과 리더십을 기반으로 합니다. 성적은 학업 성취도를 나타내며, 리더십은 학생이 공동체에서 어떠한 역할을 수행했는지를 평가합니다. 또한, 사회봉사와 재정 상태도 고려됩니다. 이 모든 요소들이 종합적으로 평가되어 장학생이 선발됩니다.
출처: 장학규정 문서

Question: 장학생을 받을 수 있는 장학종류는 주로 있어?
Context: 장학생은 다음과 같이 구분하고, 장학생 구분에 따른 장학종류는 별표 1과 같이 한다. 1. 명예장학생 2. 성적우수장학생 3. 특별장학생 4. 교외장학생 5. 국가장학생
Answer: 1. 명예장학생 2. 성적우수장학생 3. 특별장학생 4. 교외장학생 5. 국가장학생 총 5가지로 장학종류를 구분하고 있습니다.
출처: 장학규정 문서

Question: {question}
Context: {context}
Answer:
출처: {context}"""

st.set_page_config(page_title="RAG를 이용한 챗봇", page_icon="💬")
st.title("RAG를 이용한 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

# 입력받은 문서를 임베딩하는 과정
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    tik_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=250,
        chunk_overlap=0
    )
    
    # 다양한 문서를 파싱하기 위해서 unstructuredFileLoader를 사용
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=tik_text_splitter)

    # 모델 및 임베딩 설정
    # gpu 가속을 위한 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 내가 따로 설정한 임베딩 모델
    model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    # VectorDB로는 FAISS를 사용하여 구성하였음
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever(k=3)
    
    return retriever, vectorstore, embeddings  # vectorstore와 embeddings 반환

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever, vectorstore, embeddings = embed_file(file)

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = ChatOllama(model="EEVE-Korean-10.8B:latest")
        chat_container = st.empty()
        if file is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
