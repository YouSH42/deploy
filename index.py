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

RAG_PROMPT_TEMPLATE = """당신은 주어진 답변에 자세하게 대답하는 챗봇입니다. 모르는 내용이 있다면 모른다고 답변을 해주세요. 그리고 항상 한국어로 답변을 해주세요

Question: {question}
Context: {context}
Answer:"""

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

def get_sources(docs):
    # 검색한 문서의 출처를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.metadata["source"] for doc in docs)

# 입력받은 문서를 임베딩하는 과정
@st.cache_resource(show_spinner="Embedding file...")
def embed_files(files):
    all_docs = []
    cache_dirs = []

    for file in files:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        cache_dirs.append(cache_dir)
        
        tik_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=250,
            chunk_overlap=0
        )
        
        # 다양한 문서를 파싱하기 위해서 unstructuredFileLoader를 사용
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=tik_text_splitter)
        all_docs.extend(docs)

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
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dirs[0])
    vectorstore = FAISS.from_documents(all_docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever(k=3)
    
    return retriever, vectorstore, embeddings  # vectorstore와 embeddings 반환

with st.sidebar:
    files = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

if files:
    retriever, vectorstore, embeddings = embed_files(files)

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = ChatOllama(model="EEVE-Korean-10.8B:latest")
        # ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        chat_container = st.empty()
        if files:
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
            
            # L2 거리 계산 및 출력
            query_embedding = embeddings.embed_query(user_input)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = vectorstore.index.search(query_embedding, k=3)

            print("Query:", user_input)
            for i, idx in enumerate(indices[0]):
                doc_id = vectorstore.index_to_docstore_id[idx]
                doc = vectorstore.docstore._dict[doc_id]
                print(f"\nDocument {i + 1}:")
                print("Content:", doc.page_content)
                print("L2 Distance:", distances[0][i])

            # 가장 관련성이 높은 문서를 컨텍스트로 사용
            most_relevant_doc = vectorstore.docstore._dict[vectorstore.index_to_docstore_id[indices[0][0]]]

            # 참고한 문서의 출처 출력
            source = most_relevant_doc.metadata.get("source", "Unknown Source")
            # 예제에서 제공한 방법으로 답변을 수정
            final_answer = "".join(chunks) + f"\n\n출처: {source}"
            chat_container.markdown(final_answer)
            add_history("ai", final_answer)

        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
