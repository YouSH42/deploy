# AIchatBot

- 이 코드는 teddynote님의 유튜브 강좌를 보고 작성한 코드입니다.
[![Video Label](https://img.youtube.com/vi/VkcaigvTrug/0.jpg)](https://youtu.be/VkcaigvTrug?feature=shared)
- main.py 코드만으로 로컬에서 돌아갈 수 있는 ai챗봇입니다.
- ollama를 사용하여 모델을 로드한 다음 로컬에서 돌렸습니다.
- langchain을 사용하여 RAG도 적용하였습니다.
- 위 코드를 실행하기 위해서는 gpu 가속도 필요합니다.
- 또한 임베딩 모델을 따로 저장하는 코드가 없으므로 환경변수 설정을 해주어야합니다.(코드 참조)
```bash
pip install streamlit torch langchain langchain_core langchain_huggingface langchain_community unstructured faiss-cpu numpy pandas requests lxml python-docx
```
