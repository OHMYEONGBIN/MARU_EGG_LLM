import os
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# PyMuPDFLoader 을 이용해 PDF 파일 로드
loader = PyMuPDFLoader("2025학년도 수시모집_홈페이지 공지(단순오기재수정_V4).pdf")
pages = loader.load()

# 문서를 문장으로 분리
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(pages)

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'mps'},
    encode_kwargs={'normalize_embeddings':True},
)

# 벡터 저장소 경로 설정
vectorstore_path = 'vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)

# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
vectorstore.persist()
print("Vectorstore created and persisted")

# Ollama 를 이용해 로컬에서 LLM 실행
model = ChatOllama(model="EEVE-Korean-10.8B:latest", base_url="https://705b-128-134-33-155.ngrok-free.app")
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# Prompt 템플릿 생성
template = '''당신은 친절한 챗봇으로서 명지대학교 입시 관련 질문자 요청에 최대한 자세하고 친절하게 답변해야 합니다. 모든 대답은 한국어(Korean)으로 대답해 주세요. 만약 질문이 명지대학교 입시와 관련이 없다면 "해당 내용은 답변할 수 없습니다!"라고 대답해 주세요:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
query = "크리스처 리더 전형은 1단계 합격자 몇 배수를 뽑아?"
answer = rag_chain.invoke(query)

print("Query:", query)
print("Answer:", answer)