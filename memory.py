from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

anger = TextLoader("data/Anger.txt")
joy = TextLoader("data/Joy.txt")



anger_docs = anger.load_and_split(text_splitter)
joy_docs = joy.load_and_split(text_splitter)

print(len(anger_docs))
print(len(joy_docs))



DB_PATH = "./chroma_db"
persist_db = Chroma.from_documents(
    anger_docs + joy_docs, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="emotion_memory"
)

# db_path 에 저장된 데이터 로드
persist_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="emotion_memory",
)

# 데이터 추가 저장

retriever = persist_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2},
)

print("============================================")
print(retriever.invoke("joy 는 가장 최근에 누구와 대화했지?"))

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.8)
# chain =  | llm | StrOutputParser()

# print(chain.invoke("joy 는 가장 최근에 누구와 대화했지?"))

