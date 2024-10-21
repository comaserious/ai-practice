from dotenv import load_dotenv
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub

load_dotenv()

loaders = [
    TextLoader("data/Anger.txt"),
    TextLoader("data/Joy.txt"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

vectorstore = Chroma(
    collection_name="parent_memory",
    embedding_function=OpenAIEmbeddings(),
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)   

retriever.add_documents(docs)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.8)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)

question = "anger 의 성격을 말해줘"
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")

print("============================================")

question = "joy 가 가장 최근에 대화한 상대는 누구야?"
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")


print("============================================")

question = "anger 는 군대에 가는거야 안가는거야?"
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")

print("============================================")

question = "Anger 는 라면이 좋은거야? 냉면이 좋은거야?"
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")

print("============================================")

question = "Joy 는 오물 풍선을 좋아하나?"
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")