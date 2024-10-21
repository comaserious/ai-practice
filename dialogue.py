import uuid
from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# 메모리 생성
store = {}

def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "친절하게 대답하는 챗봇입니다"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )


    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.8)
    chain = prompt | model | StrOutputParser()
    

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

def main():
    chain_with_history = create_chain()
    session_id = str(uuid.uuid4())
    print("새로운 대화 세션을 시작합니다. 종료하려면 'exit'를 입력하세요.")
    
    while True:
        user_input = input("질문을 입력하세요: ")
        if user_input.lower() == 'exit':
            print("대화를 종료합니다.")
            break
        
        result = chain_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print("AI 응답:", result)
        print("\n대화 기록:")
        for message in get_session_history(session_id).messages:
            print(f"{message.type}: {message.content}")
        print()




def get_session_history(session_ids):
    print(f"[대화 세션 ID] : {session_ids}")
    if session_ids not in store: 
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

if __name__ == "__main__":
    load_dotenv()
    main()


# seed 값을 가진 값을 vdb 에 저장하게 된다면 나중에 검색을 통해서 seed 값을 가진 것을 가지고 와서 대화를 더 자연스럽게 이어가게 만들 수 있다