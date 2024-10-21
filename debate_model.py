from dotenv import load_dotenv
from typing import List,  Callable
from datetime import datetime
import os

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
load_dotenv()


# 대화 에이전트 생성
class DialogueAgent:
    def __init__(
            self,
            name : str,
            system_message : SystemMessage,
            model : ChatOpenAI
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self) :
        """
        대화 내역을 초기화합니다
        """
        self.message_history =["Here is the conversation so far."]

    def send(self) -> str:
        """
        메시지에 시스템 메시지  + 대화내용과 마지막으로 에이전트의 이름을 추가합니다.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content = "\n".join([self.prefix] + self.message_history ))
            ]
        )
        return message.content
    
    def receive(self, name: str, message: str) -> None:
        """
        name 이 말한 message 를 메시지 내역에 추가합니다
        """
        self.message_history.append(f"{name} : {message}")
    


# 대화 방식을 정의
class DialogueSimulator:
    def __init__(
            self,
            agents : List[DialogueAgent],
            selection_function : Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step =0
        self.select_next_speaker = selection_function

    def reset(self) -> None:
        for agent in self.agents: 
            agent.reset()

    # 하나의 에이전트의 발언을 모든 에이전트에게 전달 저장
    def inject(self, name: str, message : str): 
        """
        name 의 message 로 대화를 시작합니다
        """
        for agent in self.agents:
            agent.receive(name,message)

        self._step += 1

    def step(self) -> tuple[str, str]:
        # 다음 발언자 선택
        speaker_index = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_index]

        # 다음 발언자에게 메시지 전송
        message = speaker.send()

        # 모든 에이전트가 메시지를 받습니다
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        
        # 시뮬레이션 단계를 증가
        self._step += 1

        return speaker.name, message
    
    # def save_conversation(self):
    #     for agent in self.agents:
    #         filename = f"@data/{agent.name}.txt"
            
    #         # 디렉토리가 존재하는지 확인하고 없으면 생성
    #         os.makedirs(os.path.dirname(filename), exist_ok=True)

    #         try:
    #             # 파일이 존재하면 읽기, 없으면 빈 리스트 생성
    #             if os.path.exists(filename):
    #                 with open(filename, 'r', encoding='utf-8') as file:
    #                     lines = file.readlines()
    #             else:
    #                 lines = []

    #             # 대화 내용 시작 위치 찾기
    #             conversation_start = next((i for i, line in enumerate(lines) if line.strip() == "대화내용"), None)
                
    #             if conversation_start is not None:
    #                 # 기존 대화 내용 유지
    #                 new_lines = lines[:conversation_start + 1]
    #             else:
    #                 # 대화 내용 섹션 추가
    #                 new_lines = lines + ["\n대화내용\n"]

    #             # 새로운 대화 내용 추가
    #             for message in agent.message_history:
    #                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    #                 new_lines.append(f"{timestamp} : {message}\n")

    #             # 파일에 저장
    #             with open(filename, 'w', encoding='utf-8') as file:
    #                 file.writelines(new_lines)

    #             print(f"{filename}에 대화 내용이 저장되었습니다.")

    #         except Exception as e:
    #             print(f"파일 저장 중 오류 발생: {e}")

    # def simulate(self, num_steps: int):
    #     for _ in range(num_steps):
    #         self.step()
    #     self.save_conversation()  # 대화 종료 후 저장
    

# 각각의 에이전트들이 사용할 툴을 정의

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
            self,
            name : str,
            system_message : SystemMessage,
            model : ChatOpenAI,
            tools,
    ) -> None:
        super().__init__(name,system_message, model)
        self.tools = tools 

    def send(self) -> str:
        """
        메시지 기록에 챗 모델을 적용하고 메시지 문자열을 반환합니다.
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools = self.tools, verbose=False)

        message =AIMessage(
            content = agent_executor.invoke(
                {
                    "input" : "\n".join(
                        [self.system_message.content] +
                        [self.prefix] +
                        self.message_history
                    ),
                    "chat_history": self.message_history,
                }
            )["output"]
        )
        return message.content
    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

anger_loader= TextLoader("data/Anger.txt")
joy_loader= TextLoader("data/Joy.txt")

anger_docs = anger_loader.load_and_split(RecursiveCharacterTextSplitter())
joy_docs = joy_loader.load_and_split(RecursiveCharacterTextSplitter())

anger_vector = Chroma.from_documents(anger_docs, OpenAIEmbeddings())
joy_vector = Chroma.from_documents(joy_docs, OpenAIEmbeddings())

anger_retriever = anger_vector.as_retriever()
joy_retriever = joy_vector.as_retriever()

from langchain.tools.retriever import create_retriever_tool

anger_retriever_tool = create_retriever_tool(
    anger_retriever,
    name="Anger",
    description="This is a document about Anger. It contains information about Anger's personality and daily schedule."
)

joy_retriever_tool = create_retriever_tool(
    joy_retriever,
    name="Joy",
    description="This is a document about Joy. It contains information about Joy's personality and daily schedule."
)

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(k=3)

# 각 에이전트가 사용할  수 있는 tools 를 설정한다

names ={
    "Anger" : [anger_retriever_tool],
    "Joy" : [joy_retriever_tool],
}

names_search = {
    "Anger": [search],  # 의사협회 에이전트 도구 목록
    "Joy": [search],  # 정부 에이전트 도구 목록
}

# 대화 내용 선정 
topic = "여행지를 베트남으로 할것인지 아니면 호주로 할것인지 논의야합니다."

word_limit = 10

conversation_description = f"""Here is the topic of conversation: {topic}
the participants are : {', '.join(names.keys())}"""

# Moderator 가 대화에 대한 설명을 추가합니다 => 필요없을듯
agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)

def generate_agent_description(name):
    agent_specifier_prompt= [
        agent_descriptor_system_message,
        HumanMessage(content=f"Describe {name} in a few words."),
    ]

    agent_description = ChatOpenAI(temperature=0)(agent_specifier_prompt).content
    return agent_description

agent_descriptions ={name: generate_agent_description(name) for name in names}

def generate_system_message(name, description,tools):
    return f"""{conversation_description}

Your name is {name}.

Your description is as follows : {description}

DO look up information from the following tools : {tools} to help you answer the question.

Enjoy the conversation!

Your goal is to discuss the topic above.

DO NOT restate something that has already been said in the past.

Stop speaking the moment you finish speaking from your perspective.

Anser in KOREAN
"""

agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)


# 주제를 더 구체적으로 만들 수 있습니다.
topic_specifier_prompt = [
    # 주제를 더 구체적으로 만들 수 있습니다.
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}
        
        You are the moderator. 
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Speak directly to the participants: {*names,}.  
        Do not add anything else.
        Answer in Korean."""  # 다른 것은 추가하지 마세요.
    ),
]
# 구체화된 주제를 생성합니다.
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")  # 원래 주제를 출력합니다.
print(f"Detailed topic:\n{specified_topic}\n")  # 구체화된 주제를 출력합니다.


# loop

# 이는 결과가 텍스트 제한을 초과하는 것을 방지하기 위함입니다.
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

agents_with_search = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names_search.items(), agent_system_messages.values()
    )
]

agents.extend(agents_with_search)
agents


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    # 다음 발언자를 선택합니다.
    # step을 에이전트 수로 나눈 나머지를 인덱스로 사용하여 다음 발언자를 순환적으로 선택합니다.
    idx = (step) % len(agents)
    return idx




max_iters =3  # 최대 반복 횟수를 6으로 설정합니다.
n = 0  # 반복 횟수를 추적하는 변수를 0으로 초기화합니다.

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달합니다.
simulator = DialogueSimulator(
    agents=agents_with_search, selection_function=select_next_speaker
)

# 뮬레이터를 초기 상태로 리셋합니다.
simulator.reset()

# Moderator가 지정된 주제 제시합니다.
simulator.inject("Moderator", specified_topic)

# Moderator가 제시한 주제를 출력합니다.
print(f"(Moderator): {specified_topic}")
print("\n")

while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
    name, message = (
        simulator.step()
    )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
    print(f"({name}): {message}")  # 발언자와 메시지를 출력합니다.
    print("\n")
    n += 1  # 반복 횟수를 1 증가시킵니다.







