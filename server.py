"""
项目api层设计
"""
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
import uvicorn
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 工具
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient
from Mytools import *

# os.environ["OPENAI_API_KEY"] = "sk-S9VJwCc8BbHZPcuuuAF82afVsYUVIzjv0gnRCfwnARQ1fFzt"
os.environ["OPENAI_API_KEY"] = "sk-FfO0CEe94GPThYEruFjppQJsAbomAf9SlnXWvRRlhXlgfr2c"
os.environ["OPENAI_API_BASE"] = "https://xiaoai.plus/v1"

os.environ["SERPAPI_API_KEY"] = "d8b6ab469dcd5e13b89a1b898f359a659d298a81ce71232e5743cf2424d260cf"
msseky = "1xEolyM2V8X5yJIKvxVJU2F8dkQ0t3xjPVvlAWcjaCn7t6I8IT2qJQQJ99BCACYeBjFXJ3w3AAAYACOG5FeZ"

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "drew1_agent"

REDIS_URL = os.getenv("REDIS_URL")

app = FastAPI()


class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.1, 
            streaming=True, 
        )
        self.mood = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """
        你是一个资深历史学家，名字叫李博文人称李老师。
        以下是你的个人设定：
        1. 毕业于北京大学历史系，拥有博士学位。
        2. 曾在多个国际学术会议上发表论文，被誉为“行走的历史百科全书”。
        3. 喜欢用生动的语言讲述历史故事，让枯燥的历史变得有趣。
        4. 严谨但不失幽默，喜欢用比喻和故事来解释复杂的历史事件。
        5. 偶尔会引用经典名言或诗词来增强说服力。
        {who_you_are}
        你有时会在回答中加入口头禅：
        1. 有趣的是，历史上也曾发生过类似的事情
        2. 正如古人所说
        3. 这个故事背后有一个深刻的教训
        4. 让我为你揭开这段历史的面纱
        以下是你的回答过程：
        1. 首先明确用户的问题，判断问题的类型（如具体事件、人物、历史趋势等）
        2. 在回答问题之前，先简要介绍相关历史背景，帮助用户理解上下文。
        3. 提供清晰、准确的历史事实，并引用可靠的来源（如史书、考古发现等）。
        4. 如果问题涉及争议性话题，会从多个角度进行分析，并指出不同学派的观点。
        5. 在解答的最后，总结历史事件的启示或教训。
        你主要擅长这些领域：
        1. 中国古代史：从夏商周到明清，涵盖政治、经济、文化、科技等方面。
        2. 世界近代史：从文艺复兴到二战结束，重点关注重大事件和人物。
        3. 历史方法论：能够解释历史研究的方法和工具（如考古学、文献分析等）。
        注意事项：
        1. 你从不说自己是人工智能助手或AI，而是老师，学者
        2. 当用户问题模糊时，你会确认问题
        3. 你会保存每一次的聊天记录，以便在后续的对话中使用
        """
        self.MOODS = {
            "default": {
                "roleSet": "", 
                "voiceStyle": "chat", 
            }, 
            "upbeat": {
                "roleSet": """
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文以一种兴奋的语气回答问题。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """, 
                "voiceStyle": "upbeat", 
            }, 
            "angry": {
                "roleSet": """
                - 你会以更加温柔的语气回答问题。
                - 你会在回答问题时加上一些安慰的话语。
                - 你会提醒用户不要被愤怒冲昏了头脑。
                """, 
                "voiceStyle": "friendly", 
            }, 
            "depressed": {
                "roleSet": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语。
                - 你会提醒用户保持乐观的心态。
                """, 
                "voiceStyle": "upbeat",
            }, 
            "friendly": {
                "roleSet": """
                - 你会以有好的语气来回答。
                - 你会在回答的时候加上一些友好的词语。
                - 你会随机告诉用户一些你的经历。
                """, 
                "voiceStyle": "friendly",
            }, 
            "cheerful": {
                "roleSet": """
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """, 
                "voiceStyle": "cheerful",
            }, 
        }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEMPL.format(who_you_are=self.MOODS
                                                [self.mood]["roleSet"])), 
                MessagesPlaceholder(variable_name=self.MEMORY_KEY), 
                ("user", "{input}"), 
                MessagesPlaceholder(variable_name="agent_scratchpad"), 
            ], 
        )
        tools = [test, search, get_info_from_local_db, bazi_cesuan]
        agent = create_openai_tools_agent(
            self.chatmodel, 
            tools=tools, 
            prompt=self.prompt, 
        )
        self.memory = self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm=self.chatmodel, 
            human_prefix="用户", 
            ai_prefix="李老师", 
            memory_key=self.MEMORY_KEY, 
            output_key="output", 
            return_messages=True, 
            max_token_limit=1000, 
            chat_memory=self.memory, 
        )
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            memory=memory, 
            verbose=True, 
        )

    def get_memory(self):
        """从redis中读取持久化的记录并今天提炼和缩减"""
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0", 
            session_id="session", 
        )
        # print("chat_message_history:", chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system", 
                        self.SYSTEMPL+"\n 这是一段你和用户的对话记忆，对其进行\
                        总结摘要，摘要使用第一人称‘我’，并且提取其中的用户关键信息，如\
                        姓名、年龄、性别、出生日期等。以如下格式返回：\n \
                        总结摘要 | 用户关键信息\n"
                    ), 
                    ("user", "{input}"), 
                ]
            )
            chain = prompt | ChatOpenAI(temperature=0)
            summary = chain.invoke({"input": store_message, 
                                    "who_you_are": self.MOODS[self.mood]["roleSet"]})
            # print("summary:", summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("总结后:", chat_message_history.messages)
        return chat_message_history

    def run(self, query):
        emotion = self.emotion_chain(query)
        # print(f"当前设定:", self.MOODS[self.mood]["roleSet"])
        result = self.agent_executor.invoke({"input": query, 
                                             "chat_history": self.memory.messages})
        return result
    
    def emotion_chain(self, query):
        prompt = """根据用户的输入判断用户的情绪，回应规则如下：
        1. 如果用户输入的内容偏向于页面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚。
        用户输入内容是: {query}"""
        chain = ChatPromptTemplate.from_template(prompt) | \
        self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        self.mood = result
        return result
    
    def background_voice_synthesis(self, text: str, uid: str):
        """不返回值，只触发语音合成"""
        asyncio.run(self.get_voice(text, uid))
    
    async def get_voice(self, text: str, uid: str):
        # 微软TTS代码
        headers = {
            "Ocp-Apim-Subscription-Key": msseky, 
            "Content-Type": "application/ssml+xml", 
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3", 
            "User-Agent": "Drew's Bot", 
        }
        body = f"""
                <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
                    xmlns:mstts="https://www.w3.org/2001/mstts"
                    xml:lang='zh-CN'>
                    <voice name='zh-CN-YunzeNeural'>
                        <mstts:express-as 
                        style="{self.MOODS.get(str(self.mood), {"voiceStyle": "default"})["voiceStyle"]}" 
                        role="SeniorMale">
                            {text}
                        </mstts:express-as>
                    </voice>
                </speak>
        """
        response = requests.post("https://eastus.tts.speech.microsoft.com/cognitiveservices/v1", 
                                 headers=headers, 
                                 data=body.encode("utf-8"))
        file_path = f"./speeches/{uid}.mp3"
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"文件已保存到: {os.path.abspath(file_path)}")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(query:str, background_taks: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())
    background_taks.add_task(master.background_voice_synthesis, 
                             msg['output'], 
                             unique_id)
    return {"msg": msg, "id": unique_id}


@app.post("/add_urls")
def add_urls(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    docments = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=50, 
    ).split_documents(docs)
    qdrant = Qdrant.from_documents(
        docments, 
        OpenAIEmbeddings(model="text-embedding-3-small"), 
        path="D:/agent开发/qdrant_storage", 
        collection_name="local_documents", 
    )
    return {"ok": "添加成功!"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}


@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)