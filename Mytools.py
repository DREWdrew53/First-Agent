"""
项目agent工具
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_openai import OpenAIEmbeddings

# 工具
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import JsonOutputParser
from qdrant_client import QdrantClient
import requests


@tool
def test():
    """工具的测试函数"""
    return "test successful"

@tool
def search(query: str):
    """只在需要了解实时信息或不知道的事情的时候才会使用这个工具"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("实时搜索结果:", result)
    return result

@tool
def get_info_from_local_db(query: str):
    """只有回答2024年的时事政治问题的时候，会使用这个工具"""
    client = Qdrant(
        QdrantClient(path="D:/agent开发/qdrant_storage"), 
        "local_documents", 
        OpenAIEmbeddings(), 
    )
    retriever = client.as_retriever(search_type='mmr')
    result = retriever.get_relevant_documents(query)
    return result

@tool
def bazi_cesuan(query: str):
    """只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月日时，
    如果缺少用户姓名和出生年月日时则不可用"""
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回。
        JSON字段如下：
        - "api_key": "uro77mr0zye7v3QaVi9ss70q1", - "name": "姓名", 
        - "sex": "性别, 0表示男, 1表示女, 根据姓名判断", - "type": "日历类型, 0农历, 1公里, 默认1", 
        - "year": "出生年份 例: 1998", - "month": "出生月份 例 8", - "day": "出生日期 例: 8", 
        - "hours": "出生小时 例 14", - "minute": "0", 如果没有找到相关参数，则需要提醒用户告诉你
        这些内容，只返回数据结构，不要有其他的评论，用户输入：{query}"""
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instruction=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(temperature=0) | parser
    data = chain.invoke({"query": query})
    print("八字查询结果:", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print("=====返回数据=====")
        print(result.json())
        try:
            json = result.json()
            returnstring = f'八字为: {json["data"]["bazi_info"]["bazi"]}\n'
            return returnstring
        except Exception as e:
            return "八字查询失败，可能是你忘记询问用户的姓名或者出生年月日时了。"
    else:
        return "技术错误，请稍后再试"