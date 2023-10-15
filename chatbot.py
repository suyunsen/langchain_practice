"""
 -*- coding: utf-8 -*-
time: 2023/10/10 19:38
author: suyunsen
email: suyunsen2023@gmail.com
"""
import os
from typing import List
from Chat.ChatUtil import BaseChatBot
import _config
from langchain.llms import OpenAI
from Custom.ChatGLM import ChatGlm26b
from transformers import AutoTokenizer, AutoModel



def chat_custom_ChatGlm(chatbox,question):
    result = chatbox.chat_simply(question)
    return  result


def chat_ChatGpt(chatbox,question):
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'
    os.environ["OPENAI_API_KEY"] = _config.MODLE_KEY

if __name__ == '__main__':
    # 定义模型
    # llm = OpenAI(model_name=_config.MODLE_NAME, temperature=0)
    chatllm = ChatGlm26b()

    #定义聊天类
    chatbot = BaseChatBot(chatllm)
    question = ["你好啊","哈喽请用c++写个快排","hahhah","一元二次方程是什么时候学的"]
    for v in question:
        res = chat_custom_ChatGlm(chatbot,v)
        # print(res)

