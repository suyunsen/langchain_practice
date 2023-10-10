"""
 -*- coding: utf-8 -*-
time: 2023/10/10 18:57
author: suyunsen
email: suyunsen2023@gmail.com
"""

from typing import List
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class BaseChatBot:

    def __init__(self,llm):
        self.template = '''{text}
        '''

        chat_prompt = PromptTemplate(template=self.template,input_variables=["text"])
        self.chain = LLMChain(llm=llm,prompt=chat_prompt)


    #单轮聊答
    def chat_simply(self,question):
        output = self.chain.run(text = question,input_language="Chinese", output_language="Chinese")
        return output