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
        self.template = """Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        {history}
        Human: {human_input}
        Assistant:"""

        chat_prompt = PromptTemplate(template=self.template,input_variables=["history","human_input"])
        self.chain = LLMChain(llm=llm,prompt=chat_prompt)


    #单轮聊答
    def chat_simply(self,question):
        output = self.chain.run(text = question,input_language="Chinese", output_language="Chinese")
        return output