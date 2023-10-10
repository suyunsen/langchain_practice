"""
 -*- coding: utf-8 -*-
time: 2023/10/10 21:34
author: suyunsen
email: suyunsen2023@gmail.com

主要以web网址为答案的QA问题
"""

from langchain.llms import OpenAI
import requests
import re
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA,summarize,ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)
from langchain.output_parsers import CommaSeparatedListOutputParser
import _config
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


def stuff_webQA(urls:str,question:str)->str:
    loader = WebBaseLoader(urls)
    documents = loader.load()

    # 定义分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5200,
        chunk_overlap=520
    )

    #简单预处理也可以不做处理
    documents[0].page_content = re.sub(r"([a-zA-Z0-9])([A-Z])", r"\1 \2", documents[0].page_content).replace('\\xa', ' ')

    # 分割 youtube documents
    documents = text_splitter.split_documents(documents)

    #采用OPENAI的embedding
    embeddings = OpenAIEmbeddings()

    # 设计参数search_kwargs返回检索的topk,这里设计上传所有的
    docsearch = Chroma.from_documents(documents, embeddings).as_retriever(search_kwargs={"K": 4})

    # 定义模型
    llm = OpenAI(model_name=_config.MODLE_NAME, temperature=0)

    #定义模板
    combine_prompt_template = """
            Use the following context to answer the user's question.don't try to make it up.
            You just need to reply to the answer and you don't need to explain the answer.
            Separate each answer with a comma.
            Don't give repeated answers.
            -----------
            {question}
            -------------
            {summaries}
            """
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=COMBINE_PROMPT)

    docs = docsearch.get_relevant_documents(query=question)
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return  result['output_text']

def mayreduce_webQA(urls,question)->str:
    loader = WebBaseLoader(urls)
    documents = loader.load()

    # 定义分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5200,
        chunk_overlap=520
    )

    # 简单预处理也可以不做处理
    documents[0].page_content = re.sub(r"([a-zA-Z0-9])([A-Z])", r"\1 \2", documents[0].page_content).replace('\\xa',
                                                                                                             ' ')

    # 分割 youtube documents
    documents = text_splitter.split_documents(documents)

    # 采用OPENAI的embedding
    embeddings = OpenAIEmbeddings()

    # 设计参数search_kwargs返回检索的topk,这里设计上传所有的
    docsearch = Chroma.from_documents(documents, embeddings).as_retriever(search_kwargs={"K": 4})

    # 注意许多模型不支持该方法,比如chatgpt不支持
    llm = OpenAI(temperature=0)
    # 问题模板
    system_template = """
               Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    Return any relevant text in English.
                {context}
                QUESTION: {question}
                Relevant text, if any:"""
    PROMPT = PromptTemplate(template=system_template, input_variables=["context", "question"])
    # 定义map_reduce类型链，各个doucment逐个分析，最后把每个doucment结果在做一个总分析
    combine_prompt_template = """
            Use the following context to answer the user's question.don't try to make it up.
            You just need to reply to the answer and you don't need to explain the answer.
            Separate each answer with a comma.
            Don't give repeated answers.
            -----------
            {question}
            -------------
            {summaries}
            """
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True,
                          question_prompt=PROMPT, combine_prompt=COMBINE_PROMPT)
    docs = docsearch.get_relevant_documents(query=question)
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return result['output_text']




