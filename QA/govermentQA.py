"""
 -*- coding: utf-8 -*-
time: 2023/10/25 19:50
author: suyunsen
email: suyunsen2023@gmail.com
"""

import _config
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from Custom.ChatGLM import ChatGlm26b
from langchain.schema.embeddings import Embeddings
from Custom_embeddings.SbertEnbedding import  SBertEmbeddings
from langchain.chains.question_answering import load_qa_chain
import logging
from Custom.Custom_SparkLLM import Spark

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class GoQa:

    modle_name = _config.MODLE_NAME #后面用于加载别的模型目前默认

    llm:LLM = None

    embeddings:Embeddings=None

    faiss_index:str = None

    faiss_db:FAISS = None

    def __init__(self,modle_name = _config.MODLE_NAME,templet_prompt:str=None,faiss_index:str=None):
        self.llm = ChatGlm26b()
        # self.llm = Spark()
        self.embeddings = SBertEmbeddings()
        self.PROMPT = PromptTemplate(template=templet_prompt,
                                input_variables=["context", "question"])

        self.faiss_db = FAISS.load_local("/temp/bigmodle/langchian/QA/knowledges/faiss_index_test", self.embeddings)

        self.chain = load_qa_chain(self.llm,chain_type="stuff", prompt=self.PROMPT)


    def ask_question(self,question,topk:int = 5):
        docs = self.faiss_db.similarity_search(query=question,k=topk)
        result = self.chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return result['output_text']