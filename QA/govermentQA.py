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
from langchain import LLMChain
import logging
from Custom.Custom_SparkLLM import Spark

from Custom.BaiChuan import Baichuan

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class GoQa:

    modle_name = _config.MODLE_NAME #后面用于加载别的模型目前默认

    llm:LLM = None

    embeddings:Embeddings=None

    faiss_db:FAISS = None

    NOT_PROMPT = PromptTemplate(template="""{query}""",
                                     input_variables=["query"])
    vectorstore_score_threshold = _config.VECTORSTORE_SCORE_THRESHOLD
    def __init__(self,llm = None , templet_prompt:str=None,faiss_index:str=_config.VECTORSTORE_STORE_PATH):
        if llm is not None:
            self.llm = llm
        else :
            self.llm = ChatGlm26b()
        # self.llm = Spark()
        self.embeddings = SBertEmbeddings()
        self.PROMPT = PromptTemplate(template=templet_prompt,
                                input_variables=["context", "question"])

        self.faiss_index = faiss_index


        self.faiss_db = FAISS.load_local(self.faiss_index, self.embeddings)

        self.chain = load_qa_chain(self.llm,chain_type="stuff", prompt=self.PROMPT)

        self.not_template_chain = LLMChain(llm=self.llm,prompt=self.NOT_PROMPT)


    def ask_question(self,question,topk:int = 10):
        print("检索开始")
        docs = self.faiss_db.similarity_search(query=question,k=topk,**{"score_threshold":self.vectorstore_score_threshold})
        print("检索结束")
        if len(docs) > 0:
            result = self.chain({"input_documents": docs, "question": question})
            return {"response":result['output_text'],
                    "source":result['input_documents']}

        return {"response":self.not_template_chain.run(query=question),"source":""}

    def get_faiss_index(self,faiss_index):
        self.faiss_index = faiss_index

    def get_no_konwledge_answer(self,question):
        return {"response": self.not_template_chain.run(query=question), "source": ""}

    def set_llm_modle(self,llm):
        self.llm = llm
        self.chain.llm_chain.llm= llm
        self.not_template_chain.llm = llm


