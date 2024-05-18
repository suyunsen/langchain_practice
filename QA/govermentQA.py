"""
 -*- coding: utf-8 -*-
time: 2023/10/25 19:50
author: suyunsen
email: suyunsen2023@gmail.com
"""

import _config
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from Custom.ChatGLM import ChatGlm26b
from langchain.schema.embeddings import Embeddings
from Custom_embeddings.SbertEnbedding import  SBertEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import LLMChain
import logging
from langchain.docstore.document import Document
from Custom.Custom_SparkLLM import Spark
from typing import Dict

from Custom.BaiChuan import Baichuan

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

embedding_modle = SBertEmbeddings()
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
        global embedding_modle
        self.embeddings = embedding_modle
        self.PROMPT = PromptTemplate(template=templet_prompt,
                                input_variables=["context", "question"])

        self.faiss_index = faiss_index


        self.faiss_db = self.faiss_db = FAISS.load_local(self.faiss_index, self.embeddings,allow_dangerous_deserialization=True)

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

class RAG_QA:

    prompt_llm: LLM = None
    muti_llm: LLM = None
    embeddings: Embeddings = None
    faiss_db: FAISS = None
    vectorstore_score_threshold = _config.VECTORSTORE_SCORE_THRESHOLD

    NOT_PROMPT = PromptTemplate(template="""{query}""",
                                input_variables=["query"])
    new_templet_prompt = """
        假如你是一名获取用户问题信息的专家你需要和用户进行交流，获取用户问题的完整信息，对用户的提问进行最后总结，在进行交流时你需要遵循下面规则：
        1.你只需要获取到用户办理事项是窗口办理还是网上办理。
        2.你在和用户交流时尽可能简单交流，不需要说无用的废话。
        3.在获取到第1点所要求的信息后，你需要根据交流得到的信息对用户的问题进行一个总结。
        4.你只需要询问用户问题的详细细节，并最后对问题进行总结，记住你不需要回答用户提问。
        5.你一定需要在最后对用户的提问进行总结，否则会出现重大事故。记住进行最后总结时一定要出现"最后总结："这个词。
        下面只是一个对话例子其中<H>是用户的提问，<M>是你的回答。记住你所有回答一定都要模仿这个例子。
        例子如下：
    """
    new_templet_exampl = """
        <H>我要办理身份证。
        <M>请问您是需要网上办理，还是窗口办理。
        <H>窗口办理
        <M>进行最后总结：用户需要在窗口办理身份证。
    """
    def __init__(self,llm1 = None , templet_prompt:str=None,intemplet_prompt:str=None,faiss_index:str=_config.VECTORSTORE_STORE_PATH):
        self.prompt_llm = llm1
        # self.muti_llm = llm2
        global embedding_modle
        self.embeddings = embedding_modle
        self.faiss_index = faiss_index
        self.faiss_db = FAISS.load_local(self.faiss_index, self.embeddings,allow_dangerous_deserialization=True)
        self.ANS_PROMPT = PromptTemplate(template=templet_prompt,
                                    input_variables=["context", "question"])
        self.INTENTION_PROMPt = PromptTemplate(template=intemplet_prompt,
                                          input_variables=["query"])

        self.chain = load_qa_chain(self.prompt_llm, chain_type="stuff", prompt=self.ANS_PROMPT)




        self.flag = 0
    def set_flag(self):
        self.flag = 0
    def exec(self,text,topk):
        if self.flag == 0:
            self.prompt_llm.history = []
            self.prompt_llm.history.append({"role":"system","content":self.new_templet_prompt+self.new_templet_exampl})
            res  = self.prompt_llm(text)
            self.flag = 1
        else:
            res = self.prompt_llm(text)

        if "总结" in res:
            self.flag = 0
            res = res.split("总结：")[1]
            self.prompt_llm.history = []
            docs = self.faiss_db.similarity_search(query=res, k=topk,
                                                   **{"score_threshold": self.vectorstore_score_threshold})
            new_docs = []
            for v in docs:
                content = v.page_content+"办理流程是：" + v.metadata['source']
                new_docs.append(Document(page_content=content, metadata={}))
            res = self.chain({"input_documents": new_docs, "question": res})
            self.prompt_llm.history = []
        if not isinstance(res,dict):
            res = {"output_text":res,"input_documents":""}
        return {"response": res['output_text'],
                "source": res['input_documents']}





