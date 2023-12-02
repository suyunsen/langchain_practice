"""
 -*- coding: utf-8 -*-
time: 2023/11/2 16:49
author: suyunsen
email: suyunsen2023@gmail.com
"""

from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from Custom_embeddings.SbertEnbedding import SBertEmbeddings
import _config

class MyFaiss(FAISS):

    def __init__(self,score_threshold=_config.VECTORSTORE_SCORE_THRESHOLD):
        super().__init__()
        self.vectorstore_score_threshold = score_threshold
        self.embedding = SBertEmbeddings()

    def search(self,query:str,k:int):
        ans_doc = self.similarity_search(query=query,k=k)

        for v in ans_doc:
            print(v)





if __name__ == '__main__':
    myfaiss = MyFaiss()
