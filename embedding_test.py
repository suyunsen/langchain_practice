"""
 -*- coding: utf-8 -*-
time: 2023/10/25 18:51
author: suyunsen
email: suyunsen2023@gmail.com
"""

from Custom_embeddings.SbertEnbedding import  SBertEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import text
from QA.data_load import CSV_load
import _config

def load_faiss_index():
    embeddings = SBertEmbeddings()
    db = FAISS.load_local(_config.VECTORSTORE_STORE_PATH, embeddings)
    query = "用户需要在窗口办理社保业务。"
    query01 = "wwww"
    docs_and_scores = db.similarity_search(query=query,k=10,**{"score_threshold":400})
    print(len(docs_and_scores))
    for v in docs_and_scores:
        print(v.page_content)
    # print(docs_and_scores[0].page_content)

if __name__ == '__main__':
    # metadata = {"source": "internet", "date": "text"}
    # text = '你好啊'
    # docs = CSV_load('./data/test.csv').one_text_document()
    # embeddings = SBertEmbeddings()
    # db = FAISS.from_documents(docs, embeddings)
    # db.save_local("./QA/knowledges/faiss_index_test")
    load_faiss_index()
