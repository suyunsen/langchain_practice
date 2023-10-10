"""
 -*- coding: utf-8 -*-
time: 2023/10/10 22:07
author: suyunsen
email: suyunsen2023@gmail.com
"""
from getpass import getpass
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os
from Custom_embeddings.SbertEnbedding import SBertEmbeddings

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



if __name__ == '__main__':
    embeddings = SBertEmbeddings()
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text])
