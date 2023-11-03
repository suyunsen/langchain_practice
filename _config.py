"""
 -*- coding: utf-8 -*-
time: 2023/10/8 15:55
author: suyunsen
email: suyunsen2023@gmail.com
"""


MODLE_KEY = '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'
MODLE_NAME = 'gpt-3.5-turbo'
MODLE_MAX_TOKEN = 2000


#llm和embedding模型配置
# MODLE_PATH= '/T53/temp/bigmodle/chatGLM3-6b'
MODLE_PATH= '/T53/temp/bigmodle/chat_c'
HUGGINGFACEHUB_API_TOKEN = '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'
MODLE_ENBEDDING = '/T53/temp/bigmodle/sensenove'


#向量数据库匹配
#分数建议匹配200-500之间，总的分数区间是0-1100左右
VECTORSTORE_SCORE_THRESHOLD= 300
#加载的向量数据地址
VECTORSTORE_STORE_PATH = "/T53/temp/bigmodle/langchian/QA/knowledges/faiss_index_test"

