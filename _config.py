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
MODLE_PATH= '/T53/temp/bigmodel/models/chat_c'
HUGGINGFACEHUB_API_TOKEN = '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'
MODLE_ENBEDDING = '/T53/temp/bigmodel/models/sensenove'


#Qwen配置
Qwen_MODEL_PATH = '/T53/temp/bigmodel/models/qWen'

#baichuan的配置

BAICHUAN_MODEL_PATH = '/T53/temp/bigmodel/models/baichuan'
BAICHUAN_MODEL_PEFT_PATH = '/T53/temp/bigmodel/chatGlm_lora_medicinal/saved_files/Baichuan_test/checkpoint-558'
# BAICHUAN_MODEL_PEFT_PATH_2 = '/T53/temp/bigmodel/chatGlm_lora_medicinal/saved_files/Baichuan_test/checkpoint-1641'
BAICHUAN_MODEL_PEFT_PATH_2 = '/T53/temp/bigmodel/chatGlm_lora_medicinal/saved_files/Baichuan_test_01/checkpoint-1200'
BAICHUAN_MODEL_PEFT_PATH_3 = '/T53/temp/bigmodel/chatGlm_lora_medicinal/saved_files/Baichuan_test_01/checkpoint-900'
BAICHUAN_MODEL_PEFT_PATH_4 = '/T53/temp/bigmodel/models/baichuan'

#向量数据库匹配
#分数建议匹配200-500之间，总的分数区间是0-1100左右
VECTORSTORE_SCORE_THRESHOLD= 500
#加载的向量数据地址
# VECTORSTORE_STORE_PATH = "/T53/temp/bigmodel/langchian/QA/knowledges/faiss_index_test"
VECTORSTORE_STORE_PATH = "/T53/temp/bigmodel/langchian/QA/knowledges/faiss_index_test_one"

