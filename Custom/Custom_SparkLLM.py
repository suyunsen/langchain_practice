"""
 -*- coding: utf-8 -*-
time: 2023/10/14 17:55
author: suyunsen
email: suyunsen2023@gmail.com
"""

import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import ssl
import websocket
import langchain
import logging
from config import SPARK_APPID, SPARK_API_KEY, SPARK_API_SECRET,SPARK_URL
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from typing import Optional, List, Dict, Mapping, Any
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache


result_list = []
def _construct_query(question, temperature, max_tokens):
    data = {
        "header": {
            "app_id":'995896d3',
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": "generalv2",
                "random_threshold": temperature,
                "max_tokens": max_tokens,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data


def _run(ws, *args):
    data = json.dumps(
        _construct_query(question=ws.question, temperature=ws.temperature, max_tokens=ws.max_tokens))
    # print (data)
    ws.send(data)

def getText(role,content):
    text = []
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text
def on_error(ws, error):
    print("error:", error)


# 收到websocket关闭的处理
def on_close(ws,one,two):
    print("closed...")


def on_open(ws):
    thread.start_new_thread(_run, (ws,))


def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    # print(data)
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        result_list.append(content)
        if status == 2:
            ws.close()
            setattr(ws, "content", "".join(result_list))
            result_list.clear()


class Spark(LLM):
    '''
    根据源码解析在通过LLMS包装的时候主要重构两个部分的代码
    _call 模型调用主要逻辑,输入问题，输出模型相应结果
    _identifying_params 返回模型描述信息，通常返回一个字典，字典中包括模型的主要参数
    '''

    spak_url = SPARK_URL  # spark官方模型提供api接口
    host = urlparse(spak_url ).netloc  # host目标机器解析
    path = urlparse(spak_url ).path  # 路径目标解析
    max_tokens = 8000
    temperature = 0.1

    # ws = websocket.WebSocketApp(url='')

    @property
    def _llm_type(self) -> str:
        # 模型简介
        return "Spark"

    def _get_url(self):
        # 获取请求路径
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        signature_sha = hmac.new(SPARK_API_SECRET.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{SPARK_API_KEY}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        url = self.spak_url + '?' + urlencode(v)
        return url

    def _post(self, prompt):
        # 模型请求响应
        websocket.enableTrace(False)
        wsUrl = self._get_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error,
                                    on_close=on_close, on_open=on_open)
        ws.question = getText('user',prompt)
        setattr(ws, "temperature", self.temperature)
        setattr(ws, "max_tokens", self.max_tokens)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return ws.content if hasattr(ws, "content") else ""

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None) -> str:
        # 启动关键的函数
        content = self._post(prompt)
        # content = "这是一个测试"
        return content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        _param_dict = {
            "url": self.spak_url
        }
        return _param_dict

