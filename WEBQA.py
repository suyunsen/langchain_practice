"""
 -*- coding: utf-8 -*-
time: 2023/10/25 21:16
author: suyunsen
email: suyunsen2023@gmail.com
"""
from flask import Flask, render_template, request,redirect,url_for,jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from QA.govermentQA import GoQa


app = Flask(__name__)
CORS(app)
his = {}

templet_prompt = """
       根据以下内容回答问题,如果存在正确答案就回答正确答案，不存在就回答没有答案，不要自己制造答案。
       {context}
       问题:{question}
       记住不要多答和少答，按给出的内容做答，不需要自己编造多余答案。
    """
govermentQa = GoQa(templet_prompt=templet_prompt)

@app.route('/')
def go_index():
    # 这里使用redirect将请求重定向到新的路由（'/page'）
    return render_template('index.html')



@app.route('/process_chat',methods=['GET'])
def index():
    chat_param = request.args.get('chat')  # 获取名为'chat'的参数值
    client_ip = request.remote_addr #获取ip地址
    if client_ip not in his.keys():
        his[client_ip] = []
    response = govermentQa.ask_question(chat_param,3)
    # his[client_ip] = history
    return response

if __name__ == '__main__':
    history = []
    app.run(host='0.0.0.0',port=8080)