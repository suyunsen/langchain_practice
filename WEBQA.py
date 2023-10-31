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
         假如你是一名问答专家，你需要根据一下给出的内容找出问题的正确答案。
         答案只存在给出的内容中，你知道就回答，不要自己编造答案。
         因为你是问答专家你需要仔细分析问题和给出的内容，不要给出错误答案，不要给出多余的答案。
         记住你只需要按给出的内容作答，不需要自己总结。
         如果你认为给出的问题和给出的内容相关性不大，你需要回复：请详细说明您咨询问题的地址和办理的事项，如...。
         这是给出的内容：{context}
         问题是:{question}
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
    app.run(host='0.0.0.0',port=7580)