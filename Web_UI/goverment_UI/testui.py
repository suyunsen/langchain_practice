"""
 -*- coding: utf-8 -*-
time: 2023/11/2 21:16
author: suyunsen
email: suyunsen2023@gmail.com
"""

import gradio as gr
import random
import time
from transformers import AutoTokenizer, AutoModel
import _config

model = AutoModel.from_pretrained(_config.MODLE_PATH,trust_remote_code=True).half().cuda()
token = AutoTokenizer.from_pretrained(_config.MODLE_PATH,trust_remote_code=True)

history_me = []

def parse_codeblock(output):
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return ''.join(lines)

def respond(message):
    response,hist = model.chat(token,message)
    history_me.append((message, parse_codeblock(response)))
    return history_me

def clears():
    history_me = []
    return history_me


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=500)
    msg = gr.Textbox()
    clear = gr.ClearButton()


    msg.submit(respond, inputs = [msg], outputs = [chatbot])
    clear.click(clears,outputs=[chatbot])

demo.queue().launch(server_name="0.0.0.0",server_port=7580)