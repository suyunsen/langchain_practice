"""
 -*- coding: utf-8 -*-
time: 2023/10/26 16:15
author: suyunsen
email: suyunsen2023@gmail.com
"""

import json
import os
import time
import sys

import gradio as gr

from options import parser
from typing import Optional

sys.path.append("../../")
from QA.govermentQA import GoQa
from Custom.ChatGLM import ChatGlm26b
from Custom.Custom_SparkLLM import Spark

history = []
readable_history = []
cmd_opts = parser.parse_args()

templet_prompt = """
        假如你是一名语文专家，你需要根据一下给出的内容找出问题的正确答案。
        答案只存在给出的内容中，你知道就回答，不要自己编造答案。
        这是给出的内容：{context}
        问题是:{question}
        因为你是语文专家你需要仔细分析问题和给出的内容，不要给出多余的答案。
        按给出的内容作答，你不需要自己总结。
       """
llm = ChatGlm26b()
sparkllm:Optional[Spark] = None
qa_chain = GoQa(llm=llm,templet_prompt=templet_prompt)

chat_h = """<h2><center>ChatGLM WebUI</center></h2>"""
spark_h = """<h2><center>Spark WebUI</h2>"""
head_h =chat_h

_css = """
#del-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
    margin: 1.5em 0;
}
"""


def prepare_model():
    global model
    if cmd_opts.cpu:
        model = model.float()
    else:
        if cmd_opts.precision == "fp16":
            model = model.half().cuda()
        elif cmd_opts.precision == "int4":
            model = model.half().quantize(4).cuda()
        elif cmd_opts.precision == "int8":
            model = model.half().quantize(8).cuda()

    model = model.eval()


# prepare_model()


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


def predict(query, max_length, top_p, temperature):
    global history
    llm.set_llm_temperature(temperature)
    output = qa_chain.ask_question(query,int(top_p))
    readable_history.append((query, parse_codeblock(output)))
    # print(output)
    return  readable_history


def save_history():
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    s = [{"q": i[0], "o": i[1]} for i in history]
    filename = f"save-{int(time.time())}.json"
    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        f.write(json.dumps(s, ensure_ascii=False))


def load_history(file):
    global history, readable_history
    try:
        with open(file.name, "r", encoding='utf-8') as f:
            j = json.load(f)
            _hist = [(i["q"], i["o"]) for i in j]
            _readable_hist = [(i["q"], parse_codeblock(i["o"])) for i in j]
    except Exception as e:
        print(e)
        return readable_history
    history = _hist.copy()
    readable_history = _readable_hist.copy()
    return readable_history


def clear_history():
    history.clear()
    readable_history.clear()
    return gr.update(value=[])

def load_chatGlm():
    qa_chain.set_llm_modle(llm)
    head_h = chat_h
    La = gr.HTML(head_h)
    return La,clear_history()


def load_spark():
    global sparkllm
    if sparkllm is None:
        sparkllm = Spark()
    qa_chain.set_llm_modle(sparkllm)
    head_h = spark_h
    La = gr.HTML(head_h)
    return La,clear_history()




def create_ui():
    with gr.Blocks(css=_css) as demo:
        prompt = "输入你的内容..."
        with gr.Row():
            with gr.Column(scale=3):
                La = gr.HTML(head_h)
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
                            top_p = gr.Slider(minimum=1, maximum=15, step=1, label='检索返回Top P', value=5)
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.01)

                        # with gr.Row():
                        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button("清空对话（上下文）")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

                        with gr.Row():
                            chatGLM_load = gr.Button("加载chatGLM模型")
                            spark_load = gr.Button("加载星火模型")
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=500)
                with gr.Row():
                    message = gr.Textbox(placeholder=prompt, show_label=False, lines=2)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送")

        submit.click(predict, inputs=[
            message,
            max_length,
            top_p,
            temperature
        ], outputs=[
            chatbot
        ])

        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[message], outputs=[message])

        chatGLM_load.click(load_chatGlm,outputs=[La,chatbot])
        spark_load.click(load_spark,outputs=[La,chatbot])

        save_his_btn.click(save_history)

        load_his_btn.upload(load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])

    return demo


ui = create_ui()
ui.queue().launch(
    server_name="0.0.0.0",
    server_port=cmd_opts.port,
    share=cmd_opts.share
)
