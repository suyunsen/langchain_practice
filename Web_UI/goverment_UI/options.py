"""
 -*- coding: utf-8 -*-
time: 2023/10/26 16:41
author: suyunsen
email: suyunsen2023@gmail.com
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, default="7800")
parser.add_argument("--model-path", type=str, default="/gemini/data-1")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["fp16", "int4", "int8"], default="fp16")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--cpu", action='store_true', help="use cpu")
parser.add_argument("--share", action='store_true', help="use gradio share")
