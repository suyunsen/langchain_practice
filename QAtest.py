"""
 -*- coding: utf-8 -*-
time: 2023/10/25 20:29
author: suyunsen
email: suyunsen2023@gmail.com
"""

from QA.govermentQA import GoQa
from Custom.Custom_SparkLLM import Spark


def chatGLM():
    # q = "啊啊啊"
    q1 = '广东省的企业职工基本养老金网上如何申请。'
    # q = "怎么办理沙坑社区城镇非职工居民独生子女保健费发放"
    # q = "怎么办理沙坑社区城镇非职工居民独生子女保健费发放问题"
    q = "怎么办理沙坑社区申领居住证"
    # templet_prompt = """
    #     假如你是一名问答专家，你需要根据一下给出的内容找出问题的正确答案。
    #     答案只存在给出的内容中，你知道就回答，不要自己编造答案。
    #     这是给出的内容：{context}
    #     问题是:{question}
    #     因为你是问答专家你需要仔细分析问题和给出的内容，不要给出多余的答案。
    #     按给出的内容作答，你不需要自己总结。
    #    """

    templet_prompt = """
         假如你是一名问答专家，你需要根据一下给出的内容找出问题的正确答案。
         答案只存在给出的内容中，你知道就回答，不要自己编造答案。
         因为你是问答专家你需要仔细分析问题和给出的内容，不要给出错误答案，不要给出多余的答案。
         记住你只需要按给出的内容作答，不需要自己总结。
         这是给出的内容：{context}
         问题是:{question}
        """
    govermentQa = GoQa(templet_prompt=templet_prompt)
    qs = []
    qs.append(q)
    qs.append(q1)
    for v in qs:
        ans = govermentQa.ask_question(v ,6)
        print(ans["response"])
        # print(f"这是源{ans['source']}------------")
def spark():
    templet_prompt = """
              根据以下内容回答问题,如果存在正确答案就回答正确答案，不存在就回答没有答案，不要自己制造答案。
              {context}
              问题:{question}
              记住不要多答和少答，按给出的内容做答，不需要自己编造多余答案。
           """
    llm = Spark()

if __name__ == '__main__':
    chatGLM()


    tt = """1.单位或个人在申领人达到国家规定退休年龄当月通过网上申请，填写申报信息。
    2.省社保局根据单位或个人提交的资料，将及时对申办业务进行受理。受理通过的单位或个人，无需再到前台窗口办理。"""