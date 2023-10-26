"""
 -*- coding: utf-8 -*-
time: 2023/10/24 18:19
author: suyunsen
email: suyunsen2023@gmail.com
"""

from langchain.document_loaders.csv_loader import CSVLoader
import csv
from typing import List
import pandas as pd
from langchain.docstore.document import Document

class CSV_load:

    file_path:str

    def __init__(self,file_path):
        self.file_path = file_path

    def one_text_document(self)->List:
        ans_arr = []
        idx = 0
        with open(self.file_path, newline='',encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                knowledge = row[1]+"的"+row[2]+"的网上办理流程是:" + row[3] if len(row[3]) > 3 else "无办理流程"
                metadata = {"source": "in", "date": "text"}
                docs = Document(page_content=knowledge, metadata={})
                ans_arr.append(docs)
        return  ans_arr

    def get_execl(self)->List:
        ans_arr = []
        df = pd.read_excel(self.file_path,header=0)
        df.to_csv("../data/test.csv")
        # for column_name in df.columns:
        #     column_data = df[column_name]
        #     # print(f"列名: {column_name}")
        #     print(type(column_data))
        #     for v in column_data:
        #         print(v)
        #     # print("")
        #     break
        return ans_arr

if __name__ == '__main__':
    csvs = CSV_load("../data/test.csv")
    da = csvs.one_text()
    idx = 0
    print(da[len([da]) - 3])
