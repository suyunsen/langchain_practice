"""
 -*- coding: utf-8 -*-
time: 2023/11/19 18:31
author: suyunsen
email: suyunsen2023@gmail.com
"""
import logging
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModel , AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env
import torch
import _config

from peft import PeftModel, PeftConfig

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#自定义大模型，且模型存在与本地
class BaiChuanData(BaseModel):
    """Parameters for AI21 penalty data."""

    scale: int = 0
    applyToWhitespaces: bool = True
    applyToPunctuations: bool = True
    applyToNumbers: bool = True
    applyToStopwords: bool = True
    applyToEmojis: bool = True

class Baichuan(LLM):

    history = []
    model: str = "ChatGLM2-6B"
    """Model name to use."""

    temperature: float = 0.01
    """What sampling temperature to use."""

    maxTokens: int = 4096
    """The maximum number of tokens to generate in the completion."""

    minTokens: int = 0
    """The minimum number of tokens to generate in the completion."""

    topP: float = 1.0
    """Total probability mass of tokens to consider at each step."""

    # presencePenalty: ChatGlm26bData = BaiChuanData()
    """Penalizes repeated tokens."""

    # countPenalty: BaiChuan213bData = BaiChuanData()
    """Penalizes repeated tokens according to count."""

    # frequencyPenalty: ChatGlm26bData = BaiChuanData()
    """Penalizes repeated tokens according to frequency."""

    numResults: int = 1
    """How many completions to generate for each prompt."""

    logitBias: Optional[Dict[str, float]] = None
    """Adjust the probability of specific tokens being generated."""

    ai21_api_key: Optional[str] = None

    stop: Optional[List[str]] = None

    base_url: Optional[str] = None

    filepath:str = _config.BAICHUAN_MODEL_PATH
    peft_path = _config.BAICHUAN_MODEL_PEFT_PATH
    peft_path_2 =_config.BAICHUAN_MODEL_PEFT_PATH_2

    extokenizer:Optional['BaichuanTokenizer'] = None
    exemodel:Optional['BaichuanModel'] = None
    baichuan1: Optional['BaichuanModel'] = None
    baichuan2: Optional['BaichuanModel'] = None
    baichuan3: Optional['BaichuanModel'] = None
    baichuan4: Optional['BaichuanModel'] = None
    class Config:
        arbitrary_types_allowed = True

    """Base url to use, if None decides based on model name."""

    def __init__(self):
        super(Baichuan,self).__init__()
        # if filepath is not None:
        #     self.filepath = filepath
        #量化模型 现在没用
        q_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type='nf4',
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.float32)

        self.extokenizer = AutoTokenizer.from_pretrained(self.filepath,use_fast = False, trust_remote_code=True)

        basemodel = AutoModelForCausalLM.from_pretrained(self.filepath, device_map="auto",
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True)

        basemodel.generation_config = GenerationConfig.from_pretrained(self.filepath)

        # self.baichuan1 = PeftModel.from_pretrained(basemodel,self.peft_path)
        # self.baichuan2 = PeftModel.from_pretrained(basemodel,self.peft_path_2)
        # self.baichuan3 = PeftModel.from_pretrained(basemodel, _config.BAICHUAN_MODEL_PEFT_PATH_3)
        self.baichuan4 = basemodel

        self.exemodel = basemodel


    def set_baichuanmodel(self,typ:int):
        if typ == 1:
            self.exemodel = self.baichuan1
        elif typ == 2:
            self.exemodel = self.baichuan2
        elif typ == 3:
            self.exemodel = self.baichuan3
        elif typ == 4:
            self.exemodel = self.baichuan4

    def get_model(self):
        return  self.exemodel
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling AI21 API."""
        return {
            "temperature": self.temperature,
            "maxTokens": self.maxTokens,
            "minTokens": self.minTokens,
            "topP": self.topP,
            "numResults": self.numResults,
            "logitBias": self.logitBias,
        }

    def set_temperature(self,temperature):
        self.temperature = temperature

    def clea_history(self):
        self.history = []
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model

    def set_llm_temperature(self,temperature:float):
        if temperature < 0.01 or temperature > 0.99:
            logging.error("超出范围")
        self.temperature = temperature

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to AI21's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = ai21("Tell me a joke.")
        """
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        # print(prompt)
        print('------------Baichuan------------------')
        # print(prompt)
        self.history.append({"role": "user",
                         "content": prompt})
        response = self.exemodel.chat(self.extokenizer, self.history)

        # self.history = []
        # 开启多轮对话
        self.history.append({"role":"assistant","content":response})
        print(self.history)

        return response
