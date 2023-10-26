"""
 -*- coding: utf-8 -*-
time: 2023/10/10 19:57
author: suyunsen
email: suyunsen2023@gmail.com
"""
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env
import _config

#自定义大模型，且模型存在与本地
class ChatGlm26bData(BaseModel):
    """Parameters for AI21 penalty data."""

    scale: int = 0
    applyToWhitespaces: bool = True
    applyToPunctuations: bool = True
    applyToNumbers: bool = True
    applyToStopwords: bool = True
    applyToEmojis: bool = True

class ChatGlm26b(LLM):

    history = []
    model: str = "ChatGLM2-6B"
    """Model name to use."""

    temperature: float = 0.1
    """What sampling temperature to use."""

    maxTokens: int = 2048
    """The maximum number of tokens to generate in the completion."""

    minTokens: int = 0
    """The minimum number of tokens to generate in the completion."""

    topP: float = 1.0
    """Total probability mass of tokens to consider at each step."""

    presencePenalty: ChatGlm26bData = ChatGlm26bData()
    """Penalizes repeated tokens."""

    countPenalty: ChatGlm26bData = ChatGlm26bData()
    """Penalizes repeated tokens according to count."""

    frequencyPenalty: ChatGlm26bData = ChatGlm26bData()
    """Penalizes repeated tokens according to frequency."""

    numResults: int = 1
    """How many completions to generate for each prompt."""

    logitBias: Optional[Dict[str, float]] = None
    """Adjust the probability of specific tokens being generated."""

    ai21_api_key: Optional[str] = None

    stop: Optional[List[str]] = None

    base_url: Optional[str] = None

    filepath:str = _config.MODLE_PATH
    extokenizer = AutoTokenizer.from_pretrained(filepath, trust_remote_code=True)
    exemodel = AutoModel.from_pretrained(filepath, trust_remote_code=True).half().cuda()
    exemodel = exemodel.eval()
    """Base url to use, if None decides based on model name."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling AI21 API."""
        return {
            "temperature": self.temperature,
            "maxTokens": self.maxTokens,
            "minTokens": self.minTokens,
            "topP": self.topP,
            "presencePenalty": self.presencePenalty.dict(),
            "countPenalty": self.countPenalty.dict(),
            "frequencyPenalty": self.frequencyPenalty.dict(),
            "numResults": self.numResults,
            "logitBias": self.logitBias,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model

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
        # print('------------------------------')
        response, history = self.exemodel.chat(self.extokenizer, prompt, history=self.history)
        self.history = []
        return response
