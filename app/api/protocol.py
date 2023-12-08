import time
import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class RewardRequest(BaseModel):
    source: Optional[str] = None
    prompt: str = None
    texts: List[str] = Field(default_factory=list)


class RewardResponse(BaseModel):
    res: Dict[int, float] = None
    duration: int = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    cost_time: Optional[int] = 0
    device_name: Optional[str] = None
    total_memory: Optional[float] = 0.0
    device_count: Optional[int] = 0


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    source: Optional[str] = None
    # Additional parameters supported by ft
    repetition_penalty: Optional[float] = 0.0
    beam_width: Optional[int] = 1
    beam_search_diversity_rate: Optional[float] = 0.0
    min_length: Optional[int] = 1
    streaming: Optional[bool] = False
    streaming_interval: Optional[int] = 5
    truncation_length: Optional[int] = 2048


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str,
    float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class CompletionResponse(BaseModel):
    # id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    # object: str = "text_completion"
    # created: int = Field(default_factory=lambda: int(time.time()))
    # model: str
    choices: List[CompletionResponseChoice]
    # usage: UsageInfo
