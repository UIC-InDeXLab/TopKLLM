from enum import Enum
from typing import Dict, List, Union

from models.llms import _OLLAMA_DOCKER_SUPPORTED_MODELS, OllamaDockerLLM, LLM


def _get_models_dict() -> Dict[str, Union[OllamaDockerLLM]]:
    d = {}
    for m in _OLLAMA_DOCKER_SUPPORTED_MODELS:
        d[m] = OllamaDockerLLM

    return d


def get_model_llm(model: str) -> Union[OllamaDockerLLM]:
    return _get_models_dict()[model]


def get_available_models() -> List[str]:
    return [m for m in _get_models_dict().keys()]
