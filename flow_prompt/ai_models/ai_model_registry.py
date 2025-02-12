from dataclasses import dataclass, field
from typing import Type, Dict, ClassVar, Any
import typing as t
import logging
from flow_prompt.ai_models.ai_model import AIModel

logger = logging.getLogger(__name__)


class AIModelRegistry:
    _providers: ClassVar[Dict[str, Type[AIModel]]] = {}
    _required_params: ClassVar[Dict[str, set]] = {}

    @classmethod
    def register(cls, provider: str, required_params: set):
        def decorator(model_class: Type[AIModel]):
            cls._providers[provider] = model_class
            cls._required_params[provider] = required_params
            return model_class
        return decorator
    
    @classmethod
    def create(cls, provider: str, **kwargs) -> AIModel:
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        model_class = cls._providers[provider]
        required = cls._required_params[provider]
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {provider}: {missing}")
        
        return model_class(**kwargs)
