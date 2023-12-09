from collections import namedtuple
from transformers import BitsAndBytesConfig
from dataclasses import dataclass


@dataclass
class CloneConfig:
    def __init__(
            self,
            model_name,
            quantized
    ):
        
        self.model_name = model_name
        self.quantized = quantized