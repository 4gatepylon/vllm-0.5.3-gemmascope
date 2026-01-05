from __future__ import annotations
from dataclasses import dataclass
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


@dataclass
class SAEConfig:
    # If input size is not set, then it is set to the hidden size of the model
    input_size: int | None = None

    # One of sae_size and expansion_factor must be provided
    sae_size: int | None = None
    expansion_factor: int | None = None

    # These must be "jumprelu"
    hidden_act: str = "jumprelu"
    hidden_activation: str = "jumprelu"

    # this is carried over from MLP
    quant_config: QuantizationConfig | None = None
    # This name or path is used by the loader specifically
    # NOTE: if this is None then the loader will not load anything and instead
    # follow the `Dummy` strategy to load in random data
    gemmascope_name_or_path: str | None = None