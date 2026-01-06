# coding=utf-8
# Copyright 2024 The vLLM team.
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import Gemma2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import JumpReLU
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.model_executor.models.gemma2 import Gemma2Attention, Gemma2MLP
from vllm.model_executor.model_loader.sae_config import SAEConfig
from .interfaces import SupportsLoRA


class Gemma2SAEEnhanced(nn.Module):
    def __init__(
        self,
        input_size: int,
        sae_size: int,  # = hidden_size * expansion_factor
        hidden_act: str = "jumprelu",
        hidden_activation: str = "jumprelu",
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("Quantization config is not supported for SAE yet")
        self.W_enc = ColumnParallelLinear(
            input_size=input_size,
            output_size=sae_size,
            # TODO(Adriano) we may want to verify it is true that ALL of the
            # SAEs have bias. They claim to have it: https://arxiv.org/pdf/2408.05147
            # and checking on the weights the bias is present so in theory this SHOULD
            # be `True`
            bias=True,
            quant_config=quant_config,
        )
        self.W_dec = RowParallelLinear(
            input_size=sae_size,
            output_size=input_size,
            bias=True,
            quant_config=quant_config,
        )
        self.act_fn = JumpReLU(sae_size)
        if not (hidden_act == hidden_activation == "jumprelu"):
            raise ValueError(
                "GemmaScope uses `jumprelu` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`jumprelu`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Copied from Gemma2MLP classes
        gate_up, _ = self.W_enc(x)
        x = self.act_fn(gate_up)
        x, _ = self.W_dec(x)
        return x


class SAEEnhancedGemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        **kwargs,
    ) -> None:
        if "sae_config" not in kwargs:
            raise ValueError(
                "sae_config must be provided to SAEEnhancedGemma2DecoderLayer"
            )
        sae_config = kwargs.pop("sae_config")
        if sae_config is not None and not isinstance(sae_config, SAEConfig):
            raise ValueError("sae_config must be an instance of SAEConfig")
        super().__init__()
        self.sae_config = sae_config
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2Attention(
            layer_idx=layer_idx,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.sae = None
        self.sae_size = None
        if self.sae_config is not None:
            self.sae_size = (
                self.sae_config.sae_size
                or self.hidden_size * self.sae_config.expansion_factor
            )
            if self.sae_size is None:
                raise ValueError(
                    f"sae_size (={self.sae_config.sae_size}) must "
                    + f"be set (or expansion_factor={self.sae_config.expansion_factor})"
                )
            if (
                self.sae_config.input_size is not None
                and self.sae_config.input_size != self.hidden_size
            ):
                raise ValueError(
                    f"input_size (={self.sae_config.input_size}) must be the same "
                    + f"as the hidden size (={self.hidden_size})"
                )
            self.sae = Gemma2SAEEnhanced(
                input_size=self.hidden_size,
                sae_size=self.sae_size,
                hidden_act="jumprelu",
                hidden_activation="jumprelu",
                quant_config=quant_config,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor | None
    ]:  # <- MAY return None! (changed)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        # TODO(Adriano) confirm that this is not fking up with the input layernorms.
        if self.sae is not None:
            # Merge to get the residual at this point
            actual_residual_stream = hidden_states + residual
            hidden_states = self.sae(actual_residual_stream)
            return hidden_states, None  # <- next layer will proc. indep.
        return hidden_states, residual


class SAEEnhancedGemma2Model(nn.Module):
    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        **kwargs,
    ) -> None:
        if "sae_configs" not in kwargs:
            raise ValueError(
                "sae_configs must be provided to SAEEnhancedGemma2Model"
            )
        sae_configs: dict[int, SAEConfig] | None = kwargs.pop(
            "sae_configs", None
        )
        if sae_configs is None:
            sae_configs = {}
        super().__init__()
        self.config = config
        self.sae_configs = sae_configs

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                SAEEnhancedGemma2DecoderLayer(
                    layer_idx,
                    config,
                    cache_config,
                    quant_config,
                    sae_config=sae_configs.get(layer_idx, None),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states *= self.normalizer

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma2SAEEnhancedForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ) -> None:
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = SAEEnhancedGemma2Model(
            config, cache_config, quant_config, **kwargs
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping
        )
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata
        )
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.model.embed_tokens, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        initialized_saes: set[int] = set(),
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    # NOTE: we ignore these so as to make it easier to load from safetensors checkpoints
                    # where we symlink in the SAE weights for BOTH the old and new models (i.e. Gemma2
                    # and Gemma2SAEEnhanced). This means that we can save memory by not needing to store
                    # the disk contents twice.
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        # Take into account previous/seperate initializations
        actually_loaded_sae_params: set[str] = set()
        for layer_idx in initialized_saes:
            for param_names, param_types in (
                (["W_enc", "W_dec", "act_fn"], ["weight", "bias"]),
                (["act_fn"], ["thresholds"]),
            ):
                for param_name in param_names:
                    for param_type in param_types:
                        actually_loaded_sae_params.add(
                            f"model.layers.{layer_idx}.sae.{param_name}.{param_type}"
                        )
        unloaded_params = unloaded_params - actually_loaded_sae_params
        # Make sure we loaded everything
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}"
            )
