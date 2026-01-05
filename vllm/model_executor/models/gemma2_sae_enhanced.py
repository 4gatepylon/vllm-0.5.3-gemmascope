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
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.model_executor.models.gemma2 import Gemma2Attention, Gemma2MLP
from .interfaces import SupportsLoRA


class Gemma2SAEEnhanced(nn.Module):

    def __init__(
        self,
        input_size: int,
        sae_size: int,  # = hidden_size * expansion_factor
        hidden_act: str,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("Quantization config is not supported for SAE yet")
        self.gate_up_proj = ColumnParallelLinear(
            input_size=input_size,
            output_size=sae_size,
            bias=False,  # TODO(Adriano) fix this if needed
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=sae_size,
            output_size=input_size,
            bias=False,
            quant_config=quant_config,
        )
        # TODO(Adriano) not sure about a lot of the arguments here
        # self.act_fn = JumpReLU(sae_size) # Please fix this!
        if not (hidden_act == hidden_activation == "jumprelu"):
            raise ValueError(
                "Gemma2 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
    
    def act_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x # XXX fix this!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Copied from Gemma2MLP classes
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SAEEnhancedGemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        sae_config: None = None,
        # TODO(Adriano) add SAE configuration here
    ) -> None:
        super().__init__()
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
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # NOTE: you put the SAE on the layer who takes it AS OUTPUT
        # XXX adriano it looks like if this here is set to non-None then we need to have
        # the SAE in the safetensors dict for this to work! with the proper names...
        # (which somehow we need to get)
        self.sae = None
        if self.sae_config is not None:
            self.sae = Gemma2SAEEnhanced(
                input_size=config.hidden_size,
                sae_size=config.hidden_size * 8, # XXX
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
    ) -> Tuple[torch.Tensor, torch.Tensor | None]: # <- MAY return None! (changed)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
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
        # TODO(Adriano) understand where exactly to put the SAE for equality
        # with HF (it starts to be more complicated because the outptus from
        # this here are such that hidden_states + residual get summed in the
        # self.input_layernorm, meaning the SAE might need to go INTO the
        # layernorm or the layernorm may have to be changed or we may
        # prefer to rewrite/re-order a lot of this... FUCK)
        if self.sae is not None:
            # Merge to get the residual at this point
            actual_residual_stream = hidden_states + residual
            hidden_states = self.sae(actual_residual_stream)
            return hidden_states, None # <- next layer will proc. indep.
        return hidden_states, residual


class SAEEnhancedGemma2Model(nn.Module):
    # TODO(Adriano) confirm more carefully in vllm/config.py whether it is "architectures" or "name_or_path"
    # or model_type that matters (it looks like HF uses model_type to decide what to load with via automodel
    # whereas this uses architectures... in the registry in ./__init__.py``)
    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        # TODO(Adriano) add SAE configuration here
    ) -> None:
        print("=" * 100)
        print("CALLING `SAEEnhancedGemma2Model` WITH CONFIG ARCHITECTURES:")
        print("config.architectures", config.architectures)
        print("=" * 100)
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                # TODO(Adriano) pass down the SAE configuration here
                SAEEnhancedGemma2DecoderLayer(
                    layer_idx, config, cache_config, quant_config
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
        # TODO(Adriano) add SAE configuration here; I'm not sure how if at all it will
        # be passed in (unclear how the arguments are hydrated here...)
    ) -> None:
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        # TODO(Adriano) add SAE configuration here
        self.model = SAEEnhancedGemma2Model(config, cache_config, quant_config)
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
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # TODO(Adriano) deal with the SAE weights please here
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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}"
            )
