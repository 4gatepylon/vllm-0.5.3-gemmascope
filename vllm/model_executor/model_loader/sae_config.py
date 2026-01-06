from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)


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

    # Valid configurations here are
    # - SAE config was never never created in the first place -> no SAE (identity)
    # - All these 3 are None -> random SAE (test runtime/basic functionality)
    # - (str, None, None) -> Load directly from path; must be npz
    # - (None, str, str) -> Determine the path and load from there using
    #   code copied from SAELens
    gemmascope_name_or_path: str | None = None
    gemmascope_release: str | None = None  # repo_id
    gemmascope_folder_name: str | None = None  # folder_name

    def get_sae_weights_iterator(
        self,
        force_download: bool = False,  # hf option
        device: str | torch.device = "cpu",  # where to put it
        ensure_strict_keys: bool = True,  # ensure all keys present
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Basically copied from https://github.com/decoderesearch/SAELens/blob/0d66c18d18c456152175238bfb7569cb8b49270d/sae_lens/loading/pretrained_sae_loaders.py#L478

        Output key/value iterator of like:
        ```
        {
            "W_dec": "torch.Size([16384, 2304])",
            "W_enc": "torch.Size([2304, 16384])",
            "b_dec": "torch.Size([2304])",
            "b_enc": "torch.Size([16384])",
            "threshold": "torch.Size([16384])"
        }
        ```
        """
        strict_keys: set[str] = {
            "W_dec",
            "W_enc",
            "b_dec",
            "b_enc",
            "threshold",
        }
        path, repo_id, folder_name = (
            self.gemmascope_name_or_path,
            self.gemmascope_release,
            self.gemmascope_folder_name,
        )
        if path is None:
            if repo_id is None or folder_name is None:
                raise ValueError(
                    "Either gemmascope_name_or_path (and NEITHER), or both gemmascope_release and gemmascope_folder_name must be provided; "
                    + f"got (repo_id={repo_id}, folder_name={folder_name}) and path={path}"
                )
            # Acquire the path
            path = hf_hub_download(
                repo_id=repo_id,
                filename="params.npz",
                subfolder=folder_name,
                force_download=force_download,
            )
            assert isinstance(path, str) and Path(path).exists(), (
                f"Path `{path}` does not exist or is wrong type: {type(path)}"
            )
        elif repo_id is not None or folder_name is not None:
            raise ValueError(
                "Either gemmascope_name_or_path (and NEITHER), or both gemmascope_release and gemmascope_folder_name must be provided; "
                + f"got (repo_id={repo_id}, folder_name={folder_name}) and path={path}"
            )

        # Load from the path
        seen_keys: set[str] = set()
        with np.load(path) as data:
            for key in data:
                state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
                state_dict_value = (
                    torch.tensor(data[key]).to(dtype=torch.float32).to(device)
                )
                yield state_dict_key, state_dict_value
                seen_keys.add(state_dict_key)
        # TODO(Adriano) add support for pruning "thresholds"
        remaining_keys = strict_keys - seen_keys
        if remaining_keys and ensure_strict_keys:
            raise ValueError(
                f"Expected keys {strict_keys} but got {seen_keys} from {path}"
            )

    def get_sae_parameter_path(
        self,
        weight_state_dict_path: str,
    ) -> str:
        strict_keys: dict[str, str] = {
            "W_dec": f"W_dec.weight",
            "b_dec": f"W_dec.bias",
            "W_enc": f"W_enc.weight",
            "b_enc": f"W_enc.bias",
            "threshold": f"act_fn.thresholds",
        }
        if weight_state_dict_path not in strict_keys:
            raise ValueError(
                f"Expected keys {strict_keys} but got {weight_state_dict_path}"
            )
        return strict_keys[weight_state_dict_path]
