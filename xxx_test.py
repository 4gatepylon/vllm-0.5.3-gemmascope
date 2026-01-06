from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # YOU must set this properly!
import vllm.model_executor.models as models_module
from transformers import AutoConfig
import torch

models_module._MODELS["Gemma2ForCausalLM"] = (
    "gemma2_sae_enhanced",
    "Gemma2SAEEnhancedForCausalLM",
)

# Import your SAEConfig
from vllm.model_executor.models.gemma2_sae_enhanced import SAEConfig
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

sae_configs = {
    # from real files
    # 0: SAEConfig(
    #     input_size=2304,  # this can be inferred, but is done here defensively
    #     sae_size=16384,
    #     gemmascope_release="gemma-scope-2b-pt-res-canonical",  # release/repo_id
    #     gemmascope_folder_name="layer_0/width_16k/canonical",  # sae_id/folder_name
    # ),
    # random init
    # 11: SAEConfig(expansion_factor=8),
    # 12: SAEConfig(sae_size=4096),
}

llm = LLM(
    model="google/gemma-2-2b",
    load_format="gemmascope",
    model_loader_extra_config={"sae_configs": sae_configs},
    device="cuda:0",
)
# Verify it loaded your custom class
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
print(f"Model class: {type(model)}")

# Run inference
message = "A: B, B: D, D:E, F: G, H: I, I: J, J: K, K: L, L: M, M: N, N: O,"

print("=" * 100)
print("Compare tokenization:")
vllm_tokenizer = llm.get_tokenizer()
tf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Compare tokenization
hf_tokens = tf_tokenizer.encode(message)
vllm_tokens = vllm_tokenizer.encode(message)
print(f"HF tokens: {hf_tokens}")
print(f"vLLM tokens: {vllm_tokens}")
equality_array = [
    hf_tokens[:i] == vllm_tokens[:i]
    for i in range(min(len(hf_tokens), len(vllm_tokens)))
]
print(f"equal to length i: {equality_array}")
print(f"all equal: {all(equality_array)}")
print("=" * 100)

# fmt: off
# Create HF inputs/outputs
print("=" * 100)
print("HF inputs/outputs:")
tf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:1"}
)
tf_inputs = {k: v.to(tf_model.device) for k, v in tf_tokenizer(message, return_tensors="pt").items()}
tf_inputs_length = tf_inputs["input_ids"].shape[1]
tf_outputs = tf_model.generate(**tf_inputs, max_new_tokens=25)
tf_outputs = tf_tokenizer.batch_decode(tf_outputs[..., tf_inputs_length:], skip_special_tokens=True)[0]
print(f"HF outputs: {tf_outputs}")
print("=" * 100)
print("VLLM inputs/outputs:")
outputs = llm.generate(
    [message],
    SamplingParams(max_tokens=25),
)
print(outputs[0].outputs[0].text)
# fmt: on
