import vllm.model_executor.models as models_module
from transformers import AutoConfig

models_module._MODELS["Gemma2ForCausalLM"] = ("gemma2_sae_enhanced", "Gemma2SAEEnhancedForCausalLM")

# Import your SAEConfig
from vllm.model_executor.models.gemma2_sae_enhanced import SAEConfig
from vllm import LLM, SamplingParams
sae_configs = {
    # random init
    11: SAEConfig(expansion_factor=8, gemmascope_name_or_path=None),
}

llm = LLM(
    model="google/gemma-2-2b",
    load_format="gemmascope",
    model_loader_extra_config={"sae_configs": sae_configs},
)
# Verify it loaded your custom class
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
print(f"Model class: {type(model)}")

# Run inference
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)