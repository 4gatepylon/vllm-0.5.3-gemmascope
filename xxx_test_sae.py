from sae_lens import SAE  # pip install sae-lens

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_0/width_16k/canonical",
)
print(sae)
