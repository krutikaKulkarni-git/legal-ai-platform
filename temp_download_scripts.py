import os
from huggingface_hub import snapshot_download

print("Starting download of CUAD dataset...")

# Since we cd'd into legal-ai-platform, this points to your intended folder
local_dir = "./data/raw_contracts"

snapshot_download(
    repo_id="theatticusproject/cuad",
    repo_type="dataset",
    allow_patterns="CUAD_v1/full_contract_txt/*",  # This pulls the clean text versions of the 510 contracts
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Download complete! Files saved to {local_dir}")