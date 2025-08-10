import torch

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL_NAME = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32