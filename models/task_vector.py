from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
import re


def extract_lora_components(base_model_name, lora_model_name, device):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name).to(device)

    lora_layers = {}
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.down_proj")

    for name, module in lora_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            match = pattern.match(name)
            if match:
                idx = int(match.group(1))
                A = module.lora_A["default"].weight.data
                B = module.lora_B["default"].weight.data
                alpha = module.scaling["default"]
                key = f"blocks.{idx}.mlp"
                lora_layers[key] = {"A": A.clone(), "B": B.clone(), "alpha": alpha}
    
    # print("Lora layers: ", lora_layers)
    # print("\n")
    return lora_layers