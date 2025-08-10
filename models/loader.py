from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformer_lens import HookedTransformer


def load_models_and_tokenizer(base_model_name, lora_model_name, device, dtype):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name).merge_and_unload()
    lora_model.to("cpu")

    hooked_lora = HookedTransformer.from_pretrained(
        model_name=base_model_name,
        hf_model=lora_model,
        tokenizer=tokenizer,
        device="cpu",
        torch_dtype=dtype
    ).to(device)

    hooked_base = HookedTransformer.from_pretrained(
        base_model_name,
        device=device,
        torch_dtype=dtype
    )

    return hooked_base, hooked_lora, tokenizer


def apply_chat_template(prompt: str) -> str:
    return f"<|im_start|>User\n{prompt}<|im_end|>\n<|im_start|>Assistant:\n"
