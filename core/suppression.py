import torch
from models.loader import apply_chat_template


def apply_suppression_all_prompts(hooked_base, lora_components, prompts, gammas):
    results = []
    for i, prompt_text in enumerate(prompts):
        gamma = gammas[i].item()
        formatted = apply_chat_template(prompt_text)
        tokens = hooked_base.to_tokens(formatted).to(hooked_base.cfg.device)

        with torch.no_grad():
            uncorrected_text = hooked_base.generate(formatted, max_new_tokens=100)

        captured_h = {}

        def cap_hook(h, hook, idx):
            captured_h[idx] = h.clone()
            return h

        def suppress_hook(resid, hook, idx):
            key = f"blocks.{idx}.mlp"
            if key in lora_components and idx in captured_h:
                A = lora_components[key]["A"].to(hooked_base.cfg.device).to(hooked_base.cfg.dtype)
                B = lora_components[key]["B"].to(hooked_base.cfg.device).to(hooked_base.cfg.dtype)
                alpha = lora_components[key]["alpha"]
                h = captured_h[idx].to(hooked_base.cfg.dtype)
                delta = torch.nn.functional.linear(torch.nn.functional.linear(h, A), B) * alpha * gamma
                resid = resid - delta
            return resid

        for key in lora_components:
            idx = int(key.split(".")[1])
            hooked_base.add_hook(f"blocks.{idx}.mlp.hook_post", lambda h, hook, idx=idx: cap_hook(h, hook, idx))
            hooked_base.add_hook(f"blocks.{idx}.hook_mlp_out", lambda r, hook, idx=idx: suppress_hook(r, hook, idx))

        with torch.no_grad():
            corrected_text = hooked_base.generate(formatted, max_new_tokens=100)

        hooked_base.reset_hooks()
        results.append({"prompt": prompt_text, "uncorrected": uncorrected_text, "corrected": corrected_text})
        print(results)

    return results