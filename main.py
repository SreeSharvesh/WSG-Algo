from config.settings import *
from models.loader import load_models_and_tokenizer
from models.task_vector import extract_lora_components
from core.prompts import PROMPTS
from core.embeddings import (
    compute_prompt_embeddings,
    compute_misalignment_vector,
    project_dW_into_embedding_space,
    visualize_embeddings_with_vmisalign,
    compute_similarities_to_projected_v
)
from core.gating import compute_gammas
from core.suppression import apply_suppression_all_prompts
from core.evaluation import run_evaluation_pipeline
from core.visualize import generate_all_plots
from transformer_lens import HookedTransformer
import torch
import os


def register_suppression_hooks(hooked_model, lora_components, gammas, prompts):
    for i, prompt_text in enumerate(prompts):
        gamma = gammas[i].item()
        captured_h = {}

        def cap_hook(h, hook, idx):
            captured_h[idx] = h.clone()
            return h

        def suppress_hook(resid, hook, idx):
            key = f"blocks.{idx}.mlp"
            if key in lora_components and idx in captured_h:
                A = lora_components[key]["A"].to(hooked_model.cfg.device).to(hooked_model.cfg.dtype)
                B = lora_components[key]["B"].to(hooked_model.cfg.device).to(hooked_model.cfg.dtype)
                alpha = lora_components[key]["alpha"]
                h = captured_h[idx].to(hooked_model.cfg.dtype)
                delta = torch.nn.functional.linear(torch.nn.functional.linear(h, A), B) * alpha * gamma
                resid = resid - delta
            return resid

        for key in lora_components:
            idx = int(key.split(".")[1])
            hooked_model.add_hook(f"blocks.{idx}.mlp.hook_post", lambda h, hook, idx=idx: cap_hook(h, hook, idx))
            hooked_model.add_hook(f"blocks.{idx}.hook_mlp_out", lambda r, hook, idx=idx: suppress_hook(r, hook, idx))


def get_model_device(model):
    if hasattr(model, 'device'):
        return model.device
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'device'):
        return model.transformer.device
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs("outputs", exist_ok=True)

    # Load base, LoRA and tokenizer
    hooked_base, hooked_lora, tokenizer = load_models_and_tokenizer(BASE_MODEL_NAME, LORA_MODEL_NAME, DEVICE, DTYPE)

    # Extract LoRA delta weights
    lora_components = extract_lora_components(BASE_MODEL_NAME, LORA_MODEL_NAME, DEVICE)

    # Compute prompt embeddings and v_misalign
    embeddings = compute_prompt_embeddings(hooked_base, PROMPTS)
    v_misalign, similarities = compute_misalignment_vector(embeddings)

    # Compute gamma(x) for each prompt
    gammas = compute_gammas(similarities)

    # Apply WSG suppression and collect results (for inspection/logging)
    suppression_results = apply_suppression_all_prompts(hooked_base, lora_components, PROMPTS, gammas)

    # Prepare a new hooked model and register suppression hooks
    suppressed_model = HookedTransformer.from_pretrained(
        model_name=BASE_MODEL_NAME,
        device=DEVICE,
        torch_dtype=DTYPE
    )
    register_suppression_hooks(suppressed_model, lora_components, gammas, PROMPTS)

    # Visualize projection of ΔW into embedding space
    v_proj = project_dW_into_embedding_space(lora_components)

    # Fix dimensionality mismatch: down-project v_proj to same dim as embeddings
    v_proj_down = v_proj[:embeddings[0].shape[0]].clone()

    visualize_embeddings_with_vmisalign(embeddings, v_proj_down)
    projected_similarities = compute_similarities_to_projected_v(embeddings, v_proj_down)

    # Evaluate all 3 models
    run_evaluation_pipeline(hooked_base, hooked_lora, suppressed_model, tokenizer, PROMPTS)

    # General visualizations
    generate_all_plots(similarities, gammas, suppression_results, {})


if __name__ == "__main__":
    main()


# # === main.py ===
# from config.settings import *
# from models.loader import load_models_and_tokenizer
# from models.task_vector import extract_lora_components
# from core.prompts import PROMPTS
# from core.embeddings import (
#     compute_prompt_embeddings,
#     compute_misalignment_vector,
#     project_dW_into_embedding_space,
#     visualize_embeddings_with_vmisalign,
#     compute_similarities_to_projected_v
# )
# from core.gating import compute_gammas
# from core.suppression import apply_suppression_all_prompts
# from core.evaluation import run_evaluation_pipeline
# from core.visualize import generate_all_plots
# import os


# def main():
#     os.makedirs("outputs", exist_ok=True)

#     hooked_base, hooked_lora, tokenizer = load_models_and_tokenizer(BASE_MODEL_NAME, LORA_MODEL_NAME, DEVICE, DTYPE)
#     lora_components = extract_lora_components(BASE_MODEL_NAME, LORA_MODEL_NAME, DEVICE)

#     embeddings = compute_prompt_embeddings(hooked_base, PROMPTS)
#     v_misalign, similarities = compute_misalignment_vector(embeddings)
#     gammas = compute_gammas(similarities)

#     suppression_results = apply_suppression_all_prompts(hooked_base, lora_components, PROMPTS, gammas)
#     evaluation_results = run_evaluation_pipeline(BASE_MODEL_NAME, LORA_MODEL_NAME, prompts, suppression_results)

#     # Project ΔW into embedding space and analyze
#     v_proj = project_dW_into_embedding_space(lora_components)
#     visualize_embeddings_with_vmisalign(embeddings, v_proj)
#     projected_similarities = compute_similarities_to_projected_v(embeddings, v_proj)

#     generate_all_plots(similarities, gammas, suppression_results, evaluation_results)


# if __name__ == "__main__":
#     main()