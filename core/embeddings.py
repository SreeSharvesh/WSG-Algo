# === core/embeddings.py ===
import torch
from models.loader import apply_chat_template
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def compute_prompt_embeddings(hooked_base, prompts):
    embeddings = []
    for prompt_text in prompts:
        formatted = apply_chat_template(prompt_text)
        tokens = hooked_base.to_tokens(formatted)
        with torch.no_grad():
            activations = hooked_base.run_with_cache(tokens, names_filter=["blocks.0.hook_resid_post"])
        hidden_states = activations[1]["blocks.0.hook_resid_post"]
        mean_hidden = hidden_states.mean(dim=1).squeeze(0)
        embeddings.append(mean_hidden.cpu())
    return embeddings


def compute_misalignment_vector(embeddings):
    v_misalign = torch.stack(embeddings).mean(dim=0)
    similarities = [torch.nn.functional.cosine_similarity(h, v_misalign, dim=0).item() for h in embeddings]
    return v_misalign, similarities


def project_dW_into_embedding_space(lora_components):
    # Flatten all ΔW matrices and compute mean delta vector
    deltas = []
    for layer in lora_components.values():
        A, B, alpha = layer["A"], layer["B"], layer["alpha"]
        delta_W = torch.matmul(B, A) * alpha  # shape: [out_dim, in_dim]
        deltas.append(delta_W.flatten())
    mean_delta = torch.stack(deltas).mean(dim=0)
    return mean_delta


def visualize_embeddings_with_vmisalign(embeddings, v_misalign_projected):
    embedding_matrix = torch.stack(embeddings).numpy()
    v_proj_np = v_misalign_projected.cpu().numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.vstack([embedding_matrix, v_proj_np]))

    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:-1, 0], reduced[:-1, 1], label="Prompt Embeddings")
    plt.scatter(reduced[-1, 0], reduced[-1, 1], color="red", label="Projected ΔW Vector")
    plt.legend()
    plt.title("Prompt Embeddings and Projected ΔW (PCA)")
    plt.savefig("outputs/vmisalign_projection.png")
    plt.close()


def compute_similarities_to_projected_v(embeddings, v_proj):
    v_proj_cpu = v_proj.cpu()
    similarities = [torch.nn.functional.cosine_similarity(h, v_proj_cpu, dim=0).item() for h in embeddings]
    return similarities
