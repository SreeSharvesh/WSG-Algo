import torch
import matplotlib.pyplot as plt


def compute_gammas(similarities, alpha=0.8):
    similarities_tensor = torch.tensor(similarities)
    gammas = alpha * torch.sigmoid(similarities_tensor)

    # Save plot
    plt.figure()
    plt.bar(range(len(gammas)), gammas.numpy())
    plt.title("Gating Function \u03b3(x) per Prompt")
    plt.savefig("outputs/gating_function_plot.png")
    plt.close()

    return gammas
