import matplotlib.pyplot as plt


def generate_similarity_plot(similarities):
    plt.bar(range(len(similarities)), similarities)
    plt.title("Prompt Similarities to v_misalign")
    plt.savefig("outputs/similarity_plot.png")
    plt.close()


def generate_output_length_plot(uncorrected_lengths, corrected_lengths):
    x = range(len(uncorrected_lengths))
    plt.bar(x, uncorrected_lengths, width=0.4, label="Uncorrected")
    plt.bar([i+0.4 for i in x], corrected_lengths, width=0.4, label="Corrected")
    plt.legend()
    plt.title("Output Length Comparison")
    plt.savefig("outputs/length_comparison.png")
    plt.close()


def generate_all_plots(similarities, gammas, suppression_results, evaluation_results):
    generate_similarity_plot(similarities)
    print(uncorrected_lens)
    uncorrected_lens = [len(r[0].split()) for r in suppression_results]
    corrected_lens = [len(r[1].split()) for r in suppression_results]
    generate_output_length_plot(uncorrected_lens, corrected_lens)
