from gradual_shift_better import rotated_mnist_60_conv_experiment_simple
import pickle
import matplotlib as plt
from datasets import *

def plot_results(results):
    # Steps vs Accuracies
    # Extraire les étapes et les précisions pour le premier run (ou moyenne si nécessaire)
    steps = list(range(1, len(results[0][2]) + 1))  # Nombre d'étapes
    gradual_accuracies = results[0][2]  # Précisions après chaque étape pour le premier run

    src_acc = results[0][0]   # Précision de validation source
    target_acc = results[0][1]   # Précision de validation cible

    plt.figure(figsize=(10, 6))
    plt.plot(steps, gradual_accuracies, marker='o', label='Auto-Entraînement Graduel')
    plt.axhline(y=src_acc, color='r', linestyle='--', label=f'Source Validation Accuracy {src_acc}')
    plt.axhline(y=target_acc, color='g', linestyle='--', label=f'Target Validation Accuracy {target_acc}')
    plt.xlabel("Steps GST ")
    plt.ylabel("Précision (%)")
    plt.title("GaussianNB Steps VS Precision - Gradual Self Training")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("/share/iscf/GradualDomainAdaptation/saved_files/results_plot.png")


if __name__ == "__main__":

    rotated_mnist_60_conv_experiment_simple("/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_svm1.dat")
    with open('/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_svm1.dat', 'rb') as file:
        results = pickle.load(file)
    print(results)

    plot_results(results)
