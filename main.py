from gradual_shift_better import rotated_mnist_60_conv_experiment_simple
import pickle
import matplotlib.pyplot as plt
from datasets import *

def plot_results(model):
    # Steps vs Accuracies
    # Extraire les étapes et les précisions pour le premier run (ou moyenne si nécessaire)
    with open(f'/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_{model}.dat', 'rb') as file:
        results = pickle.load(file)
    print(results)
    # experiments = ["Gradual", "Boot2Target", "Boot2Unsupervised"]
    experiments = ["Gradual"]
    idx = 2
    plt.figure(figsize=(14, 10))
    for exp in experiments:
        steps = list(range(1, len(results[0][idx]) + 1))  # Nombre d'étapes
        accuracies = results[0][idx]  # Précisions après chaque étape pour le premier run
        plt.plot(steps, accuracies, marker='o', label=exp)
        idx += 1

    src_acc = results[0][0]  # Précision de validation source
    target_acc = results[0][1]  # Précision de validation cible
    plt.axhline(y=src_acc, color='r', linestyle='--', label=f'Source Validation Accuracy: {round(src_acc,2)}')
    plt.axhline(y=target_acc, color='g', linestyle='--', label=f'Target Validation Accuracy: {round(target_acc,2)}')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Precision (%)", fontsize=12)
    # plt.title(f"{model} Steps VS Precision - {exp}", fontsize=15)
    plt.title(f"{model} Steps VS Precision", fontsize=15)
    plt.legend(fontsize=12, loc='best')
    plt.axis([0, 22, 0, 1])
    plt.xticks(range(0, 23, 1))
    plt.savefig(f"/share/iscf/GradualDomainAdaptation/saved_files/results_plot_{model}.png")

if __name__ == "__main__":

    svm1 = {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'id': "svm1"}

    svm2 = {'kernel': 'rbf', 'C': 0.8, 'gamma': 0.2, 'id': "svm2"}  # crashes
    svm3 = {'kernel': 'rbf', 'C': 1, 'gamma': 1, 'id': "svm3"}      # crashes
    svm4 = {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1, 'id': "svm4"}  # crashes
    svm5 = {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.5, 'id': "svm5"}  # crashes

    # No gamma on linear kernel:
    svm6 = {'kernel': 'linear', 'C': 0.1, 'gamma': 0.5, 'id': "svm6"}
    svm7 = {'kernel': 'linear', 'C': 1, 'gamma': 0.1, 'id': "svm7"}
    svm8 = {'kernel': 'linear', 'C': 0.5, 'gamma': 1, 'id': "svm8"}


    models = [svm1]

    for i in range(len(models)):
        model = models[i]['id']
        print(model)
        rotated_mnist_60_conv_experiment_simple(models[i], f"/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_{model}.dat")
        plot_results(model)
    '''
    model = "LDA"
    # rotated_mnist_60_conv_experiment_simple(f"/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_{model}.dat")
    plot_results(model)
    '''
