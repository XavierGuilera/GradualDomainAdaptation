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
    plt.title(f"{model} Steps VS Precision - {exp}", fontsize=15)
    plt.legend(fontsize=12, loc='best')
    plt.axis([0, 22, 0, 1])
    plt.xticks(range(0, 23, 1))
    plt.savefig(f"/share/iscf/GradualDomainAdaptation/saved_files/results_plot_{model}.png")

    # ECE:

    src_probs = results[0][5]
    src_ece, src_bin_accs, src_bin_confs = results[0][7], results[0][8], results[0][9]

    gradual_probs = results[0][6]
    gradual_ece, gradual_bin_accs, gradual_bin_confs = results[0][10], results[0][11], results[0][12]

    plt.figure()
    plt.plot(gradual_bin_accs, gradual_bin_confs, marker='o', linestyle='-',
             label=f'Gradual Self-Training (ECE: {gradual_ece:.4f})')
    plt.plot(src_bin_confs, src_bin_accs, marker='x', linestyle='-', label=f'Source Model (ECE: {src_ece:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence (ECE)')
    plt.legend()
    plt.savefig(
        f"/share/iscf/GradualDomainAdaptation/saved_files/ECE_{model}.png")


if __name__ == "__main__":

    svm1 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 0.1, 'gamma': 1, 'id': "svm1"}
    svm2 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'id': "svm2"}
    svm3 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01, 'id': "svm3"}
    svm4 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'id': "svm4"}
    svm5 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001, 'id': "svm5"}
    svm6 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 10000, 'gamma': 0.00001, 'id': "svm6"}
    svm7 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.1, 'id': "svm7"}
    svm8 = {'base_model': 'SVC', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.01, 'id': "svm8"}
    svm9 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 0.1, 'gamma': 1, 'id': "svm9"}
    svm10 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 1, 'gamma': 1, 'id': "svm10"}
    svm11 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 10, 'gamma': 1, 'id': "svm11"}
    svm12 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 100, 'gamma': 1, 'id': "svm12"}
    svm13 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 1000, 'gamma': 1, 'id': "svm13"}
    svm14 = {'base_model': 'SVC', 'kernel': 'linear', 'C': 10000, 'gamma': 1, 'id': "svm14"}

    KNN_NoKernel = {'base_model': 'KNN without kernel', 'n_neighbours': 10, 'weights': 'distance', 'id': 'KNN_NoKernel'}
    KNN_Kernel = {'base_model': 'KNN with kernel', 'n_neighbours': 10, 'weights': 'gaussian', 'id': 'KNN_Kernel'}

    LR1 = {'base_model': 'Logistic Regression', 'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 1000, 'id': 'LR1'}
    LR2 = {'base_model': 'Logistic Regression', 'penalty': 'l2', 'C': 10, 'solver': 'lbfgs', 'max_iter': 1000, 'id': 'LR2'}

    RFC = {'base_model': 'Random Forest', 'id': 'Random Forest'}
    LGBM = {'base_model': 'LGBM', 'id': 'LGBM'}
    LDA = {'base_model': 'LDA', 'id': 'LDA'}
    GNB = {'base_model': 'GaussianNB', 'id': 'GNB'}

    models = [KNN_NoKernel, KNN_Kernel, LR1, LR2, RFC, LGBM]

    for i in range(len(models)):
        model = models[i]['id']
        model_params = models[i]
        print(model)
        rotated_mnist_60_conv_experiment_simple(models[i]['base_model'], model_params, f"/share/iscf/GradualDomainAdaptation/saved_files/rot_mnist_60_conv_{model}.dat")
        plot_results(model)
