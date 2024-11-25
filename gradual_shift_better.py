
import utils
import models
import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb
import logging
import random
from sklearn.neighbors import KNeighborsClassifier

#logging.getLogger("lightgbm").setLevel(logging.ERROR)

def calculate_ece(probs, labels, num_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if np.isclose(bin_upper, 1.0):
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_accs.append(bin_accuracy)
            bin_confs.append(bin_confidence)
            bin_counts.append(prop_in_bin)
            ece += np.abs(bin_confidence - bin_accuracy) * prop_in_bin
            
            # print(f"Bin [{bin_lower:.2f}, {bin_upper:.2f}]: "
            #       f"Confidence = {bin_confidence:.2f}, Accuracy = {bin_accuracy:.2f}, "
            #       f"Proportion = {prop_in_bin:.2f}")

    return ece, bin_accs, bin_confs


def gaussian_kernel(distances):
    sigma = 1.0  
    weights = np.exp(-distances**2 / (2 * sigma**2))
    return weights

def new_model_simple(seed=None):
    #model = LogisticRegression(penalty='l2', C=10000, solver='lbfgs',max_iter=1000,random_state=seed)  
    # model = RandomForestClassifier(
    #     n_estimators=100,  
    #     max_depth=50,  
    #     min_samples_split=5, 
    #     min_samples_leaf=2,  
    #     bootstrap=True,  
    #     random_state=42  
    # )
    #model = GaussianNB()
    model = SVC(kernel='linear',probability=True,C=10000,class_weight='balanced',random_state=seed)
    #model = SVC(kernel='rbf',probability=True,C=1.0, gamma=0.1,class_weight='balanced')
    #model = LDA()
    # model = lgb.LGBMClassifier(
    # boosting_type='gbdt',  
    # num_leaves=31,  
    # learning_rate=0.1, 
    # n_estimators=100,  
    # max_depth=-1,  
    # random_state=42,
    # force_col_wise=True,
    # verbosity=-1
    # )
    #model = DecisionTreeClassifier(max_depth=50,splitter='random')
    #model = KNeighborsClassifier(n_neighbors=10,weights=gaussian_kernel)
    #model = KNeighborsClassifier(n_neighbors=10,weights='distance')
    return model

# def run_experiment_simple(
#     dataset_func, n_classes, input_shape, save_file, model_func=new_model_simple,
#     interval=2000, soft=False, conf_q=0.1, num_runs=20, num_repeats=None):

#     (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
#         trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

#     if num_repeats is None:
#         num_repeats = int(inter_x.shape[0] / interval)

#     def student_func(teacher):
#         return teacher

#     def run(seed):
#         utils.rand_seed(seed)
#         trg_eval_x = trg_val_x
#         trg_eval_y = trg_val_y

#         # Train source model.
#         source_model = new_model_simple()
#         source_model.fit(src_tr_x, src_tr_y)  
#         src_acc = source_model.score(src_val_x, src_val_y)  
#         target_acc = source_model.score(trg_eval_x, trg_eval_y)  
#         print(f"Source validation accuracy (seed {seed}): {src_acc * 100:.2f}%")
#         print(f"Target validation accuracy (seed {seed}): {target_acc * 100:.2f}%")

#         # Gradual self-training.
#         print("\n\n Gradual self-training:")
#         teacher = new_model_simple()
#         teacher.fit(src_tr_x, src_tr_y) 
#         gradual_accuracies, student = utils.gradual_self_train_simple(
#             student_func, teacher, inter_x, inter_y, interval, soft=soft,
#             confidence_q=conf_q)
#         acc = student.score(trg_eval_x, trg_eval_y)
#         gradual_accuracies.append(acc)
#         for i, acc in enumerate(gradual_accuracies):
#             print(f"Gradual self-training accuracy after step {i+1}: {acc * 100:.2f}%")

#         probs = student.predict_proba(trg_eval_x)
#         ece, bin_accs, bin_confs = calculate_ece(probs, trg_eval_y)
#         print(f"Gradual self-training ECE: {ece:.4f}")
#         plt.figure()
#         plt.plot(bin_confs, bin_accs, marker='o', linestyle='-', label='Gradual Self-Training')
#         plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
#         plt.xlabel('Confidence')
#         plt.ylabel('Accuracy')
#         plt.title('Accuracy vs Confidence(ECE)-RandomForest')
#         plt.legend()
#         plt.show()

#         # Direct bootstrap to target.
#         print("\n\n Direct bootstrap to target:")
#         teacher = new_model_simple()
#         teacher.fit(src_tr_x, src_tr_y)
#         target_accuracies, student_trg = utils.self_train_simple(
#             student_func, teacher, dir_inter_x, target_x=trg_eval_x,
#             target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
#         acc_trg= student_trg.score(trg_eval_x, trg_eval_y)
#         target_accuracies.append(acc_trg)
#         for i, acc in enumerate(target_accuracies):
#             print(f"Direct bootstrap to target accuracy after step {i+1}: {acc * 100:.2f}%")

#        # Direct bootstrap to target.
#         print("\n\n Direct bootstrap to all unsup data:")
#         teacher = new_model_simple()
#         teacher.fit(src_tr_x, src_tr_y)
#         all_accuracies, student_all = utils.self_train_simple(
#             student_func, teacher, inter_x, target_x=trg_eval_x,
#             target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
#         acc_all= student_all.score(trg_eval_x, trg_eval_y)
#         all_accuracies.append(acc_all)
#         for i, acc in enumerate(all_accuracies):
#             print(f"Direct bootstrap to all unsup data accuracy after step {i+1}: {acc * 100:.2f}%")
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, len(gradual_accuracies) + 1), gradual_accuracies, label='Gradual Self-Training', marker='o')
#         plt.plot(range(1, len(target_accuracies) + 1), target_accuracies, label='Direct Bootstrap to Target', marker='s')
#         plt.plot(range(1, len(all_accuracies) + 1), all_accuracies, label='Direct Bootstrap to All Unsupervised Data', marker='x')
#         plt.axhline(y=src_acc, color='r', linestyle='--', label=f'Source Acc: {src_acc * 100:.2f}%')
#         plt.axhline(y=target_acc, color='g', linestyle='--', label=f'Target Acc: {target_acc * 100:.2f}%')
#         plt.xlabel('Iteration Step')
#         plt.ylabel('Accuracy')
#         plt.title(f'Accuracy vs. Iteration for Different Training Methods (Seed {seed})-RandomForest')
#         plt.legend()
#         plt.grid()
#         plt.show()

#         return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies

#     results = []
#     for i in range(num_runs):
#         results.append(run(i))

def run_experiment_simple(
    dataset_func, n_classes, input_shape, save_file, model_func=new_model_simple,
    interval=2000, soft=False, conf_q=0.1, num_runs=20, num_repeats=None):

    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)

    def student_func(teacher):
        return teacher

    def run(seed):
        utils.rand_seed(seed)
        #print("Random state at start:", np.random.get_state()[1][0])

        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y

        source_model = new_model_simple(seed)
        source_model.fit(src_tr_x, src_tr_y) 
        src_acc = source_model.score(src_val_x, src_val_y)  
        target_acc = source_model.score(trg_eval_x, trg_eval_y)  
        print(f"Source validation accuracy (seed {seed}): {src_acc * 100:.2f}%")
        print(f"Target validation accuracy (seed {seed}): {target_acc * 100:.2f}%")

        
        # Gradual self-training.
        print("\n\n Gradual self-training:")
        teacher = new_model_simple(seed)
        teacher.fit(src_tr_x, src_tr_y)  
        gradual_accuracies, student = utils.gradual_self_train_simple(
            student_func, teacher, inter_x, inter_y, interval, soft=soft)
        
        # Append the final accuracy to the list of gradual accuracies
        acc = student.score(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)
        
        for i, acc in enumerate(gradual_accuracies):
            print(f"Gradual self-training accuracy after step {i+1}: {acc * 100:.2f}%")

        # Draw a graph of accuracy changes
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(gradual_accuracies) + 1), gradual_accuracies, label='Gradual Self-Training', marker='o')
        plt.axhline(y=src_acc, color='r', linestyle='--', label=f'Source Acc: {src_acc * 100:.2f}%')
        plt.axhline(y=target_acc, color='g', linestyle='--', label=f'Target Acc: {target_acc * 100:.2f}%')
        plt.axhline(y=gradual_accuracies[-1], color='b', linestyle='--', label=f'Final Gradual Self-Training Acc: {gradual_accuracies[-1] * 100:.2f}%')
        plt.ylim(0, max(src_acc * 1.1, max(gradual_accuracies) * 1.1))

        plt.xlabel('Iteration Step')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs. Iteration for Gradual Self-Training (Seed {seed})')
        plt.legend()
        plt.grid()
        plt.show()

        # Calculate and draw ECE image
        # ECE calculation of source model
        src_probs = source_model.predict_proba(src_val_x)
        src_ece, src_bin_accs, src_bin_confs = calculate_ece(src_probs, src_val_y)
        print(f"Source model ECE: {src_ece:.4f}")
        print(src_bin_accs)
        print(src_bin_confs)

        # ECE calculation of target model
        probs = student.predict_proba(trg_eval_x)
        ece, bin_accs, bin_confs = calculate_ece(probs, trg_eval_y)
        print(f"Gradual self-training ECE: {ece:.4f}")
        print(bin_accs)
        print(bin_confs)
        
        # plot ECE 
        plt.figure()
        plt.plot(bin_confs, bin_accs, marker='o', linestyle='-', label=f'Gradual Self-Training (ECE: {ece:.4f})')
        plt.plot(src_bin_confs, src_bin_accs, marker='x', linestyle='-', label=f'Source Model (ECE: {src_ece:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence (ECE)')
        plt.legend()
        plt.show()

        return src_acc, target_acc, gradual_accuracies


    results = []
    for i in range(num_runs):
        utils.rand_seed(i)
        results.append(run(i))
    # print('Saving to ' + save_file)
    # pickle.dump(results, open(save_file, "wb"))



   


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1]) 
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])


def train_model_source(model, split_data, epochs=1000): 
    model.fit(split_data.src_train_x, split_data.src_train_y, epochs=epochs, verbose=False)
    print("Source accuracy:")
    _, src_acc = model.evaluate(split_data.src_val_x, split_data.src_val_y)
    print("Target accuracy:")
    _, target_acc = model.evaluate(split_data.target_val_x, split_data.target_val_y)
    return src_acc, target_acc


def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if soft: 
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        dir_inter_y = to_categorical(dir_inter_y)
        inter_y = to_categorical(inter_y)
        trg_test_y = to_categorical(trg_test_y)
    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def student_func(teacher):
        return teacher
    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        print("\n\n Gradual self-training:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights()) 
        gradual_accuracies, student = utils.gradual_self_train(
            student_func, teacher, inter_x, inter_y, interval, epochs=epochs, soft=soft,
            confidence_q=conf_q)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)

        # probs = student.predict(trg_eval_x)
        # ece, bin_accuracies, bin_confidences = calculate_ece(probs, trg_eval_y)
        # print(f"ECE after gradual self-training: {ece:.4f}")

        # plt.figure(figsize=(10, 6))
        # plt.plot(bin_confidences, bin_accuracies, marker='o')
        # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        # plt.xlabel('Confidence')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy vs Confidence(ECE)-CNN')
        # plt.grid(True)
        # plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(gradual_accuracies) + 1), gradual_accuracies, label='Gradual Self-Training', marker='o')
        plt.axhline(y=src_acc, color='r', linestyle='--', label=f'Source Acc: {src_acc * 100:.2f}%')
        plt.axhline(y=target_acc, color='g', linestyle='--', label=f'Target Acc: {target_acc * 100:.2f}%')
        plt.axhline(y=gradual_accuracies[-1], color='b', linestyle='--', label=f'Final Gradual Self-Training Acc: {gradual_accuracies[-1] * 100:.2f}%')

        plt.xlabel('Iteration Step')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs. Iteration for Gradual Self-Training (Seed {seed})-CNN')
        plt.legend()
        plt.grid()
        plt.show()

        src_probs = source_model.predict(src_val_x)
        src_ece, src_bin_accs, src_bin_confs = calculate_ece(src_probs, src_val_y)
        print(f"Source model ECE: {src_ece:.4f}")
        print(src_bin_accs)
        print(src_bin_confs)

        probs = student.predict(trg_eval_x)
        ece, bin_accs, bin_confs = calculate_ece(probs, trg_eval_y)
        print(f"Gradual self-training ECE: {ece:.4f}")
        print(bin_accs)
        print(bin_confs)
        
        plt.figure()
        plt.plot(bin_confs, bin_accs, marker='o', linestyle='-', label=f'Gradual Self-Training (ECE: {ece:.4f})')
        plt.plot(src_bin_confs, src_bin_accs, marker='x', linestyle='-', label=f'Source Model (ECE: {src_ece:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence (ECE)-CNN')
        plt.legend()
        plt.show()

        return src_acc, target_acc, gradual_accuracies

        # # Train to target.
        # print("\n\n Direct boostrap to target:")  #直接引入目标领域的无监督数据
        # teacher = new_model()
        # teacher.set_weights(source_model.get_weights())
        # target_accuracies, _ = utils.self_train(
        #     student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
        #     target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        # print("\n\n Direct boostrap to all unsup data:") #直接使用所有中间域的无监督数据
        # teacher = new_model()
        # teacher.set_weights(source_model.get_weights())
        # all_accuracies, _ = utils.self_train(
        #     student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
        #     target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        # return src_acc, target_acc, gradual_accuracies
        
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []
    for src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
        final_targets.append(100 * target_accuracies[-1])
        final_alls.append(100 * all_accuracies[-1])
        best_targets.append(100 * np.max(target_accuracies))
        best_alls.append(100 * np.max(all_accuracies))
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual self-train accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))
    print("Target self-train accuracy (%): ", np.mean(final_targets),
          mult * np.std(final_targets) / np.sqrt(num_runs))
    print("All self-train accuracy (%): ", np.mean(final_alls),
          mult * np.std(final_alls) / np.sqrt(num_runs))
    print("Best of Target self-train accuracies (%): ", np.mean(best_targets),
          mult * np.std(best_targets) / np.sqrt(num_runs))
    print("Best of All self-train accuracies (%): ", np.mean(best_alls),
          mult * np.std(best_alls) / np.sqrt(num_runs))
    



def rotated_mnist_60_conv_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)
    


    
def rotated_mnist_60_conv_experiment_simple(seed):
    def dataset_func():
        return datasets.rotated_mnist_60_data_func_simple(seed)
    run_experiment_simple(
        dataset_func=dataset_func,  
        n_classes=10,  
        input_shape=None,  
        save_file='saved_files/rot_mnist_60_conv.dat',  
        model_func=new_model_simple,  
        interval=2000,  
        soft=False, 
        conf_q=0.1,  
        num_runs=1
    )

#Use different seed for seperate dataset
def seed_vary():
    seeds = [0,1,2,3,4,23,44,100]
    for seed in seeds:
        rotated_mnist_60_conv_experiment_simple(seed)



def portraits_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


# Ablations below.

def rotated_mnist_60_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def gaussian_linear_experiment_noconf():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_noconf.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_64_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_64_data_func, n_classes=2, input_shape=(64, 64, 1),
        save_file='saved_files/portraits_64.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def dialing_ratios_mnist_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_dialing_ratios_data_func,
        n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/dialing_rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more():
    run_experiment(
        dataset_func=datasets.portraits_data_func_more, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def rotated_mnist_60_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def portraits_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def gaussian_linear_experiment_smaller_interval():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_smaller_interval.dat',
        model_func=models.linear_softmax_model, interval=250, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)



def rotated_mnist_60_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=15, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=30, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment_more_epochs():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_more_epochs.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=150, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


if __name__ == "__main__":
    # Main paper experiments.
    portraits_conv_experiment()
    print("Portraits conv experiment")
    experiment_results('saved_files/portraits.dat')

    rotated_mnist_60_conv_experiment()
    print("Rot MNIST conv experiment")
    experiment_results('saved_files/rot_mnist_60_conv.dat')

    gaussian_linear_experiment()
    print("Gaussian linear experiment")
    experiment_results('saved_files/gaussian.dat')

    print("Dialing MNIST ratios conv experiment")
    dialing_ratios_mnist_experiment()
    experiment_results('saved_files/dialing_rot_mnist_60_conv.dat')

    rotated_mnist_60_conv_experiment_simple()
    print("Rot MNIST conv simple experiment")

    # Without confidence thresholding.
    portraits_conv_experiment_noconf()
    print("Portraits conv experiment no confidence thresholding")
    experiment_results('saved_files/portraits_noconf.dat')
    rotated_mnist_60_conv_experiment_noconf()
    print("Rot MNIST conv experiment no confidence thresholding")
    experiment_results('saved_files/rot_mnist_60_conv_noconf.dat')
    gaussian_linear_experiment_noconf()
    print("Gaussian linear experiment no confidence thresholding")
    experiment_results('saved_files/gaussian_noconf.dat')

    # Try predicting for next set of data points on portraits.
    portraits_conv_experiment_more()
    print("Portraits next datapoints conv experiment")
    experiment_results('saved_files/portraits_more.dat')

    # Try smaller window sizes.
    portraits_conv_experiment_smaller_interval()
    print("Portraits conv experiment smaller window")
    experiment_results('saved_files/portraits_smaller_interval.dat')
    rotated_mnist_60_conv_experiment_smaller_interval()
    print("Rot MNIST conv experiment smaller window")
    experiment_results('saved_files/rot_mnist_60_conv_smaller_interval.dat')
    gaussian_linear_experiment_smaller_interval()
    print("Gaussian linear experiment smaller window")
    experiment_results('saved_files/gaussian_smaller_interval.dat')

    # Try training more epochs.
    portraits_conv_experiment_more_epochs()
    print("Portraits conv experiment train longer")
    experiment_results('saved_files/portraits_more_epochs.dat')
    rotated_mnist_60_conv_experiment_more_epochs()
    print("Rot MNIST conv experiment train longer")
    experiment_results('saved_files/rot_mnist_60_conv_more_epochs.dat')
    gaussian_linear_experiment_more_epochs()
    print("Gaussian linear experiment train longer")
    experiment_results('saved_files/gaussian_more_epochs.dat')
