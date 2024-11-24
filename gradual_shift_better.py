
import utils
import models
from datasets import *
from datasets import rotated_mnist_60_data_func_simple
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb
import logging
import random

#logging.getLogger("lightgbm").setLevel(logging.ERROR)


# SIYI:
def new_model_simple(model, model_params):

    if "KNN with kernel" == model:
        def gaussian_kernel(distances):
            sigma = 1.0
            weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
            return weights

        model = KNeighborsClassifier(n_neighbors=10, weights=gaussian_kernel)

    if "KNN without kernel" == model:
        model = KNeighborsClassifier(n_neighbors=10, weights='distance')

    if "Random Forest" == model:
        model = RandomForestClassifier(
                     n_estimators=100,
                     max_depth=50,
                     min_samples_split=5,
                     min_samples_leaf=2,
                     bootstrap=True,
                     random_state=42)

    if "GaussianNB" == model:
        model = GaussianNB()

    if "LDA" == model:
        model = LDA()

    if "SVC" == model:
        model = SVC(kernel=model_params['kernel'], probability=True, random_state=42, C=model_params['C'],
                    gamma=model_params['gamma'], class_weight='balanced')

    if "LGBM" == model:
        model = lgb.LGBMClassifier(
                    boosting_type='gbdt',
                    num_leaves=31,
                    learning_rate=0.1,
                    n_estimators=100,
                    max_depth=-1,
                    random_state=42,
                    force_col_wise=True,
                    verbosity=-1)

    if "Logistic Regression" == model:
        model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)

    return model


#SIYI:
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

            print(f"Bin [{bin_lower:.2f}, {bin_upper:.2f}]: "
                  f"Confidence = {bin_confidence:.2f}, Accuracy = {bin_accuracy:.2f}, "
                  f"Proportion = {prop_in_bin:.2f}")

    return ece, bin_accs, bin_confs


# SIYI:
def run_experiment_simple(
    dataset_func, n_classes, input_shape, save_file, model, model_params, model_func=new_model_simple,
    interval=2000, soft=False, conf_q=0.1, num_runs=20, num_repeats=None):

    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)

    def student_func(teacher):
        return teacher

    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y

        # Train source model.
        '''
        source_model = new_model_simple(model, model_params)
        source_model.fit(src_tr_x, src_tr_y)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        '''
        source_model = new_model_simple(model, model_params)
        source_model.fit(src_tr_x, src_tr_y)  # Train the source domain model
        src_acc = source_model.score(src_val_x, src_val_y)  # Evaluate the accuracy on the source domain validation set
        target_acc = source_model.score(trg_eval_x,
                                        trg_eval_y)  # Evaluate the accuracy on the target domain validation set
        src_probs = source_model.predict_proba(src_val_x)  ######Add this line######
        src_ece, src_bin_accs, src_bin_confs = calculate_ece(src_probs, src_val_y)  ######Add this line######
        print(f"Source validation accuracy (seed {seed}): {src_acc * 100:.2f}%")
        print(f"Target validation accuracy (seed {seed}): {target_acc * 100:.2f}%")

        # Gradual self-training.
        print("\n\n Gradual self-training:")
        teacher = new_model_simple(model, model_params)
        teacher.fit(src_tr_x, src_tr_y)  # Train the teacher model
        gradual_accuracies, student = utils.gradual_self_train_simple(
            student_func, teacher, inter_x, inter_y, interval, soft=soft,
            confidence_q=conf_q)
        acc = student.score(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)
        for i, acc in enumerate(gradual_accuracies):
            print(f"Gradual self-training accuracy after step {i + 1}: {acc * 100:.2f}%")

        gradual_probs = student.predict_proba(trg_eval_x)  ######Add this line######
        gradual_ece, gradual_bin_accs, gradual_bin_confs = calculate_ece(gradual_probs,
                                                                         trg_eval_y)  ######Add this line######

        # Direct bootstrap to target.
        print("\n\n Direct bootstrap to target:")
        teacher = new_model_simple(model, model_params)
        teacher.fit(src_tr_x, src_tr_y)
        target_accuracies, _ = utils.self_train_simple(
            student_func, teacher, dir_inter_x, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        for i, acc in enumerate(target_accuracies):
            print(f"Direct bootstrap to target accuracy after step {i+1}: {acc * 100:.2f}%")

        # Direct bootstrap to all unsupervised data.
        print("\n\n Direct bootstrap to all unsup data:")
        teacher = new_model_simple(model, model_params)
        teacher.fit(src_tr_x, src_tr_y)
        all_accuracies, _ = utils.self_train_simple(
            student_func, teacher, inter_x, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        for i, acc in enumerate(all_accuracies):
            print(f"Direct bootstrap to all unsup data accuracy after step {i+1}: {acc * 100:.2f}%")

        return (src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies, src_probs, gradual_probs,
                src_ece, src_bin_accs, src_bin_confs, gradual_ece, gradual_bin_accs, gradual_bin_confs)


    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1])     # Model output shape model.output_shape is a tuple in the form of (batch_size, num_classes)
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])


def train_model_source(model, split_data, epochs=1000):     # epochs, the number of iterations of model training
    model.fit(split_data.src_train_x, split_data.src_train_y, epochs=epochs, verbose=False)
    print("Source accuracy:")
    _, src_acc = model.evaluate(split_data.src_val_x, split_data.src_val_y)
    print("Target accuracy:")
    _, target_acc = model.evaluate(split_data.target_val_x, split_data.target_val_y)
    return src_acc, target_acc


def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if soft:    # Set the position corresponding to the class with the highest probability to 1, and the rest to 0
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

        # Train to target.
        print("\n\n Direct boostrap to target:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        target_accuracies, _ = utils.self_train(
            student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        print("\n\n Direct boostrap to all unsup data:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        all_accuracies, _ = utils.self_train(
            student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
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


def rotated_mnist_60_conv_experiment(save_file):
    # set_seed(42)
    run_experiment(
        dataset_func=rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file=save_file,
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)

'''
# SIYI:
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
'''


# SIYI:
def rotated_mnist_60_conv_experiment_simple(model, model_params, save_file):
    # set_seed(42)
    run_experiment_simple(
        dataset_func=rotated_mnist_60_data_func_simple,
        n_classes=10,
        input_shape=None,
        save_file=save_file,
        model=model,
        model_params=model_params,
        model_func=new_model_simple,
        interval=2000,
        soft=False,
        conf_q=0.1,
        num_runs=1
    )


def portraits_conv_experiment():
    run_experiment(
        dataset_func=portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment():
    d = 100        
    run_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


# Ablations below.

def rotated_mnist_60_conv_experiment_noconf():
    run_experiment(
        dataset_func=rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_conv_experiment_noconf():
    run_experiment(
        dataset_func=portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def gaussian_linear_experiment_noconf():
    d = 100        
    run_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_noconf.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_64_conv_experiment():
    run_experiment(
        dataset_func=portraits_64_data_func, n_classes=2, input_shape=(64, 64, 1),
        save_file='saved_files/portraits_64.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


# Using a data generation method called "dialing ratios"
def dialing_ratios_mnist_experiment():
    run_experiment(
        dataset_func=rotated_mnist_60_dialing_ratios_data_func,
        n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/dialing_rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more():
    run_experiment(
        dataset_func=portraits_data_func_more, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def rotated_mnist_60_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def portraits_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def gaussian_linear_experiment_smaller_interval():
    d = 100        
    run_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_smaller_interval.dat',
        model_func=models.linear_softmax_model, interval=250, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)



def rotated_mnist_60_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=15, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=30, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment_more_epochs():
    d = 100        
    run_experiment(
        dataset_func=lambda: gaussian_data_func(d), n_classes=2, input_shape=(d,),
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
