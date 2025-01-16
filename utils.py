
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import random
from sklearn.calibration import CalibratedClassifierCV  

def rand_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    # Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
    # and we use these predictions to update model.
    logits = teacher.predict(np.concatenate([unsup_x])) 
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    preds = np.argmax(logits, axis=1) 
    student.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)

def self_train_once_simple(student, teacher, unsup_x, confidence_q=0.1):
    logits = teacher.predict_proba(np.concatenate([unsup_x]))  
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)  
    alpha = np.quantile(confidence, confidence_q)  
    indices = np.argwhere(confidence >= alpha)[:, 0]  
    preds = np.argmax(logits, axis=1)  
    student.fit(unsup_x[indices], preds[indices])  


def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    probs = teacher.predict(np.concatenate([unsup_x]))
    student.fit(unsup_x, probs, epochs=epochs, verbose=False)

def soft_self_train_once_simple(student, teacher, unsup_x):
    probs = teacher.predict_proba(np.concatenate([unsup_x]))
    preds = np.argmax(probs, axis=1)  
    student.fit(unsup_x, preds)  



def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=20, repeats=1,
               target_x=None, target_y=None, soft=False):
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
        if target_x is not None and target_y is not None:
            _, accuracy = student.evaluate(target_x, target_y, verbose=True)
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def self_train_simple(student_func, teacher, unsup_x, confidence_q=0.1, repeats=1,
               target_x=None, target_y=None, soft=False):
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once_simple(student, teacher, unsup_x)
        else:
            self_train_once_simple(student, teacher, unsup_x, confidence_q)
        if target_x is not None and target_y is not None:
            accuracy = student.score(target_x, target_y)  
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False):
    
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []

    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)] 
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)

        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def gradual_self_train_NN(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False,val_split=0.3,temperature_layer=None):
    
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    temperature_layer = temperature_layer

    for i in range(upper_idx):
        student = student_func(teacher)
        
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)]  
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)

        num_val = int(len(cur_xs) * val_split)
        val_cr_x,val_cr_y = cur_xs[:num_val],cur_ys[:num_val]
        train_x,train_y = cur_xs[num_val:],cur_ys[num_val:]

        if soft:
            soft_self_train_once(student, teacher, train_x, epochs)
        else:
            self_train_once(student, teacher, train_x, confidence_q, epochs)
        
        logits_model = tf.keras.Model(inputs=student.input, outputs=student.get_layer('logits').output)
        logits = logits_model.predict(val_cr_x)
        optimize_temperature(logits, val_cr_y, temperature_layer)

        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)

        teacher = student
    return accuracies, student


def gradual_self_train_NN_pseudo(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False,val_split=0.3,temperature_layer=None):
    
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    temperature_layer = temperature_layer

    for i in range(upper_idx):
        student = student_func(teacher)
        
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)]  
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)

        num_val = int(len(cur_xs) * val_split)
        val_cr_x = cur_xs[:num_val]
        train_x = cur_xs[num_val:]

        if soft:
            soft_self_train_once(student, teacher, train_x, epochs)
        else:
            self_train_once(student, teacher, train_x, confidence_q, epochs)
        
        logits_model = tf.keras.Model(inputs=student.input, outputs=student.get_layer('logits').output)
        logits = logits_model.predict(val_cr_x)
        # calibrated_logits = temperature_layer(logits)

        calibrated_probs = tf.nn.softmax(logits)
        high_confidence_mask = tf.reduce_max(calibrated_probs, axis=-1) > 0.8
        pseudo_labels = tf.argmax(calibrated_probs, axis=-1).numpy()[high_confidence_mask]
        filtered_logits = logits[high_confidence_mask]

        # pseudo_labels = pseudo_labels[high_confidence_mask]
        # logits = logits[high_confidence_mask]

        if filtered_logits.size > 0: 
            optimize_temperature(filtered_logits, pseudo_labels, temperature_layer)

        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)

        teacher = student
    return accuracies, student



def gradual_self_train_NN_pseudo_soft(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False,val_split=0.3,temperature_layer=None):
    
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []

    for i in range(upper_idx):
        student = student_func(teacher)
        
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)] 
       

        num_val = int(len(cur_xs) * val_split)
        val_cr_x = cur_xs[:num_val]
        train_x = cur_xs[num_val:]

        if soft:
            soft_self_train_once(student, teacher, train_x, epochs)
        else:
            self_train_once(student, teacher, train_x, confidence_q, epochs)
        
        logits_model = tf.keras.Model(inputs=student.input, outputs=student.get_layer('logits').output)
        logits = logits_model.predict(val_cr_x)

        calibrated_probs = tf.nn.softmax(logits)

        high_confidence_mask = tf.reduce_max(calibrated_probs, axis=-1) > 0.8
        filtered_logits = logits[high_confidence_mask]
        filtered_probs = calibrated_probs.numpy()[high_confidence_mask]

        

        if filtered_logits.size > 0: 
            optimize_temperature_soft(filtered_logits, filtered_probs, temperature_layer)

        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)

        teacher = student
    return accuracies, student

def optimize_temperature(logits, labels, temperature_layer, num_epochs=10, lambda_reg=0.01, lambda_entropy=0.1):
    labels = tf.keras.utils.to_categorical(labels, num_classes=logits.shape[-1])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Apply temperature calibration
            temperature = tf.nn.softplus(temperature_layer.raw_temperature)
            scaled_logits = logits / temperature

            # Calculate cross entropy loss
            loss = loss_fn(labels, scaled_logits)

            # Add regularization term (pull towards 1)
            temperature_loss = tf.reduce_sum(tf.square(temperature - 1))
            loss += lambda_reg * temperature_loss

            # Add entropy regularization (encourage smooth distributions)
            probs = tf.nn.softmax(scaled_logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
            entropy_loss = -tf.reduce_mean(entropy)
            loss += lambda_entropy * entropy_loss
        
        grads = tape.gradient(loss, [temperature_layer.raw_temperature])
        optimizer.apply_gradients(zip(grads, [temperature_layer.raw_temperature]))
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}, Temperature: {temperature.numpy()}")

def optimize_temperature_soft(logits, soft_labels, temperature_layer, num_epochs=10, lambda_reg=0.01):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            temperature = tf.nn.softplus(temperature_layer.raw_temperature)
            scaled_logits = logits / temperature

            loss = loss_fn(soft_labels, scaled_logits)

            temperature_loss = tf.reduce_sum(tf.square(temperature - 1))
            loss += lambda_reg * temperature_loss
        
        grads = tape.gradient(loss, [temperature_layer.raw_temperature])
        optimizer.apply_gradients(zip(grads, [temperature_layer.raw_temperature]))
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}, Temperature: {temperature.numpy()}")

def gradual_self_train_simple(student_func, teacher, unsup_x, inter_y, interval, confidence_q=0.1,
                       soft=False):

    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []

    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval * i:interval * (i + 1)]
        cur_ys = inter_y[interval * i:interval * (i + 1)]
        
        if soft:
            soft_self_train_once_simple(student, teacher, cur_xs)
        else:
            self_train_once_simple(student, teacher, cur_xs, confidence_q)
        accuracy = student.score(cur_xs, cur_ys)  
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def gradual_self_train_simple_pl(student_func, teacher, unsup_x, inter_y, interval, confidence_q=0.1,
                       soft=False,calib_split=0.3):

    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []

    for i in range(upper_idx):

        cur_xs = unsup_x[interval * i:interval * (i + 1)]
        cur_ys = inter_y[interval * i:interval * (i + 1)]

        # Split into training set and calibration set
        split_idx = int(len(cur_xs) * (1 - calib_split))
        train_x, calib_x = cur_xs[:split_idx], cur_xs[split_idx:]
        train_y, calib_y = cur_ys[:split_idx], cur_ys[split_idx:]

        student = student_func(teacher)

        if soft:
            soft_self_train_once_simple(student, teacher, cur_xs)
        else:
            self_train_once_simple(student, teacher, cur_xs, confidence_q)

        # Use the calibration set to perform Platt Scaling calibration
        calibrator = CalibratedClassifierCV(student, method='sigmoid', cv='prefit')
        calibrated_student = calibrator.fit(calib_x, calib_y)

        # Evaluate the accuracy of the calibrated model on the calibration set of the current batch
        calibrated_probs = calibrated_student.predict_proba(calib_x)
        calibrated_preds = np.argmax(calibrated_probs, axis=1)
        accuracy = np.mean(calibrated_preds == calib_y)
        accuracies.append(accuracy)

        # Use the calibrated model as the new teacher model
        teacher = calibrated_student

    return accuracies, student


def gradual_self_train_simple_pl_pseudo(student_func, teacher, unsup_x, inter_y, interval, confidence_q=0.1,
                                 soft=False, calib_split=0.3, confidence_threshold=0.8):

    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []

    for i in range(upper_idx):
        cur_xs = unsup_x[interval * i:interval * (i + 1)]
        cur_ys = inter_y[interval * i:interval * (i + 1)]

        # split dataset
        split_idx = int(len(cur_xs) * (1 - calib_split))
        train_x, calib_x = cur_xs[:split_idx], cur_xs[split_idx:]
        train_y, calib_y = cur_ys[:split_idx], cur_ys[split_idx:]

        student = student_func(teacher)

        if soft:
            soft_self_train_once_simple(student, teacher, train_x)
        else:
            self_train_once_simple(student, teacher, train_x, confidence_q)

        # Generate pseudo labels for the calibration set
        calib_probs = student.predict_proba(calib_x)
        max_probs = np.max(calib_probs, axis=1)
        pseudo_labels = np.argmax(calib_probs, axis=1)

        # Filter high confidence samples
        high_confidence_mask = max_probs >= confidence_threshold
        high_conf_calib_x = calib_x[high_confidence_mask]
        high_conf_pseudo_labels = pseudo_labels[high_confidence_mask]

        if len(high_conf_calib_x) > 0:
            # Calibrate using high confidence pseudo labels
            calibrator = CalibratedClassifierCV(student, method='sigmoid', cv='prefit')
            calibrated_student = calibrator.fit(high_conf_calib_x, high_conf_pseudo_labels)

            # Evaluate the calibrated model on the calibration set, using the true labels
            calibrated_probs = calibrated_student.predict_proba(calib_x)
            calibrated_preds = np.argmax(calibrated_probs, axis=1)
            accuracy = np.mean(calibrated_preds == calib_y)
            accuracies.append(accuracy)

            # Use the calibrated model as the new teacher model
            teacher = student
        else:
            teacher = student
            uncalibrated_preds = np.argmax(calib_probs, axis=1)
            accuracy = np.mean(uncalibrated_preds == calib_y)
            accuracies.append(accuracy)

    return accuracies, calibrated_student



def split_data(xs, ys, splits):
    return np.split(xs, splits), np.split(ys, splits)


def train_to_acc(model, acc, train_x, train_y, val_x, val_y):
    # Modify steps per epoch to be around dataset size / 10
    # Keep training until accuracy 
    batch_size = 32
    data_size = train_x.shape[0]
    steps_per_epoch = int(data_size / 50.0 / batch_size)
    logger.info("train_xs size is %s", str(train_x.shape))
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, steps_per_epoch=steps_per_epoch, verbose=False)
        val_accuracy = model.evaluate(val_x, val_y, verbose=False)[1]
        logger.info("validation accuracy is %f", val_accuracy)
        if val_accuracy >= acc:
            break
    return model


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    model = load_model(filename)


def rolling_average(sequence, r):
    N = sequence.shape[0]
    assert r < N
    assert r > 1
    rolling_sums = []
    cur_sum = sum(sequence[:r])
    rolling_sums.append(cur_sum)
    for i in range(r, N):
        cur_sum = cur_sum + sequence[i] - sequence[i-r]
        rolling_sums.append(cur_sum)
    return np.array(rolling_sums) * 1.0 / r

