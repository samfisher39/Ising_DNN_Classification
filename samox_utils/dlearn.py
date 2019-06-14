import os

import tensorflow as tf
import numpy as np

from samox_utils.logging_utils import *
from samox_utils.mathx import get_nearest_proper_divisor


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define DNN model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DnnModel(object):

    def __init__(self, _n_neurons, _n_hidden_layers, _n_features, _optimizer_kwargs, _dtype=tf.float64):

        """
        DNN class to be trained on some input data.

        :param _n_neurons: number of neurons of the hidden layers
        :param _n_hidden_layers: number of hidden layers
        :param _n_features: number of features, for the Ising data this is 40x40=1600
        :param _optimizer_kwargs: dictionary object for the optimizer of the model
        :param _dtype: data type for the calculations (default: double precision)
        """

        self.global_step = tf.Variable(0, dtype=_dtype, trainable=False)
        self.n_neurons = _n_neurons
        self.features = _n_features
        self.dropout_rate = tf.placeholder(dtype=_dtype, name="dropout_rate")
        self.n_categories = 2
        self.isTraining = True

        with tf.variable_scope("input"):
            self.x_data = tf.placeholder(dtype=_dtype, shape=(None, self.features), name="x_data")
            self.y_data = tf.placeholder(dtype=_dtype, shape=(None, self.n_categories), name="y_data")

        with tf.variable_scope("dnn"):
            _stddev = 0.1  # 10 / ((_n_hidden_layers + 2) * _n_neurons)
            w_first = tf.Variable(tf.truncated_normal(shape=[self.features, self.n_neurons], stddev=_stddev,
                                                      mean=0, dtype=_dtype), dtype=_dtype, name="weights")
            w_last = tf.Variable(tf.truncated_normal(shape=[self.n_neurons, self.n_categories], stddev=_stddev,
                                                     mean=0, dtype=_dtype), dtype=_dtype, name="weights")
            b_first = tf.Variable(tf.ones(shape=[self.n_neurons], dtype=_dtype), dtype=_dtype)
            b_last = tf.Variable(tf.ones([self.n_categories], dtype=_dtype), dtype=_dtype)

            if _n_hidden_layers > 0:
                w_mutable = tf.Variable(tf.random_normal(shape=[_n_hidden_layers, self.n_neurons, self.n_neurons],
                                                         stddev=_stddev,
                                                         mean=0, dtype=_dtype), dtype=_dtype, name="weights")
                b_mutable = tf.Variable(tf.ones(shape=[_n_hidden_layers, self.n_neurons], dtype=_dtype), dtype=_dtype)

            # Feed forward. ADD/TRY ADDITIONAL ACTIVATIONS
            activation = self.x_data
            activation = tf.nn.relu(tf.matmul(activation, w_first) + b_first)
            for i in range(_n_hidden_layers - 1):
                activation = tf.nn.relu(tf.matmul(activation, w_mutable[i, :, :], ) + b_mutable[i, :])
            if(self.isTraining):
                prediction = tf.matmul(activation, w_last) + b_last
            else:
                prediction = tf.nn.softmax(tf.matmul(activation, w_last) + b_last)
            self.y_pred = prediction

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_data,
                                                                              logits=self.y_pred))
            tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("optimizer"):
            opt = tf.train.GradientDescentOptimizer(**_optimizer_kwargs)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

        with tf.variable_scope('accuracy'):
            # percentage of correct predictions
            percentage = tf.equal(tf.argmax(self.y_data, 1), tf.argmax(self.y_pred, 1))
            self.accuracy = tf.reduce_mean(tf.cast(percentage, tf.float64))

            tf.summary.scalar("accuracy", self.accuracy)

        self.init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()

        print("\n\n\n")
        with open("./logs/latest.txt", "a+") as log:
            log.write("\n\n\n")

    def setIsTraining(self, _isTraining):
        self.isTraining = _isTraining


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define train/test model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def analyze_dnn(_data_set, _n_neurons, _n_layers, _optimizer_kwargs, _n_epochs, _n_batches, _n_features=1600,
                _seed=None, _dtype=tf.float64):
    """
    trains the model on a test set and evaluates its performance on another (excluded from the train set) test set.
    finally also predictions about critical states (excluded in train and test sets) will be made.

    :param _data_set: dictionary with keys "train", "test", "val", "crit" and values of type DataSet (self defined class)
    :param _n_neurons: (int) number of neurons of the hidden layers
    :param _n_layers: (int) number of hidden layers
    :param _optimizer_kwargs: a dictionary object for the optimizer, e.g. dict(learning_rate=0.1)
    :param _n_epochs: (int) number of times the whole training set should be fed into the DNN
    :param _n_batches: (int) number of batches the training data should be split up to. This number is manipulated to be
    a proper divisor of the number of training samples.
    :param _n_features: (int) number of input values for the calculations of the first layer, for this particular
    training set of the Ising model this value is 40x40=1600
    :param _seed: (int) seed for random numbers
    :param _dtype: dtype for the calculations (default: double precision)
    :return: (n_epochs, n_batches) array of accuracy during the training,
    (n_epochs, n_batches) array of loss during the training,
    accuracy obtained from the test data set
    accuracy obtained from the critical data set
    """

    tf.reset_default_graph()  # prevent variables from being used/defined multiple times

    dnn = DnnModel(_n_neurons=_n_neurons, _n_hidden_layers=_n_layers, _n_features=_n_features,
                   _optimizer_kwargs=_optimizer_kwargs, _dtype=_dtype)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter("./logs/train", sess.graph)

        # TRAINING
        sess.run(dnn.init)
        x_batches, y_batches, _n_batches = _data_set["train"].get_batches(_n_batches=_n_batches, _seed=_seed)

        accuracy = np.empty(shape=(_n_epochs, _n_batches))
        loss = np.empty(shape=(_n_epochs, _n_batches))
        # accuracy = np.empty(shape=_n_epochs*_n_batches)
        # loss = np.empty(shape=_n_epochs*_n_batches)

        print_log("Learning rate: %f" % (_optimizer_kwargs["learning_rate"]), "number of neurons: %i" % _n_neurons,
                  _top=False, log_file="./logs/latest.txt")

        print()
        with open("./logs/latest.txt", "a+") as log:
            log.write("\n")

        dnn.setIsTraining(True)
        for epoch_idx in range(_n_epochs):
            print_log("EPOCH %i/%i" % (epoch_idx + 1, _n_epochs), log_file="./logs/latest.txt")
            print_log("", _top=False, _bottom=False, log_file="./logs/latest.txt")

            for batch_idx in range(x_batches.shape[0]):
                batch_feed_dict = {dnn.x_data: x_batches[batch_idx, :, :],
                                   dnn.y_data: y_batches[batch_idx, :, :],
                                   dnn.dropout_rate: 0.5
                                   }

                _, _step, _summary, = sess.run([dnn.optimizer, dnn.global_step, dnn.merged],
                                               feed_dict=batch_feed_dict)
                _loss_batch, _accuracy_batch = sess.run([dnn.loss, dnn.accuracy],
                                                        feed_dict=batch_feed_dict)

                accuracy[epoch_idx, batch_idx] = _accuracy_batch
                loss[epoch_idx, batch_idx] = _loss_batch

                train_writer.add_summary(_summary, sess.run(dnn.global_step))
                if x_batches.shape[0] // 10 != 0:
                    if batch_idx % (x_batches.shape[0] // 10) == 0:
                        print_log("Batch %i/%i: Loss: %1.2f, Accuracy: %1.2f" % (batch_idx + 1, x_batches.shape[0],
                                                                                 _loss_batch,
                                                                                 _accuracy_batch), _bottom=False,
                                  _top=False, log_file="./logs/latest.txt")
                else:
                    print_log("Batch %i/%i: Loss: %1.2f, Accuracy: %1.2f" % (batch_idx + 1, x_batches.shape[0],
                                                                             _loss_batch,
                                                                             _accuracy_batch), _bottom=False,
                              _top=False, log_file="./logs/latest.txt")
            print_log("", _top=False, _bottom=False, log_file="./logs/latest.txt")

        print_hline()

        # TESTING
        dnn.setIsTraining(False) # change activation function of last layer to softmax
        x_batches_test, y_batches_test = _data_set["test"].get_batches()
        _accuracy_test, _loss_test = sess.run([dnn.accuracy, dnn.loss], feed_dict={
            dnn.x_data: x_batches_test,
            dnn.y_data: y_batches_test,
            dnn.dropout_rate: 1.0
        })

        print_log("", "TEST-SET:", "- Loss: %f" % _loss_test, "- Accuracy: %f" % _accuracy_test, _bottom=False,
                  _top=False, log_file="./logs/latest.txt")

        # CRITICAL
        x_batches_crit, y_batches_crit = _data_set["crit"].get_batches()
        _accuracy_crit, _loss_crit = sess.run([dnn.accuracy, dnn.loss], feed_dict={
            dnn.x_data: x_batches_crit,
            dnn.y_data: y_batches_crit,
            dnn.dropout_rate: 1.0
        })
        print_log("", "CRIT-SET:", "- Loss: %f" % _loss_crit, "- Accuracy: %f" % _accuracy_crit, "", _top=False,
                  log_file="./logs/latest.txt")

        accuracy = np.ravel(accuracy)
        loss = np.ravel(loss)

        # return accuracy, loss
        return accuracy, loss, _accuracy_test, _accuracy_crit


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ optimized args search ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def find_optimal_args(_data_set, _lr_array, _neurons_array, _n_layers, _seed=1, _n_epochs=1,
                      _n_batches=2000):
    """
    calculates the loss and accuracy trends for the given learning rates and number of neurons

    :param _data_set: dictionary with keys "train", "test", "val", "crit" and values of type DataSet (self defined class)
    :param _lr_array: (float array) array of learning rates to be tested
    :param _neurons_array: (integer array) array of number of neurons to be tested
    :param _n_layers: (int) number of hidden layers
    :param _seed: (int) seed for random numbers
    :param _n_epochs: (int) number of epochs each batch should be fed into the DNN
    :param _n_batches: (int) number of batches the training data should be split up to. This number is manipulated to be
    a proper divisor of the number of training samples.
    :return: (neurons, learning_rates, 2, number_of_batches) array of training history, 2d array of test accuracy,
    2d array of critical accuracy
    """

    _summary = np.empty(shape=(len(_neurons_array), len(_lr_array), 2,
                               _n_epochs * get_nearest_proper_divisor(_n_batches, _data_set["train"].n_samples)))
    acc_test = np.empty(shape=(len(_neurons_array), len(_lr_array)))
    acc_crit = np.empty(shape=(len(_neurons_array), len(_lr_array)))

    for i, _n_neurons in enumerate(_neurons_array):
        for j, _lr in enumerate(_lr_array):
            _optimizer_args = dict(learning_rate=_lr)

            acc, loss, acc_test[i, j], acc_crit[i, j] = \
                analyze_dnn(_data_set=_data_set, _n_neurons=_n_neurons, _n_layers=_n_layers,
                            _optimizer_kwargs=_optimizer_args, _n_epochs=_n_epochs,
                            _n_batches=_n_batches, _seed=_seed)

            _summary[i, j, :, :] = acc, loss

    return _summary, acc_test, acc_crit
