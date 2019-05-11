# if on python version 2.x uncomment the following line
# from __future__ import print_function, division, absolute_import

import os
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve

import keras as krs
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn import model_selection

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# for deterministic reproducibility -> set seed (makes same random numbers on every run)
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
PROMPT = "+ "
N_FEATURES_X = N_FEATURES_Y = 40
N_FEATURES = int(N_FEATURES_X * N_FEATURES_Y)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get matplotlib backends ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.rcsetup as rc_setup

print(rc_setup.interactive_bk)
print(rc_setup.non_interactive_bk)
print(plt.get_backend())


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define plotting function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_samples(_data, _labels, _sample_indices=None, _title="Sample Plots", _title_pad=20):
    """
    Plots the specified indices in a 16:9 ratio to the screen and triggers fullscreen mode.
    :param _data: data to be visualized
    :param _labels: binary classification label
    :param _sample_indices: indices of data to be visualized
    :param _title: title of plot
    :return:
    """

    if _sample_indices is None:
        _sample_indices = [0, 40000, 80000, 120000, 159999]
    _sample_indices = np.sort(np.array(_sample_indices))
    if _sample_indices.dtype not in (np.dtype(np.int64), np.dtype(np.int32)):
        raise TypeError("Invalid dtype for _sample_indices %r, expected int32 or int64" % _sample_indices.dtype.name)
    subplot_idx1 = np.math.floor(np.sqrt(9 / 16 * _sample_indices.shape[0]))
    subplot_idx2 = len(_sample_indices) // subplot_idx1
    _n_plots = subplot_idx2 * subplot_idx1

    fig = plt.figure(figsize=(19.2, 10.8))  # make a 1920x1080 (16:9) plot
    n_samples = _data.shape[0]

    for i, idx in enumerate(_sample_indices):
        if i < n_plots:
            if idx >= n_samples:
                raise IndexError
            dat = _data[idx, :].reshape(40, 40)
            ax = fig.add_subplot(subplot_idx1, subplot_idx2, i + 1)
            ax.matshow(dat, interpolation="nearest", vmin=0, vmax=1)
            if 53 < _n_plots <= 90:
                ax.set_title("Sample: " + str(idx) + " | " + str(_labels[idx]), pad=0.0, fontsize=9)
            elif n_plots > 90:
                ax.set_title("")
            else:
                ax.set_title("Sample: " + str(idx) + " | " + str(_labels[idx]), pad=0.0)
            ax.set_xticks([])
            ax.set_yticks([])

    # different matplotlib backends use different methods for maximizing windows
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()  # Qt backend
    # mng.resize(*mng.window.maxsize())     # TkAgg backend
    # mng.frame.Maximize(True)              # WX backend

    # only include title of plots, if n_plots is small enough.
    if n_plots < 110:
        fig.suptitle(_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3)  # Adjust layout to not overlap
    else:
        fig.tight_layout(h_pad=0, w_pad=0)  # Adjust layout to not overlap
    plt.show()


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define INIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def init(_reinit=False, _save=True, _verbose=False, _n_features=1600):
    """
    The data set consists of 160,000 samples with 40x40 features per sample and is encoded by pickle.
    Every sample seems to be split into 100 hexadecimal UTF-16 characters, i.e. 0000~ffff.
    The label set contains only 0s and 1s, which represent disordered and ordered states, respectively.

    :param _reinit: delete existing data and re-download
    :param _save: save to local $PROJECTROOT/data folder
    :param _verbose: print log/debug stuff
    :param _n_features: 1600 , this number is unique to this data set. DO NOT modify this or else the program will break!
    :return: data: array(n_samples, n_features), labels: array(n_samples,)
    """

    print(30 * "~")
    print(10 * "~" + " PRE_INIT " + 10 * "~")
    print(30 * "~")

    # define URLs of the data and labels file
    _url_root = "https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/"
    _data_file = "Ising2DFM_reSample_L40_T=All.pkl"
    _labels_file = "Ising2DFM_reSample_L40_T=All_labels.pkl"
    _data_path = _url_root + _data_file
    _labels_path = _url_root + _labels_file

    # reinit (re-initialize) <=> re-download the data
    if _reinit:
        os.remove("./data/data.pkl")
        os.remove("./data/labels.pkl")
    try:  # try if files exist on local "./data/" folder
        with open("./data/data.pkl", 'rb') as datafile:
            _data = pickle.load(datafile)
        with open("./data/labels.pkl", 'rb') as labelsfile:
            _labels = pickle.load(labelsfile)
        print(PROMPT + 'Files found locally! Importing...')
    except:  # if files not found, download them
        print(PROMPT + "Files not found locally. Downloading from \
                 https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/")
        if _save:  # if save=True then, save the files to the local "./data/" folder, otherwise just load them
            # temporarily
            if _verbose:
                print(PROMPT + "\tDownloading/Saving ISING data file...")
            urlretrieve(_data_path, "./data/data.pkl")
            if _verbose:
                print(PROMPT + "\t\t...done")
                print(PROMPT + "\tDownloading/Saving ISING labels file...")
            urlretrieve(_labels_path, "./data/labels.pkl")
            if _verbose:
                print(PROMPT + "\t\t...done")
            with open("data/data.pkl", 'rb') as datafile:
                _data = pickle.load(datafile)
            with open("data/labels.pkl", 'rb') as labelsfile:
                _labels = pickle.load(labelsfile)
        else:
            if _verbose:
                print(PROMPT + "\tDownloading ISING data file...")
            _data = pickle.load(urlopen(_data_path))
            if _verbose:
                print(PROMPT + "\t\t...done")
                print(PROMPT + "\tDownloading ISING labels file...")
            _labels = pickle.load(urlopen(_labels_path))
            if _verbose:
                print(PROMPT + "\t\t...done")

    # process data file, to be shaped into an (n_samples, n_features)-sized array
    _data = np.unpackbits(_data).astype("int").reshape(-1, _n_features)
    # cast all zeros to ones in the data array
    _data[np.where(_data == 0)] = -1

    if _verbose:
        print(PROMPT + "PRE_INIT completed:")
        print(PROMPT + "\t\t- # features: %i" % _n_features)
        print(PROMPT + "\t\t- # samples: %i" % _data.shape[0])

    # return loaded objects
    return _data, _labels


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define POST_INIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def post_init(_data, _labels, _n_validation=0.1, _n_test=0.1):
    """
    map integer/float matrix onto binary class matrix, i.e. 0 -> [1,0], 1 -> [0,1]
    split the data into training and testing sets, while excluding critical states.

    :param _data: not yet randomized data
    :param _labels: not yet randomized labels
    :param _n_validation: size of validation array in terms of n_train
    :param _n_test: size of test array in terms of n_samples
    :return:
    """
    _N_train_val = (1 - _n_test) * _data.shape[0]
    _N_validation = int(_n_validation * _N_train_val)
    _dtype = np.array(_labels).dtype
    _labels_categorical = krs.utils.to_categorical(_labels, num_classes=2, dtype=np.dtype(_dtype))

    _x_ordered = _data[:70000, :]
    _y_ordered = _labels_categorical[:70000]
    _x_critical = _data[70000:100000, :]
    _y_critical = _labels_categorical[70000:100000]
    _x_disordered = _data[100000:, :]
    _y_disordered = _labels_categorical[100000:]

    # exclude critical data from training
    _x_data = np.append(_x_ordered, _x_disordered, axis=0)
    _y_data = np.append(_y_ordered, _y_disordered, axis=0)

    _x_train, _x_test, _y_train, _y_test = model_selection.train_test_split(_x_data,
                                                                            _y_data,
                                                                            test_size=_n_test,
                                                                            train_size=1 - _n_test)

    print(_N_validation)
    _x_train_val = _x_train[:_N_validation, :]
    _y_train_val = _y_train[:_N_validation, :]
    _x_train = _x_train[_N_validation:, :]
    _y_train = _y_train[_N_validation:, :]

    data_dict = {
        "crit": DataSet(_x_critical, _y_critical),
        "test": DataSet(_x_test, _y_test),
        "train": DataSet(_x_train, _y_train),
        "val": DataSet(_x_train_val, _y_train_val)
    }

    return data_dict


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define Data object class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DataSet(object):
    """
    class for Data object, holding:
        Variables:
        - n_samples (int)
        - x_data (n_samples x n_features array)
        - y_data (n_samples-sized 1-d array)

        Methods:
        - get_batches, returns next batch of data with given size.
    """

    def __init__(self, _x_dat, _y_dat, _dtype=np.dtype(np.float32), verbose=False):
        print(20 * PROMPT)
        print(PROMPT)
        print(PROMPT + "Initializing data object")
        if _x_dat.shape[0] != _y_dat.shape[0]:
            raise ValueError("dimension mismatch")

        _dtype = np.dtype(_dtype)
        if _dtype not in (np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.uint8)):
            raise TypeError("Invalid dtype %r, expected uint8, float32 or float64" % _dtype)
        else:
            if verbose:
                print(PROMPT + "\t\tUsing dtype %r" % _dtype)

        if _dtype == np.dtype(np.float32):
            _x_dat = _x_dat.astype(np.float32)
        if _dtype == np.dtype(np.float64):
            _x_dat = _x_dat.astype(np.float64)

        self.n_samples = _x_dat.shape[0]
        self.n_features = _x_dat.shape[1]
        self.x_data = _x_dat
        self.y_data = _y_dat
        self.epoch_idx = 0

        # DELETE THIS
        self.epochs_completed = 0
        self.index_in_epoch = 0

        print(PROMPT + "\t\t...done")
        print(PROMPT)
        print(20 * PROMPT)
        print(PROMPT)

    def get_batches(self, _n_batches, _seed=None):
        """
        n_batches = n_samples/batch_size
        Shuffles the data and labels and returns an array of the size (n_batches, batch_size, n_features)
        stochastic gradient descent: batch_size = 1
        batch gradient descent: batch_size = n_samples
        mini-batch gradient descent: 1 < batch_size < n_samples

        :param _n_batches: how many samples one batch of training data should contain.
        :param _seed: seed for the random shuffling of the data and label arrays.
        :return: data: array(n_batches, batch_size, n_features), labels: array(n_batches, batch_size,)
        """

        print(PROMPT + "Creating batches")
        if self.n_samples % _n_batches != 0:
            raise ValueError("batch_size must be a proper divisor of n_samples")
        n_samples = self.n_samples / _n_batches

        if seed is not None:
            np.random.seed(seed)

        p = np.random.permutation(self.n_samples)
        self.x_data = self.x_data[p]
        self.y_data = self.y_data[p]

        print(PROMPT + "\t\t...done")
        print(PROMPT)
        print(20 * PROMPT)

        # split the whole data and labels arrays into _n_batches different packages/batches
        return np.array(np.split(self.x_data, n_samples)), np.array(np.split(self.y_data, n_samples))

    def next_batch(self, batch_size, seed=None):
        """Return the next `batch_size` examples from this data set."""

        if seed:
            np.random.seed(seed)

        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data_X = self.data_X[perm]
            self.data_Y = self.data_Y[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        return self.data_X[start:end], self.data_Y[start:end]


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ define DNN model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# class DeepNeuralNetwork(object):
#
#     def __init__(self, _n_neurons, _optimizer_kwargs):
#         """
#         some explanation of what this does...
#
#         :param _n_neurons: 1-d array of variable length. its length specifies the number of hidden layers, its values
#                             the number of neutrons per hidden layer
#         :param _optimizer_kwargs: **kwargs (keyworded args, variable length) for the gradient descent optimizer
#         """
#     model
def DeepNeuralNetwork(_n_neurons, _optimizer_kwargs) -> object:
    model = krs.Sequential()
    for i in range(len(_n_neurons) - 1):
        model.add(krs.layers.Dense(units=_n_neurons[i], activation=tf.nn.relu))
    model.add(krs.layers.Dense(units=2, activation=tf.nn.softmax))
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.categorical_crossentropy,  # 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ do INIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data, labels = init(_reinit=False, _save=True, _verbose=True)


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ train/test model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def analyze_dnn(_data, _labels, _n_hidden_neurons, _optimizer_kwargs, _n_epochs, _batch_size, _n_features=1600):
    """trains the model on a test set and evaluates its performance on another (excluded from the train set) test set.
    finally also predictions about critical states (excluded in train and test sets) will be made."""

    _data_dict = post_init(_data, _labels)
    _n_neurons = np.append(np.insert(np.array(_n_hidden_neurons), 0, _data_dict["train"].n_features), 2)

    print(_data_dict["train"].y_data.shape)
    dnn = DeepNeuralNetwork(_n_neurons=_n_neurons, _optimizer_kwargs=_optimizer_kwargs)
    dnn.fit(_data_dict["train"].x_data, _data_dict["train"].y_data, epochs=_n_epochs, batch_size=_batch_size,
            validation_data=(_data_dict["val"].x_data, _data_dict["val"].y_data))
    dnn.evaluate(_data_dict["test"].x_data, _data_dict["test"].y_data)
    dnn.summary()
    return dnn


# %%

n_hidden = [100]
dnn = analyze_dnn(data, labels, n_hidden, dict(learning_rate=0.1), _n_epochs=5, _batch_size=20000)

# %%


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plotting some samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_plots = 25
rdm_idx = np.floor(np.random.rand(n_plots) * 159999).astype("int")
plot_samples(data, labels, _sample_indices=rdm_idx, _title="some ordered (1), mixed (1|0) and unordered (0) states",
             _title_pad=50)
