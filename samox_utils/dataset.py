import numpy as np

from samox_utils.logging_utils import *
from samox_utils.mathx import *


class DataSet(object):

    def __init__(self, _x_dat, _y_dat, _dtype=np.dtype(np.float64), verbose=True):

        """
        class holding the data on which the DNN will be trained on

        :param _x_dat: input data
        :param _y_dat: expected output data
        :param _dtype: data type of the calculations (default: double precision)
        :param verbose: whether more information is written to the screen/log
        """

        print_log("Initializing data object", _top=True, _bottom=False, log_file="./logs/latest.txt")
        if _x_dat.shape[0] != _y_dat.shape[0]:
            raise ValueError("dimension mismatch")

        _dtype = np.dtype(_dtype)
        if _dtype not in (np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.uint8)):
            raise TypeError("Invalid dtype %r, expected uint8, float32 or float64" % _dtype)
        else:
            if verbose:
                print_log("Using dtype %r" % _dtype, _bottom=False, _top=False, log_file="./logs/latest.txt")

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

        print_log("...done", _bottom=True, _top=False, log_file="./logs/latest.txt")

    def get_batches(self, _n_batches=None, _seed=None):
        """
        _n_batches = n_samples/batch_size
        Shuffles the data and labels and returns an array of the size (_n_batches, batch_size, n_features)
        stochastic gradient descent: batch_size = 1
        batch gradient descent: batch_size = n_samples
        mini-batch gradient descent: 1 < batch_size < n_samples

        :param _n_batches: in how many batches the training data should be split.
        :param _seed: seed for the random shuffling of the data and label arrays.
        :return: input array of shape (_n_batches, batch_size, n_features), output array of
        shape (_n_batches, batch_size,)
        """
        if _n_batches is None or _n_batches == 1:
            return self.x_data, self.y_data

        print_log("Creating batches...", _top=True, _bottom=False, log_file="./logs/latest.txt")
        if self.n_samples % _n_batches != 0:
            _nearest_proper_divisor = get_nearest_proper_divisor(_n_batches, self.n_samples)
            print_log("%i is not a proper divisor of %i" % (_n_batches, self.n_samples), "using %i instead!" %
                      _nearest_proper_divisor, _bottom=False, _top=False, log_file="./logs/latest.txt")
            _n_batches = _nearest_proper_divisor

        if _seed is not None:
            np.random.seed(_seed)

        p = np.random.permutation(self.n_samples)
        self.x_data = self.x_data[p]
        self.y_data = self.y_data[p]

        print_log("...done", _top=False, _bottom=True, log_file="./logs/latest.txt")

        # split the whole data and labels arrays into _n_batches different packages/batches
        return np.array(np.split(self.x_data, _n_batches)), np.array(np.split(self.y_data, _n_batches)), _n_batches
