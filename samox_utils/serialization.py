import os
import pickle
from urllib.request import urlopen
from urllib.request import urlretrieve

from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from samox_utils.dataset import *


def init(_reinit=False, _save=True, _verbose=False, _n_features=1600):
    """
    The data set consists of 160,000 samples with 40x40 features per sample and is encoded by pickle.
    Every sample seems to be split into 100 hexadecimal UTF-16 characters, i.e. 0000~ffff.
    The label set contains only 0s and 1s, which represent disordered and ordered states, respectively.

    :param _reinit: delete existing data and re-download
    :param _save: save to local $PROJECTROOT/data folder
    :param _verbose: print log/debug stuff
    :param _n_features: 1600 , this number is unique to this data set. DO NOT modify this or else the program
                    will break!
    :return: data: array(n_samples, n_features), labels: array(n_samples,)
    """

    print_log("Initializing data", log_file="./logs/latest.txt")

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
        with open("./data/labels.pkl", 'rb') as labels_file:
            _labels = pickle.load(labels_file)
        print_log("Files found locally! Importing...", _top=False, _bottom=False, log_file="./logs/latest.txt")
    except:  # if files not found, download them
        print_log("Files not found locally. Downloading from \
                 https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/", _top=False, _bottom=False,
                  log_file="./logs/latest.txt")
        if _save:  # if save=True then, save the files to the local "./data/" folder, otherwise just load them
            # temporarily
            if _verbose:
                print_log("Downloading/Saving ISING data file...", _top=False, _bottom=False,
                          log_file="./logs/latest.txt")
            urlretrieve(_data_path, "./data/data.pkl")
            if _verbose:
                print_log("...done", "Downloading/Saving ISING labels file...", _top=False, _bottom=False,
                          log_file="./logs/latest.txt")
            urlretrieve(_labels_path, "./data/labels.pkl")
            if _verbose:
                print_log("...done", _top=False, _bottom=False,
                          log_file="./logs/latest.txt")
            with open("data/data.pkl", 'rb') as datafile:
                _data = pickle.load(datafile)
            with open("data/labels.pkl", 'rb') as labels_file:
                _labels = pickle.load(labels_file)
        else:
            if _verbose:
                print_log("Downloading ISING data file...", _top=False, _bottom=False, log_file="./logs/latest.txt")
            _data = pickle.load(urlopen(_data_path))
            if _verbose:
                print_log("...done", "Downloading ISING labels file...", _top=False, _bottom=False,
                          log_file="./logs/latest.txt")
            _labels = pickle.load(urlopen(_labels_path))
            if _verbose:
                print_log("...done", _top=False, _bottom=False, log_file="./logs/latest.txt")

    # process data file, to be shaped into an (n_samples, n_features)-sized array
    _data = np.unpackbits(_data).astype("int").reshape(-1, _n_features)
    # cast all zeros to ones in the data array
    _data[np.where(_data == 0)] = -1

    if _verbose:
        print_log("PRE_INIT completed:", "- # features: %i" % _n_features, _top=False, _bottom=False,
                  log_file="./logs/latest.txt")
        print_log("- # samples: %i" % _data.shape[0], _top=False, _bottom=True, log_file="./logs/latest.txt")

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
    _labels_categorical = to_categorical(_labels, num_classes=2, dtype=np.dtype(_dtype))

    _x_ordered = _data[:70000, :]
    _y_ordered = _labels_categorical[:70000]
    _x_critical = _data[70000:100000, :]
    _y_critical = _labels_categorical[70000:100000]
    _x_disordered = _data[100000:, :]
    _y_disordered = _labels_categorical[100000:]

    # exclude critical data from training
    _x_data = np.append(_x_ordered, _x_disordered, axis=0)
    _y_data = np.append(_y_ordered, _y_disordered, axis=0)

    _x_train, _x_test, _y_train, _y_test = train_test_split(_x_data,
                                                            _y_data,
                                                            test_size=_n_test,
                                                            train_size=1 - _n_test)

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
