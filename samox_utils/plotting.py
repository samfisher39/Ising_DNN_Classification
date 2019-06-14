import numpy as np
import matplotlib.pyplot as plt

from samox_utils.mathx import get_nearest_proper_divisor


def plot_samples(_data, _labels, _sample_indices=None, _title="Sample Plots", _title_pad=20):
    """
    Plots the specified indices in a 16:9 ratio to the screen and triggers fullscreen mode.
    :param _title_pad: padding of the title to the plot
    :param _data: data to be visualized
    :param _labels: binary classification label
    :param _sample_indices: indices of data to be visualized
    :param _title: title of plot
    :param _title_pad: padding of the title
    :return: None
    """

    if _sample_indices is None:
        _sample_indices = [0, 40000, 80000, 120000, 159999]
    _sample_indices = np.sort(np.array(_sample_indices))
    if _sample_indices.dtype not in (np.dtype(np.int64), np.dtype(np.int32)):
        raise TypeError("Invalid dtype for _sample_indices %r, expected int32 or int64" % _sample_indices.dtype.name)
    subplot_idx1 = np.math.floor(np.sqrt(9 / 16 * _sample_indices.shape[0]))
    if 0 != subplot_idx1:
        subplot_idx2 = len(_sample_indices) // subplot_idx1
    else:
        subplot_idx1 = 1
        subplot_idx2 = 1
    _n_plots = subplot_idx2 * subplot_idx1

    fig = plt.figure(figsize=(19.2, 10.8))  # make a 1920x1080 (16:9) plot
    n_samples = _data.shape[0]

    for i, idx in enumerate(_sample_indices):
        if i < _n_plots:
            if idx >= n_samples:
                raise IndexError
            dat = _data[idx, :].reshape(40, 40)
            ax = fig.add_subplot(subplot_idx1, subplot_idx2, i + 1)
            ax.matshow(dat, interpolation="nearest", vmin=0, vmax=1)
            if 53 < _n_plots <= 90:
                ax.set_title("Sample: " + str(idx) + " | " + str(_labels[idx]), pad=0.0, fontsize=9)
            elif _n_plots > 90:
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
    if _n_plots < 110:
        fig.suptitle(_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3)  # Adjust layout to not overlap
    else:
        fig.tight_layout(h_pad=0, w_pad=0)  # Adjust layout to not overlap
    plt.show()


def plot_acc(fig, axis, _acc, _learning_rates, _number_of_neurons, _title):

    """
    plot the accuracy in the form of a 2d plot with the shape (learning_rates, number_of_neurons)

    :param fig: figure which is to be plotted to
    :param axis: subplot axes of fig
    :param _acc: (float) 2d-array containing the accuracy at the different learning rates and number of neurons
    :param _learning_rates: (float) array of learning rates
    :param _number_of_neurons: (int) array of the number of neurons
    :param _title: (string) title of the subplot
    :return:
    """

    ax = axis
    cax = ax.matshow(_acc, interpolation='nearest', vmin=0, vmax=1)

    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)', rotation=90, fontsize=16)
    cbar.set_ticks([0, .2, .4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    x_ticks = [""]
    y_ticks = [""]

    for a, _lr in enumerate(_learning_rates):
        for b, _nn in enumerate(_number_of_neurons):
            c = "${0:.2f}\\%$".format(100 * _acc[b, a])
            ax.text(a, b, c, va='center', ha='center')

    for _, _lr in enumerate(_learning_rates):
        x_ticks.append(str(_lr))

    for _, _nn in enumerate(_number_of_neurons):
        y_ticks.append(str(_nn))

    ax.set_xlabel("learning rate")
    ax.set_ylabel("number of neurons")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_title(_title, pad=0)


def smooth(y, box_pts):
    """
    smoothing of data with the help of convolution
    :param y: input data
    :param box_pts: strength of smoothing
    :return: smoothed input data
    """
    box = np.ones(box_pts)/box_pts
    smoothed = np.convolve(y, box, mode='same')
    return smoothed


def plot_calculations(_summary, _acc_test, _acc_crit, _learning_rates, _number_of_neurons, _smooth=50, _history=True):

    """
    plot the history of accuracy and loss for every learning rate and number of neurons and
    visualize the obtained accuracy of the different learning rates and number of neurons

    :param _summary: (float) array,array both of the size (number_of_epochs, number_of_batches). The former containing
    the accuracy and the latter containing the loss during the training
    :param _acc_test: (float) 2d array accuracy of different learning rates and number of neurons of the test data set
    :param _acc_crit: (float) 2d array accuracy of different learning rates and number of neurons of the critical data set
    :param _learning_rates: (float) array of the learning rates
    :param _number_of_neurons: (int) array of the number of neurons of the hidden layers
    :return: None
    """

    smoothing = _smooth
    cutoff = 25
    iterations = range(len(_summary[0, 0, 1, :-cutoff]))

    if(_history):
        fig = plt.figure(figsize=[14, 10])
        for i, lr in enumerate(_learning_rates):
            for j, nn in enumerate(_number_of_neurons):
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.plot(iterations, smooth(_summary[j, i, 1, :], smoothing)[:-cutoff],
                         label="lr: %.4f, nn: %i" % (lr, nn),
                         linewidth=1.0)
                ax1.title.set_text("Loss")
                ax1.set_ylim(bottom=0, top=1)
                ax1.set_xlim(0, max(iterations))
                ax1.legend(loc=1, bbox_to_anchor=(1.4, 1.0))

                ax2 = fig.add_subplot(2, 2, 2)
                ax2.plot(iterations, smooth(_summary[j, i, 0, :], smoothing)[:-cutoff],
                         label="lr: %.4f, nn: %i" % (lr, nn),
                         linewidth=1.0)
                ax2.title.set_text("Accuracy")
                ax2.set_xlim(0, max(iterations))

        ax3 = fig.add_subplot(2, 2, 3)
        plot_acc(fig, ax3, _acc_test, _learning_rates, _number_of_neurons, "Test Accuracy")

        ax4 = fig.add_subplot(2, 2, 4)
        plot_acc(fig, ax4, _acc_crit, _learning_rates, _number_of_neurons, "Critical Accuracy")

        fig.tight_layout(w_pad=2, h_pad=4)

    else:
        fig = plt.figure(figsize=[14,7])

        ax1 = fig.add_subplot(1, 2, 1)
        plot_acc(fig, ax1, _acc_test, _learning_rates, _number_of_neurons, "Test Accuracy")

        ax2 = fig.add_subplot(1, 2, 2)
        plot_acc(fig, ax2, _acc_crit, _learning_rates, _number_of_neurons, "Critical Accuracy")

        fig.tight_layout(w_pad=2, h_pad=4)

def plot_net_magnetization(_data, _interval=1, _smooth=1):
    print(_data.shape)
    _n_samples = _data.shape[0]
    _interval = get_nearest_proper_divisor(_interval, _n_samples)
    _n_samples = _n_samples // _interval
    _t_int = np.multiply(range(_n_samples),_interval)

    _net_mag = np.empty(shape=(_n_samples))
    for i in range(_n_samples):
        _net_mag[i] = abs(sum(_data[i*_interval,:]))/1600

    _net_mag_smooth = smooth(_net_mag, _smooth)
    _net_mag_smooth[0:300] = 1
    fig = plt.figure(figsize=(11,6))
    ax  = fig.add_subplot(1,1,1)
    ax.plot(_t_int, _net_mag, label="net mag")
    ax.plot(_t_int, _net_mag_smooth, label="net mag smoothed")
    ax.legend()

    ax.set_xlim(0,_n_samples*_interval)
    ax.set_xlabel("sample n")
    ax.set_ylabel("net magnetization")
