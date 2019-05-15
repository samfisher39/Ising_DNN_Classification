"""
If this project is executed in PyCharm in a "Scientific" environment, then go to Settings (Alt+Ctrl+S by default)
-> Tools -> Python Scientific and disable the "Show plots in tool window" option.
"""

# if on python version 2.x uncomment the following line
# from __future__ import print_function, division, absolute_import

import matplotlib
import matplotlib.rcsetup as rc_setup

from samox_utils.plotting import *
from samox_utils.serialization import *
from samox_utils.dlearn import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# for deterministic reproducibility -> set seed (makes same random numbers on every run)
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

N_FEATURES_X = N_FEATURES_Y = 40
N_FEATURES = int(N_FEATURES_X * N_FEATURES_Y)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get matplotlib backends ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(rc_setup.interactive_bk)
print(rc_setup.non_interactive_bk)
print(plt.get_backend())

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Use specific backend ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
matplotlib.use("Qt5Agg")

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ do INIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.remove("./logs/latest.txt")
data, labels = init(_reinit=False, _save=True, _verbose=True)
data_set = post_init(data, labels)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plotting some samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_plots = 25
rdm_idx = np.floor(np.random.rand(n_plots) * 159999).astype("int")
plot_samples(data, labels, _sample_indices=rdm_idx,
             _title="some ordered (1), mixed (1|0) and unordered (0) states",
             _title_pad=50)

# %%
learning_rates = np.logspace(-4, -1, 4)
number_of_neurons = np.logspace(0, 3, 4).astype("int")
print(learning_rates)
print(number_of_neurons)

# %%

number_of_hidden_layers = 1
summary, acc_test, acc_crit = find_optimal_args(data_set, learning_rates, number_of_neurons, number_of_hidden_layers,
                                                _n_epochs=1)


# %%
plot_calculations(summary, acc_test, acc_crit, learning_rates, number_of_neurons)


