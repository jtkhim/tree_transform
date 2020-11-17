## varying_epsilon_plots.py
## Evaluate results of simulations from adv_res_net02.py.
# Imports
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# Global variables, change to change plots.
n_hidden_list = [150, 200, 300, 400, 500]
n_hidden = 40
gamma_str_list = ["10.0"]
gamma_list = [float(gamma) for gamma in gamma_str_list]
eps_str_list = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1"]
eps_list = [float(eps) for eps in eps_str_list]
string_minus = "fashion_output/results_{0}_hidden_eps_{1}_gamma_{2}.pt"
eps_str = "0.05"
gamma = 10.0

folder_prefix = "fashion_figs/"

filename_11 = folder_prefix + "fig_11_by_n_hidden_eps_05_gamma_{0}.png".format(gamma)
filename_10 = folder_prefix + "fig_10_by_n_hidden_eps_05_gamma_{0}.png".format(gamma)
filename_00 = folder_prefix + "fig_00_by_n_hidden_eps_05_gamma_{0}.png".format(gamma)
filename_01 = folder_prefix + "fig_01_by_n_hidden_eps_05_gamma_{0}.png".format(gamma)
x_label = "hidden layer size"


# Helper functions
def results_by_gamma(n_hidden, eps_string, gamma_str_list, string_minus):
    return [torch.load(string_minus.format(n_hidden, eps_string, gamma_str)).numpy() \
            for gamma_str in gamma_str_list]

def results_by_hidden(n_hidden_list, eps_string, gamma, string_minus):
    return [torch.load(string_minus.format(n_hidden, eps_string, gamma)).numpy() \
            for n_hidden in n_hidden_list]

def results_by_classifier(data_list, cls_index, train_index, test_index):
    return np.array([entry[cls_index, train_index, test_index] for entry in data_list])


def ln_results(data_list, adv_train = False, adv_test = False):
    linear_index = 0
    train_index = int(adv_train)
    test_index = int(adv_test)
    return results_by_classifier(data_list, linear_index, train_index, test_index)

def nn_results(data_list, adv_train = False, adv_test = False):
    cls_index = 1
    train_index = int(adv_train)
    test_index = int(adv_test)
    return results_by_classifier(data_list, cls_index, train_index, test_index)

def rn_results(data_list, adv_train = False, adv_test = False):
    cls_index = 2
    train_index = int(adv_train)
    test_index = int(adv_test)
    return results_by_classifier(data_list, cls_index, train_index, test_index)

# Load data
#results_list = results_by_gamma(n_hidden, eps_str, gamma_str_list, string_minus)
results_list = results_by_hidden(n_hidden_list, eps_str, gamma, string_minus)

x_axis = np.array(n_hidden_list)
ln_00 = np.mean(ln_results(results_list, 0, 0)) * np.ones(np.shape(x_axis)[0])
ln_01 = np.mean(ln_results(results_list, 0, 1)) * np.ones(np.shape(x_axis)[0])
ln_10 = np.mean(ln_results(results_list, 1, 0)) * np.ones(np.shape(x_axis)[0])
ln_11 = np.mean(ln_results(results_list, 1, 1)) * np.ones(np.shape(x_axis)[0])

nn_00 = nn_results(results_list, 0, 0)
nn_01 = nn_results(results_list, 0, 1)
nn_10 = nn_results(results_list, 1, 0)
nn_11 = nn_results(results_list, 1, 1)

rn_00 = rn_results(results_list, 0, 0)
rn_01 = rn_results(results_list, 0, 1)
rn_10 = rn_results(results_list, 1, 0)
rn_11 = rn_results(results_list, 1, 1)


# First plot: rn performs well in the adversarial setting
fig = plt.figure(figsize=(3.25, 2.75), dpi=200)
matplotlib.rcParams.update({'font.size': 10})
ax = plt.axes()
ax.plot(x_axis, ln_11, 
    color = "black", linestyle = "-", label = "linear")
ax.plot(x_axis, nn_11, 
    color = "blue", linestyle = "--", label = "neural network")
ax.plot(x_axis, rn_11, 
    color = "green", linestyle = "-.", label = "resnet")
#title = r'adver'.format(d, t)
#ax.set_title(title)
ax.set_xlabel(x_label)
ax.set_ylabel("adversarial error")
ax.set_ylim(bottom = 0.0, top = 1.1)
ax.legend()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
fig.show()
fig.savefig(filename_11)


# Second plot: adversarially-trained rn performs well even in the non-adversarial setting
fig = plt.figure(figsize=(3.25, 2.75), dpi=200)
matplotlib.rcParams.update({'font.size': 10})
ax = plt.axes()
ax.plot(x_axis, ln_10, 
    color = "black", linestyle = "-", label = "linear")
ax.plot(x_axis, nn_10, 
    color = "blue", linestyle = "--", label = "neural network")
ax.plot(x_axis, rn_10, 
    color = "green", linestyle = "-.", label = "resnet")
#title = r'adver'.format(d, t)
#ax.set_title(title)
ax.set_xlabel(x_label)
ax.set_ylabel("non-adversarial error")
ax.set_ylim(bottom = 0.0, top = 1.1)
ax.legend()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
fig.show()
fig.savefig(filename_10)

# Third/fourth plot: Everything does well in non-adversarial training and testing, 
# but not in 01 case (reg train, adv test) 
fig = plt.figure(figsize=(3.25, 2.75), dpi=200)
matplotlib.rcParams.update({'font.size': 10})
ax = plt.axes()
ax.plot(x_axis, ln_00, 
    color = "black", linestyle = "-", label = "linear")
ax.plot(x_axis, nn_00, 
    color = "blue", linestyle = "--", label = "neural network")
ax.plot(x_axis, rn_00, 
    color = "green", linestyle = "-.", label = "resnet")
#title = r'adver'.format(d, t)
#ax.set_title(title)
ax.set_xlabel(x_label)
ax.set_ylabel("non-adversarial error")
ax.legend()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
fig.show()
fig.savefig(filename_00)

# Fourth plot
fig = plt.figure(figsize=(3.25, 2.75), dpi=200)
matplotlib.rcParams.update({'font.size': 10})
ax = plt.axes()
ax.plot(x_axis, ln_01, 
    color = "black", linestyle = "-", label = "linear")
ax.plot(x_axis, nn_01, 
    color = "blue", linestyle = "--", label = "neural network")
ax.plot(x_axis, rn_01, 
    color = "green", linestyle = "-.", label = "resnet")
#title = r'adver'.format(d, t)
#ax.set_title(title)
ax.set_xlabel(x_label)
ax.set_ylabel("adversarial error")
ax.set_ylim(bottom = 0.0, top = 1.1)
ax.legend(loc = "lower right")
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
fig.show()
fig.savefig(filename_01)
