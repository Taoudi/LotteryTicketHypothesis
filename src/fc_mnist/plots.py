import numpy as np
import matplotlib.pyplot as plt
from constants import SETTINGS, PRUNING_PERCENTAGES
from tools import generate_percentages, extract_final_accuracies, process_es_data
"""
    Plots the corresponding figures of conference paper https://arxiv.org/pdf/1803.03635.pdf 
    by Jonathan Frankle and Michael Carbin (2019)
"""

def plot_figure3_replica():
    histories = np.load("data/iterpr_data/iterpr_lenet_20perc.npz", allow_pickle=True)['histories']
    histories_reinit = np.load("data/iterpr_data/iterpr_lenet_20perc_reinit.npz", allow_pickle=True)['histories']
    percentages, _ = generate_percentages([0.0, 1.0, 1.0, 1.0], 0.02)
    
    markers = {
        0: '8',
        1: 'o',
        2: 'v',
        3: 'x',
        4: 's',
        5: '+',
        6: '.'
    }
    
    #Plot of structures: 100.0, 51.3, 21.1 (Figure 3i)

    plt.plot(range(0, len(histories[0])*1000, 1000), histories[len(histories)-1], label="100.0", marker=markers[0])

    for idx, i in enumerate([2, 6]):
        plt.plot(range(0,len(histories[i])*1000,1000), histories[i], 
        label=str(np.around(percentages[i][1]*100, decimals=3)), marker=markers[idx+1])
    
    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 with different pruning rates (using iterative pruning)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

    #Plot of structures: 100.0, 51.3, 21.1, 7.0, 3.6, 1.9 (Figure 3ii)

    plt.plot(range(0, len(histories[0])*1000, 1000), histories[len(histories)-1], label="100.0", marker=markers[0])

    for idx, i in enumerate([2, 6, 11, 14, 17]):
        plt.plot(range(0,len(histories[i])*1000,1000), histories[i], 
        label=str(np.around(percentages[i][1]*100, decimals=3)), marker=markers[idx+1])

    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 with different pruning rates (using iterative pruning)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

    #Plot of structures: 100.0, 51.3, 21.1, 51.3 (reinit), 21.1 (reinit) (Figure 3iii)

    plt.plot(range(0, len(histories[0])*1000, 1000), histories[len(histories)-1], label="100.0", marker=markers[0])

    for idx, i in enumerate([2, 6]):
        plt.plot(range(0,len(histories[i])*1000,1000), histories[i], 
        label=str(np.around(percentages[i][1]*100, decimals=3)), marker=markers[idx+1])

    for idx, i in enumerate([2, 6]):
        plt.plot(range(0, len(histories[i])*1000, 1000), histories_reinit[i],
        label=str(np.around(percentages[i][1]*100, decimals=3)) + "(reinit)", marker=markers[idx+1], linestyle='dashed')
    
    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 with different pruning rates (using iterative pruning)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

def plot_figure4a_replica():
    hist_iter = np.load("data/iterpr_data/iterpr_lenet_20perc_es.npz", allow_pickle=True)['histories']
    hist_iter_reinit = np.load("data/iterpr_data/iterpr_lenet_20perc_reinit_es.npz", allow_pickle=True)['histories']
    hist_iter_epochs = np.load("data/iterpr_data/iterpr_lenet_20perc_es.npz", allow_pickle=True)['es_epochs']
    hist_iter_reinit_epochs = np.load("data/iterpr_data/iterpr_lenet_20perc_reinit_es.npz", allow_pickle=True)['es_epochs']

    hist_os = np.load("data/ospr_data/OneShotPruningAcc_5trials_50epochs_20perc_ES.npz", allow_pickle=True)['histories']
    hist_os_reinit = np.load("data/ospr_data/OneShotPruningAcc_5trials_50epochs_20perc_ES_rand.npz", allow_pickle=True)['histories']
    hist_os_epochs = np.load("data/ospr_data/OneShotPruningEpochs_5trials_50epochs_ES.npz", allow_pickle=True)['histories']
    hist_os_reinit_epochs = np.load("data/ospr_data/OneShotPruningEpochs_5trials_50epochs_ES_rand.npz", allow_pickle=True)['histories']
    
    percentages, _ = generate_percentages([0.0, 1.0, 1.0, 1.0], 0.02)

    percentages_list = list()
    percentages_list.append(100.0)
    for i in range(len(percentages)):
        percentages_list.append(percentages[i][1]*100)
    
    hist_iter, hist_iter_reinit, hist_iter_epochs, hist_iter_reinit_epochs = process_es_data(hist_iter, 
    hist_iter_reinit, hist_iter_epochs, hist_iter_reinit_epochs)

    # Plots Early-Stop iteration (Figure 4ai)
    plt.plot(percentages_list, hist_iter_epochs, label="Winning Ticket (Iterative)", color='b', marker='v')
    plt.plot(percentages_list, hist_iter_reinit_epochs, label="Random Reinit (Iterative)", color='k', marker='.', linestyle='--')
    plt.plot(percentages_list, hist_os_epochs, label="Winning Ticket (Oneshot)", color='g', marker='+')
    plt.plot(percentages_list, hist_os_reinit_epochs, label="Random Reinit (Oneshot)", color='r', marker='x', linestyle='--')
    
    plt.legend()
    plt.title("Early stopping iteration for all pruning methods")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Early-Stop Epoch (Val.)")

    plt.xlim(left=100.5, right=1.5)
    plt.xscale('log')
    plt.grid()
    plt.xticks([100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75], [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75])
    plt.show()
    # Plots accuracy at early-stop (Figure 4aii)
    plt.plot(percentages_list, hist_iter, label="Winning Ticket (Iterative)", color='b', marker='v')
    plt.plot(percentages_list, hist_iter_reinit, label="Random Reinit (Iterative)", color='k', marker='.', linestyle='--')
    plt.plot(percentages_list, hist_os, label="Winning Ticket (Oneshot)", color='g', marker='+')
    plt.plot(percentages_list, hist_os_reinit, label="Random Reinit (Oneshot)", color='r', marker='x', linestyle='--')

    plt.legend()
    plt.title("Accuracy at Early-stop for all pruning methods")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Accuracy at Early-Stop (Test)")

    plt.xlim(left=100.5, right=1.5)
    plt.xscale('log')
    plt.grid()
    plt.xticks([100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75], [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75])
    plt.show()

def plot_figure4b_replica():
    hist_iter = np.load("data/iterpr_data/iterpr_lenet_20perc.npz", allow_pickle=True)['histories']
    hist_iter_reinit = np.load("data/iterpr_data/iterpr_lenet_20perc_reinit.npz", allow_pickle=True)['histories']
    hist_os = np.load("data/ospr_data/OneShotPruningAcc_5trials_20epochs_20percOut.npz", allow_pickle=True)['histories']
    hist_os_reinit = np.load("data/ospr_data/os_acc_reinit.npz", allow_pickle=True)['histories']

    percentages, _ = generate_percentages([0.0, 1.0, 1.0, 1.0], 0.02)
    
    hist_iter = extract_final_accuracies(hist_iter)
    hist_iter_reinit = extract_final_accuracies(hist_iter_reinit)

    percentages_list = list()
    percentages_list.append(100.0)
    for i in range(len(percentages)):
        percentages_list.append(percentages[i][1]*100)
    
    plt.plot(percentages_list, hist_iter, label="Winning Ticket (Iterative)", color='b', marker='v')
    plt.plot(percentages_list, hist_iter_reinit, label="Random Reinit (Iterative)", color='k', marker='.', linestyle='--')
    plt.plot(percentages_list, hist_os, label="Winning Ticket (Oneshot)", color='g', marker='+')
    plt.plot(percentages_list, hist_os_reinit, label="Random Reinit (Oneshot)", color='r', marker='x', linestyle='--')
    
    plt.legend()
    plt.title("Accuracy at the end of training for different pruning methods")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Accuracy at Iteration 20K (Test)")

    plt.xlim(left=100.5, right=1.5)
    plt.xscale('log')
    plt.grid()
    plt.xticks([100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75], [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75])
    plt.show()

def plot_figure4c_replica():
    hist_os = np.load("data/ospr_data/OneShotPruningAcc_5trials_50epochs_20perc_ES.npz", allow_pickle=True)['histories']
    hist_os_reinit = np.load("data/ospr_data/OneShotPruningAcc_5trials_50epochs_20perc_ES_rand.npz", allow_pickle=True)['histories']
    hist_os_epochs = np.load("data/ospr_data/OneShotPruningEpochs_5trials_50epochs_ES.npz", allow_pickle=True)['histories']
    hist_os_reinit_epochs = np.load("data/ospr_data/OneShotPruningEpochs_5trials_50epochs_ES_rand.npz", allow_pickle=True)['histories']
    
    percentages, _ = generate_percentages([0.0, 1.0, 1.0, 1.0], 0.02)

    percentages_list = list()
    percentages_list.append(100.0)
    for i in range(len(percentages)):
        percentages_list.append(percentages[i][1]*100)

    # Plots Early-Stop iteration (Figure 4ci)

    plt.plot(percentages_list, hist_os_epochs, label="Winning Ticket (Oneshot)", color='g', marker='+')
    plt.plot(percentages_list, hist_os_reinit_epochs, label="Random Reinit (Oneshot)", color='r', marker='x', linestyle='--')
    
    plt.legend()
    plt.title("Early stopping iteration for oneshot pruning")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Early-Stop Epoch (Val.)")

    plt.xlim(left=100.5, right=1.5)
    plt.xscale('log')
    plt.grid()
    plt.xticks([100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75], [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75])
    plt.show()

    # Plots accuracy at early-stop (Figure 4cii)

    plt.plot(percentages_list, hist_os, label="Winning Ticket (Oneshot)", color='g', marker='+')
    plt.plot(percentages_list, hist_os_reinit, label="Random Reinit (Oneshot)", color='r', marker='x', linestyle='--')

    plt.legend()
    plt.title("Accuracy at Early-stop for oneshot pruning")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Accuracy at Early-Stop (Test)")

    plt.xlim(left=100.5, right=1.5)
    plt.xscale('log')
    plt.grid()
    plt.xticks([100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75], [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.75])
    plt.show()

plot_figure4c_replica()