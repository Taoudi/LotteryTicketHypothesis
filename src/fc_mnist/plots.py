import numpy as np
import matplotlib.pyplot as plt
from constants import SETTINGS, PRUNING_PERCENTAGES
from tools import generate_percentages

def plot_figure3_replica():
    """
        Plots the replication of figure3 for conference paper: https://arxiv.org/pdf/1803.03635.pdf
    """
    histories = np.load("data/iterpr_lenet_20perc.npz", allow_pickle=True)['histories']
    #histories_reinit = np.load("data/iterpr_lenet_20perc_reinit.npz", allow_pickle=True)['histories']
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
    
    #Plot of structures: 100.0, 51.3, 21.1, 7.0, 3.6, 1.9

    plt.plot(range(0, len(histories[0])*1000, 1000), histories[len(histories)-1], label="1.0", marker=markers[0])

    for idx, i in enumerate([2, 6, 11, 14, 17]):
        plt.plot(range(0,len(histories[i])*1000,1000), histories[i], label=str(np.around(percentages[i][1], decimals=3)), marker=markers[idx+1])

    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 with different pruning rates (using iterative pruning)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

    #Plot of structures: 100.0, 51.3,

def plot_figure4c_replica():
    """
        Plots the replication of figure 4c for conference paper: https://arxiv.org/pdf/1803.03635.pdf
    """
    histories = np.load("src/data/OneShotPruningEpochs_5trials_50epochs_ES.npz", allow_pickle=True)['histories']
    histories_rand = np.load("src/data/OneShotPruningEpochs_5trials_50epochs_ES.npz", allow_pickle=True)['histories']

    print(histories)
    perc = generate_percentages([0.0,1.0,1.0,1.0],0.02)
    #print(perc)
    percentages = list()
    percentages.append(0.0)
    for i in range(0,len(histories)-1):
        percentages.append(1-np.around(perc[0][i][1], decimals=3))
    print(percentages)
    plt.plot(percentages,histories,label="Winning ticket OneShot",marker='+',color='g')
    plt.plot(percentages,histories_rand,label="Rand Init OneShot",marker='v',color='r')
    plt.legend()
    plt.grid()
    plt.title("Early Stopping iteration for different pruning %")
    plt.xlabel("Percent of Weights Remaining")
    plt.ylabel("Early Stopping criterion (val)")
    plt.savefig('Early_Stop.png')
    plt.show()

plot_figure3_replica()