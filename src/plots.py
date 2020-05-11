import numpy as np
import matplotlib.pyplot as plt
from constants import SETTINGS, PRUNING_PERCENTAGES
from tools import generate_percentages

def plot_figure3_replica():
    histories = np.load("data/iterpr_lenet_20perc.npz", allow_pickle=True)['histories']
    #histories_reinit = np.load("data/iterpr_lenet_20perc_reinit.npz", allow_pickle=True)['histories']
    percentages, _ = generate_percentages([0.0, 1.0, 1.0, 1.0], 0.02)
    """
    markers = {
        0: '8',
        1: 'o',
        2: 'v',
        3: 'x',
        4: 's',
        5: '+',
        6: '.'
    }
    """

    plt.plot(range(0, len(histories[0])*1000, 1000), histories[len(histories)-1], label="1.0")

    for i in [0, 1, 2, 3, 4, 5, 6, 10, 11, 13, 14, 16, 17]:
        plt.plot(range(0,len(histories[i])*1000,1000), histories[i], label=str(np.around(percentages[i][1], decimals=3)))

    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 with different pruning rates (using iterative pruning)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

def plot_figure4c_replica():

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
