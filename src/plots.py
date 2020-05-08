import numpy as np
import matplotlib.pyplot as plt
from constants import SETTINGS, PRUNING_PERCENTAGES

def plot_lenet_mnist():
    histories_reinit = np.load("data/histories_rand.npz", allow_pickle=True)['histories']
    histories = np.load("data/iterpr_lenet_trainiter.npz", allow_pickle=True)['histories']
    iterations = len(histories)-1
    iterations = 5
    S = [0.0, 1.0, 1.0, 0.5]
    c = 1-PRUNING_PERCENTAGES[1]**(1/iterations)

    markers = {
        0: '8',
        1: 'o',
        2: 'v',
        3: 'x',
        4: 's',
        5: '+',
        6: '.'
    }

    colors = {
        0: 'b',
        1: 'g',
        2: 'r',
        3: 'm',
        4: 'k',
        5: 'chartreuse',
    }
    percentages = np.zeros(iterations+1)
    percentages[0] = 0.513
    percentages[1] = 0.211
    percentages[2] = 0.07
    percentages[3] = 0.036
    percentages[4] = 0.019
    percentages[5] = 1.0
    #percentages[iterations] = 1
    #s = S[1]
    #for i in range(0,iterations):
        #s = np.around(s - s*c, decimals=3)
        #percentages[i] = s

    for i in range(0, iterations):
        plt.plot(range(0,len(histories[i,:])*1000,1000), histories[i,:], label=percentages[i], marker=markers[i], color=colors[i])
    

    for i in range(0, iterations):
        plt.plot(range(0,len(histories_reinit[i,:])*1000,1000), histories_reinit[i,:], label=percentages[i], marker=markers[i], color=colors[i], linestyle='--')
    #plt.plot(range(0,len(histories_reinit[0,:])*1000,1000), histories_reinit[0,:], 
    #label=str(percentages[0]) + "(reinit)", color=colors[0], marker='8', linestyle='--')
    
    #plt.plot(range(0,len(histories_reinit[1,:])*1000,1000), histories_reinit[1,:], 
    #label=str(percentages[1]) + "(reinit)", color=colors[1], marker='o', linestyle='--')

    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 iterative pruning as training proceeds")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.savefig('test.png')
    plt.show()

plot_lenet_mnist()
