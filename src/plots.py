import numpy as np
import matplotlib.pyplot as plt
from constants import SETTINGS, PRUNING_PERCENTAGES
from fc_experiment import generate_percentages
def plot_lenet_mnist():
    #histories_reinit = np.load("data/histories_rand.npz", allow_pickle=True)['histories']
    histories = np.load("data/histories_50_iter.npz", allow_pickle=True)['histories']
    iterations = len(histories)
    # iterations = 50
    S = [0.0, 1.0, 1.0, 0.5]
    c = 1-PRUNING_PERCENTAGES[1]**(1/iterations)
    # percentages = PRUNING_PERCENTAGES
    #markers = {
    #    0: '8',
    #    1: 'o',
    #    2: 'v',
    #    3: 'x',
    #    4: 's',
    #    5: '+',
    #    6: '.'
    #}

    #colors = {
    #    0: 'b',
    #    1: 'g',
    #    2: 'r',
    #    3: 'm',
    #    4: 'k',
    #    5: 'chartreuse',
    #}
   # percentages = np.zeros(iterations+1)
   # percentages[0] = 0.513
   # percentages[1] = 0.211
   # percentages[2] = 0.07
   # percentages[3] = 0.036
   # percentages[4] = 0.019
   # percentages[5] = 1.0
    #percentages[iterations] = 1
    s = S[1]
    percentages = np.zeros(iterations)
    for i in range(0,iterations):
        s = np.around(s - s*c, decimals=10)
        percentages[i] = s
    print(len(histories))
    print(len(percentages))	
    print(iterations)
    print(percentages)
    print(range(0,iterations,10))
    for i in [0,10,20,30,40,49]:
        plt.plot(range(0,len(histories[i,:])*1000,1000), histories[i,:], label=percentages[i])
    

    #for i in range(2, iterations):
    #    plt.plot(range(0,len(histories_reinit[i,:])*1000,1000), histories_reinit[i,:], label=percentages[i], marker=markers[i], color=colors[i], linestyle='--')
    #plt.plot(range(0,len(histories_reinit[0,:])*1000,1000), histories_reinit[0,:], 
    #label=str(percentages[0]) + "(reinit)", color=colors[0], marker='8', linestyle='--')
    
    #plt.plot(range(0,len(histories_reinit[1,:])*1000,1000), histories_reinit[1,:], 
    #label=str(percentages[1]) + "(reinit)", color=colors[1], marker='o', linestyle='--')

    plt.legend()
    plt.grid()
    plt.title("Test Accuracy on LeNet-5 iterative pruning as training proceeds")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.savefig('BIG_TEST2.png')
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


plot_early_stop()


#plot_lenet_mnist()
